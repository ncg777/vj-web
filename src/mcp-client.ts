import type { VideoSink } from "./video-export";

export type BrowserCommand = {
  id: string;
  command: string;
  args: Record<string, unknown>;
};

export type BrowserConnection = {
  sink: (jobId: string) => VideoSink;
  progress: (jobId: string, completed: number, total: number) => void;
  signal: AbortSignal;
};

// Only an explicit connection URL on the local MCP server enables remote control.
export function connectMcp(handle: (command: BrowserCommand, connection: BrowserConnection) => Promise<unknown>) {
  const token = new URLSearchParams(location.hash.slice(1)).get("mcp");
  if (!token || location.hostname !== "127.0.0.1") return;
  const status = document.createElement("div");
  status.className = "panel-section small";
  status.textContent = "MCP: connecting…";
  document.querySelector(".panel")?.prepend(status);
  let stopped = false;
  let socket: WebSocket;
  let reconnect: number | undefined;
  const open = () => {
    const controller = new AbortController();
    const url = new URL("/control", location.href);
    url.protocol = "ws:";
    url.searchParams.set("token", token);
    socket = new WebSocket(url);
    const send = (message: unknown) => {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
    };
    const connection: BrowserConnection = {
      signal: controller.signal,
      progress: (jobId, completed, total) => send({ event: "progress", jobId, completed, total }),
      sink: (jobId) => ({
        async write(data, position) {
          const response = await fetch(`/exports/${encodeURIComponent(jobId)}?offset=${position}`, {
            method: "PUT",
            headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/octet-stream" },
            body: data.slice().buffer as ArrayBuffer,
            signal: controller.signal,
          });
          if (!response.ok) throw new Error(`Video upload failed: ${await response.text()}`);
        },
        close: async () => {},
        abort: async () => {},
      }),
    };
    socket.addEventListener("open", () => { status.textContent = "MCP: connected"; });
    socket.addEventListener("message", async (event) => {
      let command: BrowserCommand;
      try { command = JSON.parse(String(event.data)); } catch { return; }
      if (typeof command.id !== "string" || typeof command.command !== "string") return;
      try { send({ id: command.id, result: await handle(command, connection) }); }
      catch (error) { send({ id: command.id, error: error instanceof Error ? error.message : String(error) }); }
    });
    socket.addEventListener("close", (event) => {
      controller.abort(new Error("MCP browser connection closed."));
      status.textContent = event.code === 1008 ? "MCP: another tab is already connected" : "MCP: disconnected";
      if (!stopped && event.code !== 1008) reconnect = window.setTimeout(open, 2000);
    });
    socket.addEventListener("error", () => { status.textContent = "MCP: connection unavailable"; });
  };
  open();
  window.addEventListener("pagehide", () => {
    stopped = true;
    clearTimeout(reconnect);
    socket.close();
  }, { once: true });
}
