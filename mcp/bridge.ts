import { randomBytes, randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdir, open, readFile, rename, rm, type FileHandle } from "node:fs/promises";
import { join } from "node:path";
import { WebSocket, WebSocketServer } from "ws";
import type { VideoOptions } from "../src/control";

type ExportStatus = "running" | "cancelling" | "completed" | "failed" | "cancelled";
type ExportJob = {
  id: string;
  status: ExportStatus;
  completedFrames: number;
  totalFrames: number;
  path: string;
  bytes: number;
  error?: string;
};
type InternalJob = ExportJob & { file?: FileHandle; writes: Promise<void> };
type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void; timer: NodeJS.Timeout };

export class BrowserBridge {
  private readonly token = randomBytes(32).toString("hex");
  private readonly sockets = new WebSocketServer({ noServer: true, maxPayload: 1024 * 1024 });
  private readonly http = createServer((req, res) => {
    void this.handleHttp(req, res).catch(() => {
      if (!res.headersSent) res.writeHead(500);
      res.end("Request failed.");
    });
  });
  private client?: WebSocket;
  private readonly pending = new Map<string, Pending>();
  private readonly jobs = new Map<string, InternalJob>();
  private readonly tasks = new Set<Promise<void>>();
  private origin = "";

  constructor(private readonly webRoot: string, private readonly exportRoot: string) {}

  async listen(port = 0) {
    this.http.on("upgrade", (req, socket, head) => {
      let url: URL;
      try { url = new URL(req.url ?? "/", this.origin); }
      catch {
        socket.end("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
        return;
      }
      if (req.headers.host !== new URL(this.origin).host || req.headers.origin !== this.origin ||
          url.pathname !== "/control" || url.searchParams.get("token") !== this.token) {
        socket.end("HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n");
        return;
      }
      this.sockets.handleUpgrade(req, socket, head, (client) => {
        if (this.client) {
          client.close(1008, "Another browser tab is already connected.");
          return;
        }
        this.client = client;
        client.on("message", (data) => {
          try {
            const message = JSON.parse(data.toString());
            if (message.event === "progress") {
              const job = this.jobs.get(message.jobId);
              if (job?.status === "running" && Number.isInteger(message.completed) &&
                  message.completed >= job.completedFrames && message.completed <= job.totalFrames) {
                job.completedFrames = message.completed;
              }
              return;
            }
            const pending = this.pending.get(message.id);
            if (!pending) return;
            clearTimeout(pending.timer);
            this.pending.delete(message.id);
            if (typeof message.error === "string") pending.reject(new Error(message.error));
            else pending.resolve(message.result);
          } catch { /* Ignore malformed messages without disturbing pending requests. */ }
        });
        client.on("error", () => client.terminate());
        client.on("close", () => {
          if (this.client !== client) return;
          this.client = undefined;
          this.rejectPending("Browser disconnected. Reopen the connection URL.");
        });
      });
    });
    await new Promise<void>((resolve, reject) => {
      this.http.once("error", reject);
      this.http.listen(port, "127.0.0.1", () => {
        this.http.off("error", reject);
        resolve();
      });
    });
    const address = this.http.address();
    if (!address || typeof address === "string") throw new Error("Could not start browser bridge.");
    this.origin = `http://127.0.0.1:${address.port}`;
  }

  get connection() {
    return {
      connected: this.client?.readyState === WebSocket.OPEN,
      url: `${this.origin}/#mcp=${this.token}`,
      exportDirectory: this.exportRoot,
      instructions: "Open this URL in Chrome or Edge. Keep that tab open for controls and video exports. Only one tab connects at a time.",
    };
  }

  request(command: string, args: Record<string, unknown> = {}, timeout = 15_000): Promise<unknown> {
    const client = this.client;
    if (!client || client.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("No browser connected. Call get_connection and open its URL in Chrome or Edge."));
    }
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error("Browser request timed out. Its outcome is unknown; check the browser state before retrying."));
      }, timeout);
      this.pending.set(id, { resolve, reject, timer });
      client.send(JSON.stringify({ id, command, args }), (error) => {
        if (!error) return;
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      });
    });
  }

  async startExport(options: VideoOptions) {
    if (!this.connection.connected) throw new Error("Connect a browser using get_connection before exporting.");
    if ([...this.jobs.values()].some((job) => ["running", "cancelling"].includes(job.status))) {
      throw new Error("A video export is already running. Check or cancel it before starting another.");
    }
    const id = randomUUID();
    const job: InternalJob = {
      id, status: "running", completedFrames: 0, totalFrames: Math.ceil(options.duration * options.fps),
      path: join(this.exportRoot, `vj-${id}.mp4`), bytes: 0, writes: Promise.resolve(),
    };
    // Reserve the export slot before awaiting filesystem operations.
    this.jobs.set(id, job);
    try {
      await mkdir(this.exportRoot, { recursive: true });
      job.file = await open(`${job.path}.part`, "wx");
    } catch (error) {
      job.status = "failed";
      job.error = String(error);
      throw error;
    }
    const task = (async () => {
      try {
        await this.request("export_video", { ...options, jobId: id }, 12 * 60 * 60 * 1000);
        await job.writes;
        await job.file?.close();
        job.file = undefined;
        if (job.status === "cancelling") throw new Error("Export cancelled.");
        if (!job.bytes || job.completedFrames !== job.totalFrames) throw new Error("Browser did not finish writing all video frames.");
        await rename(`${job.path}.part`, job.path);
        job.status = "completed";
      } catch (error) {
        await job.writes.catch(() => {});
        await job.file?.close().catch(() => {});
        job.file = undefined;
        await rm(`${job.path}.part`, { force: true }).catch(() => {});
        job.error = error instanceof Error ? error.message : String(error);
        job.status = job.status === "cancelling" ? "cancelled" : "failed";
      }
    })();
    this.tasks.add(task);
    void task.finally(() => this.tasks.delete(task));
    return this.exportStatus(id);
  }

  exportStatus(id: string): ExportJob {
    const job = this.jobs.get(id);
    if (!job) throw new Error(`Unknown export job: ${id}`);
    return {
      id: job.id, status: job.status, completedFrames: job.completedFrames, totalFrames: job.totalFrames,
      path: job.path, bytes: job.bytes, ...(job.error ? { error: job.error } : {}),
    };
  }

  async cancelExport(id: string) {
    const job = this.jobs.get(id);
    if (!job) throw new Error(`Unknown export job: ${id}`);
    if (["running", "cancelling"].includes(job.status)) {
      job.status = "cancelling";
      await this.request("cancel_export", { jobId: id });
    }
    return this.exportStatus(id);
  }

  private async handleHttp(req: IncomingMessage, res: ServerResponse) {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("Cache-Control", "no-store");
    if (!this.origin || req.headers.host !== new URL(this.origin).host) {
      res.writeHead(403).end("Invalid host.");
      return;
    }
    const url = new URL(req.url ?? "/", this.origin);
    if (req.method === "PUT" && url.pathname.startsWith("/exports/")) {
      if (req.headers.origin !== this.origin || req.headers.authorization !== `Bearer ${this.token}`) {
        res.writeHead(403).end("Unauthorized.");
        return;
      }
      const job = this.jobs.get(url.pathname.slice("/exports/".length));
      if (!job || job.status !== "running" || !job.file) {
        res.writeHead(409).end("Export is not running.");
        return;
      }
      const offset = Number(url.searchParams.get("offset"));
      if (!url.searchParams.has("offset") || !Number.isSafeInteger(offset) || offset < 0 || offset > job.bytes) {
        res.writeHead(400).end("Invalid video offset.");
        return;
      }
      const chunks: Buffer[] = [];
      let length = 0;
      for await (const chunk of req) {
        length += chunk.length;
        if (length > 8 * 1024 * 1024) {
          res.writeHead(413).end("Video chunk is too large.");
          return;
        }
        chunks.push(chunk);
      }
      const data = Buffer.concat(chunks);
      job.writes = job.writes.then(async () => {
        if (!job.file || job.status !== "running") throw new Error("Export stopped.");
        let written = 0;
        while (written < data.length) {
          const result = await job.file.write(data, written, data.length - written, offset + written);
          if (!result.bytesWritten) throw new Error("Could not write video data.");
          written += result.bytesWritten;
        }
        job.bytes = Math.max(job.bytes, offset + written);
      });
      await job.writes;
      res.writeHead(204).end();
      return;
    }
    if (!["GET", "HEAD"].includes(req.method ?? "")) {
      res.writeHead(405).end("Method not allowed.");
      return;
    }
    const path = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    if (path !== "index.html" && !/^assets\/[\w.-]+\.(js|css)$/.test(path)) {
      res.writeHead(404).end("Not found.");
      return;
    }
    try {
      const data = await readFile(join(this.webRoot, path));
      res.setHeader("Content-Type", path.endsWith(".js") ? "text/javascript" : path.endsWith(".css") ? "text/css" : "text/html");
      res.writeHead(200).end(req.method === "HEAD" ? undefined : data);
    } catch {
      res.writeHead(404).end("App build missing. Run npm run build:mcp.");
    }
  }

  private rejectPending(message: string) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error(message));
    }
    this.pending.clear();
  }

  async close() {
    this.rejectPending("MCP server stopped.");
    for (const client of this.sockets.clients) client.terminate();
    await Promise.allSettled(this.tasks);
    this.sockets.close();
    this.http.closeAllConnections();
    await new Promise<void>((resolve) => this.http.close(() => resolve()));
  }
}
