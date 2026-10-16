import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";

export async function startServer(t, mode = { pin: "2026-07-28" }) {
  const directory = await mkdtemp(join(tmpdir(), "vj-mcp-test-"));
  const client = new Client({ name: "vj-web-test", version: "1.0.0" }, { versionNegotiation: { mode } });
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [fileURLToPath(new URL("../dist-mcp/server/index.js", import.meta.url))],
    cwd: directory,
    env: { VJ_EXPORT_DIR: directory, VJ_MCP_PORT: "0" },
    stderr: "pipe",
  });
  let stderr = "";
  transport.stderr?.on("data", (data) => { stderr += data; });
  t.after(async () => {
    await client.close();
    const target = resolve(directory);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes("vj-mcp-test-"));
    await rm(target, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  });
  try { await client.connect(transport); }
  catch (error) { throw new Error(`${error}\n${stderr}`); }
  const call = async (name, args = {}) => {
    const result = await client.callTool({ name, arguments: args });
    assert.ok(!result.isError, JSON.stringify(result));
    return JSON.parse(result.content[0].text);
  };
  const connection = await call("get_connection");
  return { client, call, connection, directory };
}

export async function waitForExport(call, id, timeout = 60_000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) {
    const status = await call("get_export_status", { jobId: id });
    if (!["running", "cancelling"].includes(status.status)) return status;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Export ${id} did not finish within ${timeout}ms`);
}
