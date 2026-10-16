import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { BrowserBridge, createMcpServer } from "./server";

const port = Number(process.env.VJ_MCP_PORT ?? 0);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error("VJ_MCP_PORT must be 0–65535.");
const bridge = new BrowserBridge(
  fileURLToPath(new URL("../web/", import.meta.url)),
  process.env.VJ_EXPORT_DIR ? resolve(process.env.VJ_EXPORT_DIR) : fileURLToPath(new URL("../../exports/", import.meta.url)),
);
await bridge.listen(port);
console.error(`vj-web MCP ready. Open ${bridge.connection.url}`);
console.error(`Video exports: ${bridge.connection.exportDirectory}`);
const handle = serveStdio(() => createMcpServer(bridge));
let closing = false;
async function close() {
  if (closing) return;
  closing = true;
  await bridge.close();
  await handle.close();
}
process.stdin.once("end", () => { void close(); });
process.once("SIGINT", () => { void close(); });
process.once("SIGTERM", () => { void close(); });
