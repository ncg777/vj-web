import assert from "node:assert/strict";
import { test } from "node:test";
import { once } from "node:events";
import { readFile, readdir } from "node:fs/promises";
import { WebSocket } from "ws";
import { startServer, waitForExport } from "./helpers.mjs";

test("current MCP protocol exposes the full catalog and validates tool arguments", async (t) => {
  const { client, call, connection } = await startServer(t);
  const { tools } = await client.listTools();
  assert.equal(tools.length, 11);
  assert.ok(tools.some((tool) => tool.name === "export_video"));
  const scenes = await call("list_scenes");
  assert.ok(scenes.length >= 40);
  assert.equal(new Set(scenes.map((scene) => scene.id)).size, scenes.length);
  const smoke = await call("list_scenes", { query: "rainbow plumes" });
  assert.equal(smoke[0].id, "smoke-rainbow-plumes");
  const scene = await call("get_scene", { sceneId: "neon" });
  assert.equal(scene.parameters.find((param) => param.id === "components").max, 64);
  const resource = await client.readResource({ uri: "vj-web://scenes" });
  assert.equal(JSON.parse(resource.contents[0].text).length, scenes.length);
  for (const [name, args] of [
    ["get_scene", { sceneId: "missing" }],
    ["set_parameters", { sceneId: "neon", values: { components: 65 } }],
    ["set_parameters", { sceneId: "neon", values: { components: 1.5 } }],
    ["set_parameters", { sceneId: "neon", values: { seed: -1 } }],
    ["set_parameters", { sceneId: "neon", values: { constructor: 1 } }],
    ["export_video", { width: 321 }],
    ["get_state", {}],
  ]) {
    const result = await client.callTool({ name, arguments: args });
    assert.equal(result.isError, true, name);
  }
  assert.equal(connection.connected, false);
  const origin = new URL(connection.url).origin;
  assert.equal((await fetch(origin)).status, 200);
  assert.equal((await fetch(`${origin}/package.json`)).status, 404);
  assert.equal((await fetch(`${origin}/exports/test?offset=0`, { method: "PUT", body: "test" })).status, 403);
});

test("legacy MCP clients can discover and call tools", async (t) => {
  const { call } = await startServer(t, "legacy");
  assert.equal((await call("get_scene", { sceneId: "neon" })).id, "neon");
});

test("browser bridge authorizes connections and handles streamed export success, cancellation and disconnect", async (t) => {
  const { call, client, connection, directory } = await startServer(t);
  const origin = new URL(connection.url).origin;
  const token = new URLSearchParams(new URL(connection.url).hash.slice(1)).get("mcp");
  const wsUrl = `${origin.replace("http:", "ws:")}/control?token=${token}`;
  for (const [url, originHeader] of [[wsUrl + "bad", origin], [wsUrl, "https://example.com"]]) {
    const bad = new WebSocket(url, { origin: originHeader });
    const [error] = await once(bad, "error");
    assert.match(error.message, /403/);
  }
  const socket = new WebSocket(wsUrl, { origin });
  t.after(() => socket.terminate());
  await once(socket, "open");
  assert.equal((await call("get_connection")).connected, true);
  const second = new WebSocket(wsUrl, { origin });
  assert.equal((await once(second, "close"))[0], 1008);
  let exportRequest;
  socket.on("message", (raw) => {
    const request = JSON.parse(raw);
    if (request.command === "export_video") exportRequest = request;
    else {
      socket.send(JSON.stringify({ id: request.id, result: { sceneId: request.args.sceneId ?? "neon" } }));
      if (request.command === "cancel_export" && exportRequest) {
        socket.send(JSON.stringify({ id: exportRequest.id, error: "Export cancelled." }));
      }
    }
  });
  assert.equal((await call("select_scene", { sceneId: "neon" })).sceneId, "neon");
  const start = async () => {
    exportRequest = undefined;
    const job = await call("export_video", { duration: 1, fps: 30, width: 320, height: 180 });
    while (!exportRequest) await new Promise((resolve) => setTimeout(resolve, 5));
    return job;
  };
  const job = await start();
  const headers = { Origin: origin, Authorization: `Bearer ${token}` };
  const upload = (offset, body) => fetch(`${origin}/exports/${job.id}?offset=${offset}`, { method: "PUT", headers, body });
  assert.equal((await upload(100, "bad")).status, 400);
  assert.equal((await upload(0, "hello-video")).status, 204);
  assert.equal((await upload(0, "HELLO")).status, 204);
  socket.send(JSON.stringify({ event: "progress", jobId: job.id, completed: 30, total: 30 }));
  socket.send(JSON.stringify({ id: exportRequest.id, result: {} }));
  const complete = await waitForExport(call, job.id);
  assert.equal(complete.status, "completed");
  assert.equal(await readFile(complete.path, "utf8"), "HELLO-video");
  const cancel = await start();
  const duplicate = await client.callTool({ name: "export_video", arguments: {} });
  assert.equal(duplicate.isError, true);
  await call("cancel_export", { jobId: cancel.id });
  assert.equal((await waitForExport(call, cancel.id)).status, "cancelled");
  const disconnected = await start();
  socket.close();
  assert.equal((await waitForExport(call, disconnected.id)).status, "failed");
  assert.equal((await readdir(directory)).filter((name) => name.endsWith(".part")).length, 0);
  await client.close();
  await assert.rejects(fetch(origin));
});
