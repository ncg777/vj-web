import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile, readdir } from "node:fs/promises";
import { chromium } from "playwright";
import { startServer, waitForExport } from "./helpers.mjs";

test("real browser controls scenes and exports a playable MP4 without a save dialog", { timeout: 120_000 }, async (t) => {
  const { client, call, connection, directory } = await startServer(t);
  const browser = await chromium.launch({
    channel: process.env.VJ_BROWSER_CHANNEL || (process.platform === "win32" ? "msedge" : undefined),
    headless: true,
    args: ["--enable-unsafe-swiftshader"],
  });
  t.after(() => browser.close());
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(connection.url);
  await page.getByText("MCP: connected", { exact: true }).waitFor();
  await call("select_scene", { sceneId: "neon" });
  await call("set_parameters", { sceneId: "neon", values: { components: 4, seed: 123 } });
  assert.equal((await call("get_state")).values.components, 4);
  assert.equal(await page.locator("#scene-list").inputValue(), "neon");
  const invalid = await client.callTool({ name: "set_parameters", arguments: { sceneId: "neon", values: { components: 10, isoBands: 500 } } });
  assert.equal(invalid.isError, true);
  assert.equal((await call("get_state")).values.components, 4);
  const job = await call("export_video", { duration: 2, fps: 60, width: 1280, height: 720 });
  const video = await waitForExport(call, job.id);
  assert.equal(video.status, "completed", JSON.stringify(video));
  assert.ok(video.bytes > 1024 * 1024, "Exercise multiple streamed video chunks");
  const data = await readFile(video.path);
  assert.equal(data.toString("ascii", 4, 8), "ftyp");
  const metadata = await page.evaluate(async (base64) => {
    const element = document.createElement("video");
    const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "video/mp4" }));
    try {
      element.src = url;
      await new Promise((resolve, reject) => {
        element.onloadeddata = resolve;
        element.onerror = () => reject(new Error("Exported MP4 cannot be decoded."));
      });
      return { width: element.videoWidth, height: element.videoHeight, duration: element.duration };
    } finally { URL.revokeObjectURL(url); }
  }, data.toString("base64"));
  assert.deepEqual(metadata, { width: 1280, height: 720, duration: 2 });
  assert.equal((await call("get_state")).exporting, false);
  await call("reset_scene", { sceneId: "neon" });
  assert.equal((await call("get_state")).values.components, 64);
  const cancel = await call("export_video", { duration: 60, fps: 30, width: 320, height: 180 });
  const locked = await client.callTool({ name: "select_scene", arguments: { sceneId: "thunderstorm" } });
  assert.equal(locked.isError, true);
  await call("cancel_export", { jobId: cancel.id });
  assert.equal((await waitForExport(call, cancel.id)).status, "cancelled");
  assert.equal((await call("get_state")).exporting, false);
  assert.equal((await readdir(directory)).filter((name) => name.endsWith(".part")).length, 0);
  assert.deepEqual(errors, []);
});
