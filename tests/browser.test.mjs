import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { chromium } from "playwright";
import { startServer, waitForExport } from "./helpers.mjs";

test("interference scenes render, animate, freeze and respond to phase and seed controls", { timeout: 120_000 }, async (t) => {
  const { call, connection } = await startServer(t);
  const browser = await chromium.launch({
    channel: process.env.VJ_BROWSER_CHANNEL || (process.platform === "win32" ? "msedge" : undefined),
    headless: true, args: ["--enable-unsafe-swiftshader"],
  });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 960, height: 640 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(connection.url);
  await page.getByText("MCP: connected", { exact: true }).waitFor();
  const scenes = await call("list_scenes", { query: "interference" });
  assert.equal(scenes.length, 4);
  const capture = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => {
    // Sample immediately after the app's draw, before WebGL clears the drawing buffer.
    const canvas = document.createElement("canvas");
    canvas.width = 160; canvas.height = 120;
    const context = canvas.getContext("2d");
    context.drawImage(document.querySelector("#gl-canvas"), 0, 0, 160, 120);
    resolve(Array.from(context.getImageData(0, 0, 160, 120).data).filter((_, i) => i % 4 !== 3));
  })));
  const difference = (a, b) => a.reduce((sum, value, i) => sum + Math.abs(value - b[i]), 0) / a.length;
  const fingerprints = [];
  for (const { id: sceneId, requiresFloatTargets } of scenes) {
    assert.equal(requiresFloatTargets, false);
    // Exercise scene discovery in the picker as well as the remote control path.
    await page.locator("#scene-list").selectOption(sceneId);
    assert.equal((await call("get_state")).sceneId, sceneId);
    await call("set_parameters", { sceneId, values: { speed: 0, seed: 123 } });
    const frozen = await capture();
    assert.ok(Math.max(...frozen) - Math.min(...frozen) > 60, `${sceneId} must contain visible fringes`);
    assert.deepEqual(await capture(), frozen, `${sceneId} must freeze at speed zero`);
    fingerprints.push(frozen);
    if (process.env.VJ_INTERFERENCE_PREVIEW_DIR) {
      const data = await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => {
        resolve(document.querySelector("#gl-canvas").toDataURL("image/png").split(",")[1]);
      })));
      await writeFile(join(process.env.VJ_INTERFERENCE_PREVIEW_DIR, `${sceneId}.png`), Buffer.from(data, "base64"));
    }
    await call("set_parameters", { sceneId, values: { phase: 1.7 } });
    assert.ok(difference(frozen, await capture()) > 1, `${sceneId} must respond to relative phase`);
    await call("set_parameters", { sceneId, values: { phase: 0, seed: 456 } });
    assert.ok(difference(frozen, await capture()) > 1, `${sceneId} must respond to reseeding`);
    await call("set_parameters", { sceneId, values: { seed: 123 } });
    assert.deepEqual(await capture(), frozen, `${sceneId} must reproduce a seed`);
    await call("set_parameters", { sceneId, values: { speed: 1 } });
    const moving = await capture();
    await page.evaluate(() => new Promise(resolve => {
      let count = 0;
      const tick = () => ++count === 6 ? resolve() : requestAnimationFrame(tick);
      requestAnimationFrame(tick);
    }));
    assert.ok(difference(moving, await capture()) > 0.1, `${sceneId} must animate`);
    await page.getByRole("button", { name: "Reset", exact: true }).click();
    assert.equal((await call("get_state")).values.phase, 0);
    assert.equal(await page.evaluate(() => document.querySelector("#gl-canvas").getContext("webgl2").getError()), 0);
  }
  for (let i = 0; i < fingerprints.length; i++) {
    for (let j = i + 1; j < fingerprints.length; j++) {
      assert.ok(difference(fingerprints[i], fingerprints[j]) > 5, "Each interference scene must look distinct");
    }
  }
  const job = await call("export_video", { duration: 1, fps: 30, width: 320, height: 180 });
  const result = await waitForExport(call, job.id);
  assert.equal(result.status, "completed", JSON.stringify(result));
  assert.ok(result.bytes > 1024);
  assert.deepEqual(errors, []);
});

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

test("optional distortion and audio soundtrack survive seek, looping and offline export", { timeout: 120_000 }, async (t) => {
  const { call, connection } = await startServer(t);
  const browser = await chromium.launch({
    channel: process.env.VJ_BROWSER_CHANNEL || (process.platform === "win32" ? "msedge" : undefined),
    headless: true, args: ["--enable-unsafe-swiftshader"],
  });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", msg => {
    if (msg.type() === "error" && !msg.text().includes("Failed to load resource")) errors.push(msg.text());
  });
  await page.goto(connection.url);
  await page.getByText("MCP: connected", { exact: true }).waitFor();
  assert.equal(await page.locator("#distortion-enabled").isChecked(), false);
  await call("select_scene", { sceneId: "neon" });
  await call("set_parameters", { sceneId: "neon", values: { seed: 123 } });
  await page.locator("#distortion-enabled").check();
  const fingerprints = [];
  for (const mode of ["0", "1", "2"]) {
    await page.locator("#distortion-mode").selectOption(mode);
    const job = await call("export_video", { duration: 1, fps: 30, width: 160, height: 90 });
    const result = await waitForExport(call, job.id);
    assert.equal(result.status, "completed", JSON.stringify(result));
    const data = await readFile(result.path);
    assert.equal(data.includes(Buffer.from("soun")), false);
    const pixels = await page.evaluate(async base64 => {
      const video = document.createElement("video");
      const url = URL.createObjectURL(new Blob([Uint8Array.from(atob(base64), c => c.charCodeAt(0))], { type: "video/mp4" }));
      try {
        video.src = url;
        await new Promise((resolve, reject) => { video.onloadeddata = resolve; video.onerror = reject; });
        video.currentTime = 0.5;
        await new Promise(resolve => { video.onseeked = resolve; });
        const canvas = document.createElement("canvas"); canvas.width = 160; canvas.height = 90;
        const context = canvas.getContext("2d"); context.drawImage(video, 0, 0);
        return Array.from(context.getImageData(0, 0, 160, 90).data).filter((_, i) => i % 4 !== 3);
      } finally { URL.revokeObjectURL(url); }
    }, data.toString("base64"));
    assert.ok(pixels.reduce((a, b) => a + b, 0) > 10000, `Distorted video must not be blank (mode ${mode}, sum ${pixels.reduce((a, b) => a + b, 0)})`);
    fingerprints.push(pixels);
  }
  assert.notDeepEqual(fingerprints[0], fingerprints[1], "Different effects must produce different images");
  assert.notDeepEqual(fingerprints[0], fingerprints[2]);
  // A one-second mono WAV: 110 Hz in the first half, silence in the second.
  await page.locator("#soundtrack-controls summary").click();
  const rate = 48000;
  const wav = Buffer.alloc(44 + rate * 2);
  wav.write("RIFF"); wav.writeUInt32LE(wav.length - 8, 4); wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 2, 28);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write("data", 36); wav.writeUInt32LE(rate * 2, 40);
  for (let i = 0; i < rate / 2; i++) wav.writeInt16LE(Math.round(Math.sin(i * 2 * Math.PI * 110 / rate) * 16000), 44 + i * 2);
  await page.locator("#audio-file").setInputFiles({ name: "pulse.wav", mimeType: "audio/wav", buffer: wav });
  await page.getByText("pulse.wav", { exact: true }).waitFor();
  await page.locator("#audio-duration").click();
  assert.equal(await page.locator("#offline-duration").inputValue(), "1");
  await page.locator("#audio-play").click();
  await page.getByRole("button", { name: "Pause", exact: true }).waitFor();
  await page.locator("#audio-play").click();
  await page.locator("#audio-seek").fill("0.25");
  // Decode the MP4 soundtrack to verify actual audio, looping, and silent tail.
  const decode = async (data) => page.evaluate(async base64 => {
    const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
    const context = new AudioContext();
    try {
      const audio = await context.decodeAudioData(bytes.buffer);
      const pcm = audio.getChannelData(0);
      const rms = (start, end) => {
        const a = Math.floor(start * audio.sampleRate), b = Math.floor(end * audio.sampleRate);
        let sum = 0; for (let i = a; i < b; i++) sum += (pcm[i] ?? 0) ** 2;
        return Math.sqrt(sum / (b - a));
      };
      return { duration: audio.duration, tone: rms(0.1, 0.4), tail: rms(1.1, 1.4) };
    } finally { await context.close(); }
  }, data.toString("base64"));
  for (const loop of [true, false]) {
    await page.locator("#audio-loop").setChecked(loop);
    const job = await call("export_video", { duration: 2, fps: 30, width: 160, height: 90 });
    const result = await waitForExport(call, job.id);
    assert.equal(result.status, "completed", JSON.stringify(result));
    const data = await readFile(result.path);
    assert.ok(data.includes(Buffer.from("soun")));
    const decoded = await decode(data);
    assert.ok(Math.abs(decoded.duration - 2) < 0.1, JSON.stringify(decoded));
    assert.ok(decoded.tone > 0.2, JSON.stringify(decoded));
    assert.ok(loop ? decoded.tail > 0.2 : decoded.tail < 0.01, JSON.stringify(decoded));
  }
  await page.locator("#audio-include").uncheck();
  const silentJob = await call("export_video", { duration: 1, fps: 30, width: 160, height: 90 });
  const silentResult = await waitForExport(call, silentJob.id);
  assert.equal(silentResult.status, "completed");
  assert.equal((await readFile(silentResult.path)).includes(Buffer.from("soun")), false);
  await page.locator("#audio-include").check();
  await page.locator("#audio-seek").fill("0");
  const downloadReady = page.waitForEvent("download");
  await page.locator("#rec-btn").click();
  await page.getByRole("button", { name: "Stop", exact: true }).waitFor();
  await page.waitForFunction(() => Number(document.querySelector("#audio-seek").value) > 0.65);
  await page.locator("#rec-btn").click();
  const recording = await downloadReady;
  assert.ok((await decode(await readFile(await recording.path()))).tone > 0.2, "Live recording includes audio");
  await page.locator("#audio-clear").click();
  assert.equal(await page.locator("#audio-play").isDisabled(), true);
  await call("select_scene", { sceneId: "smoke-rainbow-plumes" });
  const fluidJob = await call("export_video", { duration: 1, fps: 30, width: 160, height: 90 });
  assert.equal((await waitForExport(call, fluidJob.id)).status, "completed");
  if (process.env.VJ_SCREENSHOT_PATH) await page.screenshot({ path: process.env.VJ_SCREENSHOT_PATH });
  assert.equal((await call("get_state")).exporting, false);
  assert.equal(await page.evaluate(() => document.querySelector("#gl-canvas").getContext("webgl2").getError()), 0);
  assert.deepEqual(errors, []);
});
