import "./style.css";
import { Soundtrack } from "./audio";
import { Distortion } from "./distortion";
import { createAudioPanel } from "./audio-panel";
import vertexSource from "./shaders/fullscreen.vert?raw";
import { animations, type AnimationConfig, type ParamSpec } from "./animations";
import {
  createFullscreenTriangle,
  createProgram,
  getUniformLocations,
  resizeCanvasToDisplaySize,
  type UniformLocations,
} from "./gl";
import { renderVideo, type VideoSink } from "./video-export";
import { connectMcp } from "./mcp-client";
import { findScene, validateParameters, type VideoOptions } from "./control";

type ProgramInfo = {
  program: WebGLProgram;
  uniforms: UniformLocations;
};

type ParamValues = Record<string, number>;
type StatefulResources = {
  size: number;
  textures: [WebGLTexture, WebGLTexture];
  fbos: [WebGLFramebuffer, WebGLFramebuffer];
  index: number;
  needsInit: boolean;
  lastTime?: number;
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app root element");
}

app.innerHTML = `
  <div class="app-shell">
    <div class="layout">
      <aside class="panel">
        <div class="panel-section">
          <div class="section-title">Scenes</div>
          <select class="scene-select" id="scene-list"></select>
        </div>
        <div class="panel-section">
          <div class="section-title">Controls</div>
          <div class="panel-actions" id="panel-actions"></div>
          <div class="control-list" id="control-list"></div>
        </div>
        <div class="panel-section">
          <div class="section-title">Record</div>
          <div class="rec-row">
            <button class="ghost small" id="rec-btn">Record</button>
            <span class="rec-badge hidden" id="rec-badge"></span>
          </div>
        </div>
        <div class="panel-section">
          <div class="section-title">Export video</div>
          <div class="offline-controls">
            <div class="offline-row">
              <label>Duration</label>
              <input type="number" id="offline-duration" value="10" min="1" max="3600" step="1" />
              <span class="offline-unit">sec</span>
            </div>
            <div class="offline-row">
              <label>FPS</label>
              <select id="offline-fps">
                <option value="30">30</option>
                <option value="60" selected>60</option>
              </select>
            </div>
            <div class="offline-row">
              <label>Resolution</label>
              <select id="offline-res">
                <option value="1280x720">720p</option>
                <option value="1920x1080" selected>1080p</option>
                <option value="2560x1440">1440p</option>
                <option value="3840x2160">4K</option>
              </select>
            </div>
            <button class="ghost small" id="offline-btn">Export MP4</button>
            <div class="offline-progress hidden" id="offline-progress">
              <div class="offline-progress-bar" id="offline-bar"></div>
            </div>
            <div class="offline-status hidden" id="offline-status"></div>
          </div>
        </div>
        <div class="panel-section small">
          <div class="section-title">Keys</div>
          <div class="key-help" id="key-help"></div>
        </div>
      </aside>
      <main class="stage">
        <canvas id="gl-canvas"></canvas>
        <button class="sidebar-toggle" data-action="toggle-sidebar"></button>
        <div class="hud">
          <div class="hud-title" id="hud-title"></div>
          <div class="hud-desc" id="hud-desc"></div>
        </div>
      </main>
    </div>
  </div>
`;

const canvas = document.querySelector<HTMLCanvasElement>("#gl-canvas")!;
const sceneList = document.querySelector<HTMLSelectElement>("#scene-list")!;
const controlList = document.querySelector<HTMLDivElement>("#control-list")!;
const panelActions = document.querySelector<HTMLDivElement>("#panel-actions")!;
const keyHelp = document.querySelector<HTMLDivElement>("#key-help")!;
const hudTitle = document.querySelector<HTMLDivElement>("#hud-title")!;
const hudDesc = document.querySelector<HTMLDivElement>("#hud-desc")!;
const sidebarToggleButton =
  document.querySelector<HTMLButtonElement>("[data-action='toggle-sidebar']")!;
const stage = document.querySelector<HTMLElement>(".stage")!;

if (
  !canvas ||
  !sceneList ||
  !controlList ||
  !panelActions ||
  !keyHelp ||
  !hudTitle ||
  !hudDesc ||
  !sidebarToggleButton ||
  !stage
) {
  throw new Error("Missing required UI elements");
}

const gl = canvas.getContext("webgl2", { antialias: true })!;
if (!gl) {
  stage.innerHTML = `
    <div class="fallback">
      <h2>WebGL2 unavailable</h2>
      <p>Your browser or GPU does not expose WebGL2. Try a different browser.</p>
    </div>
  `;
  throw new Error("WebGL2 unavailable");
}

const floatRenderExt = gl.getExtension("EXT_color_buffer_float");
const supportsFloatTargets = !!floatRenderExt;

gl.disable(gl.DEPTH_TEST);
gl.disable(gl.BLEND);

const vao = createFullscreenTriangle(gl);
const soundtrack = new Soundtrack();
const distortion = new Distortion(gl, vao);
const audioPanel = createAudioPanel(soundtrack, distortion, () => {
  distortion.reset();
  markStateNeedsInit(activeAnimation.id);
  startTime = performance.now();
});
const programById = new Map<string, ProgramInfo>();
const paramSpecById: Record<string, Record<string, ParamSpec>> = {};
const stateByAnimation: Record<string, ParamValues> = {};
const defaultByAnimation: Record<string, ParamValues> = {};
const statefulById = new Map<string, StatefulResources>();

for (const animation of animations) {
  const program = createProgram(gl, vertexSource, animation.fragment);
  const uniformNames = new Set<string>();
  uniformNames.add(animation.resolutionUniform);
  uniformNames.add(animation.timeUniform);
  if (animation.loopUniform) {
    uniformNames.add(animation.loopUniform);
  }
  if (animation.stateful) {
    uniformNames.add(animation.passUniform ?? "uPass");
    uniformNames.add(animation.stateUniform ?? "uState");
    uniformNames.add(animation.gridUniform ?? "uGridSize");
  }
  for (const param of animation.params) {
    uniformNames.add(param.uniform);
  }
  const uniforms = getUniformLocations(gl, program, Array.from(uniformNames));
  programById.set(animation.id, { program, uniforms });

  const paramMap: Record<string, ParamSpec> = {};
  const values: ParamValues = {};
  for (const param of animation.params) {
    paramMap[param.id] = param;
    values[param.id] =
      param.type === "seed" ? Math.floor(Math.random() * 1_000_000) : param.value;
  }
  paramSpecById[animation.id] = paramMap;
  stateByAnimation[animation.id] = { ...values };
  defaultByAnimation[animation.id] = { ...values };
}

let activeAnimation = animations[0];
let activeControls: Record<
  string,
  { range?: HTMLInputElement; number?: HTMLInputElement }
> = {};
let startTime = performance.now();
let animationFrame = 0;
let hudTimeout: number | null = null;
let sidebarTimeout: number | null = null;

// ── Recording state ──────────────────────────────────────────
let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let recordingStartTime = 0;
let recordingTimerHandle: number | null = null;
let isRecording = false;

function pickMimeType(): string {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) return mime;
  }
  return "";
}

function extensionForMime(mime: string): string {
  if (mime.startsWith("video/mp4")) return "mp4";
  return "webm";
}

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const s = String(totalSec % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function updateRecordingTimer() {
  const badge = document.getElementById("rec-badge");
  if (!badge || !isRecording) return;
  badge.textContent = `⏺ ${formatDuration(performance.now() - recordingStartTime)}`;
}

async function startRecording() {
  if (audioPanel.loading) { alert("Wait for soundtrack analysis to finish."); return; }
  if (audioPanel.includeAudio && soundtrack.buffer && soundtrack.player.paused) {
    try { await soundtrack.play(); }
    catch (error) { alert(`Cannot play soundtrack: ${error instanceof Error ? error.message : error}`); return; }
  }
  const mime = pickMimeType();
  if (!mime) {
    alert("Recording is not supported in this browser.");
    return;
  }

  const stream = canvas.captureStream(60);
  if (audioPanel.includeAudio && soundtrack.buffer) {
    for (const track of soundtrack.stream?.getAudioTracks() ?? []) stream.addTrack(track.clone());
  }
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream, {
    mimeType: mime,
    videoBitsPerSecond: 16_000_000,
  });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const ext = extensionForMime(mime);
    const blob = new Blob(recordedChunks, { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeAnimation.id}-${Date.now()}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
    recordedChunks = [];
    stream.getTracks().forEach(track => track.stop());
  };

  mediaRecorder.start(500); // collect data every 500ms
  isRecording = true;
  recordingStartTime = performance.now();

  updateRecordButton();
  recordingTimerHandle = window.setInterval(updateRecordingTimer, 250);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  if (recordingTimerHandle !== null) {
    clearInterval(recordingTimerHandle);
    recordingTimerHandle = null;
  }
  updateRecordButton();
}

function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

function updateRecordButton() {
  const btn = document.getElementById("rec-btn") as HTMLButtonElement | null;
  const badge = document.getElementById("rec-badge");
  if (!btn || !badge) return;
  if (isRecording) {
    btn.textContent = "Stop";
    btn.classList.add("recording");
    badge.classList.remove("hidden");
  } else {
    btn.textContent = "Record";
    btn.classList.remove("recording");
    badge.classList.add("hidden");
    badge.textContent = "";
  }
}

// ── Offline (non-realtime) rendering ─────────────────────────
let isOfflineRendering = false;

function setSidebarToggleHidden(hidden: boolean) {
  sidebarToggleButton.classList.toggle("hidden", hidden);
}

function showSidebarToggleTemporarily() {
  setSidebarToggleHidden(false);
  if (sidebarTimeout !== null) {
    window.clearTimeout(sidebarTimeout);
  }
  sidebarTimeout = window.setTimeout(() => {
    setSidebarToggleHidden(true);
  }, 2500);
}

function updateSidebarToggleLabel() {
  const isCollapsed = document.body.classList.contains("sidebar-collapsed");
  sidebarToggleButton.textContent = isCollapsed ? ">>" : "<<";
}

function showHudTemporarily() {
  hudTitle.parentElement?.classList.remove("hidden");
  if (hudTimeout !== null) {
    window.clearTimeout(hudTimeout);
  }
  hudTimeout = window.setTimeout(() => {
    hudTitle.parentElement?.classList.add("hidden");
  }, 10000);
}

function createStateTexture(size: number) {
  const texture = gl.createTexture();
  if (!texture) {
    throw new Error("Failed to create state texture");
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.RGBA16F,
    size,
    size,
    0,
    gl.RGBA,
    gl.HALF_FLOAT,
    null,
  );
  gl.bindTexture(gl.TEXTURE_2D, null);
  return texture;
}

function createStateResources(size: number): StatefulResources {
  const texA = createStateTexture(size);
  const texB = createStateTexture(size);
  const fbA = gl.createFramebuffer();
  const fbB = gl.createFramebuffer();
  if (!fbA || !fbB) {
    throw new Error("Failed to create framebuffer");
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbA);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texA, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbB);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texB, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  return {
    size,
    textures: [texA, texB],
    fbos: [fbA, fbB],
    index: 0,
    needsInit: true,
  };
}

function getStateResources(animation: AnimationConfig) {
  if (!animation.stateful) {
    return null;
  }
  let existing = statefulById.get(animation.id);
  const size = animation.bufferSize ?? 192;
  if (!existing || existing.size !== size) {
    existing = createStateResources(size);
    statefulById.set(animation.id, existing);
  }
  return existing;
}

function markStateNeedsInit(animationId: string) {
  const state = statefulById.get(animationId);
  if (state) {
    state.needsInit = true;
    state.lastTime = undefined;
  }
}

function clampValue(param: ParamSpec, value: number) {
  let next = value;
  if (param.min !== undefined) {
    next = Math.max(param.min, next);
  }
  if (param.max !== undefined) {
    next = Math.min(param.max, next);
  }
  if (param.type === "int") {
    next = Math.round(next);
  }
  return next;
}

function formatValue(param: ParamSpec, value: number) {
  if (param.type === "int") {
    return String(Math.round(value));
  }
  const step = param.step ?? 0.01;
  const decimals = step < 1 ? Math.min(4, Math.max(2, Math.ceil(-Math.log10(step)))) : 0;
  return value.toFixed(decimals);
}

function normalizeKey(key: string) {
  if (key.length === 1) {
    return key.toLowerCase();
  }
  return key;
}

function isTypingTarget(target: EventTarget | null) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement
  );
}

function setParamValue(
  animationId: string,
  paramId: string,
  rawValue: number,
  updateInputs = true,
) {
  const param = paramSpecById[animationId][paramId];
  if (!param) {
    return;
  }
  const nextValue = clampValue(param, rawValue);
  stateByAnimation[animationId][paramId] = nextValue;

  if (animationId === activeAnimation.id && updateInputs) {
    const controls = activeControls[paramId];
    if (controls?.range) {
      controls.range.value = String(nextValue);
    }
    if (controls?.number) {
      controls.number.value = formatValue(param, nextValue);
    }
  }
}

function reseed(animationId: string) {
  for (const param of animations.find((a) => a.id === animationId)?.params ?? []) {
    if (param.type === "seed") {
      setParamValue(animationId, param.id, Math.floor(Math.random() * 1_000_000));
    }
  }
  markStateNeedsInit(animationId);
}

function resetParams(animationId: string) {
  const defaults = defaultByAnimation[animationId];
  for (const [paramId, value] of Object.entries(defaults)) {
    setParamValue(animationId, paramId, value, true);
  }
  markStateNeedsInit(animationId);
}

function buildSceneList() {
  sceneList.innerHTML = "";
  for (const animation of animations) {
    const option = document.createElement("option");
    option.value = animation.id;
    option.textContent = animation.name;
    sceneList.appendChild(option);
  }
  sceneList.addEventListener("change", () => {
    setActiveAnimation(sceneList.value);
  });
}

function buildPanelActions(animation: AnimationConfig) {
  panelActions.innerHTML = "";
  const resetButton = document.createElement("button");
  resetButton.className = "ghost small";
  resetButton.textContent = "Reset";
  resetButton.addEventListener("click", () => resetParams(animation.id));
  panelActions.appendChild(resetButton);

  const hasSeed = animation.params.some((param) => param.type === "seed");
  if (hasSeed) {
    const seedButton = document.createElement("button");
    seedButton.className = "ghost small";
    seedButton.textContent = "Reseed";
    seedButton.addEventListener("click", () => reseed(animation.id));
    panelActions.appendChild(seedButton);
  }
}

function buildControls(animation: AnimationConfig) {
  controlList.innerHTML = "";
  activeControls = {};
  for (const param of animation.params) {
    if (param.type === "seed") {
      continue;
    }
    const wrapper = document.createElement("div");
    wrapper.className = "control";
    const header = document.createElement("div");
    header.className = "control-header";

    const label = document.createElement("label");
    label.textContent = param.label;
    header.appendChild(label);

    if (param.key) {
      const keyLabel = document.createElement("span");
      keyLabel.className = "key-cap";
      keyLabel.textContent = `${param.key.inc.toUpperCase()}/${param.key.dec.toUpperCase()}`;
      header.appendChild(keyLabel);
    }

    const inputs = document.createElement("div");
    inputs.className = "control-inputs";

    const range = document.createElement("input");
    range.type = "range";
    range.min = String(param.min ?? 0);
    range.max = String(param.max ?? 1);
    range.step = String(param.step ?? (param.type === "int" ? 1 : 0.01));
    range.value = String(stateByAnimation[animation.id][param.id]);
    range.addEventListener("input", (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      if (!Number.isNaN(value)) {
        setParamValue(animation.id, param.id, value);
      }
    });

    const number = document.createElement("input");
    number.type = "number";
    number.min = range.min;
    number.max = range.max;
    number.step = range.step;
    number.value = formatValue(param, stateByAnimation[animation.id][param.id]);
    number.addEventListener("input", (event) => {
      const value = Number((event.target as HTMLInputElement).value);
      if (!Number.isNaN(value)) {
        setParamValue(animation.id, param.id, value);
      }
    });

    inputs.appendChild(range);
    inputs.appendChild(number);
    wrapper.appendChild(header);
    wrapper.appendChild(inputs);
    controlList.appendChild(wrapper);
    activeControls[param.id] = { range, number };
  }
}

function buildKeyHelp(animation: AnimationConfig) {
  keyHelp.innerHTML = "";
  for (const param of animation.params) {
    if (!param.key || param.type === "seed") {
      continue;
    }
    const row = document.createElement("div");
    row.className = "key-row";
    row.textContent = `${param.key.inc.toUpperCase()}/${param.key.dec.toUpperCase()}  ${param.label}`;
    keyHelp.appendChild(row);
  }
  if (!keyHelp.childElementCount) {
    keyHelp.textContent = "No mapped keys for this scene.";
  }
}

function setActiveAnimation(id: string) {
  const next = animations.find((animation) => animation.id === id);
  if (!next) {
    return;
  }
  activeAnimation = next;
  distortion.reset();
  if (next.stateful) {
    getStateResources(next);
    markStateNeedsInit(next.id);
  }
  hudTitle.textContent = next.name;
  hudDesc.textContent = next.description;
  showHudTemporarily();
  buildPanelActions(next);
  buildControls(next);
  buildKeyHelp(next);

  sceneList.value = next.id;
}

function handleKey(event: KeyboardEvent) {
  if (isOfflineRendering) return;
  if (isTypingTarget(event.target)) {
    return;
  }
  const key = normalizeKey(event.key);
  const params = activeAnimation.params;
  for (const param of params) {
    if (!param.key || param.type === "seed") {
      continue;
    }
    const isIncrease = key === param.key.inc;
    const isDecrease = key === param.key.dec;
    if (!isIncrease && !isDecrease) {
      continue;
    }
    const step = event.shiftKey && param.key.shiftStep ? param.key.shiftStep : param.key.step;
    const current = stateByAnimation[activeAnimation.id][param.id];
    const next = current + step * (isIncrease ? 1 : -1);
    setParamValue(activeAnimation.id, param.id, next);
    event.preventDefault();
    break;
  }
}

function applyUniforms(
  animation: AnimationConfig,
  info: ProgramInfo,
  timeSeconds: number,
  width: number,
  height: number,
) {
  const values = stateByAnimation[animation.id];
  const uniforms = info.uniforms;

  const resLoc = uniforms[animation.resolutionUniform];
  if (resLoc) {
    gl.uniform2f(resLoc, width, height);
  }

  const timeLoc = uniforms[animation.timeUniform];
  if (timeLoc) {
    if (animation.timeMode === "phase") {
      const loop = animation.loopDuration ?? 8;
      const phase = (timeSeconds % loop) / loop;
      gl.uniform1f(timeLoc, phase);
    } else if (animation.timeMode === "looped") {
      const loop = animation.loopDuration ?? 8;
      const t = timeSeconds % loop;
      gl.uniform1f(timeLoc, t);
      if (animation.loopUniform) {
        const loopLoc = uniforms[animation.loopUniform];
        if (loopLoc) {
          gl.uniform1f(loopLoc, loop);
        }
      }
    } else {
      gl.uniform1f(timeLoc, timeSeconds);
    }
  }

  const vec3Groups: Record<string, [number, number, number]> = {};

  for (const param of animation.params) {
    const loc = uniforms[param.uniform];
    const value = values[param.id];
    if (param.component !== undefined) {
      const group = vec3Groups[param.uniform] ?? [0, 0, 0];
      group[param.component] = value;
      vec3Groups[param.uniform] = group;
      continue;
    }
    if (!loc) {
      continue;
    }
    if (param.type === "int") {
      gl.uniform1i(loc, Math.round(value));
    } else {
      gl.uniform1f(loc, value);
    }
  }

  for (const [uniformName, color] of Object.entries(vec3Groups)) {
    const loc = uniforms[uniformName];
    if (loc) {
      gl.uniform3f(loc, color[0], color[1], color[2]);
    }
  }
}

function applyStateUniforms(
  animation: AnimationConfig,
  info: ProgramInfo,
  state: StatefulResources,
  pass: number,
) {
  const passLoc = info.uniforms[animation.passUniform ?? "uPass"];
  if (passLoc) {
    gl.uniform1i(passLoc, pass);
  }
  const gridLoc = info.uniforms[animation.gridUniform ?? "uGridSize"];
  if (gridLoc) {
    gl.uniform2f(gridLoc, state.size, state.size);
  }
  const stateLoc = info.uniforms[animation.stateUniform ?? "uState"];
  if (stateLoc) {
    gl.uniform1i(stateLoc, 0);
  }
}

/**
 * Render a single frame of `animation` at the given virtual time.
 * The caller is responsible for setting the canvas size and calling
 * gl.viewport / gl.clear beforehand when needed.
 */
function renderSceneFrame(
  animation: AnimationConfig,
  timeSeconds: number,
  width: number,
  height: number,
  target: WebGLFramebuffer | null,
) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target);
  const info = programById.get(animation.id);
  if (!info) return;

  if (animation.stateful) {
    if (!supportsFloatTargets) return;
    const state = getStateResources(animation);
    if (!state) return;

    const currentTex = () => state.textures[state.index];
    const nextFbo = () => state.fbos[(state.index + 1) % 2];

    gl.useProgram(info.program);
    gl.bindVertexArray(vao);

    if (state.needsInit) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, nextFbo());
      gl.viewport(0, 0, state.size, state.size);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, currentTex());
      applyUniforms(animation, info, timeSeconds, width, height);
      applyStateUniforms(animation, info, state, 2);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      state.index = (state.index + 1) % 2;
      state.needsInit = false;
    }

    if (state.lastTime !== timeSeconds) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, nextFbo());
      gl.viewport(0, 0, state.size, state.size);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, currentTex());
      applyUniforms(animation, info, timeSeconds, width, height);
      applyStateUniforms(animation, info, state, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      state.index = (state.index + 1) % 2;
      state.lastTime = timeSeconds;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentTex());
    applyUniforms(animation, info, timeSeconds, width, height);
    applyStateUniforms(animation, info, state, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return;
  }

  gl.viewport(0, 0, width, height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(info.program);
  gl.bindVertexArray(vao);
  applyUniforms(animation, info, timeSeconds, width, height);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

let previousPreviewTime = 0;
function renderSingleFrame(animation: AnimationConfig, time: number, width: number, height: number) {
  if (!isOfflineRendering && soundtrack.buffer && time < previousPreviewTime) {
    markStateNeedsInit(animation.id);
    distortion.reset();
  }
  if (!isOfflineRendering) previousPreviewTime = time;
  const target = distortion.begin(width, height);
  renderSceneFrame(animation, time, width, height, target);
  if (target) distortion.finish(time, soundtrack.sample(time), !!soundtrack.buffer);
}

function renderFrame(now: number) {
  if (isOfflineRendering) return; // paused while offline render is running
  const elapsed = soundtrack.buffer ? soundtrack.player.currentTime : (now - startTime) / 1000;
  audioPanel.update();
  const { width, height } = resizeCanvasToDisplaySize(canvas);
  renderSingleFrame(activeAnimation, elapsed, width, height);
  animationFrame = requestAnimationFrame(renderFrame);
}

// ── Offline render engine (WebCodecs + mp4-muxer) ───────────

async function runOfflineRender(
  options: VideoOptions,
  sink: VideoSink,
  signal?: AbortSignal,
  onProgress?: (completed: number, total: number) => void,
) {
  if (isOfflineRendering || isRecording || audioPanel.loading) {
    await sink.abort();
    throw new Error(audioPanel.loading ? "Wait for soundtrack analysis to finish." : "Stop the current recording or export before exporting video.");
  }
  if (gl.isContextLost()) {
    await sink.abort();
    throw new Error("WebGL context is lost. Reload the browser tab.");
  }
  const animation = activeAnimation;
  if (animation.stateful && !supportsFloatTargets) {
    await sink.abort();
    throw new Error("This scene requires floating-point WebGL render targets.");
  }
  const wasPlaying = !!soundtrack.buffer && !soundtrack.player.paused;
  soundtrack.player.pause();
  distortion.reset();
  isOfflineRendering = true;
  cancelAnimationFrame(animationFrame);
  const panel = document.querySelector<HTMLElement>(".panel")!;
  panel.inert = true;
  const previous = { width: canvas.width, height: canvas.height, styleW: canvas.style.width, styleH: canvas.style.height };
  const progressWrap = document.getElementById("offline-progress");
  const bar = document.getElementById("offline-bar");
  const status = document.getElementById("offline-status");
  progressWrap?.classList.remove("hidden");
  status?.classList.remove("hidden");
  try {
    canvas.width = options.width;
    canvas.height = options.height;
    canvas.style.width = options.width + "px";
    canvas.style.height = options.height + "px";
    markStateNeedsInit(animation.id);
    await renderVideo({
      ...options, canvas, sink, signal,
      audio: audioPanel.includeAudio ? soundtrack.buffer : undefined,
      loopAudio: soundtrack.player.loop,
      render(time, width, height) {
        renderSingleFrame(animation, time, width, height);
        gl.finish();
      },
      progress(completed, total) {
        if (bar) bar.style.width = (completed / total * 100) + "%";
        if (status) status.textContent = "Rendering frame " + completed + "/" + total;
        if (completed % options.fps === 0 || completed === total) onProgress?.(completed, total);
      },
    });
    if (status) status.textContent = "Video saved.";
  } catch (error) {
    if (status) status.textContent = signal?.aborted ? "Export cancelled." : "Export failed: " + (error instanceof Error ? error.message : String(error));
    throw error;
  } finally {
    canvas.width = previous.width;
    canvas.height = previous.height;
    canvas.style.width = previous.styleW;
    canvas.style.height = previous.styleH;
    markStateNeedsInit(animation.id);
    panel.inert = false;
    isOfflineRendering = false;
    startTime = performance.now();
    distortion.reset();
    if (wasPlaying) await soundtrack.play().catch(console.error);
    animationFrame = requestAnimationFrame(renderFrame);
  }
}

async function offlineRender(animation: AnimationConfig, duration: number, fps: number, width: number, height: number) {
  let fileHandle: FileSystemFileHandle;
  try {
    fileHandle = await window.showSaveFilePicker({
      suggestedName: animation.id + "-" + width + "x" + height + "-" + fps + "fps-" + duration + "s.mp4",
      types: [{ description: "MP4 Video", accept: { "video/mp4": [".mp4"] } }],
    });
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      alert(error instanceof Error ? error.message : String(error));
    }
    return;
  }
  try {
    const stream = await fileHandle.createWritable();
    await runOfflineRender({ duration, fps, width, height }, {
      write: (data, position) => stream.write({ type: "write", position, data: data.slice().buffer as ArrayBuffer }),
      close: () => stream.close(),
      abort: () => stream.abort(),
    });
  } catch (error) {
    console.error("Video export failed:", error);
  }
}

function currentState() {
  return {
    sceneId: activeAnimation.id,
    name: activeAnimation.name,
    values: { ...stateByAnimation[activeAnimation.id] },
    recording: isRecording,
    exporting: isOfflineRendering,
    capabilities: { floatTargets: supportsFloatTargets, webCodecs: typeof VideoEncoder !== "undefined", contextLost: gl.isContextLost() },
  };
}

let remoteExport: { jobId: string; controller: AbortController } | undefined;
connectMcp(async ({ command, args }, connection) => {
  if (command === "get_state") return currentState();
  if (command === "cancel_export") {
    if (remoteExport && remoteExport.jobId === args.jobId) remoteExport.controller.abort(new Error("Export cancelled."));
    return currentState();
  }
  if (isOfflineRendering) throw new Error("Wait for the video export to finish or cancel it before editing the scene.");
  if (gl.isContextLost()) throw new Error("WebGL context is lost. Reload the browser tab.");
  if (command === "export_video") {
    const jobId = String(args.jobId);
    const controller = new AbortController();
    const disconnect = () => controller.abort(new Error("MCP disconnected during export."));
    connection.signal.addEventListener("abort", disconnect, { once: true });
    if (connection.signal.aborted) disconnect();
    remoteExport = { jobId, controller };
    try {
      await runOfflineRender({ duration: Number(args.duration), fps: Number(args.fps), width: Number(args.width), height: Number(args.height) },
        connection.sink(jobId), controller.signal,
        (completed, total) => connection.progress(jobId, completed, total));
      return currentState();
    } finally {
      connection.signal.removeEventListener("abort", disconnect);
      remoteExport = undefined;
    }
  }
  const scene = findScene(animations, String(args.sceneId));
  if (command === "select_scene") {
    if (scene.stateful && !supportsFloatTargets) throw new Error("This scene requires floating-point WebGL render targets.");
    setActiveAnimation(scene.id);
  } else {
    if (scene.id !== activeAnimation.id) throw new Error("The active scene changed. Read get_state before editing it.");
    if (command === "set_parameters") {
      const values = args.values as Record<string, number>;
      validateParameters(scene, values);
      for (const [id, value] of Object.entries(values)) setParamValue(scene.id, id, value);
      if (scene.params.some((param) => param.type === "seed" && Object.prototype.hasOwnProperty.call(values, param.id))) markStateNeedsInit(scene.id);
    } else if (command === "reset_scene") resetParams(scene.id);
    else if (command === "reseed_scene") reseed(scene.id);
    else throw new Error("Unknown browser command: " + command);
  }
  return currentState();
});

sidebarToggleButton.addEventListener("click", () => {
  document.body.classList.toggle("sidebar-collapsed");
  updateSidebarToggleLabel();
});

stage.addEventListener("mousemove", () => {
  showSidebarToggleTemporarily();
});

document.addEventListener("keydown", handleKey);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    startTime = performance.now();
  }
});

document.getElementById("rec-btn")?.addEventListener("click", toggleRecording);

document.getElementById("offline-btn")?.addEventListener("click", () => {
  if (isOfflineRendering) return;
  const durationInput = document.getElementById("offline-duration") as HTMLInputElement;
  const fpsSelect = document.getElementById("offline-fps") as HTMLSelectElement;
  const resSelect = document.getElementById("offline-res") as HTMLSelectElement;
  const duration = Math.max(1, Math.min(3600, Number(durationInput?.value ?? 10)));
  const fps = Number(fpsSelect?.value ?? 60);
  const [w, h] = (resSelect?.value ?? "1920x1080").split("x").map(Number);
  offlineRender(activeAnimation, duration, fps, w, h);
});

buildSceneList();
setActiveAnimation(activeAnimation.id);
updateSidebarToggleLabel();
setSidebarToggleHidden(true);
animationFrame = requestAnimationFrame(renderFrame);
