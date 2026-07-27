import "./style.css";
import vertexSource from "./shaders/fullscreen.vert?raw";
import { animations, type AnimationConfig, type ParamSpec } from "./animations";
import {
  createFullscreenTriangle,
  createProgram,
  getUniformLocations,
  resizeCanvasToDisplaySize,
  type UniformLocations,
} from "./gl";
import { Muxer, FileSystemWritableFileStreamTarget } from "mp4-muxer";

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
          <div class="section-title">Offline Render</div>
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
            <button class="ghost small" id="offline-btn">Generate</button>
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

function startRecording() {
  const mime = pickMimeType();
  if (!mime) {
    alert("Recording is not supported in this browser.");
    return;
  }

  const stream = canvas.captureStream(60);
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
function renderSingleFrame(
  animation: AnimationConfig,
  timeSeconds: number,
  width: number,
  height: number,
) {
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

    gl.bindFramebuffer(gl.FRAMEBUFFER, nextFbo());
    gl.viewport(0, 0, state.size, state.size);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, currentTex());
    applyUniforms(animation, info, timeSeconds, width, height);
    applyStateUniforms(animation, info, state, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    state.index = (state.index + 1) % 2;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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

function renderFrame(now: number) {
  if (isOfflineRendering) return; // paused while offline render is running
  const elapsed = (now - startTime) / 1000;
  const { width, height } = resizeCanvasToDisplaySize(canvas);
  renderSingleFrame(activeAnimation, elapsed, width, height);
  requestAnimationFrame(renderFrame);
}

// ── Offline render engine (WebCodecs + mp4-muxer) ───────────

/** Yield one animation frame — keeps the browser from thinking the page is hung. */
function yieldFrame(): Promise<void> {
  return new Promise((r) => requestAnimationFrame(() => r()));
}

async function offlineRender(
  animation: AnimationConfig,
  durationSec: number,
  fps: number,
  targetWidth: number,
  targetHeight: number,
) {
  // Check WebCodecs availability
  if (typeof VideoEncoder === "undefined" || typeof VideoFrame === "undefined") {
    alert(
      "Offline rendering requires the WebCodecs API (Chrome/Edge 94+, Safari 16.4+).",
    );
    return;
  }

  // Check H.264 support
  const encoderConfig: VideoEncoderConfig = {
    codec: "avc1.640028", // H.264 High Profile Level 4.0
    width: targetWidth,
    height: targetHeight,
    bitrate: 16_000_000,
    framerate: fps,
  };
  const support = await VideoEncoder.isConfigSupported(encoderConfig);
  if (!support.supported) {
    alert("H.264 video encoding is not supported on this device.");
    return;
  }

  // Prompt user for save location — streams directly to disk, no memory pressure
  let fileHandle: FileSystemFileHandle;
  try {
    fileHandle = await window.showSaveFilePicker({
      suggestedName: `${animation.id}-${targetWidth}x${targetHeight}-${fps}fps-${durationSec}s.mp4`,
      types: [
        {
          description: "MP4 Video",
          accept: { "video/mp4": [".mp4"] },
        },
      ],
    });
  } catch {
    // User cancelled the save dialog
    return;
  }

  const writableStream = await fileHandle.createWritable();

  // UI handles
  const btn = document.getElementById("offline-btn") as HTMLButtonElement | null;
  const progressWrap = document.getElementById("offline-progress");
  const bar = document.getElementById("offline-bar");
  const statusEl = document.getElementById("offline-status");
  if (btn) btn.disabled = true;
  progressWrap?.classList.remove("hidden");
  statusEl?.classList.remove("hidden");

  // Pause live loop
  isOfflineRendering = true;

  // Reset stateful resources so the offline render starts clean
  if (animation.stateful) {
    getStateResources(animation);
    markStateNeedsInit(animation.id);
  }

  // Resize canvas to target output size
  const prevWidth = canvas.width;
  const prevHeight = canvas.height;
  const prevStyleW = canvas.style.width;
  const prevStyleH = canvas.style.height;
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  canvas.style.width = `${targetWidth}px`;
  canvas.style.height = `${targetHeight}px`;

  // Set up mp4-muxer — stream to disk via File System Access API
  const muxTarget = new FileSystemWritableFileStreamTarget(writableStream);
  const muxer = new Muxer({
    target: muxTarget,
    video: {
      codec: "avc",
      width: targetWidth,
      height: targetHeight,
    },
    fastStart: false, // streaming mode — no need to buffer everything in RAM
  });

  // Set up VideoEncoder
  let encoderError: Error | null = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta ?? undefined),
    error: (e) => {
      encoderError = e;
    },
  });
  encoder.configure(encoderConfig);

  const totalFrames = Math.ceil(durationSec * fps);
  const frameDurationUs = Math.round(1_000_000 / fps);
  const KEYFRAME_INTERVAL = fps * 2; // keyframe every 2 seconds
  const FLUSH_INTERVAL = fps * 30;   // flush encoder every 30 seconds of video
  const t0 = performance.now();

  // Handle context loss during render — attempt to save partial output
  let contextLost = false;
  const onContextLost = (e: Event) => {
    e.preventDefault(); // allow restoration later
    contextLost = true;
  };
  canvas.addEventListener("webglcontextlost", onContextLost);

  for (let frame = 0; frame < totalFrames; frame++) {
    if (encoderError || contextLost) break;

    const timeSeconds = frame / fps;
    renderSingleFrame(animation, timeSeconds, targetWidth, targetHeight);
    gl.finish(); // ensure GPU is done before capturing

    const videoFrame = new VideoFrame(canvas, {
      timestamp: frame * frameDurationUs,
      duration: frameDurationUs,
    });
    encoder.encode(videoFrame, {
      keyFrame: frame % KEYFRAME_INTERVAL === 0,
    });
    videoFrame.close();

    // Periodically flush the encoder to reduce memory pressure
    if (frame > 0 && frame % FLUSH_INTERVAL === 0) {
      await encoder.flush();
    }

    // Update progress
    const pct = ((frame + 1) / totalFrames) * 100;
    if (bar) bar.style.width = `${pct}%`;
    if (frame % 30 === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      const remaining =
        (elapsed / (frame + 1)) * (totalFrames - frame - 1);
      if (statusEl) {
        statusEl.textContent = `Frame ${frame + 1}/${totalFrames}  \u2014  ~${Math.ceil(remaining)}s left`;
      }
    }

    // Yield every frame via rAF to prevent the browser from killing the context.
    // Also applies backpressure if the encoder is falling behind.
    if (encoder.encodeQueueSize > 10) {
      await new Promise<void>((r) => setTimeout(r, 1));
    }
    await yieldFrame();
  }

  canvas.removeEventListener("webglcontextlost", onContextLost);

  // Flush encoder, finalize MP4, and close the file stream
  try {
    await encoder.flush();
    encoder.close();
    muxer.finalize();
    await writableStream.close();
  } catch (e) {
    console.error("Finalization failed:", e);
    try { await writableStream.close(); } catch { /* already closed */ }
  }

  // Restore canvas
  canvas.style.width = prevStyleW;
  canvas.style.height = prevStyleH;
  canvas.width = prevWidth;
  canvas.height = prevHeight;

  // Reset stateful so live rendering resumes cleanly
  if (animation.stateful) {
    markStateNeedsInit(animation.id);
  }

  // Resume live loop
  isOfflineRendering = false;
  startTime = performance.now();
  requestAnimationFrame(renderFrame);

  // Update UI
  if (btn) btn.disabled = false;
  const errorMessage = encoderError ? (encoderError as Error).message : null;
  if (statusEl) {
    if (contextLost) {
      statusEl.textContent = `Context lost — partial video saved.`;
    } else if (errorMessage) {
      statusEl.textContent = `Error: ${errorMessage}`;
    } else {
      statusEl.textContent = "Done!";
    }
  }
  const hasIssue = contextLost || !!errorMessage;
  setTimeout(() => {
    progressWrap?.classList.add("hidden");
    statusEl?.classList.add("hidden");
    if (bar) bar.style.width = "0%";
  }, hasIssue ? 8000 : 3000);
}

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
requestAnimationFrame(renderFrame);
