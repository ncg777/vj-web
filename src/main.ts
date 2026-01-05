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

type ProgramInfo = {
  program: WebGLProgram;
  uniforms: UniformLocations;
};

type ParamValues = Record<string, number>;

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app root element");
}

app.innerHTML = `
  <div class="app-shell">
    <header class="top-bar">
      <div class="brand">
        <div class="badge">VJ LAB</div>
        <div class="brand-title">GL Field Deck</div>
        <div class="brand-sub">Live shader scenes for performance and capture</div>
      </div>
      <div class="top-actions">
        <button class="ghost" data-action="toggle-ui">Hide UI</button>
        <button class="accent" data-action="fullscreen">Fullscreen</button>
      </div>
    </header>
    <div class="layout">
      <aside class="panel">
        <div class="panel-section">
          <div class="section-title">Scenes</div>
          <div class="scene-list" id="scene-list"></div>
        </div>
        <div class="panel-section">
          <div class="section-title">Controls</div>
          <div class="panel-actions" id="panel-actions"></div>
          <div class="control-list" id="control-list"></div>
        </div>
        <div class="panel-section small">
          <div class="section-title">Keys</div>
          <div class="key-help" id="key-help"></div>
        </div>
      </aside>
      <main class="stage">
        <canvas id="gl-canvas"></canvas>
        <button class="floating-toggle" data-action="toggle-ui-floating">Show UI</button>
        <div class="hud">
          <div class="hud-title" id="hud-title"></div>
          <div class="hud-desc" id="hud-desc"></div>
        </div>
      </main>
    </div>
  </div>
`;

const canvas = document.querySelector<HTMLCanvasElement>("#gl-canvas")!;
const sceneList = document.querySelector<HTMLDivElement>("#scene-list")!;
const controlList = document.querySelector<HTMLDivElement>("#control-list")!;
const panelActions = document.querySelector<HTMLDivElement>("#panel-actions")!;
const keyHelp = document.querySelector<HTMLDivElement>("#key-help")!;
const hudTitle = document.querySelector<HTMLDivElement>("#hud-title")!;
const hudDesc = document.querySelector<HTMLDivElement>("#hud-desc")!;
const toggleUiButton = document.querySelector<HTMLButtonElement>(
  "[data-action='toggle-ui']",
)!;
const floatingToggleButton = document.querySelector<HTMLButtonElement>(
  "[data-action='toggle-ui-floating']",
)!;
const fullscreenButton = document.querySelector<HTMLButtonElement>(
  "[data-action='fullscreen']",
)!;
const stage = document.querySelector<HTMLElement>(".stage")!;

if (
  !canvas ||
  !sceneList ||
  !controlList ||
  !panelActions ||
  !keyHelp ||
  !hudTitle ||
  !hudDesc ||
  !toggleUiButton ||
  !floatingToggleButton ||
  !fullscreenButton ||
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

gl.disable(gl.DEPTH_TEST);
gl.disable(gl.BLEND);

const vao = createFullscreenTriangle(gl);
const programById = new Map<string, ProgramInfo>();
const paramSpecById: Record<string, Record<string, ParamSpec>> = {};
const stateByAnimation: Record<string, ParamValues> = {};
const defaultByAnimation: Record<string, ParamValues> = {};

for (const animation of animations) {
  const program = createProgram(gl, vertexSource, animation.fragment);
  const uniformNames = new Set<string>();
  uniformNames.add(animation.resolutionUniform);
  uniformNames.add(animation.timeUniform);
  if (animation.loopUniform) {
    uniformNames.add(animation.loopUniform);
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
let uiHidden = false;
let startTime = performance.now();

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
}

function resetParams(animationId: string) {
  const defaults = defaultByAnimation[animationId];
  for (const [paramId, value] of Object.entries(defaults)) {
    setParamValue(animationId, paramId, value, true);
  }
}

function buildSceneList() {
  sceneList.innerHTML = "";
  for (const animation of animations) {
    const button = document.createElement("button");
    button.className = "scene-button";
    button.textContent = animation.name;
    button.dataset.scene = animation.id;
    button.addEventListener("click", () => setActiveAnimation(animation.id));
    sceneList.appendChild(button);
  }
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
  hudTitle.textContent = next.name;
  hudDesc.textContent = next.description;
  buildPanelActions(next);
  buildControls(next);
  buildKeyHelp(next);

  sceneList.querySelectorAll(".scene-button").forEach((button) => {
    const element = button as HTMLButtonElement;
    element.classList.toggle("active", element.dataset.scene === next.id);
  });
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

function renderFrame(now: number) {
  const elapsed = (now - startTime) / 1000;
  const { width, height } = resizeCanvasToDisplaySize(canvas);
  const info = programById.get(activeAnimation.id);
  if (!info) {
    return;
  }
  gl.viewport(0, 0, width, height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(info.program);
  gl.bindVertexArray(vao);
  applyUniforms(activeAnimation, info, elapsed, width, height);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  requestAnimationFrame(renderFrame);
}

function setUiHidden(nextHidden: boolean) {
  uiHidden = nextHidden;
  document.body.classList.toggle("ui-hidden", uiHidden);
  toggleUiButton.textContent = uiHidden ? "Show UI" : "Hide UI";
  floatingToggleButton.textContent = uiHidden ? "Show UI" : "Hide UI";
}

toggleUiButton.addEventListener("click", () => setUiHidden(!uiHidden));
floatingToggleButton.addEventListener("click", () => setUiHidden(!uiHidden));
fullscreenButton.addEventListener("click", async () => {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else {
    await stage.requestFullscreen();
  }
});

document.addEventListener("keydown", handleKey);
document.addEventListener("fullscreenchange", () => {
  fullscreenButton.textContent = document.fullscreenElement
    ? "Exit Fullscreen"
    : "Fullscreen";
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    startTime = performance.now();
  }
});

buildSceneList();
setActiveAnimation(activeAnimation.id);
requestAnimationFrame(renderFrame);
