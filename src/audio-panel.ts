import { Soundtrack } from "./audio";
import { Distortion } from "./distortion";

export function createAudioPanel(sound: Soundtrack, distortion: Distortion, reset: () => void) {
  const panel = document.createElement("div");
  panel.className = "panel-section audio-panel";
  panel.innerHTML = `
    <div class="section-title">Psychedelic distortion</div>
    <label class="audio-check"><input id="distortion-enabled" type="checkbox"> Enable distortion</label>
    <div id="distortion-options" hidden>
    <label>Style <select id="distortion-mode" class="scene-select">
      <option value="0">Liquid warp</option><option value="1">Spectral kaleidoscope</option><option value="2">Psychedelic feedback</option>
    </select></label>
    <div id="distortion-sliders"></div>
    </div>
    <details id="soundtrack-controls"><summary class="section-title">Soundtrack · optional</summary>
    <input id="audio-file" type="file" accept="audio/*" aria-label="Load soundtrack">
    <div id="audio-status" role="status">Works without audio. Files stay on this device.</div>
    <div class="panel-actions"><button id="audio-play" class="ghost small" disabled>Play</button><button id="audio-clear" class="ghost small" disabled>Remove</button></div>
    <input id="audio-seek" type="range" min="0" max="1" step="0.01" value="0" aria-label="Soundtrack position" disabled>
    <output id="audio-time">0:00 / 0:00</output>
    <label class="audio-check"><input id="audio-loop" type="checkbox" checked> Loop soundtrack</label>
    <label>Preview volume <input id="audio-volume" type="range" min="0" max="1" step="0.01" value="1"></label>
    <label class="audio-check"><input id="audio-include" type="checkbox" checked> Include soundtrack in video</label>
    <button id="audio-duration" class="ghost small" disabled>Use track length for export</button>
    <p class="audio-note">Generate exports from the start of the track at its original volume. Short tracks loop when Loop is enabled; otherwise the remainder is silent. Seeking restarts fluid and feedback history.</p>
    </details>
  `;
  document.querySelector(".panel")!.insertBefore(panel, document.querySelector(".panel")!.children[1]);
  const input = (id: string) => panel.querySelector<HTMLInputElement>(`#${id}`)!;
  const button = (id: string) => panel.querySelector<HTMLButtonElement>(`#${id}`)!;
  const status = panel.querySelector<HTMLElement>("#audio-status")!;
  const play = button("audio-play"), clear = button("audio-clear"), duration = button("audio-duration");
  const seek = input("audio-seek");
  let loading = false;
  for (const [key, label, min, max, step] of [
    ["amount", "Intensity / mix", 0, 1, 0.01], ["scale", "Pattern scale", 1, 12, 0.1],
    ["color", "Color separation", 0, 1, 0.01], ["persistence", "Trail persistence", 0, 0.98, 0.01],
    ["sensitivity", "Audio sensitivity", 0, 4, 0.05],
  ] as const) {
    const labelEl = document.createElement("label");
    labelEl.textContent = label;
    const range = document.createElement("input");
    range.id = `distortion-${key}`; range.type = "range";
    range.min = String(min); range.max = String(max); range.step = String(step); range.value = String(distortion[key]);
    const output = document.createElement("output"); output.textContent = range.value;
    range.oninput = () => { distortion[key] = Number(range.value); output.textContent = range.value; };
    labelEl.append(range, output); panel.querySelector("#distortion-sliders")!.append(labelEl);
  }
  input("distortion-enabled").onchange = () => {
    distortion.enabled = input("distortion-enabled").checked;
    panel.querySelector<HTMLElement>("#distortion-options")!.hidden = !distortion.enabled;
    distortion.reset();
  };
  panel.querySelector<HTMLSelectElement>("#distortion-mode")!.onchange = event => {
    distortion.mode = Number((event.target as HTMLSelectElement).value); distortion.reset();
  };
  input("audio-file").onchange = async () => {
    const file = input("audio-file").files?.[0];
    if (!file) return;
    loading = true; play.disabled = clear.disabled = duration.disabled = seek.disabled = true;
    input("audio-file").disabled = true;
    status.textContent = "Analyzing soundtrack…";
    try {
      if (await sound.load(file)) {
        seek.max = String(sound.buffer!.duration);
        sound.volume = Number(input("audio-volume").value);
        status.textContent = file.name;
        reset();
      }
    } catch (error) { status.textContent = `Could not load audio: ${error instanceof Error ? error.message : error}`; }
    finally {
      loading = false;
      input("audio-file").disabled = false;
      play.disabled = clear.disabled = duration.disabled = seek.disabled = !sound.buffer;
    }
  };
  play.onclick = async () => {
    try { if (sound.player.paused) await sound.play(); else sound.player.pause(); }
    catch (error) { status.textContent = `Playback failed: ${error instanceof Error ? error.message : error}`; }
  };
  clear.onclick = () => {
    sound.clear(); input("audio-file").value = "";
    play.disabled = clear.disabled = duration.disabled = seek.disabled = true;
    seek.value = "0"; status.textContent = "Works without audio. Files stay on this device."; reset();
  };
  seek.oninput = () => { sound.player.currentTime = Number(seek.value); reset(); };
  input("audio-loop").onchange = () => { sound.player.loop = input("audio-loop").checked; };
  input("audio-volume").oninput = () => { sound.volume = Number(input("audio-volume").value); };
  duration.onclick = () => {
    if (sound.buffer) (document.querySelector("#offline-duration") as HTMLInputElement).value = String(Math.min(3600, Math.max(1, sound.buffer.duration)));
  };
  const format = (time: number) => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, "0")}`;
  return {
    get loading() { return loading; },
    get includeAudio() { return input("audio-include").checked; },
    update() {
      play.textContent = sound.player.paused ? "Play" : "Pause";
      if (sound.buffer) seek.value = String(sound.player.currentTime);
      panel.querySelector("#audio-time")!.textContent = `${format(sound.player.currentTime)} / ${format(sound.buffer?.duration ?? 0)}`;
    },
  };
}
