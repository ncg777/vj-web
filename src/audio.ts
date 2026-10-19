/** A seekable feature timeline shared by playback and offline rendering. */
export class Soundtrack {
  readonly player = new Audio();
  buffer?: AudioBuffer;
  private url?: string;
  private frames: Float32Array = new Float32Array();
  private generation = 0;
  private context?: AudioContext;
  private destination?: MediaStreamAudioDestinationNode;
  private gain?: GainNode;
  readonly bands = new Float32Array(64);

  constructor() { this.player.loop = true; }

  async load(file: File) {
    const generation = ++this.generation;
    const context = this.context ??= new AudioContext();
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    if (buffer.duration > 3600) throw new Error("Choose an audio file no longer than one hour.");
    const frames = await analyze(buffer);
    if (generation !== this.generation) return false;
    this.player.pause();
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = URL.createObjectURL(file);
    this.player.src = this.url;
    this.buffer = buffer;
    this.frames = frames;
    if (!this.destination) {
      const source = context.createMediaElementSource(this.player);
      this.destination = context.createMediaStreamDestination();
      this.gain = context.createGain();
      source.connect(this.gain);
      this.gain.connect(context.destination);
      // Recording and export retain the soundtrack's original volume.
      source.connect(this.destination);
    }
    return true;
  }

  clear() {
    this.generation++;
    this.player.pause();
    this.player.removeAttribute("src");
    this.player.load();
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = undefined;
    this.buffer = undefined;
    this.frames = new Float32Array();
  }

  async play() { await this.context?.resume(); await this.player.play(); }
  set volume(value: number) { if (this.gain) this.gain.gain.value = value; }
  get stream() { return this.destination?.stream; }

  sample(time: number): Float32Array {
    this.bands.fill(0);
    if (!this.buffer) return this.bands;
    const duration = this.buffer.duration;
    if (time >= duration && !this.player.loop) return this.bands;
    const position = ((time % duration) + duration) % duration * 60;
    const count = this.frames.length / 64;
    const a = Math.min(count - 1, Math.floor(position));
    const b = Math.min(count - 1, a + 1);
    const blend = position - Math.floor(position);
    for (let i = 0; i < 64; i++) this.bands[i] = this.frames[a * 64 + i] * (1 - blend) + this.frames[b * 64 + i] * blend;
    return this.bands;
  }
}

// Windowed FFT, logarithmic frequency bins, and a short release envelope.
export async function analyze(buffer: AudioBuffer): Promise<Float32Array> {
  const size = 2048;
  const count = Math.ceil(buffer.duration * 60);
  const result = new Float32Array(count * 64);
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, i) => buffer.getChannelData(i));
  const re = new Float32Array(size), im = new Float32Array(size);
  for (let frame = 0; frame < count; frame++) {
    const start = Math.round(frame * buffer.sampleRate / 60) - size / 2;
    im.fill(0);
    for (let i = 0; i < size; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[start + i] ?? 0;
      re[i] = sample / channels.length * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (size - 1)));
    }
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) [re[i], re[j]] = [re[j], re[i]];
    }
    for (let len = 2; len <= size; len <<= 1) {
      const angle = -2 * Math.PI / len;
      const cosine = Math.cos(angle), sine = Math.sin(angle);
      for (let start = 0; start < size; start += len) {
        let wr = 1, wi = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = start + j, b = a + len / 2;
          const tr = wr * re[b] - wi * im[b], ti = wr * im[b] + wi * re[b];
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
          const next = wr * cosine - wi * sine;
          wi = wr * sine + wi * cosine; wr = next;
        }
      }
    }
    for (let band = 0; band < 64; band++) {
      const low = Math.max(1, Math.floor(30 * Math.pow(600, band / 64) * size / buffer.sampleRate));
      const high = Math.min(size / 2, Math.max(low + 1, Math.ceil(30 * Math.pow(600, (band + 1) / 64) * size / buffer.sampleRate)));
      let energy = 0;
      for (let bin = low; bin < high; bin++) energy = Math.max(energy, Math.hypot(re[bin], im[bin]) * 4 / size);
      const value = Math.min(1, Math.log1p(energy * 30) / Math.log(31));
      result[frame * 64 + band] = Math.max(value, frame ? result[(frame - 1) * 64 + band] * 0.88 : 0);
    }
    if (frame % 120 === 0) await new Promise<void>(resolve => setTimeout(resolve, 0));
  }
  return result;
}
