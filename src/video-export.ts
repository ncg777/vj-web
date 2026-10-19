import { Muxer, StreamTarget } from "mp4-muxer";
import { validateVideoOptions, type VideoOptions } from "./control";

export type VideoSink = {
  write: (data: Uint8Array, position: number) => Promise<void>;
  close: () => Promise<void>;
  abort: () => Promise<void>;
};

export async function renderVideo(options: VideoOptions & {
  canvas: HTMLCanvasElement;
  audio?: AudioBuffer;
  loopAudio?: boolean;
  render: (time: number, width: number, height: number) => void;
  sink: VideoSink;
  signal?: AbortSignal;
  progress?: (completed: number, total: number) => void;
}) {
  const { canvas, render, sink, signal, progress, duration, fps, width, height } = options;
  let encoder: VideoEncoder | undefined;
  let audioEncoder: AudioEncoder | undefined;
  let writeQueue = Promise.resolve();
  let failure: unknown;
  let contextLost = false;
  const onContextLost = () => { contextLost = true; };
  canvas.addEventListener("webglcontextlost", onContextLost);
  const check = () => {
    signal?.throwIfAborted();
    if (contextLost) throw new Error("WebGL context was lost during export.");
    if (failure) throw failure;
  };
  try {
    validateVideoOptions(options);
    if (typeof VideoEncoder === "undefined" || typeof VideoFrame === "undefined") {
      throw new Error("Video export requires WebCodecs in a supported Chrome or Edge browser.");
    }
    // Level 5.2 accommodates 4K/60; the original fixed level 4.0 did not.
    const codec = width * height * fps > 1920 * 1080 * 30 ? "avc1.640034" : "avc1.640028";
    const config: VideoEncoderConfig = {
      codec, width, height, bitrate: 16_000_000, framerate: fps,
    };
    if (!(await VideoEncoder.isConfigSupported(config)).supported) {
      throw new Error("H.264 encoding is unavailable at this resolution and frame rate.");
    }
    const audio = options.audio;
    const channels = audio ? Math.min(2, audio.numberOfChannels) : 0;
    const audioConfig: AudioEncoderConfig | undefined = audio ? {
      codec: "mp4a.40.2", sampleRate: audio.sampleRate, numberOfChannels: channels, bitrate: 192_000,
    } : undefined;
    if (audioConfig && (typeof AudioEncoder === "undefined" || typeof AudioData === "undefined" ||
        !(await AudioEncoder.isConfigSupported(audioConfig)).supported)) {
      throw new Error("AAC soundtrack encoding is unavailable. Use a supported Chrome or Edge browser, or uncheck Include soundtrack in video.");
    }
    check();
    const target = new StreamTarget({
      chunked: true,
      chunkSize: 1024 * 1024,
      onData(data, position) {
        const copy = data.slice();
        writeQueue = writeQueue.then(async () => {
          check();
          await sink.write(copy, position);
        }).catch((error) => { failure ??= error; });
      },
    });
    const muxer = new Muxer({
      target, video: { codec: "avc", width, height, frameRate: fps }, fastStart: false,
      audio: audio ? { codec: "aac", sampleRate: audio.sampleRate, numberOfChannels: channels } : undefined,
      firstTimestampBehavior: "offset",
    });
    encoder = new VideoEncoder({
      output: (chunk, metadata) => {
        try { muxer.addVideoChunk(chunk, metadata); } catch (error) { failure ??= error; }
      },
      error: (error) => { failure ??= error; },
    });
    encoder.configure(config);
    if (audioConfig) {
      audioEncoder = new AudioEncoder({
        output(chunk, metadata) {
          try { muxer.addAudioChunk(chunk, metadata); } catch (error) { failure ??= error; }
        },
        error(error) { failure ??= error; },
      });
      audioEncoder.configure(audioConfig);
    }
    let audioPosition = 0;
    const audioLength = audio ? Math.ceil(duration * audio.sampleRate) : 0;
    const total = Math.ceil(duration * fps);
    for (let frame = 0; frame < total; frame++) {
      check();
      render(frame / fps, width, height);
      const videoFrame = new VideoFrame(canvas, {
        timestamp: Math.round(frame * 1_000_000 / fps),
        duration: Math.round((frame + 1) * 1_000_000 / fps) - Math.round(frame * 1_000_000 / fps),
      });
      try { encoder.encode(videoFrame, { keyFrame: frame % (fps * 2) === 0 }); }
      finally { videoFrame.close(); }
      // Interleave small audio blocks with video to keep muxer memory bounded.
      if (audio && audioEncoder) {
        const end = Math.min(audioLength, Math.ceil((frame + 1) / fps * audio.sampleRate));
        while (audioPosition < end) {
          const length = Math.min(1024, end - audioPosition);
          const data = new Float32Array(length * channels);
          for (let channel = 0; channel < channels; channel++) {
            const source = audio.getChannelData(channel);
            for (let i = 0; i < length; i++) {
              const position = audioPosition + i;
              data[channel * length + i] = options.loopAudio ? source[position % source.length] : (source[position] ?? 0);
            }
          }
          const packet = new AudioData({ format: "f32-planar", sampleRate: audio.sampleRate,
            numberOfFrames: length, numberOfChannels: channels,
            timestamp: Math.round(audioPosition * 1_000_000 / audio.sampleRate), data });
          try { audioEncoder.encode(packet); } finally { packet.close(); }
          audioPosition += length;
          if (audioEncoder.encodeQueueSize >= 8) await audioEncoder.flush();
        }
      }
      // Bound both the encoder queue and pending upload data.
      if (encoder.encodeQueueSize >= 8 || frame % fps === fps - 1) {
        await encoder.flush();
        await audioEncoder?.flush();
      }
      await writeQueue;
      check();
      progress?.(frame + 1, total);
      // A timer keeps exports progressing when the tab is in the background.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    await encoder.flush();
    await audioEncoder?.flush();
    check();
    muxer.finalize();
    await writeQueue;
    check();
    await sink.close();
  } catch (error) {
    failure ??= error;
    await writeQueue;
    await sink.abort().catch(() => {});
    throw error;
  } finally {
    if (encoder && encoder.state !== "closed") encoder.close();
    if (audioEncoder && audioEncoder.state !== "closed") audioEncoder.close();
    canvas.removeEventListener("webglcontextlost", onContextLost);
  }
}
