import { Muxer, StreamTarget } from "mp4-muxer";
import { validateVideoOptions, type VideoOptions } from "./control";

export type VideoSink = {
  write: (data: Uint8Array, position: number) => Promise<void>;
  close: () => Promise<void>;
  abort: () => Promise<void>;
};

export async function renderVideo(options: VideoOptions & {
  canvas: HTMLCanvasElement;
  render: (time: number, width: number, height: number) => void;
  sink: VideoSink;
  signal?: AbortSignal;
  progress?: (completed: number, total: number) => void;
}) {
  const { canvas, render, sink, signal, progress, duration, fps, width, height } = options;
  let encoder: VideoEncoder | undefined;
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
    });
    encoder = new VideoEncoder({
      output: (chunk, metadata) => {
        try { muxer.addVideoChunk(chunk, metadata); } catch (error) { failure ??= error; }
      },
      error: (error) => { failure ??= error; },
    });
    encoder.configure(config);
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
      // Bound both the encoder queue and pending upload data.
      if (encoder.encodeQueueSize >= 8 || frame % fps === fps - 1) await encoder.flush();
      await writeQueue;
      check();
      progress?.(frame + 1, total);
      // A timer keeps exports progressing when the tab is in the background.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    await encoder.flush();
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
    canvas.removeEventListener("webglcontextlost", onContextLost);
  }
}
