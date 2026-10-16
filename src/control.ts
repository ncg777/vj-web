import type { AnimationConfig } from "./animations";

export type VideoOptions = {
  duration: number;
  fps: number;
  width: number;
  height: number;
};

export function findScene(scenes: AnimationConfig[], id: string) {
  const scene = scenes.find((item) => item.id === id);
  if (!scene) throw new Error(`Unknown scene: ${id}. Use list_scenes to find scene IDs.`);
  return scene;
}

// Validate the entire update before changing anything, including hidden seed controls.
export function validateParameters(scene: AnimationConfig, values: Record<string, number>) {
  for (const [id, value] of Object.entries(values)) {
    const param = scene.params.find((item) => item.id === id);
    if (!param) throw new Error(`Unknown parameter ${id} for ${scene.id}.`);
    if (!Number.isFinite(value)) throw new Error(`${id} must be a finite number.`);
    if (param.type !== "float" && !Number.isInteger(value)) {
      throw new Error(`${id} must be an integer.`);
    }
    const min = param.min ?? (param.type === "seed" ? 0 : -Infinity);
    const max = param.max ?? (param.type === "seed" ? 999_999 : Infinity);
    if (value < min || value > max) throw new Error(`${id} must be between ${min} and ${max}.`);
  }
}

export function describeScene(scene: AnimationConfig) {
  return {
    id: scene.id,
    name: scene.name,
    description: scene.description,
    requiresFloatTargets: scene.stateful ?? false,
    parameters: scene.params.map(({ id, label, type, value, min, max, step }) => ({
      id, label, type, default: value,
      min: min ?? (type === "seed" ? 0 : undefined),
      max: max ?? (type === "seed" ? 999_999 : undefined),
      step,
    })),
  };
}

export function validateVideoOptions(options: VideoOptions) {
  const { duration, fps, width, height } = options;
  if (!Number.isFinite(duration) || duration < 1 || duration > 3600) {
    throw new Error("Duration must be between 1 and 3600 seconds.");
  }
  if (![30, 60].includes(fps)) throw new Error("FPS must be 30 or 60.");
  if (!Number.isInteger(width) || !Number.isInteger(height) ||
      width < 16 || height < 16 || width > 3840 || height > 2160 || width % 2 || height % 2) {
    throw new Error("Video dimensions must be even integers from 16×16 through 3840×2160.");
  }
}
