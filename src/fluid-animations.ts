import fluidFragment from "./shaders/kaleidoscopic_fluid.frag?raw";
import type { AnimationConfig, ParamSpec } from "./animations";

const variants = [
  { id: "fluid-vortex-mandala", name: "Fluid Vortex Mandala", description: "Rainbow dye swept through turbulent vortices and nested mirrored petals.", symmetry: 8, detail: 3, twist: 0.7, scale: 1.4, turbulence: 1.8 },
  { id: "fluid-prismatic-ribbons", name: "Fluid Prismatic Ribbons", description: "Silky streams of prismatic ink stretched into recursive kaleidoscopic ribbons.", symmetry: 6, detail: 2, twist: -0.5, scale: 1.9, turbulence: 1.2 },
  { id: "fluid-cellular-jewel", name: "Fluid Cellular Jewel", description: "Colorful cellular eddies mixing inside an intricate fractal jewel.", symmetry: 12, detail: 4, twist: 0.2, scale: 2.3, turbulence: 2.5 },
  { id: "fluid-spiral-bloom", name: "Fluid Spiral Bloom", description: "Swirling liquid pigments blooming along twisting fractal spiral arms.", symmetry: 5, detail: 3, twist: 2.4, scale: 1.2, turbulence: 2.0 },
];

function control(id: string, label: string, uniform: string, value: number, min: number, max: number, step: number, type: "float" | "int" = "float"): ParamSpec {
  return { id, label, uniform, type, value, min, max, step };
}

export const fluidAnimations: AnimationConfig[] = variants.map((variant, style) => ({
  id: variant.id,
  name: variant.name,
  description: variant.description,
  // Compile each flow variant with its own style while sharing simulation code.
  fragment: fluidFragment.replace("uniform int uStyle;", `const int uStyle = ${style};`),
  resolutionUniform: "uResolution",
  timeUniform: "uTime",
  timeMode: "seconds",
  stateful: true,
  bufferSize: 384,
  params: [
    control("speed", "Flow Speed", "uSpeed", 1, 0, 3, 0.05),
    control("turbulence", "Turbulence", "uTurbulence", variant.turbulence, 0, 5, 0.05),
    control("symmetry", "Kaleidoscope Sectors", "uSymmetry", variant.symmetry, 2, 24, 1, "int"),
    control("detail", "Fractal Folds", "uDetail", variant.detail, 0, 5, 1, "int"),
    control("scale", "Pattern Scale", "uScale", variant.scale, 0.3, 4, 0.05),
    control("twist", "Spiral Twist", "uTwist", variant.twist, -4, 4, 0.05),
    control("persistence", "Dye Persistence", "uPersistence", 0.992, 0.95, 0.999, 0.001),
    control("injection", "Dye Injection", "uInjection", 1, 0, 3, 0.05),
    control("hue", "Hue Shift", "uHue", 0, 0, 1, 0.01),
    control("glow", "Liquid Vein Glow", "uGlow", 0.7, 0, 3, 0.05),
    { id: "seed", label: "Seed", uniform: "uSeed", type: "seed", value: 0 },
  ],
}));
