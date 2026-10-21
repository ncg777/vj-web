import interferenceFragment from "./shaders/interference_patterns.frag?raw";
import type { AnimationConfig, ParamSpec } from "./animations";

function control(id: string, label: string, uniform: string, value: number, min: number, max: number, step: number, type: "float" | "int" = "float"): ParamSpec {
  return { id, label, uniform, type, value, min, max, step };
}

const variants = [
  {
    id: "interference-ripple-duet",
    name: "Interference Ripple Duet",
    description: "Orbiting wave sources weave luminous ripples through dark cancellation fringes.",
    frequency: 8,
    controls: [control("separation", "Source Separation", "uSeparation", 0.85, 0.1, 2, 0.01)],
  },
  {
    id: "interference-moire-weave",
    name: "Interference Moiré Weave",
    description: "Two rotating line gratings beat against one another in flowing spectral moiré bands.",
    frequency: 18,
    controls: [
      control("angle", "Crossing Angle", "uAngle", 12, 0, 90, 0.5),
      control("mismatch", "Grating Mismatch", "uMismatch", 0.04, 0, 0.3, 0.005),
    ],
  },
  {
    id: "interference-spectral-slits",
    name: "Interference Spectral Slits",
    description: "A phased row of wave sources fans into prismatic diffraction lobes and fine fringes.",
    frequency: 9,
    controls: [
      control("sources", "Slit Count", "uSources", 5, 2, 8, 1, "int"),
      control("separation", "Slit Spacing", "uSeparation", 0.3, 0.08, 0.7, 0.01),
    ],
  },
  {
    id: "interference-standing-lattice",
    name: "Interference Standing Lattice",
    description: "Crossed standing waves exchange energy across glowing, Chladni-inspired nodal lattices.",
    frequency: 2,
    controls: [
      control("modeX", "Horizontal Mode", "uModeX", 3, 1, 12, 1, "int"),
      control("modeY", "Vertical Mode", "uModeY", 4, 1, 12, 1, "int"),
    ],
  },
];

export const interferenceAnimations: AnimationConfig[] = variants.map((variant, style) => ({
  id: variant.id,
  name: variant.name,
  description: variant.description,
  fragment: interferenceFragment.replace("uniform int uStyle;", `const int uStyle = ${style};`),
  resolutionUniform: "uResolution",
  timeUniform: "uTime",
  timeMode: "seconds",
  params: [
    control("speed", "Animation Speed", "uSpeed", 1, 0, 3, 0.05),
    control("frequency", "Wave Frequency", "uFrequency", variant.frequency, 0.5, 32, 0.1),
    control("scale", "Pattern Scale", "uScale", 1, 0.4, 3, 0.05),
    ...variant.controls,
    control("phase", "Relative Phase", "uPhase", 0, -3.14, 3.14, 0.01),
    control("motion", "Pattern Motion", "uMotion", 0.5, 0, 1.5, 0.05),
    control("contrast", "Fringe Contrast", "uContrast", 1.25, 0.5, 3, 0.05),
    control("hue", "Hue Shift", "uHue", 0, 0, 1, 0.01),
    control("glow", "Wave Glow", "uGlow", 0.8, 0, 2, 0.05),
    { id: "seed", label: "Seed", uniform: "uSeed", type: "seed", value: 0 },
  ],
}));
