import smokeFragment from "./shaders/turbulent_smoke.frag?raw";
import type { AnimationConfig, ParamSpec } from "./animations";

const variants = [
  { id: "smoke-rainbow-plumes", name: "Smoke Rainbow Plumes", description: "Hot pink, cyan and golden smoke billowing upward from rainbow vents.", turbulence: 1.2, buoyancy: 1.0, lifetime: 0.993 },
  { id: "gas-chromatic-collision", name: "Gas Chromatic Collision", description: "Opposing multicolored gas jets collide and roll into turbulent mixing clouds.", turbulence: 1.5, buoyancy: 0.15, lifetime: 0.992 },
  { id: "smoke-nebula-vortex", name: "Smoke Nebula Vortex", description: "Luminous violet, turquoise and orange gases spiral into a cosmic vortex.", turbulence: 1.0, buoyancy: 0.0, lifetime: 0.996 },
  { id: "gas-neon-fog", name: "Gas Neon Fog", description: "Low banks of electric rainbow fog tumble in rolling crosswinds.", turbulence: 1.8, buoyancy: 0.25, lifetime: 0.995 },
  { id: "smoke-orbital-trails", name: "Smoke Orbital Trails", description: "Orbiting emitters leave curling ribbons of vivid smoke that disperse into wisps.", turbulence: 1.3, buoyancy: 0.1, lifetime: 0.991 },
];

function control(id: string, label: string, uniform: string, value: number, min: number, max: number, step: number): ParamSpec {
  return { id, label, uniform, type: "float", value, min, max, step };
}

export const smokeAnimations: AnimationConfig[] = variants.map((variant, style) => ({
  id: variant.id,
  name: variant.name,
  description: variant.description,
  fragment: smokeFragment.replace("uniform int uStyle;", `const int uStyle = ${style};`),
  resolutionUniform: "uResolution",
  timeUniform: "uTime",
  timeMode: "seconds",
  stateful: true,
  bufferSize: 384,
  params: [
    control("speed", "Flow Speed", "uSpeed", 1, 0, 3, 0.05),
    control("turbulence", "Turbulence", "uTurbulence", variant.turbulence, 0, 4, 0.05),
    control("eddyScale", "Eddy Scale", "uEddyScale", 1, 0.4, 3, 0.05),
    control("buoyancy", "Buoyancy", "uBuoyancy", variant.buoyancy, -1, 2, 0.05),
    control("wind", "Crosswind", "uWind", 0, -1, 1, 0.05),
    control("emission", "Gas Emission", "uEmission", 1, 0, 3, 0.05),
    control("width", "Emitter Width", "uWidth", 1, 0.4, 2.5, 0.05),
    control("persistence", "Smoke Persistence", "uPersistence", variant.lifetime, 0.97, 0.999, 0.001),
    control("density", "Optical Density", "uDensity", 1.3, 0.2, 3, 0.05),
    control("hue", "Hue Shift", "uHue", 0, 0, 1, 0.01),
    control("glow", "Scattering Glow", "uGlow", 0.8, 0, 2, 0.05),
    { id: "seed", label: "Seed", uniform: "uSeed", type: "seed", value: 0 },
  ],
}));
