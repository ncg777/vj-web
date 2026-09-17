# GL VJ Lab (Web)

This is a Vite + TypeScript WebGL2 port of the selected GL animations.

## Quick start

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Thunderstorm

The lightning uses a procedural stepped-leader model with multiscale jagged
channels, tapered branches and forks, brief return strokes, and cloud scattering.
Each discharge keeps its shape through subsequent restrikes. This is a real-time
visual approximation, not an electrical-field or volumetric simulation.

`Strike Rate` controls event frequency; `Restrike Speed` controls the spacing of
repeat flashes within a discharge. Parameter IDs and keyboard bindings are
unchanged. `Bolt Branching` at zero disables forks.

## Colorful smoke and gas

Five scenes—Smoke Rainbow Plumes, Gas Chromatic Collision, Smoke Nebula Vortex,
Gas Neon Fog, and Smoke Orbital Trails—transport persistent multicolored gas
through turbulent eddies. Adjust flow speed, turbulence, eddy scale, buoyancy,
crosswind, emission, emitter width, persistence, optical density, hue, and glow.
Seed changes reset the gas. Density shading and scattering give the clouds soft,
lit edges. These are gas-advection visual simulations with prescribed curl flow
and density-driven buoyancy, rather than full Navier–Stokes solvers. They require
WebGL2 floating-point render targets, like the existing fluid scenes.

## Kaleidoscopic fluids

Four scenes—Fluid Vortex Mandala, Fluid Prismatic Ribbons, Fluid Cellular Jewel,
and Fluid Spiral Bloom—advect persistent rainbow dye through multiscale turbulent
flow fields. Recursive mirror folds and spiral warping form kaleidoscopic fractal
patterns. Controls adjust flow speed, turbulence, sectors, folds, scale, twist,
dye persistence/injection, hue, and luminous liquid veins. Seed changes reset the
simulation. These are dye-advection visual simulations with prescribed
divergence-free stirring, rather than full Navier–Stokes solvers. They require
WebGL2 floating-point render targets, like the existing diffusion scenes.
