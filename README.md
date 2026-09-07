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
