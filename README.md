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

## MCP server and agent video export

The local MCP server uses the official TypeScript SDK **2.3.1**, the latest stable
release verified on October 16, 2026. It supports the **2026-07-28** protocol and
older MCP clients through the SDK's compatibility support. See the
[MCP v2 documentation](https://ts.sdk.modelcontextprotocol.io/v2/).

Install with Node.js **20.19+**, then build the server and its browser app:

```bash
npm ci
npm run build:mcp
```

Add the entry in [`mcp.config.example.json`](mcp.config.example.json) to your MCP
client's configuration. Change the absolute path if your checkout is elsewhere.
The client launches `node dist-mcp/server/index.js` directly. For a manual launch,
use `npm run --silent mcp`; stdout is reserved for MCP messages and startup
information goes to stderr. The server can start from any working directory.

Ask the agent to call `get_connection`, then open its URL in Chrome or Edge.
The page displays **MCP: connected**. Keep that tab open while controlling scenes
or rendering video. One browser tab connects at a time; close it before connecting
another. The URL contains a random connection token and changes on each server
restart. The server binds only to `127.0.0.1` on an available port. Ordinary web
visits without the connection token do not enable MCP control.

| Tool | Purpose |
| --- | --- |
| `get_connection` | Browser URL, connection status, and export directory |
| `list_scenes` | Browse or search the complete scene catalog |
| `get_scene` | Parameter IDs, defaults, types, and limits |
| `get_state` | Active scene, current parameters, and browser capabilities |
| `select_scene` | Change the active scene |
| `set_parameters` | Validate and apply a batch of parameters, including seeds |
| `reset_scene` / `reseed_scene` | Reset controls or generate new procedural seeds |
| `export_video` | Start an asynchronous H.264 MP4 render |
| `get_export_status` | Check progress, completion, output path, or failure |
| `cancel_export` | Stop an export and remove its incomplete file |

The `vj-web://scenes` resource also exposes the catalog and control ranges.
Catalog tools work without a browser; live tools require the connected tab.
When editing, provide the active `sceneId` to avoid changing a scene the user has
switched away from. Parameter updates are validated together, and changing a seed
resets stateful simulations. Reset restores that browser session's defaults,
including its original randomized seed.

For example, an agent can select `reverie-seascape`, inspect its controls, adjust
them, then call:

```json
{
  "name": "export_video",
  "arguments": { "duration": 10, "fps": 60, "width": 1920, "height": 1080 }
}
```

The response contains a job `id`. Poll `get_export_status` with `{ "jobId": "…" }`
until `status` is `completed`, then use the returned `path`. Videos are saved in
`exports/` inside the checkout by default, with unique filenames. No browser save
dialog is shown for agent exports. Data streams to disk as the clip renders;
failed or cancelled jobs remove their `.part` files. Completed videos remain after
the server stops; job status is held in memory for that server session.

Exports capture the current scene and distortion settings and start at time zero.
If a soundtrack is loaded and **Include soundtrack in video** is checked, it is
included as AAC audio. Duration is 1–3600 seconds, FPS is 30 or 60, and even dimensions from 16×16
through 3840×2160 are accepted. Defaults are 10 seconds at 1080p/60. Actual codec
and resolution support depends on the browser and GPU. Stateful fluid/smoke scenes
also require floating-point WebGL render targets. Scene controls are locked during
an export; cancel through MCP to stop early. Closing the tab fails the export and
removes its incomplete file. The **Export MP4** button retains its save
dialog for manual exports.

Optional environment variables on the MCP client entry:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VJ_MCP_PORT` | `0` | Fixed loopback port, or `0` to choose an available port |
| `VJ_EXPORT_DIR` | `<checkout>/exports` | Output directory; use an absolute path |

Re-run `npm run build:mcp` after changing scenes or app code. This builds into
`dist-mcp/` separately from the existing GitHub Pages build in `docs/`.

Run the protocol/bridge checks with `npm run test:mcp`. Run the browser and video
checks with `npm run test:browser`. Browser checks use installed Edge on Windows;
elsewhere install Chromium with `npx playwright install chromium`. Set
`VJ_BROWSER_CHANNEL=chrome` to use installed Chrome instead.

## Psychedelic distortion and optional soundtrack

Enable **Psychedelic distortion** to process any scene through an independent
visual layer. Scene parameters are not changed. Choose **Liquid warp** for flowing
displacement and spectral ripples, **Spectral kaleidoscope** for twisting radial
folds, or **Psychedelic feedback** for swirling trails and echoes. Adjust intensity,
pattern scale, color separation, trail persistence (feedback mode), and audio
sensitivity. Intensity zero bypasses the layer entirely. Distortion is off by default.

Audio is optional: without a file, the effects animate on their own. Load a local
audio file to drive their spatial patterns with its frequency spectrum. Files stay
on your device. Use Play/Pause, the seek bar, Loop soundtrack, and preview volume
to explore the track. Browser-decodable audio files up to one hour are supported.
Playback time controls the animation while a track is loaded; removing it resumes
free-running animation. Seeking and track loops restart fluid and feedback history
rather than reconstructing earlier simulation states.

To save a video with music:

1. Load a soundtrack and enable the desired distortion.
2. Leave **Include soundtrack in video** checked.
3. Click **Use track length for export**, or choose a custom duration under **Export video**.
4. Choose resolution and FPS, then click **Export MP4** and choose a save location.

Export always starts at the beginning, regardless of preview position. The same
precomputed spectrum timeline drives preview and offline frames. Short soundtracks
repeat when Loop is enabled; otherwise the remaining video has silence. Preview
volume does not alter the exported soundtrack. Uncheck Include soundtrack to
export silent visuals while retaining audio-driven distortion. AAC soundtrack
encoding requires browser support; an error explains how to export silently if
it is unavailable. Live **Record** also includes the soundtrack when checked and
starts playback from the current position if paused.

`npm run test:audio` checks frequency analysis and timeline sampling. Browser tests
also decode exported video and audio to check effects, soundtrack inclusion,
looping, and silence after the track ends.

## Reverie Seascape

Reverie Seascape combines the Seascape/Acidscape wave surface with Fractal
Reverie's drifting color fields on the water and Plasma Oil Diffraction across
the sky. The sky reflects in the wave faces, with haze blending the distant
water into the horizon. Water and sky have separate scale, speed, hue, and
brightness controls; `Sky Reflection` adjusts the reflected sky on the water.
`Time Scale` controls the whole scene, and `Reseed` changes all procedural fields.

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

## Interference patterns

Four scenes explore animated wave superposition: **Interference Ripple Duet**
combines two orbiting circular sources, **Interference Moiré Weave** overlaps
rotating line gratings, **Interference Spectral Slits** sums a phased row of
two to eight sources, and **Interference Standing Lattice** mixes crossed
standing waves into Chladni-inspired nodal patterns.

Adjust wave frequency, scale, relative phase, pattern motion, fringe contrast,
hue, and glow. Each scene also has controls for its source spacing, grating
angle/mismatch, slit count, or standing-wave modes. Animation Speed at zero
freezes the entire scene; Pattern Motion at zero holds the geometry while
the waves continue. Reseed changes the composition. The scenes support the
existing distortion, soundtrack, recording, and MP4 export controls and do not
require floating-point render targets.

## Kaleidoscopic fluids

Four scenes—Fluid Vortex Mandala, Fluid Prismatic Ribbons, Fluid Cellular Jewel,
and Fluid Spiral Bloom—advect persistent rainbow dye through multiscale turbulent
flow fields. Recursive mirror folds and spiral warping form kaleidoscopic fractal
patterns. Controls adjust flow speed, turbulence, sectors, folds, scale, twist,
dye persistence/injection, hue, and luminous liquid veins. Seed changes reset the
simulation. These are dye-advection visual simulations with prescribed
divergence-free stirring, rather than full Navier–Stokes solvers. They require
WebGL2 floating-point render targets, like the existing diffusion scenes.
