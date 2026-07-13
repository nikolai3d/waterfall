# WebGPU vs WebGL2: performance comparison

Both backends run the same MLS-MPM pipeline and the same screen-space fluid
rendering; they differ only in how the GPU work is expressed:

| | WebGL2 backend | WebGPU backend |
| --- | --- | --- |
| P2G scatter | 1 GL point per (particle × 27 cells), additive float blending | fixed-point `atomicAdd` into a storage buffer |
| Grid | 3D grid tiled into a 2D texture | flat 3D storage buffer |
| Particle state | 5-attachment RGBA32F MRT, ping-ponged | SoA storage buffers, updated in place |
| Substep | 5 draw calls across 4 FBO switches | 6 compute dispatches in one pass |

## Method

`?bench=N` (in `src/app.js`) times N frames after `?warm` pre-simulation and
freezes with the result in the stats line and `document.title`. Runs used
headless Chrome on the real GPU with `--disable-gpu-vsync
--disable-frame-rate-limit` (so frame time measures throughput, not the
display), 1200×900 canvas, `warm=600`, `bench=300`, default substeps
(`s = g/32`), SSF rendering. Active particle count at a given substep is
identical across backends (spawning is age-scheduled), so the workloads
match exactly. Machine: Apple-silicon Mac, Chrome, 2026-07.

## Results

| config (grid / particles / substeps) | WebGL2 | WebGPU | WebGPU speedup |
| --- | --- | --- | --- |
| 64³ / 65,536 / 2  | 0.95 ms/frame (1050 fps) | 1.18 ms/frame (850 fps) | 0.81× |
| 96³ / 65,536 / 3  | 3.84 ms/frame (260 fps)  | 2.10 ms/frame (477 fps) | 1.83× |
| 128³ / 147,456 / 4 | 9.95 ms/frame (100 fps) | 5.61 ms/frame (178 fps) | 1.77× |
| 128³ / 262,144 / 4 | 17.48 ms/frame (57 fps) | 8.13 ms/frame (123 fps) | 2.15× |

## Why the results look this way

- **The scatter is the story.** WebGL2 pays 27× vertex amplification per
  particle per P2G pass (two passes per substep) plus read-modify-write
  float blending into an RGBA32F target — at 128³/s4 that is ~64M point
  primitives per frame just for scatter. The WebGPU backend replaces all of
  it with 108 integer atomics per particle, and atomics on Apple-silicon
  GPUs are cheap. That's where the ~1.8× at the heavy configs comes from.
- **Small configs are overhead-bound, not work-bound.** At 64³/s2 both
  backends run near 1 ms/frame, where per-frame fixed costs dominate. The
  WebGPU path currently submits one command buffer per substep plus one for
  rendering (7/frame) and writes uniforms per substep; WebGL2's monolithic
  command stream is marginally cheaper there. Batching all substeps into
  one submission would likely close the 0.2 ms gap, but at >800 fps on both
  backends it is irrelevant in practice.
- **Scaling headroom differs.** WebGL2 cost grows with particles × 27
  vertices × blending bandwidth; WebGPU grows with particles × atomics and
  keeps the grid in a compact buffer. The gap therefore widens exactly
  where you want more budget — high grid resolutions and particle counts
  (which is also where the water looks best; particles-per-cell is the
  quality currency).

## Recommendation

**WebGPU by default, WebGL2 as the fallback** — which is what the app now
does at startup (`navigator.gpu` auto-detect, `?api=` to override).

- At the presets where quality is limited today (96³+), WebGPU is ~1.8×
  faster, turning 100 fps into 178 fps at 128³/384², and the gap keeps
  widening past the panel's presets (2.15× at 128³ with 262k particles —
  the config WebGL2 can no longer hold 60 fps at).
- At light configs WebGPU's small fixed overhead is invisible (both far
  above refresh rate).
- Keeping the WebGL2 backend costs little (the app shell is shared; the
  GLSL is unchanged) and covers browsers without WebGPU (older Safari,
  Firefox stable as of mid-2026).
