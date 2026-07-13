---
name: verify
description: Verify waterfall changes by driving the WebGL2 app in headless Chrome and screenshotting it
---

# Verifying waterfall

No test suite; the only end-to-end check is running the app and looking at it.

## Serve + screenshot

```sh
python3 -m http.server 8123   # in the repo root; ES modules need http
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --no-first-run --window-size=900,650 \
  --screenshot=out.png --virtual-time-budget=4000 \
  "http://localhost:8123/?warm=800&dbg=1"
```

- `?warm=N` pre-runs N substeps synchronously before the first frame — use it
  to reach a steady state; do NOT rely on `--virtual-time-budget` to simulate
  time (it only advances ~7 rAF frames per virtual second).
- On the real GPU (no swiftshader flags) warm=800 takes seconds. Add
  `--use-angle=swiftshader --enable-unsafe-swiftshader` only if a GPU-less
  environment forces it; then keep warm small (`?p=128&l=800&warm=600`).
- The HUD stats line (bottom-left) and `?dbg=1` overlay are the observable
  state; page console.log does not reach the Chrome log. Stats refresh only
  every 500ms of rAF time, so under virtual time they can lag several UI
  actions behind — trust chips/URL for immediate state, stats only after a
  generous tail delay.
- `?r=points` exercises the legacy impostor render path; `?r=volume` the
  grid-density raymarcher; `?r=voxel` the grid-aligned-cube DDA renderer.
  Both volume and voxel render offscreen at `?rscale=` (default 0.5, valid
  0.1–1). The `?iso=` density threshold (default 1.5, valid 0.1–16; garbage
  or out-of-range falls back to the default) drives both voxel's solid
  cells and the mesh renderer's isosurface.
- `?r=mesh` is the marching-cubes isosurface mesh — WebGPU only: on WebGL2
  the chip stays enabled but the mode renders the `ssf` path (console note,
  no error overlay), so a `?api=webgl2&r=mesh` shot must look like `r=ssf`.
- `?r=trace` is the progressive path tracer — WebGPU only, same fallback
  contract as mesh (`trace→ssf` on WebGL2). Selecting it (URL or chip)
  auto-pauses the sim; the stats line reads
  `… trace rscale=0.5 spp=N · webgpu · paused`, where `spp=` counts
  accumulated samples (frames since the last accumulation reset × `?spp=`,
  which is 1–8 paths/pixel/frame; `?bounces=` 1–8 is max path depth — both
  baked at init, garbage falls back to defaults). The accumulation resets
  on camera orbit, rock drag, sim steps, and panel changes — assert `spp=`
  dropping to a small value to prove a reset, or growing large to prove
  convergence. "Let it cook" needs REAL time: under virtual time only a
  handful of frames accumulate, so converged shots come from a real-time
  harness run. Gotcha shared by all grid-density renderers
  (volume/voxel/mesh/trace): after a grid/particles chip change while
  paused, there is no water until the sim advances (fresh grid is empty).
- Evidence for WebGPU-only modes (`r=mesh`, `r=trace`) must pin `?api=webgpu` AND
  stats-assert BOTH the mode token and the backend name. Two markers make a
  silent fallback impossible to mistake for the real thing: when the auto
  backend selection falls back (webgpu import failed, `api` not pinned) the
  backend token reads `webgl2 (fallback)`, and when a backend renders a
  substitute for the selected mode the mode token reads `selected→effective`
  (e.g. `mesh→ssf` on WebGL2, with the inert `iso=` readout suppressed). A
  plain `mesh · webgl2` can therefore never appear — if your stats say
  `mesh · webgpu`, the mesh path really ran.
- `?r=aniso` exercises the anisotropic-ellipsoid variant of the SSF splats
  (same blur/thickness/composite pipeline); its elongation gain is `?k=`
  (default 1.5, valid 0–4; garbage or out-of-range falls back to the
  default, and `?k=0` looks like `r=ssf`).
- The stats line shows the active render mode (and, in volume/voxel/trace
  modes, the effective `rscale=`; in voxel and mesh modes also `iso=`; in
  aniso mode `k=`; in trace mode the accumulated `spp=`) — assert on it in
  harnesses to confirm a mode/param took effect.
- macOS has no `timeout` command.

## Driving the UI (clicks) headlessly

`--screenshot` can't click. Write a temporary same-origin harness page in the
repo root (e.g. `_verify_ui.html`, delete after) that iframes `/?...`, waits
for `load` (module script incl. warmup runs before it), then dispatches
`.click()` on `#panel button[data-g=...]` / `[data-p=...]` chips, synthetic
`MouseEvent`s on the canvas (drags), and keyboard events on the iframe
window, logging iframe `location.search`, `#stats` text, `#err` visibility,
and `webglcontextlost`. The hover cursor (`move` over a rock) lets the
harness *find* draggable objects by scanning mousemoves and reading
`canvas.style.cursor` — assert drags via the shift of the hover-hit centroid.

**Virtual time cannot drive frame-dependent interactions.** Under
`--screenshot --virtual-time-budget`, rAF may fire only a handful of times
total (observed: 3) no matter the budget, so anything that advances per
frame while a button is held (e.g. a dragged rock chasing its target) barely
moves, and awaiting the iframe's rAF can hang forever. For those flows run
headless Chrome in **real time** instead:

- launch `--headless=new --remote-debugging-port=9333 <url>` in the
  background (the debug port keeps it alive), sleep ~30 s, kill it;
- no screenshot flag works there, so the harness ships evidence out over
  HTTP: each `log()` line as `fetch('/HARNESS/<seq>/<encoded>')` picked up
  from the `python3 -m http.server` access log, and canvas pixels as
  `canvas.toDataURL('image/png')` POSTed (`mode: 'no-cors'`) to a tiny
  BaseHTTPRequestHandler upload server on another port — capture inside a
  `requestAnimationFrame` callback so the just-drawn buffer is read before
  the non-preserved drawing buffer clears.
