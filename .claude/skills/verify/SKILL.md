---
name: verify
description: Verify waterfall changes by driving the WebGL2 app in headless Chrome and screenshotting it
---

# Verifying waterfall

No test suite; the only end-to-end check is running the app and looking at it.

## Serve + screenshot

```sh
python3 serve.py   # repo-root dev server: threaded + no-store (required)
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
  0.1–1). The `?iso=` density threshold (defaults: voxel 1.5, mesh 0.5; valid 0.1–16; garbage
  or out-of-range falls back to the default) drives both voxel's solid
  cells and the mesh renderer's isosurface.
- `?r=mesh` is the marching-cubes isosurface mesh — WebGPU only: on WebGL2
  the chip stays enabled but the mode renders the `ssf` path (console note,
  no error overlay), so a `?api=webgl2&r=mesh` shot must look like `r=ssf`.
- `?r=trace` is the progressive path tracer — WebGPU only, same fallback
  contract as mesh (`trace→ssf` on WebGL2), and the auto-pause is gated on
  the EFFECTIVE mode: on WebGL2 the fallback keeps the sim RUNNING, so a
  `?api=webgl2&r=trace` shot must look like a running `r=ssf`, not a frozen
  one. Where trace actually renders (WebGPU), selecting it (URL or chip)
  auto-pauses the sim; the stats line reads
  `… trace rscale=0.5 spp=N (xM) bounces=K · webgpu · paused`, where `spp=`
  counts accumulated samples (frames since the last accumulation reset ×
  `?spp=`), `(xM)` echoes the `?spp=` setting (1–8 paths/pixel/frame) and
  `bounces=K` the `?bounces=` setting (1–8 max path depth — both baked at
  init, garbage falls back to defaults `(x1) bounces=4`). The accumulation
  resets on camera orbit, zoom, resize, sim steps, panel changes, and rock
  drags — but a rock drag only while the sim runs (paused drags move
  nothing, so they reset nothing). Stats refresh every 500ms while the
  accumulation advances ~every frame, so the exact post-reset `spp=1` is
  never observable — assert `spp=` dropping to a small value to prove a
  reset, or growing large to prove convergence. "Let it cook" needs REAL
  time: under virtual time only a handful of frames accumulate, so
  converged shots come from a real-time harness run. Gotcha shared by all
  grid-density renderers (volume/voxel/mesh/trace): after a grid/particles
  chip change while paused, there is no water until the sim advances
  (fresh grid is empty).
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
- `?spray=` (also a live panel chip row — changing it must NOT restart the water) (0–2, default 1) scales the droplet-spray enhancement: an
  isolation-gated velocity jitter in G2P (changes the SIM, so every
  renderer sees it) plus isolation-shrunk hash-varied splat radii in
  ssf/aniso/points. `?spray=0` is an exact identity probe — sim and splats
  must look like the pre-spray build. Garbage (`?spray=banana`) or
  out-of-range falls back to the default 1, and because the ` spray=N`
  stats token is shown only when N ≠ 1, the garbage probe asserts the
  ABSENCE of a spray token (plus default-looking rendering), never
  ` spray=1`; `spray=0` and `spray=2` do show their tokens.
- The stats line shows the active render mode (and, in volume/voxel/trace
  modes, the effective `rscale=`; in voxel and mesh modes also `iso=`; in
  aniso mode `k=`; in trace mode the accumulated `spp=N (xM) bounces=K`;
  in any mode ` spray=N` when `?spray=` ≠ 1) —
  assert on it in harnesses to confirm a mode/param took effect.
- macOS has no `timeout` command.

## Sim-state diagnostics (readback beats screenshots)

`?dbg=1` exposes `window.__wf` (backend handle + live params + `rockData`,
the live rock centers/radii in grid units — needed to correlate particles
with rock surfaces after drags). For "is this a sim bug or a render bug?"
questions, read particle state directly in a real-time harness:
`await __wf.backend.readParticles()` (pos+age) twice a few seconds apart
finds stuck/hovering particles by displacement; `readC()` (WebGPU) returns
the C-matrix buffer incl. the isolation signal in `[i*12+3]` and the
stagnation counter in `[i*12+7]`; `readAux()` (both backends) returns the
density pass — `[i*4]` is per-particle ρ (≈4 in the pool, <2.8 = visually
distinct airborne bead); `readVel()` (WebGPU) the velocity buffer.

Detector that settled the "hanging droplets" bug (2026-07-13): sample
POS+AUX every 800ms and flag particles staying within 0.5 cells for ≥3
consecutive samples while ρ<2.8 and y>3. Pool-surface bobbers alias into
short runs when a wave crest briefly drops their ρ to ~2 — read the ρ
trajectory before calling something a hoverer. For the VISUAL layer, blob
tracking on a frame sequence (threshold `min(r,g,b)>150`, blobs 2–400px,
chain centroids <6px/frame) works, but every persistent track found so far
was a rock-waterline or wave-crest specular highlight — crop and look
before believing it. NOTE: the dev server must send no-store headers (use the repo's serve.py) — the plain `python3 -m http.server` lets
Chrome heuristically cache modules, so a user's tab can silently test
STALE code while fresh headless profiles test new code.

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
