// Waterfall — a GPU MLS-MPM fluid simulation in the browser.
//
// This is the backend-agnostic app shell: URL parameters, camera + input,
// rock dragging, the resolution panel, HUD, and the frame loop. The
// simulation and rendering live in a backend (WebGL2 or WebGPU) behind a
// small interface: init(config), substep(), render(frame), readParticles().

import { ROCKS } from './shaders.js';

const params = new URLSearchParams(location.search);
const GRID_SIZES = [32, 64, 96, 128];

let GRID = parseInt(params.get('g') || '96', 10);
if (!GRID_SIZES.includes(GRID)) GRID = 96;
let PTEX = parseInt(params.get('p') || '384', 10);
if (!(PTEX >= 4 && PTEX <= 2048)) PTEX = 384;
const LIFE = parseInt(params.get('l') || '2600', 10);

// Volume/voxel renderer offscreen scale (?rscale=, default 0.5 — the
// raymarch is heavy at full resolution; the low-res target is load-bearing).
let RSCALE = parseFloat(params.get('rscale') || '0.5');
if (!(RSCALE >= 0.1 && RSCALE <= 1)) RSCALE = 0.5;

// Voxel renderer density threshold (?iso=, particles/cell). The blurred
// grid rests at ~4/cell but surface cells are half-empty, so ~1.5 is the
// sweet spot; garbage or out-of-range values fall back to the default.
let ISO = parseFloat(params.get('iso') || '1.5');
if (!(ISO >= 0.1 && ISO <= 16)) ISO = 1.5;
// Mesh surfaces at the volume/trace threshold by default (0.5) so strands
// and droplets read as thick as the other modes; voxel keeps 1.5 (chunky
// look needs mostly-full cells). An explicit ?iso= drives both.
const MISO = params.has('iso') ? ISO : 0.5;

// Aniso renderer elongation gain (?k=): splats stretch along velocity by
// 1 + k*min(speed/VMAX, 1). Clamped to 0..4 so fast splash particles don't
// streak; garbage or out-of-range values fall back to the default.
let K = parseFloat(params.get('k') || '1.5');
if (!(K >= 0 && K <= 4)) K = 1.5;

// Path tracer (r=trace) knobs, baked into the shader at init: ?spp= paths
// per pixel per frame (more = faster convergence, slower frames) and
// ?bounces= max path depth. Garbage or out-of-range falls back to defaults.
let SPP = parseInt(params.get('spp') || '1', 10);
if (!(SPP >= 1 && SPP <= 8)) SPP = 1;
let BOUNCES = parseInt(params.get('bounces') || '4', 10);
if (!(BOUNCES >= 1 && BOUNCES <= 8)) BOUNCES = 4;

// Droplet spray strength (?spray=), baked into both shader headers: scales
// BOTH the isolation-gated velocity jitter in G2P (all renderers benefit)
// and the isolation-shrunk splat radii (ssf/aniso/points/thickness). 0
// disables the effect exactly (identity with the no-spray look); garbage or
// out-of-range values fall back to the default 1.
let SPRAY = parseFloat(params.get('spray') || '1');
if (!(SPRAY >= 0 && SPRAY <= 2)) SPRAY = 1;
// Live value shared with the backends (rockData precedent): panel chips
// mutate it in place; backends re-upload per substep/frame — no restart.
const sprayRef = new Float32Array([SPRAY]);

// The CFL clamp caps velocity in cells/substep, so finer grids need more
// substeps per frame to move at the same world-space speed.
const defaultSubsteps = () => Math.max(1, Math.round(GRID / 32));
const S_EXPLICIT = params.has('s');
let SUBSTEPS = S_EXPLICIT ? parseInt(params.get('s'), 10) : defaultSubsteps();

let N = PTEX * PTEX;

// Rocks: canonical state in world units ([-1,1] cube, xyz center + radius),
// draggable at runtime; the ROCKS reference layout is defined at the 64 grid.
// Grid-unit copies are derived for the backends (uniforms) and pool seeding.
const rocksW = ROCKS.map(([x, y, z, r]) => [x / 32 - 1, y / 32 - 1, z / 32 - 1, r / 32]);
let rocks = []; // grid units, mirrors rocksW
const rockData = new Float32Array(rocksW.length * 4); // uploaded as uRocks
const rockVel = new Float32Array(rocksW.length * 3);  // grid units/substep

function updateRockData() {
  rocks = rocksW.map(([x, y, z, r]) =>
    [(x + 1) * GRID / 2, (y + 1) * GRID / 2, (z + 1) * GRID / 2, r * GRID / 2]);
  for (let i = 0; i < rocks.length; i++) rockData.set(rocks[i], i * 4);
}

const canvas = document.getElementById('view');
const statsEl = document.getElementById('stats');
const errEl = document.getElementById('err');

function fail(msg) {
  errEl.textContent = msg;
  errEl.style.display = 'block';
  throw new Error(msg);
}
window.addEventListener('error', (e) => {
  errEl.textContent = 'Error: ' + e.message;
  errEl.style.display = 'block';
});

// ---------------------------------------------------------------------------
// Backend selection: ?api=webgl2|webgpu overrides; otherwise prefer WebGPU
// when the browser has it, falling back to WebGL2.

const api = params.get('api') === 'webgl2' ? 'webgl2'
  : params.get('api') === 'webgpu' ? 'webgpu'
  : (navigator.gpu ? 'webgpu' : 'webgl2');

async function loadBackend(which) {
  const mod = await import(which === 'webgpu' ? './backend-webgpu.js' : './backend-webgl2.js');
  return mod.createBackend({ canvas, fail });
}

let backend;
let backendFellBack = false; // auto-selection fell back: flagged in the stats line
try {
  backend = await loadBackend(api);
} catch (e) {
  if (api === 'webgpu' && params.get('api') !== 'webgpu') {
    console.warn('WebGPU backend unavailable, falling back to WebGL2:', e);
    backend = await loadBackend('webgl2');
    backendFellBack = true;
  } else {
    throw e;
  }
}
// Persistent stats-line backend token, so a silent fallback can't masquerade
// as the requested backend in screenshots/harness assertions.
const backendLabel = backend.name + (backendFellBack ? ' (fallback)' : '');

// ---------------------------------------------------------------------------
// Initial particle layout: a shallow pool plus staggered spout spawns.

function sdRocksJS(x, y, z) {
  let d = 1e9;
  for (const [cx, cy, cz, r] of rocks) {
    d = Math.min(d, Math.hypot(x - cx, y - cy, z - cz) - r);
  }
  return d;
}

function initialParticleData() {
  const pos = new Float32Array(N * 4);
  // ~35% of particles start as a shallow pool; the rest spawn from the
  // spout, staggered so the stream flows continuously from frame one.
  const POOL = Math.floor(N * 0.35);
  const spacing = 0.63; // ~4 particles per cell at rest density
  let i = 0;
  outer:
  for (let y = 2.6; y < GRID; y += spacing) {
    for (let x = 2.5; x < GRID - 2.4; x += spacing) {
      for (let z = 2.5; z < GRID - 2.4; z += spacing) {
        if (sdRocksJS(x, y, z) < 0.4) continue;
        const j = i * 4;
        pos[j] = x + (Math.random() - 0.5) * 0.3;
        pos[j + 1] = y + (Math.random() - 0.5) * 0.3;
        pos[j + 2] = z + (Math.random() - 0.5) * 0.3;
        pos[j + 3] = (i / POOL) * LIFE * 0.95; // recycle staggered
        if (++i >= POOL) break outer;
      }
    }
  }
  for (; i < N; i++) {
    // Dormant: negative age counts up to the particle's spawn slot.
    pos[i * 4 + 3] = -(((i - POOL) / (N - POOL)) * LIFE) - 1.0;
  }
  return pos;
}

function config() {
  N = PTEX * PTEX;
  updateRockData();
  return { GRID, PTEX, LIFE, N, RSCALE, ISO, MISO, K, SPP, BOUNCES, sprayRef, initialData: initialParticleData(), rockData, rockVel };
}

backend.init(config());

// ---------------------------------------------------------------------------
// Minimal matrix math (column-major, WebGL convention)

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function normalize3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function lookAt(eye, target, up) {
  const f = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  const r = normalize3(cross3(f, up));
  const uu = cross3(r, f);
  return {
    view: new Float32Array([
      r[0], uu[0], -f[0], 0,
      r[1], uu[1], -f[1], 0,
      r[2], uu[2], -f[2], 0,
      -(r[0] * eye[0] + r[1] * eye[1] + r[2] * eye[2]),
      -(uu[0] * eye[0] + uu[1] * eye[1] + uu[2] * eye[2]),
      f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2],
      1,
    ]),
    right: r, up: uu, fwd: f,
  };
}

function mul4(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

// ---------------------------------------------------------------------------
// Camera + input

// Initial camera is URL-overridable (?az=&el=&dist=) for shareable views.
const cam = {
  az: parseFloat(params.get('az') ?? '0.7'),
  el: parseFloat(params.get('el') ?? '0.30'),
  dist: parseFloat(params.get('dist') ?? '2.7'),
  target: [0, -0.22, 0],
};
let dragging = false, lastX = 0, lastY = 0, lastInteraction = -1e9;
let paused = false;

const FOV = 0.9;
const LIGHT = normalize3([0.5, 0.8, 0.3]);

function cameraBasis(w, h) {
  const eye = [
    cam.target[0] + cam.dist * Math.cos(cam.el) * Math.cos(cam.az),
    cam.target[1] + cam.dist * Math.sin(cam.el),
    cam.target[2] + cam.dist * Math.cos(cam.el) * Math.sin(cam.az),
  ];
  return { eye, aspect: w / h, ...lookAt(eye, cam.target, [0, 1, 0]) };
}

// World-space ray under the mouse (same construction as the background
// raytrace shader).
function mouseRay(e) {
  const { eye, aspect, right, up, fwd } = cameraBasis(canvas.width, canvas.height);
  const ux = (e.clientX / canvas.clientWidth) * 2 - 1;
  const uy = 1 - (e.clientY / canvas.clientHeight) * 2;
  const t = Math.tan(FOV / 2);
  return {
    ro: eye,
    rd: normalize3([
      fwd[0] + right[0] * ux * t * aspect + up[0] * uy * t,
      fwd[1] + right[1] * ux * t * aspect + up[1] * uy * t,
      fwd[2] + right[2] * ux * t * aspect + up[2] * uy * t,
    ]),
    fwd,
  };
}

function pickRock(ro, rd) {
  let idx = -1, tBest = 1e9;
  for (let i = 0; i < rocksW.length; i++) {
    const [cx, cy, cz, r] = rocksW[i];
    const oc = [ro[0] - cx, ro[1] - cy, ro[2] - cz];
    const b = oc[0] * rd[0] + oc[1] * rd[1] + oc[2] * rd[2];
    const h = b * b - (oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - r * r);
    if (h > 0) {
      const t = -b - Math.sqrt(h);
      if (t > 0 && t < tBest) { tBest = t; idx = i; }
    }
  }
  return idx < 0 ? null : { i: idx, t: tBest };
}

// Drag state: the rock's target follows the mouse along the camera-aligned
// plane through the grab point; the rock itself chases the target at a
// capped speed in advanceRocks so the water can respond.
let rockDrag = null; // { i, n: plane normal, p: grab point, off, target, start }

const ROCK_SPEED = 0.5; // cells per substep (safely below the VMAX 0.85 clamp)

function advanceRocks() {
  rockVel.fill(0);
  if (!rockDrag) return;
  const c = rocksW[rockDrag.i];
  const d = [rockDrag.target[0] - c[0], rockDrag.target[1] - c[1], rockDrag.target[2] - c[2]];
  const len = Math.hypot(d[0], d[1], d[2]);
  if (len < 1e-6) return;
  const k = Math.min(len, ROCK_SPEED * 2 / GRID) / len;
  for (let a = 0; a < 3; a++) {
    c[a] += d[a] * k;
    rockVel[rockDrag.i * 3 + a] = d[a] * k * GRID / 2; // world -> grid units
  }
  updateRockData();
}

canvas.addEventListener('mousedown', (e) => {
  const { ro, rd, fwd } = mouseRay(e);
  const hit = pickRock(ro, rd);
  if (hit) {
    const p = [ro[0] + rd[0] * hit.t, ro[1] + rd[1] * hit.t, ro[2] + rd[2] * hit.t];
    const c = rocksW[hit.i];
    rockDrag = {
      i: hit.i, n: fwd, p,
      off: [c[0] - p[0], c[1] - p[1], c[2] - p[2]],
      target: [c[0], c[1], c[2]],
      start: [c[0], c[1], c[2]], // gizmo anchor
    };
    dragEl.style.display = 'block';
    moveDragEl(e);
    lastInteraction = performance.now();
  } else {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
  }
});
window.addEventListener('mouseup', () => {
  dragging = false; rockDrag = null; rockVel.fill(0);
  dragEl.style.display = 'none';
});

// Floating delta readout that follows the cursor during a rock drag.
const dragEl = document.getElementById('drag');

function moveDragEl(e) {
  dragEl.style.left = (e.clientX + 16) + 'px';
  dragEl.style.top = (e.clientY + 16) + 'px';
}

function updateDragText() {
  if (!rockDrag) return;
  const c = rocksW[rockDrag.i], s = rockDrag.start;
  const d = [c[0] - s[0], c[1] - s[1], c[2] - s[2]].map((v) => v * GRID / 2);
  dragEl.textContent =
    `Δ ${d.map((v) => v.toFixed(1)).join(', ')} · ${Math.hypot(...d).toFixed(1)} cells`;
}
window.addEventListener('mousemove', (e) => {
  if (rockDrag) {
    const { ro, rd } = mouseRay(e);
    const n = rockDrag.n;
    const denom = rd[0] * n[0] + rd[1] * n[1] + rd[2] * n[2];
    if (Math.abs(denom) > 1e-6) {
      const t = ((rockDrag.p[0] - ro[0]) * n[0] + (rockDrag.p[1] - ro[1]) * n[1] + (rockDrag.p[2] - ro[2]) * n[2]) / denom;
      if (t > 0) {
        // Keep the rock inside the walls; y may sink to center-at-floor
        // (the initial rocks sit embedded in the floor).
        const r = rocksW[rockDrag.i][3];
        const B = 1 - 4 / GRID;
        const m = Math.max(B - r, 0);
        for (let a = 0; a < 3; a++) {
          const w = ro[a] + rd[a] * t + rockDrag.off[a];
          rockDrag.target[a] = a === 1 ? Math.min(m, Math.max(-B, w)) : Math.min(m, Math.max(-m, w));
        }
      }
    }
    moveDragEl(e);
    lastInteraction = performance.now();
    return;
  }
  if (!dragging) {
    const { ro, rd } = mouseRay(e);
    canvas.style.cursor = pickRock(ro, rd) ? 'move' : '';
    return;
  }
  cam.az += (e.clientX - lastX) * 0.005;
  cam.el = Math.min(1.35, Math.max(-0.15, cam.el + (e.clientY - lastY) * 0.005));
  lastX = e.clientX; lastY = e.clientY;
  lastInteraction = performance.now();
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  cam.dist = Math.min(6, Math.max(1.4, cam.dist * Math.exp(e.deltaY * 0.001)));
  lastInteraction = performance.now();
}, { passive: false });
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { paused = !paused; e.preventDefault(); }
  if (e.code === 'KeyF') {
    // Cycle through all renderer chips in panel order.
    const modes = [...panel.querySelectorAll('button[data-r]')].map((b) => b.dataset.r);
    setRenderMode(modes[(modes.indexOf(renderMode) + 1) % modes.length]);
  }
});

let renderMode = ['points', 'ssf', 'volume', 'voxel', 'aniso', 'mesh', 'trace'].includes(params.get('r')) ? params.get('r') : 'ssf';

// r=trace is a progressive accumulator ("let it cook"): entering it pauses
// the sim so the image can converge; space toggles as always (a running sim
// resets the accumulation every frame = live noisy preview). Leaving trace
// does not auto-unpause. Gated on the EFFECTIVE mode: WebGL2 renders
// trace as its ssf fallback, which must keep running, not freeze.
function pauseIfTrace(m) {
  const eff = backend.effectiveMode ? backend.effectiveMode(m) : m;
  if (eff === 'trace') paused = true;
}
pauseIfTrace(renderMode);

function setRenderMode(m) {
  if (m !== renderMode) pauseIfTrace(m);
  renderMode = m;
  syncURL();
  syncPanel();
}

// Resolution panel: changing a value rebuilds the whole sim in place
// (camera and pause state survive); the URL stays shareable.
const panel = document.getElementById('panel');

function syncPanel() {
  for (const b of panel.querySelectorAll('button[data-g]')) {
    b.classList.toggle('on', parseInt(b.dataset.g, 10) === GRID);
  }
  for (const b of panel.querySelectorAll('button[data-p]')) {
    b.classList.toggle('on', parseInt(b.dataset.p, 10) === PTEX);
  }
  for (const b of panel.querySelectorAll('button[data-r]')) {
    b.classList.toggle('on', b.dataset.r === renderMode);
  }
  for (const b of panel.querySelectorAll('button[data-spray]')) {
    b.classList.toggle('on', parseFloat(b.dataset.spray) === SPRAY);
  }
}
syncPanel();

function syncURL() {
  const q = new URLSearchParams(location.search);
  q.set('g', GRID);
  q.set('p', PTEX);
  q.set('r', renderMode);
  q.set('spray', SPRAY);
  history.replaceState(null, '', '?' + q.toString());
}

panel.addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  if (b.dataset.g) {
    const g = parseInt(b.dataset.g, 10);
    if (g === GRID) return;
    GRID = g;
    if (!S_EXPLICIT) SUBSTEPS = defaultSubsteps();
  } else if (b.dataset.p) {
    const p = parseInt(b.dataset.p, 10);
    if (p === PTEX) return;
    PTEX = p;
  } else if (b.dataset.r) {
    // Renderer switch: no sim restart, just a different render path.
    setRenderMode(b.dataset.r);
    return;
  } else if (b.dataset.spray) {
    // Spray strength is a live uniform: no restart, takes effect next frame.
    SPRAY = parseFloat(b.dataset.spray);
    sprayRef[0] = SPRAY;
    syncURL();
    syncPanel();
    return;
  } else {
    return;
  }
  syncURL();
  backend.init(config());
  syncPanel();
});

// ---------------------------------------------------------------------------
// Main loop

function frameState() {
  const w = canvas.width, h = canvas.height;
  const { eye, aspect, view, right, up, fwd } = cameraBasis(w, h);
  const proj = perspective(FOV, aspect, 0.05, 20);
  const pv = mul4(proj, view);
  const lightV = normalize3([
    view[0] * LIGHT[0] + view[4] * LIGHT[1] + view[8] * LIGHT[2],
    view[1] * LIGHT[0] + view[5] * LIGHT[1] + view[9] * LIGHT[2],
    view[2] * LIGHT[0] + view[6] * LIGHT[1] + view[10] * LIGHT[2],
  ]);
  const gizmo = rockDrag
    ? { a: rockDrag.start, b: rocksW[rockDrag.i].slice(0, 3), r: rocksW[rockDrag.i][3] }
    : null;
  return {
    w, h, eye, right, up, fwd, view, proj, pv, aspect, lightV,
    tanF: Math.tan(FOV / 2), lightW: LIGHT, mode: renderMode, gizmo,
  };
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
}
window.addEventListener('resize', resize);
resize();

let frames = 0, lastFps = performance.now();
let totalSubsteps = 0;

// r=trace accumulation generation: any change to what a path can see —
// camera, canvas size, sim state (substeps advance), rocks, or the
// grid/particle/scale config — restarts the accumulation. The app owns the
// counter (frames since the last change, 1-based) and hands it to the
// backend as frame.accFrame; the HUD's spp readout is frames x ?spp=.
let accKey = '';
let accFrames = 0;

function traceAccum(f) {
  const eff = backend.effectiveMode ? backend.effectiveMode(renderMode) : renderMode;
  if (eff !== 'trace') {
    accKey = '';
    accFrames = 0;
    return;
  }
  const key = `${cam.az}|${cam.el}|${cam.dist}|${f.w}|${f.h}|${GRID}|${PTEX}|${RSCALE}|${totalSubsteps}|${rockData.join(',')}`;
  if (key !== accKey) {
    accKey = key;
    accFrames = 0;
  }
  accFrames++;
  f.accFrame = accFrames;
}

function step() {
  advanceRocks();
  backend.substep();
  totalSubsteps++;
}

const DEBUG = params.has('dbg');
// ?dbg also exposes a diagnostics handle for readback-based harnesses.
if (DEBUG) window.__wf = { backend, cam, params: { get GRID() { return GRID; }, get PTEX() { return PTEX; } } };
const WARM = parseInt(params.get('warm') || '0', 10);
for (let i = 0; i < WARM; i++) step();

// Benchmark mode: time ?bench=N frames after warmup, then freeze and report
// (document.title is machine-readable via the DevTools /json endpoint).
// Run Chrome with --disable-gpu-vsync --disable-frame-rate-limit so rAF
// measures throughput, not the display refresh.
const BENCH = parseInt(params.get('bench') || '0', 10);
let benchFrames = 0, benchStart = 0, benchDone = false;

function benchTick() {
  if (!BENCH || benchDone) return;
  if (benchFrames === 0) benchStart = performance.now();
  benchFrames++;
  if (benchFrames > BENCH) {
    const ms = (performance.now() - benchStart) / BENCH;
    benchDone = true;
    paused = true;
    const line = `bench ${backendLabel} g=${GRID} p=${PTEX} s=${SUBSTEPS}: ` +
      `${ms.toFixed(2)} ms/frame · ${(1000 / ms).toFixed(1)} fps`;
    statsEl.textContent = line;
    document.title = line;
  }
}

async function debugDump() {
  const buf = await backend.readParticles(); // async on WebGPU (buffer map)
  let active = 0, sumY = 0, minY = 1e9, maxY = -1e9, midair = 0;
  for (let i = 0; i < N; i++) {
    const age = buf[i * 4 + 3];
    if (age < 0) continue;
    const y = buf[i * 4 + 1];
    active++; sumY += y;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (y > GRID * 0.25 && y < GRID * 0.86) midair++;
  }
  errEl.style.display = 'block';
  errEl.textContent = `[dbg] substep=${totalSubsteps} active=${active} avgY=${(sumY / active).toFixed(2)} minY=${minY.toFixed(2)} maxY=${maxY.toFixed(2)} midair=${midair}`;
}

function tick(now) {
  if (!paused) {
    for (let s = 0; s < SUBSTEPS; s++) step();
    // Idle auto-orbit.
    if (now - lastInteraction > 4000 && !dragging && !rockDrag) cam.az += 0.0012;
  }
  updateDragText();
  const frame = frameState();
  traceAccum(frame);
  backend.render(frame);
  benchTick();

  frames++;
  if (DEBUG && frames % 5 === 0) debugDump();
  if (!benchDone && now - lastFps > 500) {
    const fps = (frames * 1000) / (now - lastFps);
    // Effective-mode honesty: when the backend doesn't implement the
    // selected mode (WebGL2 + mesh renders the ssf path), the mode token
    // reads `mesh→ssf` and the param readouts follow the EFFECTIVE mode
    // (so e.g. iso=, which does nothing there, is suppressed).
    const eff = backend.effectiveMode ? backend.effectiveMode(renderMode) : renderMode;
    const modeToken = eff === renderMode ? renderMode : `${renderMode}→${eff}`;
    statsEl.textContent =
      `${fps.toFixed(0)} fps · ${N.toLocaleString()} particles · ${GRID}³ grid · ${SUBSTEPS} substeps · ${modeToken}` +
      (eff === 'volume' || eff === 'voxel' || eff === 'trace' ? ` rscale=${RSCALE}` : '') +
      (eff === 'voxel' ? ` iso=${ISO}` : '') +
      (eff === 'mesh' ? ` iso=${MISO}` : '') +
      (eff === 'aniso' ? ` k=${K}` : '') +
      (eff === 'trace' ? ` spp=${accFrames * SPP} (x${SPP}) bounces=${BOUNCES}` : '') +
      (SPRAY !== 1 ? ` spray=${SPRAY}` : '') + ` · ${backendLabel}` +
      (paused ? ' · paused' : '');
    frames = 0;
    lastFps = now;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
