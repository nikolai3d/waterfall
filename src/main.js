// Waterfall — a GPU MLS-MPM fluid simulation in WebGL2.
//
// Pipeline per substep (all on GPU, particle/grid state in float textures):
//   1. P2G-1  scatter mass + momentum to the grid (additive point rendering)
//   2. density  per-particle density/pressure via weakly compressible EOS
//   3. P2G-2  scatter pressure + viscosity forces to the grid
//   4. grid   momentum -> velocity, gravity, boundary conditions
//   5. G2P    gather velocity + affine matrix, advect, spawn/recycle

import { makeShaders, ROCKS, gridLayout } from './shaders.js';

const params = new URLSearchParams(location.search);
const GRID_SIZES = [32, 64, 96, 128];

let GRID = parseInt(params.get('g') || '64', 10);
if (!GRID_SIZES.includes(GRID)) GRID = 64;
let PTEX = parseInt(params.get('p') || '256', 10);
if (!(PTEX >= 4 && PTEX <= 2048)) PTEX = 256;
const LIFE = parseInt(params.get('l') || '2600', 10);

// The CFL clamp caps velocity in cells/substep, so finer grids need more
// substeps per frame to move at the same world-space speed.
const defaultSubsteps = () => Math.max(1, Math.round(GRID / 32));
const S_EXPLICIT = params.has('s');
let SUBSTEPS = S_EXPLICIT ? parseInt(params.get('s'), 10) : defaultSubsteps();

let N = PTEX * PTEX;
let GTEX = gridLayout(GRID).GTEX;

// Rocks: canonical state in world units ([-1,1] cube, xyz center + radius),
// draggable at runtime; the ROCKS reference layout is defined at the 64 grid.
// Grid-unit copies are derived for the shaders (uniforms) and pool seeding.
const rocksW = ROCKS.map(([x, y, z, r]) => [x / 32 - 1, y / 32 - 1, z / 32 - 1, r / 32]);
let rocks = []; // grid units, mirrors rocksW
const rockData = new Float32Array(rocksW.length * 4); // uRocks upload
const rockVel = new Float32Array(rocksW.length * 3);  // uRockVel, grid units/substep

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

const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, depth: true });
if (!gl) fail('WebGL2 is not available in this browser.');
if (!gl.getExtension('EXT_color_buffer_float')) fail('Missing EXT_color_buffer_float (cannot render to float textures).');
if (!gl.getExtension('EXT_float_blend')) fail('Missing EXT_float_blend (cannot blend into float textures).');
if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < 5) fail('Need at least 5 draw buffers.');

// ---------------------------------------------------------------------------
// GL helpers

function compile(vsSrc, fsSrc, label) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      fail(`${label} ${type === gl.VERTEX_SHADER ? 'VS' : 'FS'}: ${gl.getShaderInfoLog(s)}`);
    }
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    fail(`${label} link: ${gl.getProgramInfoLog(p)}`);
  }
  return p;
}

const uniformCache = new Map();
function u(prog, name) {
  let m = uniformCache.get(prog);
  if (!m) { m = new Map(); uniformCache.set(prog, m); }
  if (!m.has(name)) m.set(name, gl.getUniformLocation(prog, name));
  return m.get(name);
}

function createTex(w, h, data = null, internal = gl.RGBA32F, filter = gl.NEAREST) {
  const fmt = {
    [gl.RGBA32F]: [gl.RGBA, gl.FLOAT],
    [gl.R32F]: [gl.RED, gl.FLOAT],
    [gl.RG16F]: [gl.RG, gl.HALF_FLOAT],
    [gl.RGBA8]: [gl.RGBA, gl.UNSIGNED_BYTE],
  }[internal];
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, fmt[0], fmt[1], data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function createDepthTex(w, h) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function createFBO(textures, depthTex = null) {
  const f = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  const bufs = textures.map((t, i) => {
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
    return gl.COLOR_ATTACHMENT0 + i;
  });
  gl.drawBuffers(bufs);
  if (depthTex) {
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTex, 0);
  }
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    fail('Framebuffer incomplete (float render targets unsupported?).');
  }
  return f;
}

function bindTex(unit, tex, prog, name) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(u(prog, name), unit);
}

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

// ---------------------------------------------------------------------------
// Simulation state (programs + textures bake GRID/PTEX/LIFE in, so all of it
// is torn down and rebuilt by initSim when the panel changes a parameter).

let progP2G1, progP2G2, progDensity, progGrid, progG2P, progBG, progPoints,
  progPointDepth, progThick, progBlur, progComposite, progBlit, progGizmo;
let programs = [];
let cur, nxt, gridA, gridB, gridAFBO, gridBFBO, densTex, densFBO;
let substepCount = 0;

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

function makeParticleSet(posData) {
  const pos = createTex(PTEX, PTEX, posData);
  const vel = createTex(PTEX, PTEX);
  const c0 = createTex(PTEX, PTEX);
  const c1 = createTex(PTEX, PTEX);
  const c2 = createTex(PTEX, PTEX);
  const fbo = createFBO([pos, vel, c0, c1, c2]);
  return { pos, vel, c0, c1, c2, fbo };
}

function deleteParticleSet(set) {
  gl.deleteFramebuffer(set.fbo);
  for (const t of [set.pos, set.vel, set.c0, set.c1, set.c2]) gl.deleteTexture(t);
}

function initSim() {
  if (programs.length) {
    for (const p of programs) gl.deleteProgram(p);
    uniformCache.clear();
    deleteParticleSet(cur);
    deleteParticleSet(nxt);
    for (const t of [gridA, gridB, densTex]) gl.deleteTexture(t);
    for (const f of [gridAFBO, gridBFBO, densFBO]) gl.deleteFramebuffer(f);
  }

  N = PTEX * PTEX;
  GTEX = gridLayout(GRID).GTEX;
  updateRockData();

  const S = makeShaders({ GRID, PTEX, LIFE });
  progP2G1 = compile(S.vsP2G1, S.fsScatter, 'p2g1');
  progP2G2 = compile(S.vsP2G2, S.fsScatter, 'p2g2');
  progDensity = compile(S.vsQuad, S.fsDensity, 'density');
  progGrid = compile(S.vsQuad, S.fsGrid, 'grid');
  progG2P = compile(S.vsQuad, S.fsG2P, 'g2p');
  progBG = compile(S.vsQuad, S.fsBackground, 'background');
  progPoints = compile(S.vsPoint, S.fsPoint, 'points');
  progPointDepth = compile(S.vsPoint, S.fsPointDepth, 'pointDepth');
  progThick = compile(S.vsThick, S.fsThick, 'thickness');
  progBlur = compile(S.vsQuad, S.fsBlur, 'blur');
  progComposite = compile(S.vsQuad, S.fsComposite, 'composite');
  progBlit = compile(S.vsQuad, S.fsBlit, 'blit');
  progGizmo = compile(S.vsGizmo, S.fsGizmo, 'gizmo');
  programs = [progP2G1, progP2G2, progDensity, progGrid, progG2P, progBG,
    progPoints, progPointDepth, progThick, progBlur, progComposite, progBlit,
    progGizmo];

  const initData = initialParticleData();
  cur = makeParticleSet(initData);
  nxt = makeParticleSet(initData);

  gridA = createTex(GTEX, GTEX); // scatter target (momentum, mass)
  gridB = createTex(GTEX, GTEX); // updated velocities
  gridAFBO = createFBO([gridA]);
  gridBFBO = createFBO([gridB]);

  densTex = createTex(PTEX, PTEX);
  densFBO = createFBO([densTex]);

  substepCount = 0;
}
initSim();

// Screen-space fluid rendering targets (recreated on resize).
let RT = null;

function createTargets(w, h) {
  if (RT) {
    for (const t of RT.textures) gl.deleteTexture(t);
    for (const f of RT.fbos) gl.deleteFramebuffer(f);
  }
  const hw = Math.max(1, Math.ceil(w / 2));
  const hh = Math.max(1, Math.ceil(h / 2));
  const sceneColor = createTex(w, h, null, gl.RGBA8, gl.LINEAR);
  const depthTex = createDepthTex(w, h);
  const waterDepth = createTex(w, h, null, gl.R32F);
  const blurA = createTex(w, h, null, gl.R32F);
  const blurB = createTex(w, h, null, gl.R32F);
  const thick = createTex(hw, hh, null, gl.RG16F, gl.LINEAR);
  RT = {
    w, h, hw, hh,
    sceneColor, waterDepth, blurA, blurB, thick,
    sceneFBO: createFBO([sceneColor], depthTex),
    waterFBO: createFBO([waterDepth], depthTex), // shares the scene depth
    blurAFBO: createFBO([blurA]),
    blurBFBO: createFBO([blurB]),
    thickFBO: createFBO([thick]),
    textures: [sceneColor, depthTex, waterDepth, blurA, blurB, thick],
    fbos: [],
  };
  RT.fbos = [RT.sceneFBO, RT.waterFBO, RT.blurAFBO, RT.blurBFBO, RT.thickFBO];
}

// ---------------------------------------------------------------------------
// Simulation step

function substep() {
  advanceRocks();
  gl.disable(gl.DEPTH_TEST);

  // 1. clear grid + P2G-1
  gl.bindFramebuffer(gl.FRAMEBUFFER, gridAFBO);
  gl.viewport(0, 0, GTEX, GTEX);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(progP2G1);
  bindTex(0, cur.pos, progP2G1, 'uPos');
  bindTex(1, cur.vel, progP2G1, 'uVel');
  bindTex(2, cur.c0, progP2G1, 'uC0');
  bindTex(3, cur.c1, progP2G1, 'uC1');
  bindTex(4, cur.c2, progP2G1, 'uC2');
  gl.drawArrays(gl.POINTS, 0, N * 27);
  gl.disable(gl.BLEND);

  // 2. density / pressure
  gl.bindFramebuffer(gl.FRAMEBUFFER, densFBO);
  gl.viewport(0, 0, PTEX, PTEX);
  gl.useProgram(progDensity);
  bindTex(0, cur.pos, progDensity, 'uPos');
  bindTex(1, gridA, progDensity, 'uGrid');
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  // 3. P2G-2 (forces, additive into the same grid)
  gl.bindFramebuffer(gl.FRAMEBUFFER, gridAFBO);
  gl.viewport(0, 0, GTEX, GTEX);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(progP2G2);
  bindTex(0, cur.pos, progP2G2, 'uPos');
  bindTex(1, cur.c0, progP2G2, 'uC0');
  bindTex(2, cur.c1, progP2G2, 'uC1');
  bindTex(3, cur.c2, progP2G2, 'uC2');
  bindTex(4, densTex, progP2G2, 'uAux');
  gl.drawArrays(gl.POINTS, 0, N * 27);
  gl.disable(gl.BLEND);

  // 4. grid update
  gl.bindFramebuffer(gl.FRAMEBUFFER, gridBFBO);
  gl.viewport(0, 0, GTEX, GTEX);
  gl.useProgram(progGrid);
  bindTex(0, gridA, progGrid, 'uGrid');
  gl.uniform4fv(u(progGrid, 'uRocks'), rockData);
  gl.uniform3fv(u(progGrid, 'uRockVel'), rockVel);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  // 5. G2P + advect
  gl.bindFramebuffer(gl.FRAMEBUFFER, nxt.fbo);
  gl.viewport(0, 0, PTEX, PTEX);
  gl.useProgram(progG2P);
  bindTex(0, cur.pos, progG2P, 'uPos');
  bindTex(1, gridB, progG2P, 'uGrid');
  gl.uniform4fv(u(progG2P, 'uRocks'), rockData);
  gl.uniform1f(u(progG2P, 'uFrame'), substepCount);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  [cur, nxt] = [nxt, cur];
  substepCount++;
}

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

const cam = { az: 0.7, el: 0.30, dist: 2.7, target: [0, -0.22, 0] };
let dragging = false, lastX = 0, lastY = 0, lastInteraction = -1e9;
let paused = false;

function cameraBasis(w, h) {
  const eye = [
    cam.target[0] + cam.dist * Math.cos(cam.el) * Math.cos(cam.az),
    cam.target[1] + cam.dist * Math.sin(cam.el),
    cam.target[2] + cam.dist * Math.cos(cam.el) * Math.sin(cam.az),
  ];
  return { eye, aspect: w / h, ...lookAt(eye, cam.target, [0, 1, 0]) };
}

// World-space ray under the mouse (same construction as fsBackground).
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
let rockDrag = null; // { i, n: plane normal, p: grab point, off, target }

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
  if (e.code === 'KeyF') renderMode = renderMode === 'ssf' ? 'points' : 'ssf';
});

let renderMode = params.get('r') === 'points' ? 'points' : 'ssf';

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
}
syncPanel();

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
  } else {
    return;
  }
  const q = new URLSearchParams(location.search);
  q.set('g', GRID);
  q.set('p', PTEX);
  history.replaceState(null, '', '?' + q.toString());
  initSim();
  syncPanel();
});

const FOV = 0.9;
const LIGHT = normalize3([0.5, 0.8, 0.3]);

// Wireframe drag gizmo overlay: ghost circle at the grab origin, line to
// the current center, circle there. Drawn last, no depth test.
function drawGizmo(pv, right, up) {
  if (!rockDrag) return;
  const c = rocksW[rockDrag.i];
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.useProgram(progGizmo);
  gl.uniformMatrix4fv(u(progGizmo, 'uPV'), false, pv);
  gl.uniform3fv(u(progGizmo, 'uA'), rockDrag.start);
  gl.uniform3f(u(progGizmo, 'uB'), c[0], c[1], c[2]);
  gl.uniform3fv(u(progGizmo, 'uCamR'), right);
  gl.uniform3fv(u(progGizmo, 'uCamU'), up);
  gl.uniform1f(u(progGizmo, 'uR'), c[3]);
  gl.drawArrays(gl.LINES, 0, 130);
  gl.disable(gl.BLEND);
}

function render() {
  const w = canvas.width, h = canvas.height;
  if (!RT || RT.w !== w || RT.h !== h) createTargets(w, h);

  const { eye, aspect, view, right, up, fwd } = cameraBasis(w, h);
  const proj = perspective(FOV, aspect, 0.05, 20);
  const pv = mul4(proj, view);
  const lightV = normalize3([
    view[0] * LIGHT[0] + view[4] * LIGHT[1] + view[8] * LIGHT[2],
    view[1] * LIGHT[0] + view[5] * LIGHT[1] + view[9] * LIGHT[2],
    view[2] * LIGHT[0] + view[6] * LIGHT[1] + view[10] * LIGHT[2],
  ]);

  // 1. scene (cube walls + rocks) into offscreen color + depth
  gl.bindFramebuffer(gl.FRAMEBUFFER, RT.sceneFBO);
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.01, 0.015, 0.02, 1);
  gl.clearDepth(1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LESS);
  gl.useProgram(progBG);
  gl.uniform3fv(u(progBG, 'uCamPos'), eye);
  gl.uniform3fv(u(progBG, 'uCamR'), right);
  gl.uniform3fv(u(progBG, 'uCamU'), up);
  gl.uniform3fv(u(progBG, 'uCamF'), fwd);
  gl.uniform2f(u(progBG, 'uRes'), w, h);
  gl.uniform1f(u(progBG, 'uTanF'), Math.tan(FOV / 2));
  gl.uniform1f(u(progBG, 'uAspect'), aspect);
  gl.uniformMatrix4fv(u(progBG, 'uPV'), false, pv);
  gl.uniform3fv(u(progBG, 'uLightW'), LIGHT);
  gl.uniform4fv(u(progBG, 'uRocks'), rockData);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  if (renderMode === 'points') {
    // Legacy view: shaded impostors straight into the scene, then blit.
    gl.useProgram(progPoints);
    bindTex(0, cur.pos, progPoints, 'uPos');
    bindTex(1, cur.vel, progPoints, 'uVel');
    gl.uniformMatrix4fv(u(progPoints, 'uProj'), false, proj);
    gl.uniformMatrix4fv(u(progPoints, 'uView'), false, view);
    gl.uniform1f(u(progPoints, 'uPointScale'), h * proj[5]);
    gl.uniform3fv(u(progPoints, 'uLightV'), lightV);
    gl.drawArrays(gl.POINTS, 0, N);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(progBlit);
    bindTex(0, RT.sceneColor, progBlit, 'uScene');
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    drawGizmo(pv, right, up);
    return;
  }

  // 2. water surface depth (z-tested against the shared scene depth)
  gl.bindFramebuffer(gl.FRAMEBUFFER, RT.waterFBO);
  gl.clearColor(0, 0, 0, 0); // 0 = no water
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(progPointDepth);
  bindTex(0, cur.pos, progPointDepth, 'uPos');
  bindTex(1, cur.vel, progPointDepth, 'uVel');
  gl.uniformMatrix4fv(u(progPointDepth, 'uProj'), false, proj);
  gl.uniformMatrix4fv(u(progPointDepth, 'uView'), false, view);
  gl.uniform1f(u(progPointDepth, 'uPointScale'), h * proj[5]);
  gl.drawArrays(gl.POINTS, 0, N);
  gl.disable(gl.DEPTH_TEST);

  // 3. depth-aware separable blur, two iterations
  const scalePx = h * proj[5];
  let src = RT.waterDepth;
  for (const [fbo, tex, dx, dy] of [
    [RT.blurAFBO, RT.blurA, 1, 0], [RT.blurBFBO, RT.blurB, 0, 1],
    [RT.blurAFBO, RT.blurA, 1, 0], [RT.blurBFBO, RT.blurB, 0, 1],
  ]) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.useProgram(progBlur);
    bindTex(0, src, progBlur, 'uDepth');
    gl.uniform2f(u(progBlur, 'uDir'), dx, dy);
    gl.uniform1f(u(progBlur, 'uScalePx'), scalePx);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    src = tex;
  }

  // 4. thickness + foam, half resolution, additive
  gl.bindFramebuffer(gl.FRAMEBUFFER, RT.thickFBO);
  gl.viewport(0, 0, RT.hw, RT.hh);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(progThick);
  bindTex(0, cur.pos, progThick, 'uPos');
  bindTex(1, cur.vel, progThick, 'uVel');
  gl.uniformMatrix4fv(u(progThick, 'uProj'), false, proj);
  gl.uniformMatrix4fv(u(progThick, 'uView'), false, view);
  gl.uniform1f(u(progThick, 'uPointScale'), RT.hh * proj[5]);
  gl.drawArrays(gl.POINTS, 0, N);
  gl.disable(gl.BLEND);

  // 5. composite to the canvas
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, w, h);
  gl.useProgram(progComposite);
  bindTex(0, RT.sceneColor, progComposite, 'uScene');
  bindTex(1, src, progComposite, 'uDepthS');
  bindTex(2, RT.thick, progComposite, 'uThick');
  gl.uniform2f(u(progComposite, 'uRes'), w, h);
  gl.uniform1f(u(progComposite, 'uTanF'), Math.tan(FOV / 2));
  gl.uniform1f(u(progComposite, 'uAspect'), aspect);
  gl.uniform3fv(u(progComposite, 'uLightV'), lightV);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  drawGizmo(pv, right, up);
}

// ---------------------------------------------------------------------------
// Main loop

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
}
window.addEventListener('resize', resize);
resize();

let frames = 0, lastFps = performance.now();

const DEBUG = params.has('dbg');
const WARM = parseInt(params.get('warm') || '0', 10);
for (let i = 0; i < WARM; i++) substep();
function debugDump() {
  const buf = new Float32Array(PTEX * PTEX * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, cur.fbo);
  gl.readBuffer(gl.COLOR_ATTACHMENT0);
  gl.readPixels(0, 0, PTEX, PTEX, gl.RGBA, gl.FLOAT, buf);
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
  errEl.textContent = `[dbg] substep=${substepCount} active=${active} avgY=${(sumY / active).toFixed(2)} minY=${minY.toFixed(2)} maxY=${maxY.toFixed(2)} midair=${midair}`;
}

function tick(now) {
  if (!paused) {
    for (let s = 0; s < SUBSTEPS; s++) substep();
    // Idle auto-orbit.
    if (now - lastInteraction > 4000 && !dragging && !rockDrag) cam.az += 0.0012;
  }
  updateDragText();
  render();

  frames++;
  if (DEBUG && frames % 5 === 0) debugDump();
  if (now - lastFps > 500) {
    const fps = (frames * 1000) / (now - lastFps);
    statsEl.textContent =
      `${fps.toFixed(0)} fps · ${N.toLocaleString()} particles · ${GRID}³ grid · ${SUBSTEPS} substeps · ${renderMode}` +
      (paused ? ' · paused' : '');
    frames = 0;
    lastFps = now;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
