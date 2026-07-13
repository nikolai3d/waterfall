// WGSL sources for the WebGPU backend.
//
// Same physics as the GLSL version, restructured for compute: P2G scatter
// uses fixed-point atomicAdd on a flat 3D grid buffer (no tiling, no 27x
// point amplification), and particle state lives in SoA storage buffers
// updated in place. All constants are baked per config like the GLSL header.

import { ROCKS } from './shaders.js';

export function makeWGSL(opts) {
  const { GRID, LIFE, N, ISO } = opts; // ISO validated/defaulted in app.js
  const s = GRID / 64;
  const vec3f = (a) => `vec3f(${a.map((v) => v.toFixed(2)).join(', ')})`;
  const NROCK = ROCKS.length;

  const common = `
const GRIDI: i32 = ${GRID};
const GRIDF: f32 = ${GRID}.0;
const NCELL: u32 = ${GRID * GRID * GRID}u;
const NPART: u32 = ${N}u;

const MASS: f32 = 1.0;     // particle mass
const REST: f32 = 4.0;     // rest density (particles per cell)
const STIFF: f32 = 2.5;    // equation-of-state stiffness
const EOS_P: f32 = 4.0;    // equation-of-state power
const VISC: f32 = 0.06;    // dynamic viscosity
const GRAV: f32 = -0.010;  // gravity per substep (dt = 1)
const VMAX: f32 = 0.85;    // CFL velocity clamp (cells per substep)
const LIFE: f32 = ${LIFE.toFixed(1)}; // particle lifetime in substeps

const EMIT_P: vec3f = ${vec3f([7 * s, 59 * s, 32 * s])};   // spout position
const EMIT_R: vec3f = ${vec3f([2 * s, 1.5 * s, 13 * s])};  // spout extent
const EMIT_V: vec3f = vec3f(0.10, -0.05, 0.0);             // initial jet velocity

const PRADIUS: f32 = 0.021; // particle render radius, world units
const NROCK: i32 = ${NROCK};
const FX: f32 = 65536.0;    // fixed-point scale for grid atomics

fn cellIndex(g: vec3i) -> u32 {
  return u32((g.z * GRIDI + g.y) * GRIDI + g.x);
}

fn quadWeights(fx: vec3f) -> array<vec3f, 3> {
  var W: array<vec3f, 3>;
  W[0] = 0.5 * (vec3f(1.5) - fx) * (vec3f(1.5) - fx);
  W[1] = vec3f(0.75) - (fx - vec3f(1.0)) * (fx - vec3f(1.0));
  W[2] = 0.5 * (fx - vec3f(0.5)) * (fx - vec3f(0.5));
  return W;
}

fn hash3(n: u32) -> vec3f {
  var x = vec3u(n, n * 7919u, n * 104729u);
  x = ((x >> vec3u(8u)) ^ x.yzx) * vec3u(1103515245u);
  x = ((x >> vec3u(8u)) ^ x.yzx) * vec3u(1103515245u);
  x = ((x >> vec3u(8u)) ^ x.yzx) * vec3u(1103515245u);
  return vec3f(x) * (1.0 / 4294967295.0);
}

fn hash1(p: vec3f) -> f32 {
  return fract(sin(dot(p, vec3f(12.9898, 78.233, 37.719))) * 43758.5453);
}
`;

  // -------------------------------------------------------------------------
  // Simulation module: six kernels sharing one bind group.

  const sim = common + `
struct SimU {
  rocks: array<vec4f, ${NROCK}>,   // xyz center + radius, grid units
  rockVel: array<vec4f, ${NROCK}>, // xyz, grid units per substep
  frame: u32,
  _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> pos: array<vec4f>;  // xyz + age
@group(0) @binding(1) var<storage, read_write> vel: array<vec4f>;  // xyz + speed
@group(0) @binding(2) var<storage, read_write> cmat: array<vec4f>; // 3 per particle
@group(0) @binding(3) var<storage, read_write> aux: array<vec4f>;  // rho, p, V
@group(0) @binding(4) var<storage, read_write> gridA: array<atomic<i32>>; // 4 per cell
@group(0) @binding(5) var<storage, read_write> gridV: array<vec4f>; // v + mass
@group(0) @binding(6) var<uniform> U: SimU;

fn sdRocks(p: vec3f) -> f32 {
  var d = 1e9;
  for (var i = 0; i < NROCK; i++) {
    d = min(d, length(p - U.rocks[i].xyz) - U.rocks[i].w);
  }
  return d;
}

fn rockNormal(p: vec3f) -> vec3f {
  var best = 1e9;
  var n = vec3f(0.0, 1.0, 0.0);
  for (var i = 0; i < NROCK; i++) {
    let d = length(p - U.rocks[i].xyz) - U.rocks[i].w;
    if (d < best) { best = d; n = normalize(p - U.rocks[i].xyz); }
  }
  return n;
}

@compute @workgroup_size(256)
fn clearGrid(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= NCELL * 4u) { return; }
  atomicStore(&gridA[gid.x], 0);
}

// P2G pass 1: scatter mass and momentum (with affine term) to the grid.
@compute @workgroup_size(64)
fn p2g1(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NPART) { return; }
  let pa = pos[i];
  if (pa.w < 0.0) { return; }

  let p = pa.xyz;
  let v = vel[i].xyz;
  let C = mat3x3f(cmat[i * 3u].xyz, cmat[i * 3u + 1u].xyz, cmat[i * 3u + 2u].xyz);
  let base = vec3i(floor(p - vec3f(0.5)));
  let fx = p - vec3f(base);
  var W = quadWeights(fx);

  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var ii = 0; ii < 3; ii++) {
        let w = W[ii].x * W[j].y * W[k].z;
        let dpos = vec3f(f32(ii), f32(j), f32(k)) - fx;
        let mom = (MASS * w) * (v + C * dpos);
        let ci = cellIndex(base + vec3i(ii, j, k)) * 4u;
        atomicAdd(&gridA[ci], i32(mom.x * FX));
        atomicAdd(&gridA[ci + 1u], i32(mom.y * FX));
        atomicAdd(&gridA[ci + 2u], i32(mom.z * FX));
        atomicAdd(&gridA[ci + 3u], i32(MASS * w * FX));
      }
    }
  }
}

// Per-particle density and pressure (weakly compressible EOS, no solve).
@compute @workgroup_size(64)
fn density(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NPART) { return; }
  let pa = pos[i];
  if (pa.w < 0.0) { aux[i] = vec4f(0.0); return; }

  let base = vec3i(floor(pa.xyz - vec3f(0.5)));
  let fx = pa.xyz - vec3f(base);
  var W = quadWeights(fx);
  var rho = 0.0;
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var ii = 0; ii < 3; ii++) {
        let w = W[ii].x * W[j].y * W[k].z;
        let ci = cellIndex(base + vec3i(ii, j, k)) * 4u + 3u;
        rho += w * f32(atomicLoad(&gridA[ci])) / FX;
      }
    }
  }
  let pres = clamp(STIFF * (pow(rho / REST, EOS_P) - 1.0), -0.05, 40.0);
  aux[i] = vec4f(rho, pres, MASS / max(rho, 1e-5), 0.0);
}

// P2G pass 2: scatter internal forces fused into grid momentum
// (MLS-MPM, Hu et al. 2018, eq. 16).
@compute @workgroup_size(64)
fn p2g2(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NPART) { return; }
  let pa = pos[i];
  if (pa.w < 0.0) { return; }

  let p = pa.xyz;
  let C = mat3x3f(cmat[i * 3u].xyz, cmat[i * 3u + 1u].xyz, cmat[i * 3u + 2u].xyz);
  let a = aux[i];

  var stress = mat3x3f(
    vec3f(-a.y, 0.0, 0.0), vec3f(0.0, -a.y, 0.0), vec3f(0.0, 0.0, -a.y));
  stress += VISC * (C + transpose(C));
  let term = (-4.0 * a.z) * stress; // dt = 1, quadratic spline M^-1 = 4

  let base = vec3i(floor(p - vec3f(0.5)));
  let fx = p - vec3f(base);
  var W = quadWeights(fx);

  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var ii = 0; ii < 3; ii++) {
        let w = W[ii].x * W[j].y * W[k].z;
        let dpos = vec3f(f32(ii), f32(j), f32(k)) - fx;
        let f = term * (w * dpos);
        let ci = cellIndex(base + vec3i(ii, j, k)) * 4u;
        atomicAdd(&gridA[ci], i32(f.x * FX));
        atomicAdd(&gridA[ci + 1u], i32(f.y * FX));
        atomicAdd(&gridA[ci + 2u], i32(f.z * FX));
      }
    }
  }
}

// Grid update: momentum -> velocity, gravity, CFL clamp, boundaries.
@compute @workgroup_size(64)
fn gridUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NCELL) { return; }
  let ci = i * 4u;
  let m = f32(atomicLoad(&gridA[ci + 3u])) / FX;
  if (m <= 0.0) { gridV[i] = vec4f(0.0); return; }

  var v = vec3f(
    f32(atomicLoad(&gridA[ci])),
    f32(atomicLoad(&gridA[ci + 1u])),
    f32(atomicLoad(&gridA[ci + 2u]))) / (FX * m);
  v.y += GRAV;
  let sp = length(v);
  if (sp > VMAX) { v *= VMAX / sp; }

  let gi = i32(i);
  let cell = vec3i(gi % GRIDI, (gi / GRIDI) % GRIDI, gi / (GRIDI * GRIDI));
  if (cell.x < 2 && v.x < 0.0) { v.x = 0.0; }
  if (cell.x > GRIDI - 3 && v.x > 0.0) { v.x = 0.0; }
  if (cell.y < 2 && v.y < 0.0) { v.y = 0.0; }
  if (cell.y > GRIDI - 3 && v.y > 0.0) { v.y = 0.0; }
  if (cell.z < 2 && v.z < 0.0) { v.z = 0.0; }
  if (cell.z > GRIDI - 3 && v.z > 0.0) { v.z = 0.0; }

  // Rock BC relative to the nearest rock's velocity, so a dragged rock
  // pushes water instead of letting it pass through.
  var best = 1e9;
  var ri = 0;
  for (var r = 0; r < NROCK; r++) {
    let d = length(vec3f(cell) - U.rocks[r].xyz) - U.rocks[r].w;
    if (d < best) { best = d; ri = r; }
  }
  if (best < 0.5) {
    let n = normalize(vec3f(cell) - U.rocks[ri].xyz);
    let vn = dot(v - U.rockVel[ri].xyz, n);
    if (vn < 0.0) { v -= n * vn; }
  }
  gridV[i] = vec4f(v, m);
}

// G2P: gather velocity and affine matrix C, advect, handle spawning.
@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NPART) { return; }
  let pa = pos[i];
  let age = pa.w + 1.0;

  if (pa.w < 0.0 && age < 0.0) { // dormant, waiting for its spawn slot
    pos[i] = vec4f(pa.xyz, age);
    vel[i] = vec4f(0.0);
    cmat[i * 3u] = vec4f(0.0); cmat[i * 3u + 1u] = vec4f(0.0); cmat[i * 3u + 2u] = vec4f(0.0);
    return;
  }
  if (pa.w < 0.0 || age > LIFE) { // (re)spawn at the spout
    let r1 = hash3(i * 747796405u + U.frame * 2891336453u);
    let r2 = hash3(i * 2654435761u + U.frame * 1597334677u);
    let sv = EMIT_V + (r2 - vec3f(0.5)) * 0.06;
    pos[i] = vec4f(EMIT_P + (r1 - vec3f(0.5)) * EMIT_R, 0.0);
    vel[i] = vec4f(sv, length(sv));
    cmat[i * 3u] = vec4f(0.0); cmat[i * 3u + 1u] = vec4f(0.0); cmat[i * 3u + 2u] = vec4f(0.0);
    return;
  }

  var p = pa.xyz;
  let base = vec3i(floor(p - vec3f(0.5)));
  let fx = p - vec3f(base);
  var W = quadWeights(fx);

  var v = vec3f(0.0);
  var B = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  for (var k = 0; k < 3; k++) {
    for (var j = 0; j < 3; j++) {
      for (var ii = 0; ii < 3; ii++) {
        let w = W[ii].x * W[j].y * W[k].z;
        let dpos = vec3f(f32(ii), f32(j), f32(k)) - fx;
        let gv = gridV[cellIndex(base + vec3i(ii, j, k))].xyz;
        v += w * gv;
        B += mat3x3f(w * gv * dpos.x, w * gv * dpos.y, w * gv * dpos.z);
      }
    }
  }
  let C = B * 4.0;

  p += v; // dt = 1

  let sd = sdRocks(p);
  if (sd < 0.0) { // push out of rocks, kill inward velocity
    let n = rockNormal(p);
    p -= n * sd;
    let vn = dot(v, n);
    if (vn < 0.0) { v -= n * vn; }
  }
  p = clamp(p, vec3f(2.0), vec3f(GRIDF - 2.0));

  pos[i] = vec4f(p, age);
  vel[i] = vec4f(v, length(v));
  cmat[i * 3u] = vec4f(C[0], 0.0);
  cmat[i * 3u + 1u] = vec4f(C[1], 0.0);
  cmat[i * 3u + 2u] = vec4f(C[2], 0.0);
}
`;

  // -------------------------------------------------------------------------
  // Render module. NOTE: the backend pre-converts projection matrices to
  // WebGPU's [0,1] clip z, so frag depth is clip.z / clip.w directly.

  const renderUStruct = `
struct RenderU {
  pv: mat4x4f,
  proj: mat4x4f,
  view: mat4x4f,
  camPos: vec4f,
  camR: vec4f,
  camU: vec4f,
  camF: vec4f,
  lightW: vec4f,
  lightV: vec4f,
  res: vec4f,   // w, h, tanF, aspect
  misc: vec4f,  // pointScale (h * proj[5]), halfH * proj[5], volume target w, h
  rocks: array<vec4f, ${NROCK}>, // grid units
  gizmo: array<vec4f, 3>,        // a.xyz + active, b.xyz + radius, unused
};
`;

  const renderCommon = common + renderUStruct + `
@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> vel: array<vec4f>;
@group(0) @binding(3) var frontTex: texture_2d<f32>; // raw water depth (fsThick only)

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - vec2f(1.0), 0.0, 1.0);
}
`;

  // Shared hit-point shading for the analytic scene, appended into BOTH the
  // render module (fsBackground) and the volume module (whose traceScene
  // reuses the same look), so a wall/rock look tweak is one edit per
  // language (see shaders.js for GLSL). Intersection logic stays with each
  // shader.
  const sceneShade = `
const B: f32 = ${(1 - 4 / GRID).toFixed(6)}; // wall extent (cells 2..GRIDI-2)

// Rock: noise in rock-local coordinates (offset per rock), so the texture
// is attached to the rock and moves with it when dragged, instead of the
// rock sliding through a fixed world-space noise volume.
fn shadeRock(hit: vec3f, nrm0: vec3f, rockC: vec3f, rockId: f32, lightW: vec3f) -> vec3f {
  let cell = floor((hit - rockC) * 60.0 + vec3f(rockId * 23.0));
  let n1 = hash1(cell);
  let n2 = hash1(cell + vec3f(17.0));
  let n3 = hash1(cell + vec3f(43.0));
  let nrm = normalize(nrm0 + (vec3f(n1, n2, n3) - vec3f(0.5)) * 0.35);
  let diff = max(dot(nrm, lightW), 0.0);
  let ao = clamp((hit.y + B) * 2.5 + 0.25, 0.25, 1.0);
  let base = mix(vec3f(0.30, 0.27, 0.24), vec3f(0.42, 0.40, 0.37), n1);
  return base * (0.30 + 0.75 * diff) * ao;
}

// Wall/floor: base color with subtle grid lines every 8 sim cells and a
// faint glow along the cube edges.
fn shadeWall(hit: vec3f, lightW: vec3f) -> vec3f {
  let a = abs(hit) / B;
  var n: vec3f;
  if (a.x > a.y && a.x > a.z) { n = vec3f(-sign(hit.x), 0.0, 0.0); }
  else if (a.y > a.z) { n = vec3f(0.0, -sign(hit.y), 0.0); }
  else { n = vec3f(0.0, 0.0, -sign(hit.z)); }
  var base = vec3f(0.065, 0.075, 0.09);
  if (n.y > 0.5) { base = vec3f(0.10, 0.11, 0.125); }
  let diff = max(dot(n, lightW), 0.0);
  var col = base * (0.5 + 0.5 * diff);

  var tuv: vec2f;
  if (abs(n.x) > 0.5) { tuv = hit.yz; }
  else if (abs(n.y) > 0.5) { tuv = hit.xz; }
  else { tuv = hit.xy; }
  let g2 = abs(fract(tuv * (GRIDF / 16.0)) - vec2f(0.5)) / (GRIDF / 16.0);
  let line = smoothstep(0.004, 0.010, min(g2.x, g2.y));
  col *= mix(1.3, 1.0, line);

  let sd = vec3f(B) - abs(hit);
  let m1 = min(sd.x, min(sd.y, sd.z));
  let mx = max(sd.x, max(sd.y, sd.z));
  let mid = sd.x + sd.y + sd.z - m1 - mx;
  col += vec3f(0.10, 0.16, 0.20) * (1.0 - smoothstep(0.0, 0.025, mid));
  return col;
}
`;

  // Background: analytic raytrace of the cube interior and rock spheres,
  // writing real depth so water composites against it.
  const renderBG = `
struct BGOut {
  @location(0) col: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fsBackground(@builtin(position) frag: vec4f) -> BGOut {
  var uv = frag.xy / R.res.xy * 2.0 - vec2f(1.0);
  uv.y = -uv.y; // framebuffer y is top-down in WebGPU
  let rd = normalize(R.camF.xyz + R.camR.xyz * uv.x * R.res.z * R.res.w + R.camU.xyz * uv.y * R.res.z);
  let camPos = R.camPos.xyz;

  let inv = 1.0 / rd;
  let ta = (vec3f(-B) - camPos) * inv;
  let tb = (vec3f(B) - camPos) * inv;
  let tmin3 = min(ta, tb);
  let tmax3 = max(ta, tb);
  let t0 = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tE = min(min(tmax3.x, tmax3.y), tmax3.z);

  var o: BGOut;
  if (tE < max(t0, 0.0)) { // missed the cube: dark backdrop
    let g = 1.0 - length(uv) * 0.45;
    o.col = vec4f(vec3f(0.015, 0.02, 0.03) * g, 1.0);
    o.depth = 0.9999;
    return o;
  }

  var tR = 1e9;
  var nrm = vec3f(0.0);
  var rockC = vec3f(0.0);
  var rockId = 0.0;
  var isRock = false;
  for (var i = 0; i < NROCK; i++) {
    let c = R.rocks[i].xyz * (2.0 / GRIDF) - 1.0;
    let r = R.rocks[i].w * (2.0 / GRIDF);
    let oc = camPos - c;
    let b = dot(oc, rd);
    let h = b * b - (dot(oc, oc) - r * r);
    if (h > 0.0) {
      let t = -b - sqrt(h);
      if (t > max(t0, 0.0) && t < min(tR, tE)) {
        tR = t;
        nrm = normalize(camPos + rd * t - c);
        rockC = c;
        rockId = f32(i);
        isRock = true;
      }
    }
  }

  var hit: vec3f;
  var col: vec3f;
  if (isRock) {
    hit = camPos + rd * tR;
    col = shadeRock(hit, nrm, rockC, rockId, R.lightW.xyz);
  } else {
    hit = camPos + rd * tE; // exit face = visible back wall or floor
    col = shadeWall(hit, R.lightW.xyz);
  }

  let clip = R.pv * vec4f(hit, 1.0);
  o.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  o.col = vec4f(col, 1.0);
  return o;
}
`;

  // Water particles as shaded sphere impostors (instanced quads — WebGPU
  // has no sized point primitives).
  const renderPoints = `
struct PointVSOut {
  @builtin(position) pos: vec4f,
  @location(0) q: vec2f,
  @location(1) @interpolate(flat) centerV: vec3f,
  @location(2) @interpolate(flat) speed: f32,
  @location(3) @interpolate(flat) viewZ: f32,
};

fn pointCorner(vi: u32) -> vec2f {
  return vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
}

fn pointVS(vi: u32, ii: u32, scale: f32, sizeMul: f32) -> PointVSOut {
  var o: PointVSOut;
  let pa = pos[ii];
  if (pa.w < 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    o.q = vec2f(0.0); o.centerV = vec3f(0.0); o.speed = 0.0; o.viewZ = 0.0;
    return o;
  }
  let wp = pa.xyz * (2.0 / GRIDF) - vec3f(1.0);
  let center = (R.view * vec4f(wp, 1.0)).xyz;
  let z = max(-center.z, 0.05);
  let sizePx = clamp(scale * PRADIUS * sizeMul / z, 1.0, 96.0);
  let hs = sizePx * z / scale; // view-space half extent matching the px size
  let corner = pointCorner(vi);
  o.pos = R.proj * vec4f(center + vec3f(corner * hs, 0.0), 1.0);
  o.q = corner;
  o.centerV = center;
  o.speed = vel[ii].w / VMAX;
  o.viewZ = z;
  return o;
}

@vertex
fn vsPoints(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PointVSOut {
  return pointVS(vi, ii, R.misc.x, 1.0);
}

struct PointOut {
  @location(0) col: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fsPoints(v: PointVSOut) -> PointOut {
  let q = v.q;
  let r2 = dot(q, q);
  if (r2 > 1.0) { discard; }
  let zc = sqrt(1.0 - r2);
  let n = vec3f(q, zc);

  let fp = v.centerV + n * PRADIUS;
  let clip = R.proj * vec4f(fp, 1.0);
  var o: PointOut;
  o.depth = clamp(clip.z / clip.w, 0.0, 1.0);

  let lightV = R.lightV.xyz;
  let diff = max(dot(n, lightV), 0.0);
  let e = normalize(-fp);
  let spec = pow(max(dot(reflect(-lightV, n), e), 0.0), 40.0);
  let fres = pow(1.0 - max(dot(n, e), 0.0), 2.0);

  let deep = vec3f(0.03, 0.22, 0.42);
  let shal = vec3f(0.16, 0.55, 0.75);
  var col = mix(deep, shal, 0.25 + 0.75 * diff);
  col = mix(col, vec3f(0.55, 0.80, 0.90), fres * 0.6);
  let foam = smoothstep(0.5, 0.95, v.speed);
  col = mix(col, vec3f(0.93, 0.97, 1.0), foam * 0.85);
  col += vec3f(0.9) * spec * 0.6;
  o.col = vec4f(col, 1.0);
  return o;
}
`;

  // --- Screen-space fluid rendering ---------------------------------------

  // Depth pass: same impostors, but output linear view-space depth to an
  // r32float target (0 = no water); frag depth still set for the z-test
  // against the raytraced scene sharing the depth attachment.
  const renderSSF = `
struct DepthOut {
  @location(0) col: vec4f,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fsPointDepth(v: PointVSOut) -> DepthOut {
  let q = v.q;
  let r2 = dot(q, q);
  if (r2 > 1.0) { discard; }
  let fp = v.centerV + vec3f(q, sqrt(1.0 - r2)) * PRADIUS;
  let clip = R.proj * vec4f(fp, 1.0);
  var o: DepthOut;
  o.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  o.col = vec4f(-fp.z, 0.0, 0.0, 1.0);
  return o;
}

// Thickness pass: soft additive sprites into a half-res rg16float target.
// R = thickness, G = speed-weighted thickness (average speed -> foam).
@vertex
fn vsThick(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PointVSOut {
  return pointVS(vi, ii, R.misc.y, 1.7);
}

@fragment
fn fsThick(v: PointVSOut) -> @location(0) vec4f {
  let q = v.q;
  let r2 = dot(q, q);
  if (r2 > 1.0) { discard; }
  // Occlusion: water well behind the visible surface must not contribute
  // thickness or foam, or the pool draws its silhouette (and dilutes foam)
  // through the stream in front of it. Fragments outside the depth
  // silhouette (front = 0) are kept — they feather the edges.
  let s = vec2i(textureDimensions(frontTex));
  let px = clamp(vec2i(v.pos.xy * 2.0), vec2i(0), s - vec2i(1));
  let front = textureLoad(frontTex, px, 0).r;
  // Occluded water keeps coverage (B) but contributes no foam/absorption
  // (R, G) — culling it entirely would fade the composite to the
  // background wherever the near water is sparse (fake holes).
  var keep = 1.0;
  if (front > 0.0 && v.viewZ > front + 0.3) { keep = 0.0; }
  let f = 1.0 - r2;
  let th = f * f * PRADIUS * 2.0;
  return vec4f(th * keep, th * v.speed * keep, th, 1.0);
}

// Drag gizmo: attribute-less wireframe delta indicator (line + billboard
// circles), 130 vertices as line-list.
struct GizmoOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) ghost: f32,
};

@vertex
fn vsGizmo(@builtin(vertex_index) vi: u32) -> GizmoOut {
  var o: GizmoOut;
  let A = R.gizmo[0].xyz;
  let Bc = R.gizmo[1].xyz;
  let rad = R.gizmo[1].w;
  var p: vec3f;
  if (vi < 2u) {
    p = select(Bc, A, vi == 0u);
    o.ghost = 0.0;
  } else {
    let seg = i32(vi - 2u) / 2;
    let circle = seg / 32;
    let a = 6.2831853 * f32(seg % 32 + i32((vi - 2u) & 1u)) / 32.0;
    var c = A;
    var g = 1.0;
    if (circle == 1) { c = Bc; g = 0.0; }
    p = c + (R.camR.xyz * cos(a) + R.camU.xyz * sin(a)) * rad;
    o.ghost = g;
  }
  o.pos = R.pv * vec4f(p, 1.0);
  return o;
}

@fragment
fn fsGizmo(v: GizmoOut) -> @location(0) vec4f {
  if (v.ghost > 0.5) { return vec4f(0.55, 0.75, 0.95, 0.35); }
  return vec4f(0.80, 0.92, 1.0, 0.9);
}
`;

  // Separable depth-aware blur with the gap-fill closing (own module: it
  // has its own binding namespace).
  const blur = `
struct BlurU {
  dir: vec2f,     // (1,0) or (0,1), in texels
  scalePx: f32,   // pixels per world unit at z = 1
  _p: f32,
};
@group(0) @binding(0) var<uniform> BU: BlurU;
@group(0) @binding(1) var srcTex: texture_2d<f32>;

const NRANGE: f32 = 0.10; // reject neighbors further than this in depth

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - vec2f(1.0), 0.0, 1.0);
}

fn dfetch(t: vec2i) -> f32 {
  let s = vec2i(textureDimensions(srcTex));
  return textureLoad(srcTex, clamp(t, vec2i(0), s - vec2i(1)), 0).r;
}

@fragment
fn fsBlur(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let tx = vec2i(frag.xy);
  let dir = vec2i(BU.dir);
  var z0 = textureLoad(srcTex, tx, 0).r;

  // Gap fill (morphological closing): fill a no-water pixel only when water
  // lies on BOTH sides along this axis, taking the NEARER depth when the
  // sides disagree, so a near surface (the stream) stays continuous in
  // front of a far one (the pool). One-sided support is rejected.
  if (z0 <= 0.0) {
    var zp = 0.0;
    var zm = 0.0;
    var ip = 0;
    var im = 0;
    for (var i = 1; i <= 20; i++) {
      if (zp <= 0.0) { zp = dfetch(tx + dir * i); ip = i; }
      if (zm <= 0.0) { zm = dfetch(tx - dir * i); im = i; }
    }
    if (zp <= 0.0 || zm <= 0.0) { return vec4f(0.0); }
    z0 = min(zp, zm);
    let reach = clamp(0.045 * BU.scalePx / z0, 2.0, 20.0);
    if (f32(ip + im) > reach) { return vec4f(0.0); }
  }

  // Narrow-range filter (Truong & Yuksel, i3D 2018): clamp neighbors to the
  // center's depth band, anchored on the CENTER (not the window minimum —
  // that propagates isolated droplets as square halos through the separable
  // passes). Near neighbors pull the surface at most NRANGE nearer per
  // pass; across the four iterated passes a real near sheet wins.
  let radius = clamp(0.045 * BU.scalePx / z0, 2.0, 20.0);
  let lo = z0 - NRANGE;
  let hi = z0 + NRANGE;
  var sum = z0;
  var wsum = 1.0;
  for (var i = 1; i <= 20; i++) {
    let fi = f32(i);
    if (fi > radius) { break; }
    let g = exp(-fi * fi / (0.5 * radius * radius));
    let za = dfetch(tx + dir * i);
    let zb = dfetch(tx - dir * i);
    if (za > 0.0) { sum += clamp(za, lo, hi) * g; wsum += g; }
    if (zb > 0.0) { sum += clamp(zb, lo, hi) * g; wsum += g; }
  }
  return vec4f(sum / wsum, 0.0, 0.0, 1.0);
}
`;

  // Composite: reconstruct and shade the water surface over the scene
  // (own module: samples textures the main render module does not bind).
  const composite = common + renderUStruct + `
@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var linSamp: sampler;
@group(0) @binding(3) var depthTexS: texture_2d<f32>;
@group(0) @binding(4) var thickTex: texture_2d<f32>;

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - vec2f(1.0), 0.0, 1.0);
}

fn viewRay(frag: vec2f) -> vec3f {
  var uv = frag / R.res.xy * 2.0 - vec2f(1.0);
  uv.y = -uv.y; // framebuffer y is top-down in WebGPU
  return vec3f(uv.x * R.res.z * R.res.w, uv.y * R.res.z, -1.0);
}

fn dfetchS(t: vec2i) -> f32 {
  let s = vec2i(textureDimensions(depthTexS));
  return textureLoad(depthTexS, clamp(t, vec2i(0), s - vec2i(1)), 0).r;
}

fn viewPos(tx: vec2i, z: f32) -> vec3f {
  return viewRay(vec2f(tx) + vec2f(0.5)) * z;
}

@fragment
fn fsComposite(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let tx = vec2i(frag.xy);
  let tuv = frag.xy / R.res.xy;
  let scene = textureSampleLevel(sceneTex, linSamp, tuv, 0.0).rgb;
  let z0 = textureLoad(depthTexS, tx, 0).r;
  if (z0 <= 0.0) { return vec4f(scene, 1.0); }

  let P = viewRay(frag.xy) * z0;

  // Normal from finite differences, taking the smoother side on each axis.
  var zx1 = dfetchS(tx + vec2i(1, 0));
  if (zx1 <= 0.0 || abs(zx1 - z0) > 0.3) { zx1 = z0; }
  var zx2 = dfetchS(tx - vec2i(1, 0));
  if (zx2 <= 0.0 || abs(zx2 - z0) > 0.3) { zx2 = z0; }
  var zy1 = dfetchS(tx + vec2i(0, 1));
  if (zy1 <= 0.0 || abs(zy1 - z0) > 0.3) { zy1 = z0; }
  var zy2 = dfetchS(tx - vec2i(0, 1));
  if (zy2 <= 0.0 || abs(zy2 - z0) > 0.3) { zy2 = z0; }
  var dx: vec3f;
  var dy: vec3f;
  if (abs(zx1 - z0) < abs(zx2 - z0)) { dx = viewPos(tx + vec2i(1, 0), zx1) - P; }
  else { dx = P - viewPos(tx - vec2i(1, 0), zx2); }
  if (abs(zy1 - z0) < abs(zy2 - z0)) { dy = viewPos(tx + vec2i(0, 1), zy1) - P; }
  else { dy = P - viewPos(tx - vec2i(0, 1), zy2); }
  var n = normalize(cross(dx, dy));
  if (n.z < 0.0) { n = -n; }

  let t2 = textureSampleLevel(thickTex, linSamp, tuv, 0.0).rgb;
  let th = t2.r * 3.0;             // near-surface water only
  let speed = t2.g / max(t2.r, 1e-4);

  // Refract the background through the surface (y flipped: texture space
  // is top-down while view space is up).
  let off = vec2f(n.x, -n.y) * clamp(th, 0.0, 1.5) * 0.06;
  let refr = textureSampleLevel(sceneTex, linSamp, clamp(tuv + off, vec2f(0.001), vec2f(0.999)), 0.0).rgb;

  // Beer-Lambert absorption, red first (deep water goes blue-green).
  var col = refr * exp(vec3f(-2.6, -1.0, -0.55) * th);
  col += vec3f(0.04, 0.16, 0.24) * (1.0 - exp(-th * 2.5)); // in-scatter

  let e = normalize(-P);
  let fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, e), 0.0), 5.0);
  col = mix(col, vec3f(0.35, 0.50, 0.60), fres * 0.8);
  let hv = normalize(R.lightV.xyz + e);
  col += vec3f(0.9) * pow(max(dot(n, hv), 0.0), 120.0);

  let foam = smoothstep(0.45, 0.9, speed);
  col = mix(col, vec3f(0.93, 0.97, 1.0), foam * 0.75);

  // Sparse water (lone droplets, spray) fades toward the scene instead of
  // shading as a fully opaque sphere. Coverage uses TOTAL thickness (B),
  // so water occluded from foam/absorption still keeps the pixel wet.
  let cov = smoothstep(0.0, 0.09, t2.b * 3.0);
  return vec4f(mix(scene, col, cov), 1.0);
}
`;

  // --- Volumetric raymarcher (r=volume) ------------------------------------

  // One-cell tent blur of the sim grid's (velocity, mass) field into a small
  // dedicated (mass, |momentum|) buffer: the raw per-substep mass is too
  // speckly (4 particles/cell at rest) for a stable iso-surface.
  const volBlur = common + `
@group(0) @binding(0) var<storage, read> gridV: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dens: array<vec2f>;

@compute @workgroup_size(64)
fn blurGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= NCELL) { return; }
  let gi = i32(i);
  let cell = vec3i(gi % GRIDI, (gi / GRIDI) % GRIDI, gi / (GRIDI * GRIDI));
  // Tent weights (1,2,1)³ rather than a uniform box: thin sheets (the pool
  // is ~1 cell deep at 128³) keep enough peak density to cross the iso.
  var m = 0.0;
  var s = 0.0;
  for (var k = -1; k <= 1; k++) {
    for (var j = -1; j <= 1; j++) {
      for (var ii = -1; ii <= 1; ii++) {
        let c = clamp(cell + vec3i(ii, j, k), vec3i(0), vec3i(GRIDI - 1));
        let g = gridV[cellIndex(c)];
        let w = f32((2 - abs(ii)) * (2 - abs(j)) * (2 - abs(k)));
        m += w * g.w;
        s += w * length(g.xyz) * g.w;
      }
    }
  }
  dens[i] = vec2f(m, s) * (1.0 / 64.0);
}
`;

  // Fragment raymarch of the blurred density grid: iso-surface hit with
  // bisection refine, gradient normal, one refraction segment with
  // Beer-Lambert absorption from the marched interior density, and the
  // analytic scene (walls + rocks) continued along the refracted ray.
  // Renders into a scaled offscreen target (see misc.zw), upscaled after.
  const volume = common + renderUStruct + sceneShade + `
@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> dens: array<vec2f>; // blurred (mass, |momentum|)

const ISO: f32 = 0.5;            // iso-surface threshold, particles/cell
const STEP: f32 = ${(1 / GRID).toFixed(6)};  // 0.5 grid cells, world units
const MAXIT: i32 = ${Math.ceil(3.5 * GRID)}; // covers the cube diagonal
const STEP2: f32 = ${(2 / GRID).toFixed(6)}; // interior absorption step
const MAXIT2: i32 = ${2 * GRID};
const ABSORB: vec3f = vec3f(2.6, 1.0, 0.55);

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - vec2f(1.0), 0.0, 1.0);
}

// Trilinear sample of the blurred (mass, |momentum|) grid at a world point.
fn sampleD(p: vec3f) -> vec2f {
  let g = clamp((p + vec3f(1.0)) * (GRIDF * 0.5), vec3f(0.0), vec3f(GRIDF - 1.001));
  let b = vec3i(floor(g));
  let f = g - floor(g);
  let c00 = mix(dens[cellIndex(b)], dens[cellIndex(b + vec3i(1, 0, 0))], f.x);
  let c10 = mix(dens[cellIndex(b + vec3i(0, 1, 0))], dens[cellIndex(b + vec3i(1, 1, 0))], f.x);
  let c01 = mix(dens[cellIndex(b + vec3i(0, 0, 1))], dens[cellIndex(b + vec3i(1, 0, 1))], f.x);
  let c11 = mix(dens[cellIndex(b + vec3i(0, 1, 1))], dens[cellIndex(b + vec3i(1, 1, 1))], f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

fn density(p: vec3f) -> f32 { return sampleD(p).x; }

fn gradD(p: vec3f) -> vec3f {
  let h = 2.0 / GRIDF; // one cell
  return vec3f(
    density(p + vec3f(h, 0.0, 0.0)) - density(p - vec3f(h, 0.0, 0.0)),
    density(p + vec3f(0.0, h, 0.0)) - density(p - vec3f(0.0, h, 0.0)),
    density(p + vec3f(0.0, 0.0, h)) - density(p - vec3f(0.0, 0.0, h)));
}

// Analytic scene (walls + rocks) for a ray starting inside the cube:
// trimmed copy of fsBackground. Returns rgb + hit distance.
fn traceScene(ro: vec3f, rd: vec3f) -> vec4f {
  let inv = 1.0 / rd;
  let ta = (vec3f(-B) - ro) * inv;
  let tb = (vec3f(B) - ro) * inv;
  let tmax3 = max(ta, tb);
  let tE = min(min(tmax3.x, tmax3.y), tmax3.z);

  var tHit = tE;
  var nrm = vec3f(0.0);
  var rockC = vec3f(0.0);
  var rockId = 0.0;
  var isRock = false;
  for (var i = 0; i < NROCK; i++) {
    let c = R.rocks[i].xyz * (2.0 / GRIDF) - vec3f(1.0);
    let r = R.rocks[i].w * (2.0 / GRIDF);
    let oc = ro - c;
    let b = dot(oc, rd);
    let h = b * b - (dot(oc, oc) - r * r);
    if (h > 0.0) {
      let t = -b - sqrt(h);
      if (t > 1e-4 && t < tHit) {
        tHit = t;
        nrm = normalize(ro + rd * t - c);
        rockC = c;
        rockId = f32(i);
        isRock = true;
      }
    }
  }

  let hit = ro + rd * tHit;
  var col: vec3f;
  if (isRock) {
    col = shadeRock(hit, nrm, rockC, rockId, R.lightW.xyz);
  } else {
    col = shadeWall(hit, R.lightW.xyz);
  }
  return vec4f(col, tHit);
}

// Shared fsVolume/fsVoxel prologue: camera ray from the fragment coord,
// slab test against the wall cube (miss = dark backdrop), entry point,
// and entry-face normal.
struct RayStart {
  miss: bool,
  bg: vec3f,     // backdrop color, valid only when miss
  rd: vec3f,
  entry: vec3f,  // ray/cube entry point (the eye itself when inside)
  n0: vec3f,     // entry-face normal
};

fn cubeEnter(frag: vec2f) -> RayStart {
  var rs: RayStart;
  var uv = frag / R.misc.zw * 2.0 - vec2f(1.0);
  uv.y = -uv.y; // framebuffer y is top-down in WebGPU
  let rd = normalize(R.camF.xyz + R.camR.xyz * uv.x * R.res.z * R.res.w + R.camU.xyz * uv.y * R.res.z);
  let ro = R.camPos.xyz;
  rs.rd = rd;

  let inv = 1.0 / rd;
  let ta = (vec3f(-B) - ro) * inv;
  let tb = (vec3f(B) - ro) * inv;
  let tmin3 = min(ta, tb);
  let tmax3 = max(ta, tb);
  let t0 = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tE = min(min(tmax3.x, tmax3.y), tmax3.z);

  rs.miss = tE < max(t0, 0.0);
  if (rs.miss) { // missed the cube: dark backdrop
    let g = 1.0 - length(uv) * 0.45;
    rs.bg = vec3f(0.015, 0.02, 0.03) * g;
    return rs;
  }

  rs.entry = ro + rd * max(t0, 0.0);

  // Cube entry face; with the camera inside the cube fall back to the
  // dominant axis of -rd (only consumed if the eye's cell is already solid).
  if (t0 > 0.0) {
    if (t0 == tmin3.x) { rs.n0 = vec3f(-sign(rd.x), 0.0, 0.0); }
    else if (t0 == tmin3.y) { rs.n0 = vec3f(0.0, -sign(rd.y), 0.0); }
    else { rs.n0 = vec3f(0.0, 0.0, -sign(rd.z)); }
  } else {
    let a = abs(rd);
    if (a.x > a.y && a.x > a.z) { rs.n0 = vec3f(-sign(rd.x), 0.0, 0.0); }
    else if (a.y > a.z) { rs.n0 = vec3f(0.0, -sign(rd.y), 0.0); }
    else { rs.n0 = vec3f(0.0, 0.0, -sign(rd.z)); }
  }
  return rs;
}

@fragment
fn fsVolume(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let rs = cubeEnter(frag.xy);
  if (rs.miss) { return vec4f(rs.bg, 1.0); }
  let rd = rs.rd;
  let entry = rs.entry;

  // Analytic scene hit caps the march (water behind a rock stays behind it).
  let sc = traceScene(entry, rd);

  var tHit = -1.0;
  var t = STEP * hash1(vec3f(frag.xy, 0.0)); // dither to hide step banding
  for (var i = 0; i < MAXIT; i++) {
    t += STEP;
    if (t > sc.w) { break; }
    if (density(entry + rd * t) >= ISO) {
      // Bisection refine between the last two samples.
      var lo = t - STEP;
      var hi = t;
      for (var b = 0; b < 4; b++) {
        let mid = 0.5 * (lo + hi);
        if (density(entry + rd * mid) >= ISO) { hi = mid; } else { lo = mid; }
      }
      tHit = 0.5 * (lo + hi);
      break;
    }
  }
  if (tHit < 0.0) { return vec4f(sc.rgb, 1.0); }

  let P = entry + rd * tHit;
  var n = -normalize(gradD(P) + vec3f(1e-6));
  if (dot(n, rd) > 0.0) { n = -n; }

  // Refract; on total internal reflection fall back to reflection.
  var rd2 = refract(rd, n, 0.752); // 1 / 1.33
  if (dot(rd2, rd2) < 1e-6) { rd2 = reflect(rd, n); }
  rd2 = normalize(rd2);

  // Continue the analytic background along the refracted ray, integrating
  // Beer-Lambert absorption from the marched interior density.
  let sc2 = traceScene(P, rd2);
  var th = 0.0;
  for (var i = 0; i < MAXIT2; i++) {
    let t2 = (f32(i) + 0.5) * STEP2;
    if (t2 > sc2.w) { break; }
    th += density(P + rd2 * t2) / REST * STEP2;
  }
  var col = sc2.rgb * exp(ABSORB * (-6.0 * th));
  col += vec3f(0.04, 0.16, 0.24) * (1.0 - exp(-th * 12.0)); // in-scatter

  let e = -rd;
  let fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, e), 0.0), 5.0);
  col = mix(col, vec3f(0.35, 0.50, 0.60), fres * 0.8);
  let hv = normalize(R.lightW.xyz + e);
  col += vec3f(0.9) * pow(max(dot(n, hv), 0.0), 120.0);

  // Foam from the blurred momentum magnitude at the surface.
  let sd = sampleD(P);
  let speed = sd.y / max(sd.x, 1e-4) / VMAX;
  let foam = smoothstep(0.45, 0.9, speed);
  col = mix(col, vec3f(0.93, 0.97, 1.0), foam * 0.75);
  return vec4f(col, 1.0);
}

// --- Stylized voxel water (r=voxel) ---------------------------------------
// Same bindings and blurred density field as fsVolume, but the grid is
// rendered as literal grid-aligned cubes: a DDA (Amanatides & Woo) walks the
// ray through cells and the first cell with density >= VISO is a cube-face
// hit with an axis-aligned normal.

const VISO: f32 = ${ISO.toFixed(4)};  // per-cell density threshold (?iso=)
const MAXDDA: i32 = ${3 * GRID + 4};  // DDA worst case ~3*GRID cells

fn cellDens(c: vec3i) -> vec2f {
  if (any(c < vec3i(0)) || any(c >= vec3i(GRIDI))) { return vec2f(0.0); }
  return dens[cellIndex(c)];
}

struct VoxHit {
  t: f32,       // grid units along the ray; < 0 = miss
  n: vec3f,     // axis-aligned face normal of the hit cell
  cell: vec3i,
};

// Grid traversal from roG (grid units) along rd (unit direction — uniform
// world->grid scale keeps directions unchanged), capped at tEnd (grid units).
fn voxMarch(roG0: vec3f, rd0: vec3f, tEnd: f32, n0: vec3f) -> VoxHit {
  // Degenerate-axis guard preserves the component's sign (sign(0) -> +1)
  // so near-axis rays still step the right way.
  let sgn = select(sign(rd0), vec3f(1.0), rd0 == vec3f(0.0));
  let rd = sgn * max(abs(rd0), vec3f(1e-6));
  // Clamp the entry into the grid so cell and tMax agree on the origin.
  let roG = clamp(roG0, vec3f(0.0), vec3f(GRIDF));
  var cell = clamp(vec3i(floor(roG)), vec3i(0), vec3i(GRIDI - 1));
  let stp = vec3i(sgn);
  let inv = 1.0 / rd;
  let tDelta = abs(inv);
  var tMax = (vec3f(cell) + max(sgn, vec3f(0.0)) - roG) * inv;
  var t = 0.0;
  var n = n0;
  var h: VoxHit;
  h.t = -1.0; h.n = n0; h.cell = cell;
  if (tEnd <= 0.0) { return h; }
  for (var i = 0; i < MAXDDA; i++) {
    if (cellDens(cell).x >= VISO) { h.t = t; h.n = n; h.cell = cell; return h; }
    if (tMax.x < tMax.y && tMax.x < tMax.z) {
      t = tMax.x; tMax.x += tDelta.x; cell.x += stp.x; n = vec3f(-f32(stp.x), 0.0, 0.0);
    } else if (tMax.y < tMax.z) {
      t = tMax.y; tMax.y += tDelta.y; cell.y += stp.y; n = vec3f(0.0, -f32(stp.y), 0.0);
    } else {
      t = tMax.z; tMax.z += tDelta.z; cell.z += stp.z; n = vec3f(0.0, 0.0, -f32(stp.z));
    }
    if (t > tEnd || any(cell < vec3i(0)) || any(cell >= vec3i(GRIDI))) { return h; }
  }
  return h;
}

// Water path length (world units) through solid (>= VISO) cells along a ray.
fn voxThickness(roG0: vec3f, rd0: vec3f, tEnd: f32) -> f32 {
  let sgn = select(sign(rd0), vec3f(1.0), rd0 == vec3f(0.0));
  let rd = sgn * max(abs(rd0), vec3f(1e-6));
  let roG = clamp(roG0, vec3f(0.0), vec3f(GRIDF));
  var cell = clamp(vec3i(floor(roG)), vec3i(0), vec3i(GRIDI - 1));
  let stp = vec3i(sgn);
  let inv = 1.0 / rd;
  let tDelta = abs(inv);
  var tMax = (vec3f(cell) + max(sgn, vec3f(0.0)) - roG) * inv;
  var t = 0.0;
  var th = 0.0;
  for (var i = 0; i < MAXDDA; i++) {
    let tn = min(min(tMax.x, tMax.y), tMax.z);
    if (cellDens(cell).x >= VISO) { th += max(min(tn, tEnd) - t, 0.0); }
    if (tn > tEnd) { break; }
    if (tMax.x < tMax.y && tMax.x < tMax.z) { tMax.x += tDelta.x; cell.x += stp.x; }
    else if (tMax.y < tMax.z) { tMax.y += tDelta.y; cell.y += stp.y; }
    else { tMax.z += tDelta.z; cell.z += stp.z; }
    t = tn;
    if (any(cell < vec3i(0)) || any(cell >= vec3i(GRIDI))) { break; }
  }
  return th * (2.0 / GRIDF);
}

@fragment
fn fsVoxel(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let rs = cubeEnter(frag.xy);
  if (rs.miss) { return vec4f(rs.bg, 1.0); }
  let rd = rs.rd;
  let entry = rs.entry;

  // Analytic scene hit caps the DDA (water behind a rock stays behind it).
  let sc = traceScene(entry, rd);

  let G2 = GRIDF * 0.5;
  let roG = (entry + vec3f(1.0)) * G2;
  let hit = voxMarch(roG, rd, sc.w * G2, rs.n0);
  if (hit.t < 0.0) { return vec4f(sc.rgb, 1.0); }

  let P = entry + rd * (hit.t / G2);
  let n = hit.n;

  // Chunky per-face diffuse; top faces read lighter, bottoms darker.
  let diff = max(dot(n, R.lightW.xyz), 0.0);
  var shade = 0.55 + 0.45 * diff;
  if (n.y > 0.5) { shade *= 1.30; }
  if (n.y < -0.5) { shade *= 0.60; }

  // ONE refracted continuation ray (also DDA): the analytic background
  // attenuated Beer-Lambert-style by the water path length behind the face.
  var rd2 = refract(rd, n, 0.752); // 1 / 1.33
  if (dot(rd2, rd2) < 1e-6) { rd2 = reflect(rd, n); }
  rd2 = normalize(rd2);
  let sc2 = traceScene(P, rd2);
  let gp = roG + rd * hit.t; // grid-space hit point
  let th = voxThickness(gp + rd2 * 0.01, rd2, sc2.w * G2); // nudged off the face
  var bg = sc2.rgb * exp(ABSORB * (-6.0 * th));
  bg += vec3f(0.04, 0.16, 0.24) * (1.0 - exp(-th * 9.0)); // in-scatter

  // Flat face-lit water tint over the refracted background (stylized alpha).
  var col = mix(bg, vec3f(0.10, 0.38, 0.58) * shade, 0.72);

  // Whitecaps where the cell's momentum magnitude / mass crosses a threshold.
  let d = cellDens(hit.cell);
  let speed = d.y / max(d.x, 1e-4) / VMAX;
  let foam = smoothstep(0.45, 0.85, speed);
  col = mix(col, vec3f(0.93, 0.97, 1.0) * (0.75 + 0.25 * diff), foam * 0.9);

  // Cell-edge darkening on the two tangential axes of the hit face.
  let ee = select(min(fract(gp), vec3f(1.0) - fract(gp)), vec3f(1.0), abs(n) > vec3f(0.5));
  let e = min(ee.x, min(ee.y, ee.z));
  col *= mix(0.55, 1.0, smoothstep(0.02, 0.14, e));

  return vec4f(col, 1.0);
}
`;

  // Upscale blit of the scaled volume target to the canvas.
  const volUpscale = `
struct UpVSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> UpVSOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: UpVSOut;
  o.pos = vec4f(p * 2.0 - vec2f(1.0), 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

@fragment
fn fsUpscale(v: UpVSOut) -> @location(0) vec4f {
  return vec4f(textureSampleLevel(srcTex, srcSamp, v.uv, 0.0).rgb, 1.0);
}
`;

  return {
    sim,
    render: renderCommon + sceneShade + renderBG + renderPoints + renderSSF,
    blur,
    composite,
    volBlur,
    volume,
    volUpscale,
  };
}
