// WGSL sources for the WebGPU backend.
//
// Same physics as the GLSL version, restructured for compute: P2G scatter
// uses fixed-point atomicAdd on a flat 3D grid buffer (no tiling, no 27x
// point amplification), and particle state lives in SoA storage buffers
// updated in place. All constants are baked per config like the GLSL header.

import { ROCKS } from './shaders.js';

export function makeWGSL(opts) {
  const { GRID, LIFE, N } = opts;
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

  const renderCommon = common + `
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
  misc: vec4f,  // pointScale (h * proj[5]), halfH * proj[5], unused, unused
  rocks: array<vec4f, ${NROCK}>, // grid units
  gizmo: array<vec4f, 3>,        // a.xyz + active, b.xyz + radius, unused
};
@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> vel: array<vec4f>;

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - vec2f(1.0), 0.0, 1.0);
}
`;

  // Background: analytic raytrace of the cube interior and rock spheres,
  // writing real depth so water composites against it.
  const renderBG = `
struct BGOut {
  @location(0) col: vec4f,
  @builtin(frag_depth) depth: f32,
};

const B: f32 = ${(1 - 4 / GRID).toFixed(6)}; // wall extent (cells 2..GRIDI-2)

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
    // Noise in rock-local coordinates so the texture moves with the rock.
    let cell = floor((hit - rockC) * 60.0 + vec3f(rockId * 23.0));
    let n1 = hash1(cell);
    let n2 = hash1(cell + vec3f(17.0));
    let n3 = hash1(cell + vec3f(43.0));
    nrm = normalize(nrm + (vec3f(n1, n2, n3) - vec3f(0.5)) * 0.35);
    let diff = max(dot(nrm, R.lightW.xyz), 0.0);
    let ao = clamp((hit.y + B) * 2.5 + 0.25, 0.25, 1.0);
    let base = mix(vec3f(0.30, 0.27, 0.24), vec3f(0.42, 0.40, 0.37), n1);
    col = base * (0.30 + 0.75 * diff) * ao;
  } else {
    hit = camPos + rd * tE; // exit face = visible back wall or floor
    let a = abs(hit) / B;
    var n: vec3f;
    if (a.x > a.y && a.x > a.z) { n = vec3f(-sign(hit.x), 0.0, 0.0); }
    else if (a.y > a.z) { n = vec3f(0.0, -sign(hit.y), 0.0); }
    else { n = vec3f(0.0, 0.0, -sign(hit.z)); }
    let isFloor = n.y > 0.5;
    var base = vec3f(0.065, 0.075, 0.09);
    if (isFloor) { base = vec3f(0.10, 0.11, 0.125); }
    let diff = max(dot(n, R.lightW.xyz), 0.0);
    col = base * (0.5 + 0.5 * diff);

    // Subtle grid lines every 8 sim cells.
    var tuv: vec2f;
    if (abs(n.x) > 0.5) { tuv = hit.yz; }
    else if (abs(n.y) > 0.5) { tuv = hit.xz; }
    else { tuv = hit.xy; }
    let g2 = abs(fract(tuv * (GRIDF / 16.0)) - 0.5) / (GRIDF / 16.0);
    let line = smoothstep(0.004, 0.010, min(g2.x, g2.y));
    col *= mix(1.3, 1.0, line);

    // Faint glow along cube edges.
    let sd = vec3f(B) - abs(hit);
    let m1 = min(sd.x, min(sd.y, sd.z));
    let mx = max(sd.x, max(sd.y, sd.z));
    let mid = sd.x + sd.y + sd.z - m1 - mx;
    col += vec3f(0.10, 0.16, 0.20) * (1.0 - smoothstep(0.0, 0.025, mid));
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
};

fn pointCorner(vi: u32) -> vec2f {
  return vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
}

fn pointVS(vi: u32, ii: u32, scale: f32, sizeMul: f32) -> PointVSOut {
  var o: PointVSOut;
  let pa = pos[ii];
  if (pa.w < 0.0) {
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
    o.q = vec2f(0.0); o.centerV = vec3f(0.0); o.speed = 0.0;
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

  return {
    sim,
    render: renderCommon + renderBG + renderPoints,
  };
}
