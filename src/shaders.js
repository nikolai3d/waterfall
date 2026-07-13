// GLSL sources for the MLS-MPM waterfall simulation.
// All shaders share a generated header with simulation constants and helpers.

export const ROCKS = [
  // x, y, z (grid units at the reference 64 grid), radius
  [16.0, 6.0, 32.0, 8.0],
  [27.0, 4.5, 23.0, 5.5],
  [25.0, 4.0, 42.0, 5.0],
  [39.0, 3.5, 33.0, 4.0],
];

// Shared splat-tuning constants, template-interpolated into BOTH shader
// languages (wgsl.js imports these) so a retune cannot desynchronize the
// backends or the depth/thickness normalization.
export const THICK_MUL = 1.7; // thickness splat footprint vs the depth splat
export const ANISO_AGE = 24;  // substeps to ramp in aniso elongation

// z-slice tiling of the 3D grid into a 2D texture. TILES² ≥ GRID; unused
// tiles are never addressed (gridTexel only sees z < GRID).
export function gridLayout(GRID) {
  const TILES = Math.ceil(Math.sqrt(GRID));
  return { TILES, GTEX: GRID * TILES };
}

export function makeShaders(opts) {
  const { GRID, PTEX, LIFE, ISO, K } = opts; // ISO/K validated/defaulted in app.js
  const { TILES, GTEX } = gridLayout(GRID);
  const s = GRID / 64;
  const vec3 = (a) => `vec3(${a.map((v) => v.toFixed(2)).join(', ')})`;

  const header = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

const int   GRIDI = ${GRID};     // grid resolution per axis
const float GRIDF = ${GRID.toFixed(1)};
const int   TILES = ${TILES};    // z-slices tiled TILESxTILES into a 2D texture
const int   GTEX  = ${GTEX};     // grid texture size (GRIDI * TILES)
const int   PTEX  = ${PTEX};     // particle state texture size

const float MASS  = 1.0;         // particle mass
const float REST  = 4.0;         // rest density (particles per cell)
const float STIFF = 2.5;         // equation-of-state stiffness
const float EOS_P = 4.0;         // equation-of-state power
const float VISC  = 0.06;        // dynamic viscosity
const float GRAV  = -0.010;      // gravity per substep (dt = 1)
const float VMAX  = 0.85;        // CFL velocity clamp (cells per substep)
const float LIFE  = ${LIFE.toFixed(1)}; // particle lifetime in substeps

const vec3 EMIT_P = ${vec3([7 * s, 59 * s, 32 * s])};   // spout position
const vec3 EMIT_R = ${vec3([2 * s, 1.5 * s, 13 * s])};  // spout extent (a wide sheet)
const vec3 EMIT_V = vec3(0.10, -0.05, 0.0);  // initial jet velocity

const float PRADIUS = 0.021;     // particle render radius, world units

const int NROCK = ${ROCKS.length};
uniform vec4 uRocks[NROCK]; // xyz center + radius, grid units (draggable)

ivec2 gridTexel(ivec3 g) {
  return ivec2((g.z % TILES) * GRIDI + g.x, (g.z / TILES) * GRIDI + g.y);
}

vec4 gridFetch(sampler2D t, ivec3 g) {
  return texelFetch(t, gridTexel(g), 0);
}

// Quadratic B-spline weights for MLS-MPM.
void quadWeights(vec3 fx, out vec3 W[3]) {
  W[0] = 0.5 * (1.5 - fx) * (1.5 - fx);
  W[1] = 0.75 - (fx - 1.0) * (fx - 1.0);
  W[2] = 0.5 * (fx - 0.5) * (fx - 0.5);
}

float sdRocks(vec3 p) {
  float d = 1e9;
  for (int i = 0; i < NROCK; i++) {
    d = min(d, length(p - uRocks[i].xyz) - uRocks[i].w);
  }
  return d;
}

vec3 rockNormal(vec3 p) {
  float best = 1e9;
  vec3 n = vec3(0.0, 1.0, 0.0);
  for (int i = 0; i < NROCK; i++) {
    float d = length(p - uRocks[i].xyz) - uRocks[i].w;
    if (d < best) { best = d; n = normalize(p - uRocks[i].xyz); }
  }
  return n;
}

vec3 hash3(uint n) {
  uvec3 x = uvec3(n, n * 7919u, n * 104729u);
  x = ((x >> 8u) ^ x.yzx) * 1103515245u;
  x = ((x >> 8u) ^ x.yzx) * 1103515245u;
  x = ((x >> 8u) ^ x.yzx) * 1103515245u;
  return vec3(x) * (1.0 / 4294967295.0);
}

float hash1(vec3 p) {
  return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}
`;

  // Fullscreen triangle.
  const vsQuad = header + `
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

  // P2G pass 1: scatter mass and momentum (with affine term) to the grid.
  // One point per (particle, neighbor cell) pair; additive blending.
  const vsP2G1 = header + `
uniform sampler2D uPos, uVel, uC0, uC1, uC2;
flat out vec4 vOut;

void main() {
  gl_PointSize = 1.0;
  int pid = gl_VertexID / 27;
  int cell = gl_VertexID % 27;
  ivec2 pt = ivec2(pid % PTEX, pid / PTEX);
  vec4 pa = texelFetch(uPos, pt, 0);
  if (pa.w < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vOut = vec4(0.0); return; }

  vec3 p = pa.xyz;
  vec3 v = texelFetch(uVel, pt, 0).xyz;
  mat3 C = mat3(texelFetch(uC0, pt, 0).xyz,
                texelFetch(uC1, pt, 0).xyz,
                texelFetch(uC2, pt, 0).xyz);

  ivec3 off = ivec3(cell % 3, (cell / 3) % 3, cell / 9);
  ivec3 base = ivec3(floor(p - 0.5));
  vec3 fx = p - vec3(base);
  vec3 W[3];
  quadWeights(fx, W);
  float w = W[off.x].x * W[off.y].y * W[off.z].z;
  vec3 dpos = vec3(off) - fx;

  vOut = vec4(MASS * w * (v + C * dpos), MASS * w);
  ivec2 tx = gridTexel(base + off);
  gl_Position = vec4((vec2(tx) + 0.5) / float(GTEX) * 2.0 - 1.0, 0.0, 1.0);
}
`;

  // P2G pass 2: scatter internal forces (pressure + viscosity), fused into
  // grid momentum as in MLS-MPM (Hu et al. 2018, eq. 16).
  const vsP2G2 = header + `
uniform sampler2D uPos, uC0, uC1, uC2, uAux;
flat out vec4 vOut;

void main() {
  gl_PointSize = 1.0;
  int pid = gl_VertexID / 27;
  int cell = gl_VertexID % 27;
  ivec2 pt = ivec2(pid % PTEX, pid / PTEX);
  vec4 pa = texelFetch(uPos, pt, 0);
  if (pa.w < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vOut = vec4(0.0); return; }

  vec3 p = pa.xyz;
  mat3 C = mat3(texelFetch(uC0, pt, 0).xyz,
                texelFetch(uC1, pt, 0).xyz,
                texelFetch(uC2, pt, 0).xyz);
  vec4 aux = texelFetch(uAux, pt, 0); // density, pressure, volume

  mat3 stress = mat3(-aux.y);        // -pressure * I
  stress += VISC * (C + transpose(C));
  mat3 term = -4.0 * aux.z * stress; // dt = 1, quadratic spline M^-1 = 4

  ivec3 off = ivec3(cell % 3, (cell / 3) % 3, cell / 9);
  ivec3 base = ivec3(floor(p - 0.5));
  vec3 fx = p - vec3(base);
  vec3 W[3];
  quadWeights(fx, W);
  float w = W[off.x].x * W[off.y].y * W[off.z].z;
  vec3 dpos = vec3(off) - fx;

  vOut = vec4(term * (w * dpos), 0.0);
  ivec2 tx = gridTexel(base + off);
  gl_Position = vec4((vec2(tx) + 0.5) / float(GTEX) * 2.0 - 1.0, 0.0, 1.0);
}
`;

  const fsScatter = header + `
flat in vec4 vOut;
out vec4 o;
void main() { o = vOut; }
`;

  // Per-particle density and pressure from the mass grid (weakly
  // compressible equation of state — no pressure solve needed).
  const fsDensity = header + `
uniform sampler2D uPos, uGrid;
out vec4 o;

void main() {
  ivec2 pt = ivec2(gl_FragCoord.xy);
  vec4 pa = texelFetch(uPos, pt, 0);
  if (pa.w < 0.0) { o = vec4(0.0); return; }

  ivec3 base = ivec3(floor(pa.xyz - 0.5));
  vec3 fx = pa.xyz - vec3(base);
  vec3 W[3];
  quadWeights(fx, W);
  float rho = 0.0;
  for (int k = 0; k < 3; k++)
  for (int j = 0; j < 3; j++)
  for (int i = 0; i < 3; i++) {
    float w = W[i].x * W[j].y * W[k].z;
    rho += w * gridFetch(uGrid, base + ivec3(i, j, k)).w;
  }
  float pres = clamp(STIFF * (pow(rho / REST, EOS_P) - 1.0), -0.05, 40.0);
  o = vec4(rho, pres, MASS / max(rho, 1e-5), 0.0);
}
`;

  // Grid update: momentum -> velocity, gravity, CFL clamp, and boundary
  // conditions (free-slip walls and rocks).
  const fsGrid = header + `
uniform sampler2D uGrid;
uniform vec3 uRockVel[NROCK]; // grid units per substep; nonzero while dragged
out vec4 o;

void main() {
  ivec2 tx = ivec2(gl_FragCoord.xy);
  vec4 g = texelFetch(uGrid, tx, 0);
  if (g.w <= 0.0) { o = vec4(0.0); return; }

  ivec2 tile = tx / GRIDI;
  ivec2 loc = tx % GRIDI;
  ivec3 cell = ivec3(loc.x, loc.y, tile.y * TILES + tile.x);

  vec3 v = g.xyz / g.w;
  v.y += GRAV;
  float s = length(v);
  if (s > VMAX) v *= VMAX / s;

  if (cell.x < 2 && v.x < 0.0) v.x = 0.0;
  if (cell.x > GRIDI - 3 && v.x > 0.0) v.x = 0.0;
  if (cell.y < 2 && v.y < 0.0) v.y = 0.0;
  if (cell.y > GRIDI - 3 && v.y > 0.0) v.y = 0.0;
  if (cell.z < 2 && v.z < 0.0) v.z = 0.0;
  if (cell.z > GRIDI - 3 && v.z > 0.0) v.z = 0.0;

  // Rock BC relative to the nearest rock's velocity, so a dragged rock
  // pushes water instead of letting it pass through.
  float best = 1e9;
  int ri = 0;
  for (int i = 0; i < NROCK; i++) {
    float d = length(vec3(cell) - uRocks[i].xyz) - uRocks[i].w;
    if (d < best) { best = d; ri = i; }
  }
  if (best < 0.5) {
    vec3 n = normalize(vec3(cell) - uRocks[ri].xyz);
    float vn = dot(v - uRockVel[ri], n);
    if (vn < 0.0) v -= n * vn;
  }
  o = vec4(v, g.w);
}
`;

  // G2P: gather velocity and affine matrix C, advect, handle spawning.
  const fsG2P = header + `
uniform sampler2D uPos, uGrid;
uniform float uFrame;
layout(location = 0) out vec4 oPos;
layout(location = 1) out vec4 oVel;
layout(location = 2) out vec4 oC0;
layout(location = 3) out vec4 oC1;
layout(location = 4) out vec4 oC2;

void main() {
  ivec2 pt = ivec2(gl_FragCoord.xy);
  int pid = pt.y * PTEX + pt.x;
  vec4 pa = texelFetch(uPos, pt, 0);
  float age = pa.w + 1.0;

  if (pa.w < 0.0 && age < 0.0) { // dormant, waiting for its spawn slot
    oPos = vec4(pa.xyz, age);
    oVel = vec4(0.0); oC0 = vec4(0.0); oC1 = vec4(0.0); oC2 = vec4(0.0);
    return;
  }
  if (pa.w < 0.0 || age > LIFE) { // (re)spawn at the spout
    vec3 r1 = hash3(uint(pid) * 747796405u + uint(uFrame) * 2891336453u);
    vec3 r2 = hash3(uint(pid) * 2654435761u + uint(uFrame) * 1597334677u);
    vec3 sv = EMIT_V + (r2 - 0.5) * 0.06;
    oPos = vec4(EMIT_P + (r1 - 0.5) * EMIT_R, 0.0);
    oVel = vec4(sv, length(sv));
    oC0 = vec4(0.0); oC1 = vec4(0.0); oC2 = vec4(0.0);
    return;
  }

  vec3 p = pa.xyz;
  ivec3 base = ivec3(floor(p - 0.5));
  vec3 fx = p - vec3(base);
  vec3 W[3];
  quadWeights(fx, W);

  vec3 v = vec3(0.0);
  mat3 B = mat3(0.0);
  for (int k = 0; k < 3; k++)
  for (int j = 0; j < 3; j++)
  for (int i = 0; i < 3; i++) {
    float w = W[i].x * W[j].y * W[k].z;
    vec3 dpos = vec3(i, j, k) - fx;
    vec3 gv = gridFetch(uGrid, base + ivec3(i, j, k)).xyz;
    v += w * gv;
    B += w * outerProduct(gv, dpos);
  }
  mat3 C = 4.0 * B;

  p += v; // dt = 1

  float sd = sdRocks(p);
  if (sd < 0.0) { // push out of rocks, kill inward velocity
    vec3 n = rockNormal(p);
    p -= n * sd;
    float vn = dot(v, n);
    if (vn < 0.0) v -= n * vn;
  }
  p = clamp(p, vec3(2.0), vec3(GRIDF - 2.0));

  oPos = vec4(p, age);
  oVel = vec4(v, length(v));
  oC0 = vec4(C[0], 0.0);
  oC1 = vec4(C[1], 0.0);
  oC2 = vec4(C[2], 0.0);
}
`;

  // Water particles as shaded sphere impostors.
  const vsPoint = header + `
uniform sampler2D uPos, uVel;
uniform mat4 uProj, uView;
uniform float uPointScale;
flat out vec3 vCenterV;
flat out float vSpeed;

void main() {
  ivec2 pt = ivec2(gl_VertexID % PTEX, gl_VertexID / PTEX);
  vec4 pa = texelFetch(uPos, pt, 0);
  if (pa.w < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; vCenterV = vec3(0.0); vSpeed = 0.0; return; }

  vec3 wp = pa.xyz * (2.0 / GRIDF) - 1.0;
  vec4 vp = uView * vec4(wp, 1.0);
  gl_Position = uProj * vp;
  gl_PointSize = clamp(uPointScale * PRADIUS / max(-vp.z, 0.05), 1.0, 96.0);
  vCenterV = vp.xyz;
  vSpeed = texelFetch(uVel, pt, 0).w / VMAX;
}
`;

  const fsPoint = header + `
uniform mat4 uProj;
uniform vec3 uLightV; // light direction in view space
flat in vec3 vCenterV;
flat in float vSpeed;
out vec4 o;

void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  q.y = -q.y;
  float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float z = sqrt(1.0 - r2);
  vec3 n = vec3(q, z);

  vec3 fp = vCenterV + n * PRADIUS;
  vec4 clip = uProj * vec4(fp, 1.0);
  gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);

  float diff = max(dot(n, uLightV), 0.0);
  vec3 e = normalize(-fp);
  float spec = pow(max(dot(reflect(-uLightV, n), e), 0.0), 40.0);
  float fres = pow(1.0 - max(dot(n, e), 0.0), 2.0);

  vec3 deep = vec3(0.03, 0.22, 0.42);
  vec3 shal = vec3(0.16, 0.55, 0.75);
  vec3 col = mix(deep, shal, 0.25 + 0.75 * diff);
  col = mix(col, vec3(0.55, 0.80, 0.90), fres * 0.6);
  float foam = smoothstep(0.5, 0.95, vSpeed);
  col = mix(col, vec3(0.93, 0.97, 1.0), foam * 0.85);
  col += vec3(0.9) * spec * 0.6;
  o = vec4(col, 1.0);
}
`;

  // --- Screen-space fluid rendering ------------------------------------
  // Depth pass: same sphere impostors, but output linear view-space depth
  // to an R32F target (0 = no water). gl_FragDepth still set for z-test
  // against the raytraced rocks/walls sharing the depth attachment.
  const fsPointDepth = header + `
uniform mat4 uProj;
flat in vec3 vCenterV;
flat in float vSpeed;
out vec4 o;

void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  q.y = -q.y;
  float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  vec3 fp = vCenterV + vec3(q, sqrt(1.0 - r2)) * PRADIUS;
  vec4 clip = uProj * vec4(fp, 1.0);
  gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
  o = vec4(-fp.z, 0.0, 0.0, 1.0);
}
`;

  // Thickness pass: soft additive sprites into a half-res RG target.
  // R = thickness, G = speed-weighted thickness (average speed -> foam).
  const vsThick = header + `
uniform sampler2D uPos, uVel;
uniform mat4 uProj, uView;
uniform float uPointScale;
flat out float vSpeed;
flat out float vZ;

void main() {
  ivec2 pt = ivec2(gl_VertexID % PTEX, gl_VertexID / PTEX);
  vec4 pa = texelFetch(uPos, pt, 0);
  if (pa.w < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; vSpeed = 0.0; vZ = 0.0; return; }
  vec3 wp = pa.xyz * (2.0 / GRIDF) - 1.0;
  vec4 vp = uView * vec4(wp, 1.0);
  gl_Position = uProj * vp;
  gl_PointSize = clamp(uPointScale * PRADIUS * ${THICK_MUL.toFixed(4)} / max(-vp.z, 0.05), 1.0, 96.0);
  vSpeed = texelFetch(uVel, pt, 0).w / VMAX;
  vZ = -vp.z;
}
`;

  const fsThick = header + `
uniform sampler2D uFront; // smoothed water surface depth, full resolution
flat in float vSpeed;
flat in float vZ;
out vec4 o;

void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  // Occlusion: water well behind the visible surface must not contribute
  // thickness or foam, or the pool draws its silhouette (and dilutes foam)
  // through the stream in front of it. Fragments outside the depth
  // silhouette (front = 0) are kept — they feather the edges.
  ivec2 s = textureSize(uFront, 0);
  ivec2 px = clamp(ivec2(gl_FragCoord.xy * 2.0), ivec2(0), s - 1);
  float front = texelFetch(uFront, px, 0).r;
  // Occluded water keeps coverage (B) but contributes no foam/absorption
  // (R, G) — culling it entirely would fade the composite to the
  // background wherever the near water is sparse (fake holes).
  float keep = (front > 0.0 && vZ > front + 0.3) ? 0.0 : 1.0;
  float f = 1.0 - r2;
  float th = f * f * PRADIUS * 2.0;
  o = vec4(th * keep, th * vSpeed * keep, th, 1.0);
}
`;

  // --- Anisotropic ellipsoid splatting (r=aniso) ------------------------
  // Same SSF pipeline, but the depth/thickness splats are velocity-oriented
  // ellipsoids (after Yu & Turk 2013, cheap per-particle variant): major
  // axis along the view-space velocity, elongation 1 + K*min(speed/VMAX, 1)
  // (damped for freshly spawned particles), minor axes shrunk 1/sqrt(elong)
  // to conserve volume. The fragment intersects a unit sphere in the
  // ellipsoid's normalized space; the ray is approximated as view-space -z
  // within the splat (the same orthographic-silhouette approximation the
  // spherical impostors already make). Point sprites are square, so the
  // sprite size is the max of the projected ellipse's per-axis extents.
  const anisoConsts = `
const float ANISO_K = ${K.toFixed(4)};            // ?k= elongation gain
const float ANISO_AGE = ${ANISO_AGE.toFixed(1)};  // substeps to ramp in elongation
const float THICK_MUL = ${THICK_MUL.toFixed(4)};  // thickness footprint (shared with vsThick)
`;

  // Two VS variants are generated (like the WGSL entry points): the depth
  // pass at the plain radius, the thickness pass at the THICK_MUL footprint.
  const makeVsAniso = (sizeMul) => header + anisoConsts + `
uniform sampler2D uPos, uVel;
uniform mat4 uProj, uView;
uniform float uPointScale;
const float SIZE_MUL = ${sizeMul.toFixed(4)}; // 1.0 for depth, THICK_MUL for thickness
flat out vec3 vCenterV;
// Rows of the inverse semi-axis matrix, pre-multiplied so that
// n = q.x*vQx + q.y*vQy + zoff*vQz is the ellipsoid-normalized coordinate.
flat out vec3 vQx, vQy, vQz;
flat out float vHs;
flat out float vSpeed;
flat out float vZ;

void main() {
  ivec2 pt = ivec2(gl_VertexID % PTEX, gl_VertexID / PTEX);
  vec4 pa = texelFetch(uPos, pt, 0);
  if (pa.w < 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0;
    vCenterV = vec3(0.0); vQx = vec3(0.0); vQy = vec3(0.0); vQz = vec3(0.0);
    vHs = 0.0; vSpeed = 0.0; vZ = 0.0;
    return;
  }
  vec4 vl = texelFetch(uVel, pt, 0);
  vec3 wp = pa.xyz * (2.0 / GRIDF) - 1.0;
  vec4 vp = uView * vec4(wp, 1.0);
  float z = max(-vp.z, 0.05);

  // Ellipsoid basis in view space: major axis along the velocity.
  float elong = 1.0 + ANISO_K * min(vl.w / VMAX, 1.0) * clamp(pa.w / ANISO_AGE, 0.0, 1.0);
  vec3 axis = vec3(0.0, 0.0, 1.0);
  if (vl.w > 1e-4) axis = normalize((uView * vec4(vl.xyz / vl.w, 0.0)).xyz);
  else elong = 1.0; // near-rest particle: plain sphere
  vec3 refv = abs(axis.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
  vec3 m1 = normalize(cross(axis, refv));
  vec3 m2 = cross(axis, m1);
  float ra = PRADIUS * elong * SIZE_MUL;              // major semi-axis
  float rb = PRADIUS * inversesqrt(elong) * SIZE_MUL; // minors conserve volume

  // Conservative sprite: exact orthographic extents of the projected
  // ellipse per screen axis (row norms of the semi-axis matrix).
  float ex = sqrt(ra * ra * axis.x * axis.x + rb * rb * (m1.x * m1.x + m2.x * m2.x));
  float ey = sqrt(ra * ra * axis.y * axis.y + rb * rb * (m1.y * m1.y + m2.y * m2.y));
  float sizePx = clamp(uPointScale * max(ex, ey) / z, 1.0, 96.0);
  float hs = sizePx * z / uPointScale; // view-space half extent of the sprite

  gl_Position = uProj * vp;
  gl_PointSize = sizePx;
  vCenterV = vp.xyz;
  vQx = hs * vec3(axis.x / ra, m1.x / rb, m2.x / rb);
  vQy = hs * vec3(axis.y / ra, m1.y / rb, m2.y / rb);
  vQz = vec3(axis.z / ra, m1.z / rb, m2.z / rb);
  vHs = hs;
  vSpeed = vl.w / VMAX;
  vZ = z;
}
`;

  const vsPointDepthAniso = makeVsAniso(1.0);
  const vsThickAniso = makeVsAniso(THICK_MUL);

  // Depth pass: intersect the view ray (approximated as -z through the
  // fragment) with the ellipsoid — a unit sphere in normalized space.
  const fsPointDepthAniso = header + `
uniform mat4 uProj;
flat in vec3 vCenterV;
flat in vec3 vQx, vQy, vQz;
flat in float vHs;
flat in float vSpeed;
flat in float vZ;
out vec4 o;

void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  q.y = -q.y;
  vec3 q0 = q.x * vQx + q.y * vQy;
  float a2 = dot(vQz, vQz);
  float b = 2.0 * dot(q0, vQz);
  float c = dot(q0, q0) - 1.0;
  float disc = b * b - 4.0 * a2 * c;
  if (disc < 0.0) discard;
  float zoff = (-b + sqrt(disc)) / (2.0 * a2); // front surface (larger view z)
  vec3 fp = vCenterV + vec3(q * vHs, zoff);
  vec4 clip = uProj * vec4(fp, 1.0);
  gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
  o = vec4(-fp.z, 0.0, 0.0, 1.0);
}
`;

  // Thickness pass: falloff from the elliptical silhouette; amplitude is the
  // view-z chord scale 2/|qz| (reduces to the spherical 2*PRADIUS at k=0).
  const fsThickAniso = header + anisoConsts + `
uniform sampler2D uFront; // raw water depth, full resolution
flat in vec3 vCenterV;
flat in vec3 vQx, vQy, vQz;
flat in float vHs;
flat in float vSpeed;
flat in float vZ;
out vec4 o;

void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  q.y = -q.y;
  vec3 q0 = q.x * vQx + q.y * vQy;
  float a2 = dot(vQz, vQz);
  float dz = dot(q0, vQz);
  float f = 1.0 - (dot(q0, q0) - dz * dz / a2); // 1 - silhouette r^2
  if (f <= 0.0) discard;
  // Occlusion vs the raw water depth, same rule as fsThick.
  ivec2 s = textureSize(uFront, 0);
  ivec2 px = clamp(ivec2(gl_FragCoord.xy * 2.0), ivec2(0), s - 1);
  float front = texelFetch(uFront, px, 0).r;
  float keep = (front > 0.0 && vZ > front + 0.3) ? 0.0 : 1.0;
  float th = f * f * 2.0 / (sqrt(a2) * THICK_MUL);
  o = vec4(th * keep, th * vSpeed * keep, th, 1.0);
}
`;

  // Separable depth-aware blur (simplified narrow-range filter) on the
  // linear water depth. Radius shrinks with distance so smoothing is
  // roughly constant in world space.
  const fsBlur = header + `
uniform sampler2D uDepth;
uniform vec2 uDir;      // (1,0) or (0,1), in texels
uniform float uScalePx; // pixels per world unit at z = 1
out vec4 o;

const float NRANGE = 0.10; // reject neighbors further than this in depth

float dfetch(ivec2 t) {
  ivec2 s = textureSize(uDepth, 0);
  return texelFetch(uDepth, clamp(t, ivec2(0), s - 1), 0).r;
}

void main() {
  ivec2 tx = ivec2(gl_FragCoord.xy);
  ivec2 dir = ivec2(uDir);
  float z0 = texelFetch(uDepth, tx, 0).r;

  // Gap fill (morphological closing): a pixel without water is filled when
  // water lies on BOTH sides along this axis, taking the NEARER depth when
  // the two sides disagree, so a near surface (the stream) stays continuous
  // in front of a far one (the pool). One-sided support is rejected, so
  // outer silhouettes never grow.
  if (z0 <= 0.0) {
    float zp = 0.0, zm = 0.0;
    int ip = 0, im = 0;
    for (int i = 1; i <= 20; i++) {
      if (zp <= 0.0) { zp = dfetch(tx + dir * i); ip = i; }
      if (zm <= 0.0) { zm = dfetch(tx - dir * i); im = i; }
    }
    if (zp <= 0.0 || zm <= 0.0) { o = vec4(0.0); return; }
    z0 = min(zp, zm);
    float reach = clamp(0.045 * uScalePx / z0, 2.0, 20.0);
    if (float(ip + im) > reach) { o = vec4(0.0); return; }
  }

  // Narrow-range filter (Truong & Yuksel, i3D 2018): instead of rejecting
  // neighbors from a different surface, clamp them to the center's depth
  // band, anchored on the CENTER (not the window minimum — that propagates
  // isolated droplets as square halos through the separable passes). Near
  // neighbors pull the surface at most NRANGE nearer per pass; across the
  // four iterated passes a real near sheet wins over the pool behind it.
  float radius = clamp(0.045 * uScalePx / z0, 2.0, 20.0);
  float lo = z0 - NRANGE;
  float hi = z0 + NRANGE;
  float sum = z0, wsum = 1.0;
  for (int i = 1; i <= 20; i++) {
    float fi = float(i);
    if (fi > radius) break;
    float g = exp(-fi * fi / (0.5 * radius * radius));
    float za = dfetch(tx + dir * i);
    float zb = dfetch(tx - dir * i);
    if (za > 0.0) { sum += clamp(za, lo, hi) * g; wsum += g; }
    if (zb > 0.0) { sum += clamp(zb, lo, hi) * g; wsum += g; }
  }
  o = vec4(sum / wsum, 0.0, 0.0, 1.0);
}
`;

  // Composite: reconstruct the water surface from smoothed depth and shade
  // it (refraction, Beer-Lambert absorption, Fresnel, spec, foam) over the
  // scene color. Pixels without water pass the scene through.
  const fsComposite = header + `
uniform sampler2D uScene, uDepthS, uThick;
uniform vec2 uRes;
uniform float uTanF, uAspect;
uniform vec3 uLightV; // light direction in view space
out vec4 o;

vec3 viewRay(vec2 frag) {
  vec2 uv = frag / uRes * 2.0 - 1.0;
  return vec3(uv.x * uTanF * uAspect, uv.y * uTanF, -1.0);
}

float dfetch(ivec2 t) {
  ivec2 s = textureSize(uDepthS, 0);
  return texelFetch(uDepthS, clamp(t, ivec2(0), s - 1), 0).r;
}

vec3 viewPos(ivec2 tx, float z) {
  return viewRay(vec2(tx) + 0.5) * z;
}

void main() {
  ivec2 tx = ivec2(gl_FragCoord.xy);
  vec2 tuv = gl_FragCoord.xy / uRes;
  vec3 scene = texture(uScene, tuv).rgb;
  float z0 = texelFetch(uDepthS, tx, 0).r;
  if (z0 <= 0.0) { o = vec4(scene, 1.0); return; }

  vec3 P = viewPos(tx, z0);

  // Normal from finite differences, taking the smoother side on each axis.
  float zx1 = dfetch(tx + ivec2(1, 0)); if (zx1 <= 0.0 || abs(zx1 - z0) > 0.3) zx1 = z0;
  float zx2 = dfetch(tx - ivec2(1, 0)); if (zx2 <= 0.0 || abs(zx2 - z0) > 0.3) zx2 = z0;
  float zy1 = dfetch(tx + ivec2(0, 1)); if (zy1 <= 0.0 || abs(zy1 - z0) > 0.3) zy1 = z0;
  float zy2 = dfetch(tx - ivec2(0, 1)); if (zy2 <= 0.0 || abs(zy2 - z0) > 0.3) zy2 = z0;
  vec3 dx = (abs(zx1 - z0) < abs(zx2 - z0))
    ? viewPos(tx + ivec2(1, 0), zx1) - P
    : P - viewPos(tx - ivec2(1, 0), zx2);
  vec3 dy = (abs(zy1 - z0) < abs(zy2 - z0))
    ? viewPos(tx + ivec2(0, 1), zy1) - P
    : P - viewPos(tx - ivec2(0, 1), zy2);
  vec3 n = normalize(cross(dx, dy));
  if (n.z < 0.0) n = -n;

  vec3 t2 = texture(uThick, tuv).rgb;
  float th = t2.r * 3.0;             // near-surface water only
  float speed = t2.g / max(t2.r, 1e-4);

  // Refract the background through the surface.
  vec2 off = n.xy * clamp(th, 0.0, 1.5) * 0.06;
  vec3 refr = texture(uScene, clamp(tuv + off, vec2(0.001), vec2(0.999))).rgb;

  // Beer-Lambert absorption, red first (deep water goes blue-green).
  vec3 col = refr * exp(-vec3(2.6, 1.0, 0.55) * th);
  col += vec3(0.04, 0.16, 0.24) * (1.0 - exp(-th * 2.5)); // in-scatter

  vec3 e = normalize(-P);
  float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, e), 0.0), 5.0);
  col = mix(col, vec3(0.35, 0.50, 0.60), fres * 0.8);
  vec3 hv = normalize(uLightV + e);
  col += vec3(0.9) * pow(max(dot(n, hv), 0.0), 120.0);

  float foam = smoothstep(0.45, 0.9, speed);
  col = mix(col, vec3(0.93, 0.97, 1.0), foam * 0.75);

  // Sparse water (lone droplets, spray) fades toward the scene instead of
  // shading as a fully opaque sphere. Coverage uses TOTAL thickness (B),
  // so water occluded from foam/absorption still keeps the pixel wet.
  float cov = smoothstep(0.0, 0.09, t2.b * 3.0);
  o = vec4(mix(scene, col, cov), 1.0);
}
`;

  // Trivial pass-through used by the legacy points mode.
  const fsBlit = header + `
uniform sampler2D uScene;
out vec4 o;
void main() { o = vec4(texelFetch(uScene, ivec2(gl_FragCoord.xy), 0).rgb, 1.0); }
`;

  // Shared hit-point shading for the analytic scene, appended into BOTH
  // fsBackground and fsVolume (whose traceScene reuses the same look), so a
  // wall/rock look tweak is one edit per language (see wgsl.js for WGSL).
  // Intersection logic stays with each shader.
  const sceneShade = `
const float B = ${(1 - 4 / GRID).toFixed(6)}; // wall extent in world units (grid cells 2..GRIDI-2)

// Rock: noise in rock-local coordinates (offset per rock), so the texture
// is attached to the rock and moves with it when dragged, instead of the
// rock sliding through a fixed world-space noise volume.
vec3 shadeRock(vec3 hit, vec3 nrm, vec3 rockC, float rockId, vec3 lightW) {
  vec3 cell = floor((hit - rockC) * 60.0 + rockId * 23.0);
  float n1 = hash1(cell);
  float n2 = hash1(cell + 17.0);
  float n3 = hash1(cell + 43.0);
  nrm = normalize(nrm + (vec3(n1, n2, n3) - 0.5) * 0.35);
  float diff = max(dot(nrm, lightW), 0.0);
  float ao = clamp((hit.y + B) * 2.5 + 0.25, 0.25, 1.0);
  vec3 base = mix(vec3(0.30, 0.27, 0.24), vec3(0.42, 0.40, 0.37), n1);
  return base * (0.30 + 0.75 * diff) * ao;
}

// Wall/floor: base color with subtle grid lines every 8 sim cells and a
// faint glow along the cube edges.
vec3 shadeWall(vec3 hit, vec3 lightW) {
  vec3 a = abs(hit) / B;
  vec3 n;
  if (a.x > a.y && a.x > a.z)      n = vec3(-sign(hit.x), 0.0, 0.0);
  else if (a.y > a.z)              n = vec3(0.0, -sign(hit.y), 0.0);
  else                             n = vec3(0.0, 0.0, -sign(hit.z));
  bool isFloor = n.y > 0.5;
  vec3 base = isFloor ? vec3(0.10, 0.11, 0.125) : vec3(0.065, 0.075, 0.09);
  float diff = max(dot(n, lightW), 0.0);
  vec3 col = base * (0.5 + 0.5 * diff);

  vec2 tuv = (abs(n.x) > 0.5) ? hit.yz : (abs(n.y) > 0.5 ? hit.xz : hit.xy);
  vec2 g2 = abs(fract(tuv * (GRIDF / 16.0)) - 0.5) / (GRIDF / 16.0);
  float line = smoothstep(0.004, 0.010, min(g2.x, g2.y));
  col *= mix(1.3, 1.0, line);

  vec3 s = B - abs(hit);
  float m1 = min(s.x, min(s.y, s.z));
  float mx = max(s.x, max(s.y, s.z));
  float mid = s.x + s.y + s.z - m1 - mx;
  col += vec3(0.10, 0.16, 0.20) * (1.0 - smoothstep(0.0, 0.025, mid));
  return col;
}
`;

  // Background: analytic raytrace of the cube interior and rock spheres,
  // writing correct depth so particles composite against it.
  const fsBackground = header + sceneShade + `
uniform vec3 uCamPos, uCamR, uCamU, uCamF;
uniform vec2 uRes;
uniform float uTanF, uAspect;
uniform mat4 uPV;
uniform vec3 uLightW;
out vec4 o;

void main() {
  vec2 uv = (gl_FragCoord.xy / uRes) * 2.0 - 1.0;
  vec3 rd = normalize(uCamF + uCamR * uv.x * uTanF * uAspect + uCamU * uv.y * uTanF);

  vec3 inv = 1.0 / rd;
  vec3 ta = (vec3(-B) - uCamPos) * inv;
  vec3 tb = (vec3(B) - uCamPos) * inv;
  vec3 tmin3 = min(ta, tb);
  vec3 tmax3 = max(ta, tb);
  float t0 = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tE = min(min(tmax3.x, tmax3.y), tmax3.z);

  if (tE < max(t0, 0.0)) { // missed the cube: dark backdrop
    float g = 1.0 - length(uv) * 0.45;
    o = vec4(vec3(0.015, 0.02, 0.03) * g, 1.0);
    gl_FragDepth = 0.9999;
    return;
  }

  // Analytic ray-sphere against the rocks, inside the cube only.
  float tR = 1e9;
  vec3 nrm = vec3(0.0);
  vec3 rockC = vec3(0.0); // hit rock center, world units
  float rockId = 0.0;
  bool isRock = false;
  for (int i = 0; i < NROCK; i++) {
    vec3 c = uRocks[i].xyz * (2.0 / GRIDF) - 1.0;
    float r = uRocks[i].w * (2.0 / GRIDF);
    vec3 oc = uCamPos - c;
    float b = dot(oc, rd);
    float h = b * b - (dot(oc, oc) - r * r);
    if (h > 0.0) {
      float t = -b - sqrt(h);
      if (t > max(t0, 0.0) && t < min(tR, tE)) {
        tR = t;
        nrm = normalize(uCamPos + rd * t - c);
        rockC = c;
        rockId = float(i);
        isRock = true;
      }
    }
  }

  vec3 hit, col;
  if (isRock) {
    hit = uCamPos + rd * tR;
    col = shadeRock(hit, nrm, rockC, rockId, uLightW);
  } else {
    hit = uCamPos + rd * tE; // exit face = visible back wall or floor
    col = shadeWall(hit, uLightW);
  }

  vec4 clip = uPV * vec4(hit, 1.0);
  gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
  o = vec4(col, 1.0);
}
`;

  // --- Volumetric raymarcher (r=volume) ---------------------------------

  // One-cell tent blur of the grid's (velocity, mass) field into a dedicated
  // tiled (mass, |momentum|) texture: the raw per-substep mass is too
  // speckly (4 particles/cell at rest) for a stable iso-surface.
  const fsVolBlur = header + `
uniform sampler2D uGrid;
out vec4 o;

void main() {
  ivec2 tx = ivec2(gl_FragCoord.xy);
  ivec2 tile = tx / GRIDI;
  ivec2 loc = tx % GRIDI;
  ivec3 cell = ivec3(loc.x, loc.y, tile.y * TILES + tile.x);
  // Tent weights (1,2,1)³ rather than a uniform box: thin sheets (the pool
  // is ~1 cell deep at 128³) keep enough peak density to cross the iso.
  float m = 0.0;
  float s = 0.0;
  for (int k = -1; k <= 1; k++)
  for (int j = -1; j <= 1; j++)
  for (int i = -1; i <= 1; i++) {
    ivec3 c = clamp(cell + ivec3(i, j, k), ivec3(0), ivec3(GRIDI - 1));
    vec4 g = gridFetch(uGrid, c);
    float w = float((2 - abs(i)) * (2 - abs(j)) * (2 - abs(k)));
    m += w * g.w;
    s += w * length(g.xyz) * g.w;
  }
  o = vec4(m, s, 0.0, 0.0) * (1.0 / 64.0);
}
`;

  // Analytic scene (walls + rocks) for a ray starting inside the cube:
  // trimmed copy of fsBackground. Returns rgb + hit distance. Shared by
  // fsVolume and fsVoxel (expects uRocks/uLightW + sceneShade in scope).
  const traceSceneGLSL = `
vec4 traceScene(vec3 ro, vec3 rd) {
  vec3 inv = 1.0 / rd;
  vec3 ta = (vec3(-B) - ro) * inv;
  vec3 tb = (vec3(B) - ro) * inv;
  vec3 tmax3 = max(ta, tb);
  float tE = min(min(tmax3.x, tmax3.y), tmax3.z);

  float tHit = tE;
  vec3 nrm = vec3(0.0);
  vec3 rockC = vec3(0.0);
  float rockId = 0.0;
  bool isRock = false;
  for (int i = 0; i < NROCK; i++) {
    vec3 c = uRocks[i].xyz * (2.0 / GRIDF) - 1.0;
    float r = uRocks[i].w * (2.0 / GRIDF);
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float h = b * b - (dot(oc, oc) - r * r);
    if (h > 0.0) {
      float t = -b - sqrt(h);
      if (t > 1e-4 && t < tHit) {
        tHit = t;
        nrm = normalize(ro + rd * t - c);
        rockC = c;
        rockId = float(i);
        isRock = true;
      }
    }
  }

  vec3 hit = ro + rd * tHit;
  vec3 col = isRock ? shadeRock(hit, nrm, rockC, rockId, uLightW)
                    : shadeWall(hit, uLightW);
  return vec4(col, tHit);
}
`;

  // Shared fsVolume/fsVoxel main() prologue: camera ray from the fragment
  // coord, slab test against the wall cube (miss = dark backdrop + return),
  // entry point + capped analytic scene, and entry-face normal n0 (dominant
  // -rd axis when the camera is inside the cube; only consumed by the voxel
  // DDA if the eye's cell is already solid).
  const volPrologue = `
  vec2 uv = (gl_FragCoord.xy / uRes) * 2.0 - 1.0;
  vec3 rd = normalize(uCamF + uCamR * uv.x * uTanF * uAspect + uCamU * uv.y * uTanF);
  vec3 ro = uCamPos;

  vec3 inv = 1.0 / rd;
  vec3 ta = (vec3(-B) - ro) * inv;
  vec3 tb = (vec3(B) - ro) * inv;
  vec3 tmin3 = min(ta, tb);
  vec3 tmax3 = max(ta, tb);
  float t0 = max(max(tmin3.x, tmin3.y), tmin3.z);
  float tE = min(min(tmax3.x, tmax3.y), tmax3.z);

  if (tE < max(t0, 0.0)) { // missed the cube: dark backdrop
    float g = 1.0 - length(uv) * 0.45;
    o = vec4(vec3(0.015, 0.02, 0.03) * g, 1.0);
    return;
  }

  // Analytic scene hit caps the march (water behind a rock stays behind it).
  vec3 entry = ro + rd * max(t0, 0.0);
  vec4 sc = traceScene(entry, rd);

  // Cube entry face; with the camera inside the cube fall back to the
  // dominant axis of -rd.
  vec3 n0;
  if (t0 > 0.0) {
    if (t0 == tmin3.x)      n0 = vec3(-sign(rd.x), 0.0, 0.0);
    else if (t0 == tmin3.y) n0 = vec3(0.0, -sign(rd.y), 0.0);
    else                    n0 = vec3(0.0, 0.0, -sign(rd.z));
  } else {
    vec3 ad = abs(rd);
    if (ad.x > ad.y && ad.x > ad.z) n0 = vec3(-sign(rd.x), 0.0, 0.0);
    else if (ad.y > ad.z)           n0 = vec3(0.0, -sign(rd.y), 0.0);
    else                            n0 = vec3(0.0, 0.0, -sign(rd.z));
  }
`;

  // Fragment raymarch of the blurred density grid: iso-surface hit with
  // bisection refine, gradient normal, one refraction segment with
  // Beer-Lambert absorption from the marched interior density, and the
  // analytic scene (walls + rocks) continued along the refracted ray.
  // Rendered into a scaled offscreen target and upscaled after.
  const fsVolume = header + sceneShade + `
uniform sampler2D uDens; // blurred (mass, |momentum|), tiled like the grid
uniform vec3 uCamPos, uCamR, uCamU, uCamF;
uniform vec2 uRes;       // scaled target size
uniform float uTanF, uAspect;
uniform vec3 uLightW;
out vec4 o;

const float ISO = 0.5;                        // iso threshold, particles/cell
const float STEP = ${(1 / GRID).toFixed(6)};  // 0.5 grid cells, world units
const int MAXIT = ${Math.ceil(3.5 * GRID)};   // covers the cube diagonal
const float STEP2 = ${(2 / GRID).toFixed(6)}; // interior absorption step
const int MAXIT2 = ${2 * GRID};
const vec3 ABSORB = vec3(2.6, 1.0, 0.55);

// Trilinear sample of the blurred (mass, |momentum|) grid at a world point
// (per-slice fetches — the tiling breaks hardware filtering across z).
vec2 sampleD(vec3 p) {
  vec3 g = clamp((p + 1.0) * (GRIDF * 0.5), vec3(0.0), vec3(GRIDF - 1.001));
  ivec3 b = ivec3(g);
  vec3 f = g - vec3(b);
  vec2 c00 = mix(gridFetch(uDens, b).xy, gridFetch(uDens, b + ivec3(1, 0, 0)).xy, f.x);
  vec2 c10 = mix(gridFetch(uDens, b + ivec3(0, 1, 0)).xy, gridFetch(uDens, b + ivec3(1, 1, 0)).xy, f.x);
  vec2 c01 = mix(gridFetch(uDens, b + ivec3(0, 0, 1)).xy, gridFetch(uDens, b + ivec3(1, 0, 1)).xy, f.x);
  vec2 c11 = mix(gridFetch(uDens, b + ivec3(0, 1, 1)).xy, gridFetch(uDens, b + ivec3(1, 1, 1)).xy, f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

float density(vec3 p) { return sampleD(p).x; }

vec3 gradD(vec3 p) {
  float h = 2.0 / GRIDF; // one cell
  return vec3(
    density(p + vec3(h, 0.0, 0.0)) - density(p - vec3(h, 0.0, 0.0)),
    density(p + vec3(0.0, h, 0.0)) - density(p - vec3(0.0, h, 0.0)),
    density(p + vec3(0.0, 0.0, h)) - density(p - vec3(0.0, 0.0, h)));
}

` + traceSceneGLSL + `
void main() {
` + volPrologue + `
  float tHit = -1.0;
  float t = STEP * hash1(vec3(gl_FragCoord.xy, 0.0)); // dither vs banding
  for (int i = 0; i < MAXIT; i++) {
    t += STEP;
    if (t > sc.w) break;
    if (density(entry + rd * t) >= ISO) {
      // Bisection refine between the last two samples.
      float lo = t - STEP;
      float hi = t;
      for (int b = 0; b < 4; b++) {
        float mid = 0.5 * (lo + hi);
        if (density(entry + rd * mid) >= ISO) hi = mid; else lo = mid;
      }
      tHit = 0.5 * (lo + hi);
      break;
    }
  }
  if (tHit < 0.0) { o = vec4(sc.rgb, 1.0); return; }

  vec3 P = entry + rd * tHit;
  vec3 n = -normalize(gradD(P) + 1e-6);
  if (dot(n, rd) > 0.0) n = -n;

  // Refract; on total internal reflection fall back to reflection.
  vec3 rd2 = refract(rd, n, 0.752); // 1 / 1.33
  if (dot(rd2, rd2) < 1e-6) rd2 = reflect(rd, n);
  rd2 = normalize(rd2);

  // Continue the analytic background along the refracted ray, integrating
  // Beer-Lambert absorption from the marched interior density.
  vec4 sc2 = traceScene(P, rd2);
  float th = 0.0;
  for (int i = 0; i < MAXIT2; i++) {
    float t2 = (float(i) + 0.5) * STEP2;
    if (t2 > sc2.w) break;
    th += density(P + rd2 * t2) / REST * STEP2;
  }
  vec3 col = sc2.rgb * exp(ABSORB * (-6.0 * th));
  col += vec3(0.04, 0.16, 0.24) * (1.0 - exp(-th * 12.0)); // in-scatter

  vec3 e = -rd;
  float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, e), 0.0), 5.0);
  col = mix(col, vec3(0.35, 0.50, 0.60), fres * 0.8);
  vec3 hv = normalize(uLightW + e);
  col += vec3(0.9) * pow(max(dot(n, hv), 0.0), 120.0);

  // Foam from the blurred momentum magnitude at the surface.
  vec2 sd = sampleD(P);
  float speed = sd.y / max(sd.x, 1e-4) / VMAX;
  float foam = smoothstep(0.45, 0.9, speed);
  col = mix(col, vec3(0.93, 0.97, 1.0), foam * 0.75);
  o = vec4(col, 1.0);
}
`;

  // --- Stylized voxel water (r=voxel) ------------------------------------

  // Same blurred density field as fsVolume, but rendered as literal
  // grid-aligned cubes: a DDA (Amanatides & Woo) walks the ray through
  // cells and the first cell with density >= VISO is a cube-face hit with
  // an axis-aligned normal. Chunky face shading, cell-edge darkening,
  // whitecaps, and one refracted continuation ray (also DDA) attenuated
  // Beer-Lambert-style toward the analytic background.
  const fsVoxel = header + sceneShade + `
uniform sampler2D uDens; // blurred (mass, |momentum|), tiled like the grid
uniform vec3 uCamPos, uCamR, uCamU, uCamF;
uniform vec2 uRes;       // scaled target size
uniform float uTanF, uAspect;
uniform vec3 uLightW;
out vec4 o;

const float VISO = ${ISO.toFixed(4)};  // per-cell density threshold (?iso=)
const int MAXDDA = ${3 * GRID + 4};    // DDA worst case ~3*GRID cells
const vec3 ABSORB = vec3(2.6, 1.0, 0.55);
` + traceSceneGLSL + `
vec2 cellDens(ivec3 c) {
  if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(GRIDI)))) return vec2(0.0);
  return gridFetch(uDens, c).xy;
}

// Grid traversal from roG (grid units) along rd (unit direction — uniform
// world->grid scale keeps directions unchanged), capped at tEnd (grid
// units). Returns the hit t (< 0 = miss) with face normal + cell.
float voxMarch(vec3 roG, vec3 rd0, float tEnd, vec3 n0, out vec3 nOut, out ivec3 cellOut) {
  // Degenerate-axis guard preserves the component's sign (sign(0) -> +1)
  // so near-axis rays still step the right way.
  vec3 sgn = mix(sign(rd0), vec3(1.0), vec3(equal(rd0, vec3(0.0))));
  vec3 rd = sgn * max(abs(rd0), vec3(1e-6));
  // Clamp the entry into the grid so cell and tMax agree on the origin.
  roG = clamp(roG, vec3(0.0), vec3(GRIDF));
  ivec3 cell = clamp(ivec3(floor(roG)), ivec3(0), ivec3(GRIDI - 1));
  ivec3 stp = ivec3(sgn);
  vec3 inv = 1.0 / rd;
  vec3 tDelta = abs(inv);
  vec3 tMax = (vec3(cell) + max(sgn, 0.0) - roG) * inv;
  float t = 0.0;
  vec3 n = n0;
  nOut = n0;
  cellOut = cell;
  if (tEnd <= 0.0) return -1.0;
  for (int i = 0; i < MAXDDA; i++) {
    if (cellDens(cell).x >= VISO) { nOut = n; cellOut = cell; return t; }
    if (tMax.x < tMax.y && tMax.x < tMax.z) {
      t = tMax.x; tMax.x += tDelta.x; cell.x += stp.x; n = vec3(-float(stp.x), 0.0, 0.0);
    } else if (tMax.y < tMax.z) {
      t = tMax.y; tMax.y += tDelta.y; cell.y += stp.y; n = vec3(0.0, -float(stp.y), 0.0);
    } else {
      t = tMax.z; tMax.z += tDelta.z; cell.z += stp.z; n = vec3(0.0, 0.0, -float(stp.z));
    }
    if (t > tEnd || any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, ivec3(GRIDI)))) return -1.0;
  }
  return -1.0;
}

// Water path length (world units) through solid (>= VISO) cells along a ray.
float voxThickness(vec3 roG, vec3 rd0, float tEnd) {
  vec3 sgn = mix(sign(rd0), vec3(1.0), vec3(equal(rd0, vec3(0.0))));
  vec3 rd = sgn * max(abs(rd0), vec3(1e-6));
  roG = clamp(roG, vec3(0.0), vec3(GRIDF));
  ivec3 cell = clamp(ivec3(floor(roG)), ivec3(0), ivec3(GRIDI - 1));
  ivec3 stp = ivec3(sgn);
  vec3 inv = 1.0 / rd;
  vec3 tDelta = abs(inv);
  vec3 tMax = (vec3(cell) + max(sgn, 0.0) - roG) * inv;
  float t = 0.0;
  float th = 0.0;
  for (int i = 0; i < MAXDDA; i++) {
    float tn = min(min(tMax.x, tMax.y), tMax.z);
    if (cellDens(cell).x >= VISO) th += max(min(tn, tEnd) - t, 0.0);
    if (tn > tEnd) break;
    if (tMax.x < tMax.y && tMax.x < tMax.z) { tMax.x += tDelta.x; cell.x += stp.x; }
    else if (tMax.y < tMax.z) { tMax.y += tDelta.y; cell.y += stp.y; }
    else { tMax.z += tDelta.z; cell.z += stp.z; }
    t = tn;
    if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, ivec3(GRIDI)))) break;
  }
  return th * (2.0 / GRIDF);
}

void main() {
` + volPrologue + `
  float G2 = GRIDF * 0.5;
  vec3 roG = (entry + 1.0) * G2;
  vec3 n;
  ivec3 hitCell;
  float tHit = voxMarch(roG, rd, sc.w * G2, n0, n, hitCell);
  if (tHit < 0.0) { o = vec4(sc.rgb, 1.0); return; }

  vec3 P = entry + rd * (tHit / G2);

  // Chunky per-face diffuse; top faces read lighter, bottoms darker.
  float diff = max(dot(n, uLightW), 0.0);
  float shade = 0.55 + 0.45 * diff;
  if (n.y > 0.5) shade *= 1.30;
  if (n.y < -0.5) shade *= 0.60;

  // ONE refracted continuation ray (also DDA): the analytic background
  // attenuated Beer-Lambert-style by the water path length behind the face.
  vec3 rd2 = refract(rd, n, 0.752); // 1 / 1.33
  if (dot(rd2, rd2) < 1e-6) rd2 = reflect(rd, n);
  rd2 = normalize(rd2);
  vec4 sc2 = traceScene(P, rd2);
  vec3 gp = roG + rd * tHit; // grid-space hit point
  float th = voxThickness(gp + rd2 * 0.01, rd2, sc2.w * G2); // nudged off the face
  vec3 bg = sc2.rgb * exp(ABSORB * (-6.0 * th));
  bg += vec3(0.04, 0.16, 0.24) * (1.0 - exp(-th * 9.0)); // in-scatter

  // Flat face-lit water tint over the refracted background (stylized alpha).
  vec3 col = mix(bg, vec3(0.10, 0.38, 0.58) * shade, 0.72);

  // Whitecaps where the cell's momentum magnitude / mass crosses a threshold.
  vec2 d = cellDens(hitCell);
  float speed = d.y / max(d.x, 1e-4) / VMAX;
  float foam = smoothstep(0.45, 0.85, speed);
  col = mix(col, vec3(0.93, 0.97, 1.0) * (0.75 + 0.25 * diff), foam * 0.9);

  // Cell-edge darkening on the two tangential axes of the hit face.
  vec3 ee = mix(min(fract(gp), 1.0 - fract(gp)), vec3(1.0), greaterThan(abs(n), vec3(0.5)));
  float e = min(ee.x, min(ee.y, ee.z));
  col *= mix(0.55, 1.0, smoothstep(0.02, 0.14, e));

  o = vec4(col, 1.0);
}
`;

  // Upscale blit of the scaled volume target to the canvas.
  const fsVolUpscale = header + `
uniform sampler2D uScene; // linear-filtered scaled target
uniform vec2 uRes;        // canvas size
out vec4 o;
void main() { o = vec4(texture(uScene, gl_FragCoord.xy / uRes).rgb, 1.0); }
`;

  // Drag gizmo: attribute-less wireframe delta indicator — a ghost circle
  // at the drag origin, a line to the current center, and a circle there.
  // 130 vertices as GL_LINES: ids 0-1 the line, then 2x32 segments.
  const vsGizmo = header + `
uniform mat4 uPV;
uniform vec3 uA, uB;        // drag start / current rock center, world units
uniform vec3 uCamR, uCamU;  // billboard basis
uniform float uR;           // rock radius, world units
flat out float vGhost;

void main() {
  vec3 p;
  if (gl_VertexID < 2) {
    p = gl_VertexID == 0 ? uA : uB;
    vGhost = 0.0;
  } else {
    int seg = (gl_VertexID - 2) >> 1;
    int circle = seg / 32;
    float a = 6.2831853 * float(seg % 32 + ((gl_VertexID - 2) & 1)) / 32.0;
    vec3 c = circle == 0 ? uA : uB;
    p = c + (uCamR * cos(a) + uCamU * sin(a)) * uR;
    vGhost = circle == 0 ? 1.0 : 0.0;
  }
  gl_Position = uPV * vec4(p, 1.0);
}
`;

  const fsGizmo = header + `
flat in float vGhost;
out vec4 o;
void main() {
  o = vGhost > 0.5 ? vec4(0.55, 0.75, 0.95, 0.35) : vec4(0.80, 0.92, 1.0, 0.9);
}
`;

  return {
    vsQuad, vsP2G1, vsP2G2, fsScatter, fsDensity, fsGrid, fsG2P,
    vsPoint, fsPoint, fsBackground,
    fsPointDepth, vsThick, fsThick, fsBlur, fsComposite, fsBlit,
    vsPointDepthAniso, vsThickAniso, fsPointDepthAniso, fsThickAniso,
    fsVolBlur, fsVolume, fsVoxel, fsVolUpscale,
    vsGizmo, fsGizmo,
  };
}
