// GLSL sources for the MLS-MPM waterfall simulation.
// All shaders share a generated header with simulation constants and helpers.

export const ROCKS = [
  // x, y, z (grid units at the reference 64 grid), radius
  [16.0, 6.0, 32.0, 8.0],
  [27.0, 4.5, 23.0, 5.5],
  [25.0, 4.0, 42.0, 5.0],
  [39.0, 3.5, 33.0, 4.0],
];

// z-slice tiling of the 3D grid into a 2D texture. TILES² ≥ GRID; unused
// tiles are never addressed (gridTexel only sees z < GRID).
export function gridLayout(GRID) {
  const TILES = Math.ceil(Math.sqrt(GRID));
  return { TILES, GTEX: GRID * TILES };
}

export function makeShaders(opts) {
  const { GRID, PTEX, LIFE } = opts;
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

void main() {
  ivec2 pt = ivec2(gl_VertexID % PTEX, gl_VertexID / PTEX);
  vec4 pa = texelFetch(uPos, pt, 0);
  if (pa.w < 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; vSpeed = 0.0; return; }
  vec3 wp = pa.xyz * (2.0 / GRIDF) - 1.0;
  vec4 vp = uView * vec4(wp, 1.0);
  gl_Position = uProj * vp;
  gl_PointSize = clamp(uPointScale * PRADIUS * 1.7 / max(-vp.z, 0.05), 1.0, 96.0);
  vSpeed = texelFetch(uVel, pt, 0).w / VMAX;
}
`;

  const fsThick = header + `
flat in float vSpeed;
out vec4 o;

void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float f = 1.0 - r2;
  float th = f * f * PRADIUS * 2.0;
  o = vec4(th, th * vSpeed, 0.0, 1.0);
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
  // water lies on BOTH sides along this axis at a compatible depth, so
  // nearby droplets merge into one surface instead of reading as separate
  // spheres. One-sided support is rejected, so outer silhouettes never grow.
  if (z0 <= 0.0) {
    float zp = 0.0, zm = 0.0;
    int ip = 0, im = 0;
    for (int i = 1; i <= 20; i++) {
      if (zp <= 0.0) { zp = dfetch(tx + dir * i); ip = i; }
      if (zm <= 0.0) { zm = dfetch(tx - dir * i); im = i; }
    }
    if (zp <= 0.0 || zm <= 0.0 || abs(zp - zm) > NRANGE) { o = vec4(0.0); return; }
    z0 = min(zp, zm);
    float reach = clamp(0.045 * uScalePx / z0, 2.0, 20.0);
    if (float(ip + im) > reach) { o = vec4(0.0); return; }
  }

  float radius = clamp(0.045 * uScalePx / z0, 2.0, 20.0);
  float sum = z0, wsum = 1.0;
  for (int i = 1; i <= 20; i++) {
    float fi = float(i);
    if (fi > radius) break;
    float g = exp(-fi * fi / (0.5 * radius * radius));
    float za = dfetch(tx + dir * i);
    float zb = dfetch(tx - dir * i);
    if (za > 0.0 && abs(za - z0) < NRANGE) { sum += za * g; wsum += g; }
    if (zb > 0.0 && abs(zb - z0) < NRANGE) { sum += zb * g; wsum += g; }
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

  vec2 t2 = texture(uThick, tuv).rg;
  float th = t2.r * 3.0;
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
  // shading as a fully opaque sphere.
  float cov = smoothstep(0.0, 0.09, th);
  o = vec4(mix(scene, col, cov), 1.0);
}
`;

  // Trivial pass-through used by the legacy points mode.
  const fsBlit = header + `
uniform sampler2D uScene;
out vec4 o;
void main() { o = vec4(texelFetch(uScene, ivec2(gl_FragCoord.xy), 0).rgb, 1.0); }
`;

  // Background: analytic raytrace of the cube interior and rock spheres,
  // writing correct depth so particles composite against it.
  const fsBackground = header + `
uniform vec3 uCamPos, uCamR, uCamU, uCamF;
uniform vec2 uRes;
uniform float uTanF, uAspect;
uniform mat4 uPV;
uniform vec3 uLightW;
out vec4 o;

const float B = ${(1 - 4 / GRID).toFixed(6)}; // wall extent in world units (grid cells 2..GRIDI-2)

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
    // Noise in rock-local coordinates (offset per rock), so the texture is
    // attached to the rock and moves with it when dragged, instead of the
    // rock sliding through a fixed world-space noise volume.
    vec3 cell = floor((hit - rockC) * 60.0 + rockId * 23.0);
    float n1 = hash1(cell);
    float n2 = hash1(cell + 17.0);
    float n3 = hash1(cell + 43.0);
    nrm = normalize(nrm + (vec3(n1, n2, n3) - 0.5) * 0.35);
    float diff = max(dot(nrm, uLightW), 0.0);
    float ao = clamp((hit.y + B) * 2.5 + 0.25, 0.25, 1.0);
    vec3 base = mix(vec3(0.30, 0.27, 0.24), vec3(0.42, 0.40, 0.37), n1);
    col = base * (0.30 + 0.75 * diff) * ao;
  } else {
    hit = uCamPos + rd * tE; // exit face = visible back wall or floor
    vec3 a = abs(hit) / B;
    vec3 n;
    if (a.x > a.y && a.x > a.z)      n = vec3(-sign(hit.x), 0.0, 0.0);
    else if (a.y > a.z)              n = vec3(0.0, -sign(hit.y), 0.0);
    else                             n = vec3(0.0, 0.0, -sign(hit.z));
    bool isFloor = n.y > 0.5;
    vec3 base = isFloor ? vec3(0.10, 0.11, 0.125) : vec3(0.065, 0.075, 0.09);
    float diff = max(dot(n, uLightW), 0.0);
    col = base * (0.5 + 0.5 * diff);

    // Subtle grid lines every 8 sim cells.
    vec2 tuv = (abs(n.x) > 0.5) ? hit.yz : (abs(n.y) > 0.5 ? hit.xz : hit.xy);
    vec2 g2 = abs(fract(tuv * (GRIDF / 16.0)) - 0.5) / (GRIDF / 16.0);
    float line = smoothstep(0.004, 0.010, min(g2.x, g2.y));
    col *= mix(1.3, 1.0, line);

    // Faint glow along cube edges.
    vec3 s = B - abs(hit);
    float m1 = min(s.x, min(s.y, s.z));
    float mx = max(s.x, max(s.y, s.z));
    float mid = s.x + s.y + s.z - m1 - mx;
    col += vec3(0.10, 0.16, 0.20) * (1.0 - smoothstep(0.0, 0.025, mid));
  }

  vec4 clip = uPV * vec4(hit, 1.0);
  gl_FragDepth = clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
  o = vec4(col, 1.0);
}
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
    vsGizmo, fsGizmo,
  };
}
