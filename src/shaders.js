// GLSL sources for the MLS-MPM waterfall simulation.
// All shaders share a generated header with simulation constants and helpers.

export const ROCKS = [
  // x, y, z (grid units), radius
  [16.0, 6.0, 32.0, 8.0],
  [27.0, 4.5, 23.0, 5.5],
  [25.0, 4.0, 42.0, 5.0],
  [39.0, 3.5, 33.0, 4.0],
];

export function makeShaders(opts) {
  const { PTEX, LIFE } = opts;

  const rockInit = ROCKS.map(
    (r) => `vec4(${r.map((v) => v.toFixed(2)).join(', ')})`
  ).join(',\n  ');

  const header = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

const int   GRIDI = 64;          // grid resolution per axis
const float GRIDF = 64.0;
const int   TILES = 8;           // z-slices tiled 8x8 into a 2D texture
const int   GTEX  = 512;         // grid texture size (GRIDI * TILES)
const int   PTEX  = ${PTEX};     // particle state texture size

const float MASS  = 1.0;         // particle mass
const float REST  = 4.0;         // rest density (particles per cell)
const float STIFF = 2.5;         // equation-of-state stiffness
const float EOS_P = 4.0;         // equation-of-state power
const float VISC  = 0.06;        // dynamic viscosity
const float GRAV  = -0.010;      // gravity per substep (dt = 1)
const float VMAX  = 0.85;        // CFL velocity clamp (cells per substep)
const float LIFE  = ${LIFE.toFixed(1)}; // particle lifetime in substeps

const vec3 EMIT_P = vec3(7.0, 59.0, 32.0);   // spout position
const vec3 EMIT_R = vec3(2.0, 1.5, 13.0);    // spout extent (a wide sheet)
const vec3 EMIT_V = vec3(0.10, -0.05, 0.0);  // initial jet velocity

const float PRADIUS = 0.021;     // particle render radius, world units

const int NROCK = ${ROCKS.length};
const vec4 ROCKS[NROCK] = vec4[NROCK](
  ${rockInit}
);

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
    d = min(d, length(p - ROCKS[i].xyz) - ROCKS[i].w);
  }
  return d;
}

vec3 rockNormal(vec3 p) {
  float best = 1e9;
  vec3 n = vec3(0.0, 1.0, 0.0);
  for (int i = 0; i < NROCK; i++) {
    float d = length(p - ROCKS[i].xyz) - ROCKS[i].w;
    if (d < best) { best = d; n = normalize(p - ROCKS[i].xyz); }
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

  float sd = sdRocks(vec3(cell));
  if (sd < 0.5) {
    vec3 n = rockNormal(vec3(cell));
    float vn = dot(v, n);
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

  // Background: analytic raytrace of the cube interior and rock spheres,
  // writing correct depth so particles composite against it.
  const fsBackground = header + `
uniform vec3 uCamPos, uCamR, uCamU, uCamF;
uniform vec2 uRes;
uniform float uTanF, uAspect;
uniform mat4 uPV;
uniform vec3 uLightW;
out vec4 o;

const float B = 0.9375; // wall extent in world units (grid cells 2..62)

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
  bool isRock = false;
  for (int i = 0; i < NROCK; i++) {
    vec3 c = ROCKS[i].xyz / 32.0 - 1.0;
    float r = ROCKS[i].w / 32.0;
    vec3 oc = uCamPos - c;
    float b = dot(oc, rd);
    float h = b * b - (dot(oc, oc) - r * r);
    if (h > 0.0) {
      float t = -b - sqrt(h);
      if (t > max(t0, 0.0) && t < min(tR, tE)) {
        tR = t;
        nrm = normalize(uCamPos + rd * t - c);
        isRock = true;
      }
    }
  }

  vec3 hit, col;
  if (isRock) {
    hit = uCamPos + rd * tR;
    vec3 cell = floor(hit * 60.0);
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
    vec2 g2 = abs(fract(tuv * 4.0) - 0.5) / 4.0;
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

  return {
    vsQuad, vsP2G1, vsP2G2, fsScatter, fsDensity, fsGrid, fsG2P,
    vsPoint, fsPoint, fsBackground,
  };
}
