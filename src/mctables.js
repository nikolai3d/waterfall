// Marching-cubes triangle table for the mesh renderer (r=mesh, WebGPU).
//
// Generated at load instead of transcribing the classic 256x16 constant:
// per case, marching squares on each cube face yields the surface/face
// intersection segments; every cut edge then has exactly two segments, so
// the segments chain into closed loops, which are fan-triangulated.
// Ambiguous faces (diagonally opposite inside corners) always separate the
// diagonal — a rotation-invariant rule, so the two cells sharing a face
// agree on its segments and the mesh is crack-free by construction.
// Triangle winding is NOT normalized (normals come from the density
// gradient, and the mesh pipeline does not cull).
//
// Corner and edge numbering (the mcEmit kernel's CO/EA/EB arrays in
// src/wgsl.js are template-interpolated from the exports below, so JS and
// WGSL cannot drift):
//
//   corners: 0..3 = z=0 ring (0,0,0)(1,0,0)(1,1,0)(0,1,0), 4..7 = z=1 ring
//   edges:   0..3 = z=0 ring, 4..7 = z=1 ring, 8..11 = verticals c -> c+4

const CORNERS = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
const EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];
const FACES = [ // cyclic corner quads
  [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
  [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5],
];

const edgeOf = new Map();
EDGES.forEach(([a, b], i) => {
  edgeOf.set(a * 8 + b, i);
  edgeOf.set(b * 8 + a, i);
});

function caseTriangles(mask) {
  // Face segments: pairs of cube-edge indices crossed by the isosurface.
  const segs = [];
  for (const f of FACES) {
    const bit = f.map((c) => (mask >> c) & 1);
    const e = f.map((c, i) => edgeOf.get(c * 8 + f[(i + 1) & 3]));
    const cut = [0, 1, 2, 3].filter((i) => bit[i] !== bit[(i + 1) & 3]);
    if (cut.length === 2) {
      segs.push([e[cut[0]], e[cut[1]]]);
    } else if (cut.length === 4) {
      // Ambiguous: cut each inside corner off with its own segment.
      for (let i = 0; i < 4; i++) {
        if (bit[i]) segs.push([e[(i + 3) & 3], e[i]]);
      }
    }
  }

  // Every cut edge appears in exactly two segments (its two faces), so the
  // adjacency decomposes into closed loops.
  const adj = new Map();
  for (const [a, b] of segs) {
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push(b);
    adj.get(b).push(a);
  }
  for (const [edge, nb] of adj) {
    if (nb.length !== 2) throw new Error(`MC case ${mask}: edge ${edge} degree ${nb.length}`);
  }

  const tris = [];
  const seen = new Set();
  for (const start of adj.keys()) {
    if (seen.has(start)) continue;
    const loop = [start];
    seen.add(start);
    let prev = -1;
    let cur = start;
    for (;;) {
      const nb = adj.get(cur);
      const nxt = nb[0] === prev ? nb[1] : nb[0];
      if (nxt === start) break;
      loop.push(nxt);
      seen.add(nxt);
      prev = cur;
      cur = nxt;
    }
    if (loop.length < 3) throw new Error(`MC case ${mask}: degenerate loop`);
    for (let i = 1; i + 1 < loop.length; i++) {
      tris.push(loop[0], loop[i], loop[i + 1]);
    }
  }
  return tris;
}

const rows = [];
let maxLen = 0;
for (let c = 0; c < 256; c++) {
  const t = caseTriangles(c);
  rows.push(t);
  maxLen = Math.max(maxLen, t.length);
}

// Load-time validation (cheap: runs once at import, over 256 tiny rows).
// Classic MC invariants — a generator bug fails the module import loudly
// instead of rendering garbage.
if (maxLen !== 15) {
  throw new Error(`MC tables: longest case is ${maxLen / 3} triangles, expected 5`);
}
{
  // Case 1 (only corner 0 inside) is one triangle on the corner's three
  // incident edges {0, 3, 8}.
  const t1 = [...rows[1]].sort((a, b) => a - b);
  if (rows[1].length !== 3 || t1.join(',') !== '0,3,8') {
    throw new Error(`MC tables: case 1 should be one triangle on edges {0,3,8}, got [${rows[1]}]`);
  }
}
// Complementary checks. NOTE: full triangle-count symmetry under c -> 255^c
// deliberately does NOT hold here — the ambiguous-face rule ("separate the
// diagonal") is inside/outside-asymmetric, which is exactly what makes the
// table face-consistent (crack-free); e.g. case 5 (two diagonal corners) is
// two cap triangles while its complement 250 is a 4-triangle valley. What
// must hold for every case: the complement crosses the same cut edges; and
// for cases with no ambiguous face, the same triangle count.
function hasAmbiguousFace(mask) {
  return FACES.some((f) => {
    const bit = f.map((c) => (mask >> c) & 1);
    return bit.filter((b, i) => b !== bit[(i + 1) & 3]).length === 4;
  });
}
const edgeSet = (t) => [...new Set(t)].sort((a, b) => a - b).join(',');
for (let c = 0; c < 256; c++) {
  if (edgeSet(rows[c]) !== edgeSet(rows[255 ^ c])) {
    throw new Error(`MC tables: case ${c} and complement ${255 ^ c} cut different edges`);
  }
  if (!hasAmbiguousFace(c) && rows[c].length !== rows[255 ^ c].length) {
    throw new Error(
      `MC tables: unambiguous case ${c} has ${rows[c].length / 3} triangles, complement ${255 ^ c} has ${rows[255 ^ c].length / 3}`);
  }
}

// Row stride: longest case plus a -1 terminator.
export const MC_ROW = maxLen + 1;
export const MC_TRI = new Int32Array(256 * MC_ROW).fill(-1);
rows.forEach((t, c) => MC_TRI.set(t, c * MC_ROW));

// Vertex pool cap for the GPU emit (multiple of 3 so clamped emission never
// splits a triangle); pos + normal as 2 x vec4f = 32 bytes/vertex -> 48 MB.
export const MC_CAP = 1500000;

// Corner offsets / edge endpoints for the emit kernel: makeWGSL interpolates
// these into the WGSL CO/EA/EB array literals, so JS and WGSL can't drift.
export { CORNERS as MC_CORNERS, EDGES as MC_EDGES };
