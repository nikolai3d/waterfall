# waterfall

A real-time 3D fluid simulation — a waterfall splashing onto rocks inside a
cube — running entirely on the GPU in the browser. Two backends share one
app: **WebGPU** (compute shaders, used when available) and **WebGL2** (a
fallback that emulates compute with render passes).

![screenshot](docs/screenshot.png)

**Live physics, no baked animation:** ~147,000 fluid particles simulated with
**MLS-MPM** (Moving Least Squares Material Point Method, [Hu et al.,
SIGGRAPH 2018](https://yuanming.taichi.graphics/publication/2018-mlsmpm/)),
the same family of methods behind most modern real-time fluid demos.

## Running

Any static file server works (ES modules require http):

```sh
python3 -m http.server 8123
# then open http://localhost:8123
```

Controls: **drag a rock** to move it (the water is pushed aside as it goes),
**drag** elsewhere to orbit, **wheel** to zoom, **space** to pause,
**f** to cycle through the renderers. The panel
in the top-right corner adjusts grid resolution (32³–128³) and particle
count (128²–512²); changing either restarts the water in place (the camera
survives).

## How it works

Everything — particle state, the simulation grid, and all physics — lives on
the GPU. Per substep:

1. **P2G pass 1** — each particle scatters mass and momentum (including its
   affine velocity matrix *C*) to its 27 neighboring grid cells. On WebGPU
   this is a fixed-point `atomicAdd` into a 3D grid storage buffer; on
   WebGL2 it is emulated by rendering one GL point per (particle, cell)
   pair into a tiled grid texture with additive blending.
2. **Density** — a fragment pass gathers grid mass back to each particle and
   evaluates a weakly compressible equation of state (no pressure solve
   needed — this is what makes MLS-MPM so friendly to GPUs).
3. **P2G pass 2** — pressure and viscosity forces are scattered to the grid,
   fused into grid momentum (eq. 16 of the MLS-MPM paper).
4. **Grid update** — momentum → velocity, gravity, a CFL velocity clamp, and
   free-slip boundary conditions against the cube walls and the rock SDF.
   Rock positions are uniforms, so rocks can be dragged live; a dragged
   rock's velocity enters this boundary condition, shoving water aside.
5. **G2P** — particles gather their new velocity and affine matrix from the
   grid and advect. Particles are recycled through the spout on a fixed
   lifetime, so the waterfall runs forever.

On WebGL2 the 3D grid is tiled slice-by-slice into a 2D texture (WebGL2
can't render into 3D textures per-slice with blending) — e.g. 64³ into
512×512, with the tiling derived per grid size — and particle state ping-
pongs through float MRT textures; the WebGPU backend uses flat storage
buffers updated in place. Rocks and walls are raytraced analytically in a
fragment shader that writes real depth, so the water composites correctly
against them. See `docs/perf-webgpu.md` for a measured comparison of the
two backends.

### Rendering

The water surface uses **screen-space fluid rendering** (van der Laan et al.,
I3D 2009): sphere impostors write linear view-space depth to an offscreen
target (z-tested against the raytraced scene depth), a depth-aware separable
blur smooths it into a continuous surface, and an additive half-resolution
pass accumulates thickness plus a speed-weighted foam channel. The blur also
closes gaps between droplets at compatible depths (a morphological closing
that never grows outer silhouettes), and the composite fades sparse water by
thickness, so spray reads as translucent droplets rather than opaque spheres. A composite
pass reconstructs normals from the smoothed depth and shades the surface:
refraction of the scene, Beer–Lambert absorption by thickness, Fresnel,
specular, and foam. The blur uses narrow-range lower-bound clamping
(Truong & Yuksel, i3D 2018) so near surfaces win where they overlap far
ones, and the thickness pass culls water well behind the visible surface
(tested against the raw per-pixel depth), so the pool neither shows
through the stream in front of it nor punches fake holes around it.
Use `?r=points` (or cycle with **f**) for the raw shaded-particle view.

`?r=aniso` upgrades the same screen-space pipeline with **anisotropic
ellipsoid splats** (after Yu & Turk, ACM TOG 2013): each particle's splat
stretches along its velocity direction (elongation `1 + k·min(speed/VMAX, 1)`,
tuned with `?k=` and ramped in over a particle's first moments so fresh spout
spray doesn't streak) while the minor axes shrink to conserve volume, and the
depth/thickness fragments intersect the oriented ellipsoid instead of a
sphere — so thin sheets and streams reconstruct smoothly from far fewer
visible bumps (most noticeable at low particle counts, e.g. 64³/192²). The
blur, thickness accumulation, occlusion, and composite stages are shared
with `ssf` unchanged; `?k=0` reduces to the spherical splats (matching them
exactly wherever the 1–96 px sprite-size clamp is inactive).

`?r=volume` instead raymarches the simulation's own density grid as a true
3D volume: a fragment shader marches the (tent-blurred) grid mass to an
iso-surface, refines the hit by bisection, shades with a density-gradient
normal, then follows one refracted ray through the interior — integrating
Beer–Lambert absorption from the marched density — to the analytic
walls-and-rocks background, with foam from the grid's momentum magnitude;
it renders at a reduced resolution (`?rscale=`, default 0.5) and upscales;
its iso threshold is fixed at 0.5 (`?iso=` affects the voxel and mesh
renderers only).

`?r=voxel` renders that same density grid as literal grid-aligned cubes
(a deliberately chunky, Minecraft-water look that exposes the simulation's
actual data structure): a DDA raycast (Amanatides & Woo) walks each ray
through the cells and the first cell above the `?iso=` density threshold is
a cube-face hit with an axis-aligned normal — so the voxel size visibly
tracks the grid chip. Faces get flat per-face diffuse (tops lighter),
cell-edge darkening, whitecaps where a cell's momentum/mass is high, and
one refracted continuation ray (also DDA) that Beer–Lambert-attenuates the
analytic background by the water path length. It shares the volume
renderer's blurred density field and scaled target (`?rscale=`).

`?r=mesh` (WebGPU only) extracts a real triangle mesh from that same blurred
density grid with **marching cubes**, recomputed on the GPU every frame: a
compute kernel classifies each cell against the `?iso=` threshold (shared
with the voxel renderer), reserves output slots with one atomic add on the
indirect-draw args (the simple alternative to a prefix-sum scan — triangle
order is arbitrary, which an opaque surface doesn't care about), and emits
edge-interpolated vertices with density-gradient normals into a capped
vertex pool (1.5M vertices, silently clamped if ever hit) drawn via
`drawIndirect` with vertex pulling. The mesh is depth-tested in the same
pass as the raytraced background, so water and rocks occlude each other
geometrically. Shading is glassy water: Fresnel over the analytic scene
continued along one refracted ray, Beer–Lambert absorption from a few
interior density taps, and foam from the grid's momentum magnitude at the
surface. Topology changes frame to frame, so mild temporal popping is
inherent (the density blur keeps it tame). On WebGL2 the chip stays enabled
but the mode renders the `ssf` path (with a console note, and the stats line
reads `mesh→ssf` instead of `mesh`). If the WebGPU backend fails to load and
the app auto-falls back, the stats line marks it as `webgl2 (fallback)`.

`?r=trace` (WebGPU only) is the "let it cook" beauty mode: a **progressive
path tracer** that accumulates one (or `?spp=`) jittered light path per
pixel per frame into a floating-point buffer, converging from noise to a
smooth image over time. The scene needs no acceleration structure — walls
and rocks are already analytic, and the water is the same tent-blurred
density grid the volume renderer marches (iso 0.5, coarse march +
bisection). Water hits split into reflection/refraction by Schlick-Fresnel
russian roulette; interior segments attenuate Beer–Lambert from the marched
density (plus the same deterministic in-scatter tint the other water
shaders use); walls and rocks bounce diffusely (cosine-weighted) with the
albedos the other renderers shade with. At every surface event a shadow ray
runs toward the directional light through rocks (opaque) and water
(attenuating) — so rocks and water cast real colored shadows and thin water
passes caustic-ish light onto the floor; a small constant ambient stands in
for the sky (the closed cube is treated as the world, with the light
shining through the walls), and paths terminate by russian roulette after
two bounces (`?bounces=` caps depth). The accumulation is tone-mapped
(exposure + Reinhard) and upscaled from the `?rscale=` target. Selecting
trace **pauses the simulation** so the image can converge — space resumes
as always (while the sim runs, the accumulation restarts every frame: a
live noisy preview); any camera move, rock drag, sim step, or panel change
also restarts it, and the stats line counts the accumulated samples
(`spp=`). On WebGL2 the chip falls back to the `ssf` path (`trace→ssf` in
the stats line). Note that after a grid/particles change while paused, the
grid-density renderers (volume, voxel, mesh, trace) show no water until the
sim advances — the freshly rebuilt grid is empty until the first substep.

## URL parameters

| param  | default    | meaning                                            |
| ------ | ---------- | -------------------------------------------------- |
| `g`    | 96         | grid resolution per axis (32, 64, 96, or 128)      |
| `p`    | 384        | particle texture size (`p`² particles)             |
| `s`    | `g` / 32   | simulation substeps per frame                      |
| `l`    | 2600       | particle lifetime in substeps (spout recycling)    |
| `warm` | 0          | substeps to pre-simulate before the first frame    |
| `r`    | `ssf`      | rendering: `ssf` (water surface), `points`, `volume`, `voxel`, `aniso`, `mesh`/`trace` (WebGPU only) |
| `rscale`| 0.5       | offscreen target scale for `r=volume`/`r=voxel`/`r=trace` (0.1–1) |
| `iso`  | 1.5        | density threshold for `r=voxel` cells and the `r=mesh` isosurface (0.1–16) |
| `k`    | 1.5        | splat elongation gain for `r=aniso` (0–4)          |
| `spp`  | 1          | paths traced per pixel per frame for `r=trace` (1–8) |
| `bounces`| 4        | maximum path depth for `r=trace` (1–8)             |
| `api`  | auto       | backend: `webgpu` or `webgl2` (auto-detects)       |
| `bench`| off        | time N frames after warmup, then freeze + report   |
| `dbg`  | off        | overlay with GPU-readback particle statistics      |

Example: [`?p=128&s=3`](http://localhost:8123/?p=128&s=3) for slower machines.

## Requirements

WebGPU where available; otherwise WebGL2 with `EXT_color_buffer_float` and
`EXT_float_blend` (available in all current desktop browsers).

## License

MIT
