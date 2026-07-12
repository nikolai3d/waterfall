# waterfall

A real-time 3D fluid simulation — a waterfall splashing onto rocks inside a
cube — running entirely on the GPU in the browser via WebGL2.

![screenshot](docs/screenshot.png)

**Live physics, no baked animation:** ~65,000 fluid particles simulated with
**MLS-MPM** (Moving Least Squares Material Point Method, [Hu et al.,
SIGGRAPH 2018](https://yuanming.taichi.graphics/publication/2018-mlsmpm/)),
the same family of methods behind most modern real-time fluid demos.

## Running

Any static file server works (ES modules require http):

```sh
python3 -m http.server 8123
# then open http://localhost:8123
```

Controls: **drag** to orbit, **wheel** to zoom, **space** to pause.

## How it works

Everything — particle state, the simulation grid, and all physics — lives in
float textures and runs in shaders. Per substep:

1. **P2G pass 1** — each particle scatters mass and momentum (including its
   affine velocity matrix *C*) to its 27 neighboring grid cells. Scatter is
   done by rendering one GL point per (particle, cell) pair into a tiled 3D
   grid texture with additive blending.
2. **Density** — a fragment pass gathers grid mass back to each particle and
   evaluates a weakly compressible equation of state (no pressure solve
   needed — this is what makes MLS-MPM so friendly to GPUs).
3. **P2G pass 2** — pressure and viscosity forces are scattered to the grid,
   fused into grid momentum (eq. 16 of the MLS-MPM paper).
4. **Grid update** — momentum → velocity, gravity, a CFL velocity clamp, and
   free-slip boundary conditions against the cube walls and the rock SDF.
5. **G2P** — particles gather their new velocity and affine matrix from the
   grid and advect. Particles are recycled through the spout on a fixed
   lifetime, so the waterfall runs forever.

The 64³ grid is tiled into a 512×512 2D texture (WebGL2 can't render into 3D
textures per-slice with blending). Rocks and walls are raytraced analytically
in a fragment shader that writes real depth, so the particle spheres
composite correctly against them.

## URL parameters

| param  | default | meaning                                          |
| ------ | ------- | ------------------------------------------------ |
| `p`    | 256     | particle texture size (`p`² particles)           |
| `s`    | 2       | simulation substeps per frame                    |
| `l`    | 2600    | particle lifetime in substeps (spout recycling)  |
| `warm` | 0       | substeps to pre-simulate before the first frame  |
| `dbg`  | off     | overlay with GPU-readback particle statistics    |

Example: [`?p=128&s=3`](http://localhost:8123/?p=128&s=3) for slower machines.

## Requirements

WebGL2 with `EXT_color_buffer_float` and `EXT_float_blend` (available in all
current desktop browsers).

## License

MIT
