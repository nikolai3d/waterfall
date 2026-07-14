// WebGPU backend — MLS-MPM simulation in compute shaders.
//
// Same pipeline as the WebGL2 backend, without its workarounds: P2G scatter
// is fixed-point atomicAdd into a flat 3D grid storage buffer (no 27x point
// amplification, no float-blend), particle state is SoA storage buffers
// updated in place (no ping-pong MRT), and each substep is six dispatches
// recorded in a single compute pass. Rendering is the same screen-space
// fluid pipeline, with instanced quads standing in for point sprites.

import { ROCKS } from './shaders.js';
import { makeWGSL } from './wgsl.js';
import { MC_TRI, MC_CAP } from './mctables.js';

const NROCK = ROCKS.length;

export async function createBackend({ canvas, fail }) {
  if (!navigator.gpu) fail('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) fail('WebGPU: no adapter available.');
  const device = await adapter.requestDevice();
  device.onuncapturederror = (e) => fail('WebGPU error: ' + e.error.message);
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') fail('WebGPU device lost: ' + info.message);
  });

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  // GL-convention projection (clip z in [-w,w]) -> WebGPU (z in [0,w]).
  function fixProj(m) {
    const o = new Float32Array(m);
    for (let c = 0; c < 4; c++) {
      o[c * 4 + 2] = 0.5 * (m[c * 4 + 2] + m[c * 4 + 3]);
    }
    return o;
  }

  function buf(size, usage, data = null) {
    const b = device.createBuffer({ size, usage });
    if (data) device.queue.writeBuffer(b, 0, data);
    return b;
  }

  // -------------------------------------------------------------------------
  // Config-independent resources: bind group layouts, samplers, uniforms.

  const UNI = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const linSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const bufRenderU = buf(448, UNI);
  const bufBlurX = buf(16, UNI);
  const bufBlurY = buf(16, UNI);
  const bufTraceU = buf(16, UNI); // r=trace accumulation frame counter

  const simLayout = device.createBindGroupLayout({
    entries: [
      ...[0, 1, 2, 3, 4, 5].map((i) => ({
        binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' },
      })),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const renderLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      // cmat: the splat VS reads the isolation signal (spray shrink) from
      // cmat[3i].w. Binding 3 is skipped — it is fsThick's depth texture,
      // present only in thickLayout; matching indices keep one WGSL module.
      { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  // The thickness pass additionally reads the raw water depth for occlusion;
  // it needs its own layout because that texture is a render attachment in
  // the pass where the main render bind group is used.
  const thickLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } }, // cmat
    ],
  });
  const blurLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ],
  });
  const compLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    ],
  });
  // Volume renderer: grid tent blur (compute), raymarch (density buffer
  // read-only in a fragment stage), and the upscale blit.
  const volBlurLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const volLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  const volUpLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });
  // Marching-cubes mesh renderer (r=mesh): emit kernel (density + triangle
  // table in, vertex pool + indirect args out) and the vertex-pulling draw.
  const mcLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const meshLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  // Path tracer (r=trace): the accumulation kernel reads the previous
  // frame's accumulation (rgba32float: textureLoad only) while rendering
  // into the other of a ping-pong pair; the tone-map blit reads the result.
  const traceLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });
  const traceBlitLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
    ],
  });
  const simPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
  const renderPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] });
  const thickPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [thickLayout] });
  const blurPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [blurLayout] });
  const compPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [compLayout] });
  const volBlurPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [volBlurLayout] });
  const volPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [volLayout] });
  const volUpPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [volUpLayout] });
  const mcPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [mcLayout] });
  const meshPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [meshLayout] });
  const tracePipeLayout = device.createPipelineLayout({ bindGroupLayouts: [traceLayout] });
  const traceBlitPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [traceBlitLayout] });

  // MC buffers are config-independent but big (the vertex pool alone is
  // 48 MB), so they are created lazily by ensureMeshResources() on the first
  // mesh-mode frame instead of taxing the other renderers here: the triangle
  // table is constant, the vertex pool is capped (pos + normal vec4f pairs,
  // 32 B/vertex), and the indirect args buffer doubles as the emit kernel's
  // atomic vertex counter (reset per frame, clamped to the cap by
  // mcFinalize). Args word 4 sits past the 16-byte drawIndirect args and
  // carries the overflow flag mcFinalize raises.
  let bufMCTri = null;
  let bufMCVerts = null;
  let bufMCArgs = null;
  const mcArgsReset = new Uint32Array([0, 1, 0, 0, 0]); // vtxCount, instances, firsts, overflow

  // Trace accumulation targets are lazy the same way (rgba32float ping-pong
  // at the scaled size); created by ensureTraceResources() on the first
  // trace-mode frame, recreated when the scaled size changes.
  let traceRT = null;
  let traceBGs = null;
  let traceFlip = 0;
  const traceUData = new Uint32Array(4);

  const alphaBlend = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };
  const addBlend = {
    color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
  };
  const depthState = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };

  // -------------------------------------------------------------------------
  // Config-dependent state, rebuilt by init().

  let cfg = null;
  let NCELL = 0;
  let bufPos, bufVel, bufC, bufAux, bufGridA, bufGridV, bufSimU, bufVolDens;
  let simPipes = null;
  let simBG = null;
  let pipes = null;
  let renderBGGroup = null;
  let thickBGGroup = null;
  let volBlurBG = null;
  let volBG = null;
  let mcBG = null;
  let meshBG = null;
  let substepCount = 0;
  let volBlurredAt = -1; // substepCount at the last density blur (-1 = stale)
  let simBuffers = [];

  // Sim uniform staging: rocks (16f) + rockVel (16f, vec3 padded) + frame.
  const simUData = new ArrayBuffer(144);
  const simUF32 = new Float32Array(simUData);
  const simUU32 = new Uint32Array(simUData);

  // Render uniform staging (layout must match struct RenderU in wgsl.js).
  const renderUData = new ArrayBuffer(448);
  const renderUF32 = new Float32Array(renderUData);

  function init(config) {
    for (const b of simBuffers) b.destroy();
    cfg = config;
    NCELL = cfg.GRID ** 3;

    const S = makeWGSL({
      GRID: cfg.GRID, LIFE: cfg.LIFE, N: cfg.N, ISO: cfg.ISO, MISO: cfg.MISO, K: cfg.K,
      SPP: cfg.SPP, BOUNCES: cfg.BOUNCES, SPRAY: cfg.SPRAY,
    });
    const simModule = device.createShaderModule({ code: S.sim });
    const renderModule = device.createShaderModule({ code: S.render });
    const blurModule = device.createShaderModule({ code: S.blur });
    const compModule = device.createShaderModule({ code: S.composite });
    const volBlurModule = device.createShaderModule({ code: S.volBlur });
    const volModule = device.createShaderModule({ code: S.volume });
    const volUpModule = device.createShaderModule({ code: S.volUpscale });
    const mcModule = device.createShaderModule({ code: S.mcCompute });
    const meshModule = device.createShaderModule({ code: S.mesh });
    const traceModule = device.createShaderModule({ code: S.trace });
    const traceBlitModule = device.createShaderModule({ code: S.traceBlit });

    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    bufPos = buf(cfg.N * 16, ST | GPUBufferUsage.COPY_SRC, cfg.initialData);
    bufVel = buf(cfg.N * 16, ST, new Float32Array(cfg.N * 4));
    bufC = buf(cfg.N * 48, ST, new Float32Array(cfg.N * 12));
    bufAux = buf(cfg.N * 16, ST);
    bufGridA = buf(NCELL * 16, ST);
    bufGridV = buf(NCELL * 16, ST);
    bufSimU = buf(144, UNI);
    bufVolDens = buf(NCELL * 8, GPUBufferUsage.STORAGE); // blurred (mass, |momentum|)
    simBuffers = [bufPos, bufVel, bufC, bufAux, bufGridA, bufGridV, bufSimU, bufVolDens];

    const cp = (entryPoint) => device.createComputePipeline({
      layout: simPipeLayout, compute: { module: simModule, entryPoint },
    });
    simPipes = {
      clear: cp('clearGrid'), p2g1: cp('p2g1'), density: cp('density'),
      p2g2: cp('p2g2'), grid: cp('gridUpdate'), g2p: cp('g2p'),
    };
    simBG = device.createBindGroup({
      layout: simLayout,
      entries: [
        { binding: 0, resource: { buffer: bufPos } },
        { binding: 1, resource: { buffer: bufVel } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufAux } },
        { binding: 4, resource: { buffer: bufGridA } },
        { binding: 5, resource: { buffer: bufGridV } },
        { binding: 6, resource: { buffer: bufSimU } },
      ],
    });

    const rp = (desc) => device.createRenderPipeline({ layout: renderPipeLayout, ...desc });
    pipes = {
      bg: rp({
        vertex: { module: renderModule, entryPoint: 'vsFull' },
        fragment: { module: renderModule, entryPoint: 'fsBackground', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: depthState,
      }),
      bgCanvas: rp({
        vertex: { module: renderModule, entryPoint: 'vsFull' },
        fragment: { module: renderModule, entryPoint: 'fsBackground', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: depthState,
      }),
      points: rp({
        vertex: { module: renderModule, entryPoint: 'vsPoints' },
        fragment: { module: renderModule, entryPoint: 'fsPoints', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
        depthStencil: depthState,
      }),
      pointDepth: rp({
        vertex: { module: renderModule, entryPoint: 'vsPoints' },
        fragment: { module: renderModule, entryPoint: 'fsPointDepth', targets: [{ format: 'r32float' }] },
        primitive: { topology: 'triangle-strip' },
        depthStencil: depthState,
      }),
      thick: device.createRenderPipeline({
        layout: thickPipeLayout,
        vertex: { module: renderModule, entryPoint: 'vsThick' },
        fragment: { module: renderModule, entryPoint: 'fsThick', targets: [{ format: 'rgba16float', blend: addBlend }] },
        primitive: { topology: 'triangle-strip' },
      }),
      // Aniso (r=aniso) variants of the depth/thickness splats; the rest of
      // the SSF pipeline (blur, composite, occlusion) is shared unchanged.
      pointDepthAniso: rp({
        vertex: { module: renderModule, entryPoint: 'vsPointsAniso' },
        fragment: { module: renderModule, entryPoint: 'fsPointDepthAniso', targets: [{ format: 'r32float' }] },
        primitive: { topology: 'triangle-strip' },
        depthStencil: depthState,
      }),
      thickAniso: device.createRenderPipeline({
        layout: thickPipeLayout,
        vertex: { module: renderModule, entryPoint: 'vsThickAniso' },
        fragment: { module: renderModule, entryPoint: 'fsThickAniso', targets: [{ format: 'rgba16float', blend: addBlend }] },
        primitive: { topology: 'triangle-strip' },
      }),
      gizmoDepth: rp({
        vertex: { module: renderModule, entryPoint: 'vsGizmo' },
        fragment: { module: renderModule, entryPoint: 'fsGizmo', targets: [{ format, blend: alphaBlend }] },
        primitive: { topology: 'line-list' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
      }),
      gizmo: rp({
        vertex: { module: renderModule, entryPoint: 'vsGizmo' },
        fragment: { module: renderModule, entryPoint: 'fsGizmo', targets: [{ format, blend: alphaBlend }] },
        primitive: { topology: 'line-list' },
      }),
      blur: device.createRenderPipeline({
        layout: blurPipeLayout,
        vertex: { module: blurModule, entryPoint: 'vsFull' },
        fragment: { module: blurModule, entryPoint: 'fsBlur', targets: [{ format: 'r32float' }] },
        primitive: { topology: 'triangle-list' },
      }),
      composite: device.createRenderPipeline({
        layout: compPipeLayout,
        vertex: { module: compModule, entryPoint: 'vsFull' },
        fragment: { module: compModule, entryPoint: 'fsComposite', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      }),
      volBlur: device.createComputePipeline({
        layout: volBlurPipeLayout,
        compute: { module: volBlurModule, entryPoint: 'blurGrid' },
      }),
      volume: device.createRenderPipeline({
        layout: volPipeLayout,
        vertex: { module: volModule, entryPoint: 'vsFull' },
        fragment: { module: volModule, entryPoint: 'fsVolume', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
      }),
      voxel: device.createRenderPipeline({
        layout: volPipeLayout,
        vertex: { module: volModule, entryPoint: 'vsFull' },
        fragment: { module: volModule, entryPoint: 'fsVoxel', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
      }),
      volUpscale: device.createRenderPipeline({
        layout: volUpPipeLayout,
        vertex: { module: volUpModule, entryPoint: 'vsFull' },
        fragment: { module: volUpModule, entryPoint: 'fsUpscale', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      }),
      mcEmit: device.createComputePipeline({
        layout: mcPipeLayout,
        compute: { module: mcModule, entryPoint: 'mcEmit' },
      }),
      mcFinalize: device.createComputePipeline({
        layout: mcPipeLayout,
        compute: { module: mcModule, entryPoint: 'mcFinalize' },
      }),
      // No culling (default): MC triangle winding is arbitrary and the
      // fragment shader shades two-sided.
      mesh: device.createRenderPipeline({
        layout: meshPipeLayout,
        vertex: { module: meshModule, entryPoint: 'vsMesh' },
        fragment: { module: meshModule, entryPoint: 'fsMesh', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: depthState,
      }),
      trace: device.createRenderPipeline({
        layout: tracePipeLayout,
        vertex: { module: traceModule, entryPoint: 'vsFull' },
        fragment: { module: traceModule, entryPoint: 'fsTrace', targets: [{ format: 'rgba32float' }] },
        primitive: { topology: 'triangle-list' },
      }),
      traceBlit: device.createRenderPipeline({
        layout: traceBlitPipeLayout,
        vertex: { module: traceBlitModule, entryPoint: 'vsFull' },
        fragment: { module: traceBlitModule, entryPoint: 'fsTone', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
      }),
    };

    volBlurBG = device.createBindGroup({
      layout: volBlurLayout,
      entries: [
        { binding: 0, resource: { buffer: bufGridV } },
        { binding: 1, resource: { buffer: bufVolDens } },
      ],
    });
    volBG = device.createBindGroup({
      layout: volLayout,
      entries: [
        { binding: 0, resource: { buffer: bufRenderU } },
        { binding: 1, resource: { buffer: bufVolDens } },
      ],
    });
    // Mesh bind groups reference the per-config density buffer, so a rebuild
    // invalidates them; ensureMeshResources() recreates them on the next
    // mesh-mode frame (and the capped pool buffers on the first one ever).
    mcBG = null;
    meshBG = null;
    traceBGs = null; // reference the per-config density buffer too

    makeRenderBG();
    substepCount = 0;
    volBlurredAt = -1; // fresh bufVolDens: force a re-blur on the next frame
  }

  // Tent-blur the grid density into bufVolDens — but only when the grid
  // actually changed (a substep ran, or init rebuilt the buffers). Paused
  // frames — the trace renderer's "cook" frames especially — would otherwise
  // re-blur an unchanged grid every frame.
  function encodeVolBlur(enc) {
    if (volBlurredAt === substepCount) return;
    const cpass = enc.beginComputePass();
    cpass.setPipeline(pipes.volBlur);
    cpass.setBindGroup(0, volBlurBG);
    cpass.dispatchWorkgroups(Math.ceil(NCELL / 64));
    cpass.end();
    volBlurredAt = substepCount;
  }

  // Trace accumulation ping-pong at the scaled (RSCALE) size. The textures
  // are per-size (recreated when the scaled target changes), the bind groups
  // additionally per-config (bufVolDens), which init() handles by nulling
  // traceBGs. ensureTargets() must have run (RT.volW/volH).
  function ensureTraceResources() {
    if (traceRT && (traceRT.w !== RT.volW || traceRT.h !== RT.volH)) {
      for (const t of traceRT.textures) t.destroy();
      traceRT = null;
    }
    if (!traceRT) {
      const AT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
      const mk = () => device.createTexture({
        size: [RT.volW, RT.volH], format: 'rgba32float', usage: AT,
      });
      const texA = mk();
      const texB = mk();
      traceRT = {
        w: RT.volW, h: RT.volH,
        textures: [texA, texB],
        views: [texA.createView(), texB.createView()],
      };
      traceBGs = null;
    }
    if (!traceBGs) {
      const tbg = (prevView) => device.createBindGroup({
        layout: traceLayout,
        entries: [
          { binding: 0, resource: { buffer: bufRenderU } },
          { binding: 1, resource: { buffer: bufVolDens } },
          { binding: 2, resource: prevView },
          { binding: 3, resource: { buffer: bufTraceU } },
        ],
      });
      const bbg = (accView) => device.createBindGroup({
        layout: traceBlitLayout,
        entries: [{ binding: 0, resource: accView }],
      });
      // Indexed by the DESTINATION texture: trace[i] reads the other one.
      traceBGs = {
        trace: [tbg(traceRT.views[1]), tbg(traceRT.views[0])],
        blit: [bbg(traceRT.views[0]), bbg(traceRT.views[1])],
      };
    }
  }

  function ensureMeshResources() {
    if (!bufMCVerts) {
      bufMCTri = buf(MC_TRI.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, MC_TRI);
      bufMCVerts = device.createBuffer({ size: MC_CAP * 32, usage: GPUBufferUsage.STORAGE });
      bufMCArgs = buf(mcArgsReset.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    }
    if (!mcBG) {
      mcBG = device.createBindGroup({
        layout: mcLayout,
        entries: [
          { binding: 0, resource: { buffer: bufVolDens } },
          { binding: 1, resource: { buffer: bufMCTri } },
          { binding: 2, resource: { buffer: bufMCVerts } },
          { binding: 3, resource: { buffer: bufMCArgs } },
        ],
      });
      meshBG = device.createBindGroup({
        layout: meshLayout,
        entries: [
          { binding: 0, resource: { buffer: bufRenderU } },
          { binding: 1, resource: { buffer: bufMCVerts } },
          { binding: 2, resource: { buffer: bufVolDens } },
        ],
      });
    }
  }

  // The thick group depends on per-config buffers AND the per-size raw
  // water depth (fsThick's occlusion input), so both init() and
  // ensureTargets() rebuild it.
  function makeRenderBG() {
    if (!bufPos) return;
    renderBGGroup = device.createBindGroup({
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: bufRenderU } },
        { binding: 1, resource: { buffer: bufPos } },
        { binding: 2, resource: { buffer: bufVel } },
        { binding: 4, resource: { buffer: bufC } },
      ],
    });
    if (!RT) return;
    thickBGGroup = device.createBindGroup({
      layout: thickLayout,
      entries: [
        { binding: 0, resource: { buffer: bufRenderU } },
        { binding: 1, resource: { buffer: bufPos } },
        { binding: 2, resource: { buffer: bufVel } },
        { binding: 3, resource: RT.v.waterDepth },
        { binding: 4, resource: { buffer: bufC } },
      ],
    });
  }

  // -------------------------------------------------------------------------
  // Simulation substep: six dispatches in one compute pass (WebGPU inserts
  // the needed barriers between dispatches automatically).

  function substep() {
    simUF32.set(cfg.rockData, 0);
    for (let r = 0; r < NROCK; r++) {
      simUF32.set(cfg.rockVel.subarray(r * 3, r * 3 + 3), 16 + r * 4);
    }
    simUU32[32] = substepCount;
    device.queue.writeBuffer(bufSimU, 0, simUData);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setBindGroup(0, simBG);
    pass.setPipeline(simPipes.clear);
    pass.dispatchWorkgroups(Math.ceil((NCELL * 4) / 256));
    pass.setPipeline(simPipes.p2g1);
    pass.dispatchWorkgroups(Math.ceil(cfg.N / 64));
    pass.setPipeline(simPipes.density);
    pass.dispatchWorkgroups(Math.ceil(cfg.N / 64));
    pass.setPipeline(simPipes.p2g2);
    pass.dispatchWorkgroups(Math.ceil(cfg.N / 64));
    pass.setPipeline(simPipes.grid);
    pass.dispatchWorkgroups(Math.ceil(NCELL / 64));
    pass.setPipeline(simPipes.g2p);
    pass.dispatchWorkgroups(Math.ceil(cfg.N / 64));
    pass.end();
    device.queue.submit([enc.finish()]);
    substepCount++;
  }

  // -------------------------------------------------------------------------
  // Render targets (per canvas size) + their bind groups.

  let RT = null;

  function ensureTargets(w, h) {
    if (RT && RT.w === w && RT.h === h) return;
    if (RT) for (const t of RT.textures) t.destroy();
    const hw = Math.max(1, Math.ceil(w / 2));
    const hh = Math.max(1, Math.ceil(h / 2));
    const AT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (fmt, tw, th, usage = AT) =>
      device.createTexture({ size: [tw, th], format: fmt, usage });

    const sceneColor = tex('rgba8unorm', w, h);
    const sceneDepth = tex('depth24plus', w, h, GPUTextureUsage.RENDER_ATTACHMENT);
    const waterDepth = tex('r32float', w, h);
    const blurA = tex('r32float', w, h);
    const blurB = tex('r32float', w, h);
    const thick = tex('rgba16float', hw, hh);
    const volW = Math.max(1, Math.round(w * cfg.RSCALE));
    const volH = Math.max(1, Math.round(h * cfg.RSCALE));
    const volColor = tex('rgba8unorm', volW, volH);

    const v = {
      sceneColor: sceneColor.createView(),
      sceneDepth: sceneDepth.createView(),
      waterDepth: waterDepth.createView(),
      blurA: blurA.createView(),
      blurB: blurB.createView(),
      thick: thick.createView(),
      volColor: volColor.createView(),
    };
    const blurBG = (view, ubuf) => device.createBindGroup({
      layout: blurLayout,
      entries: [
        { binding: 0, resource: { buffer: ubuf } },
        { binding: 1, resource: view },
      ],
    });
    RT = {
      w, h, v, volW, volH,
      textures: [sceneColor, sceneDepth, waterDepth, blurA, blurB, thick, volColor],
      blurBGs: [blurBG(v.waterDepth, bufBlurX), blurBG(v.blurA, bufBlurY), blurBG(v.blurB, bufBlurX)],
      compBG: device.createBindGroup({
        layout: compLayout,
        entries: [
          { binding: 0, resource: { buffer: bufRenderU } },
          { binding: 1, resource: v.sceneColor },
          { binding: 2, resource: linSamp },
          { binding: 3, resource: v.blurB },
          { binding: 4, resource: v.thick },
        ],
      }),
      volUpBG: device.createBindGroup({
        layout: volUpLayout,
        entries: [
          { binding: 0, resource: v.volColor },
          { binding: 1, resource: linSamp },
        ],
      }),
    };
    makeRenderBG();
  }

  function writeRenderU(frame) {
    const f = renderUF32;
    f.set(fixProj(frame.pv), 0);
    f.set(fixProj(frame.proj), 16);
    f.set(frame.view, 32);
    f.set(frame.eye, 48);
    f.set(frame.right, 52);
    f.set(frame.up, 56);
    f.set(frame.fwd, 60);
    f.set(frame.lightW, 64);
    f.set(frame.lightV, 68);
    f.set([frame.w, frame.h, frame.tanF, frame.aspect], 72);
    f.set([frame.h * frame.proj[5], Math.ceil(frame.h / 2) * frame.proj[5], RT.volW, RT.volH], 76);
    f.set(cfg.rockData, 80);
    const g = frame.gizmo;
    f.set(g ? [...g.a, 1, ...g.b, g.r, 0, 0, 0, 0] : new Array(12).fill(0), 80 + NROCK * 4);
    device.queue.writeBuffer(bufRenderU, 0, renderUData);
  }

  function render(frame) {
    ensureTargets(frame.w, frame.h);
    writeRenderU(frame);

    const enc = device.createCommandEncoder();
    const canvasView = context.getCurrentTexture().createView();
    const clearCol = { r: 0.01, g: 0.015, b: 0.02, a: 1 };

    if (frame.mode === 'volume' || frame.mode === 'voxel') {
      // Volumetric raymarch / voxel DDA: tent-blur the grid density
      // (compute, skipped when the grid is unchanged), trace it into a
      // scaled offscreen target, upscale.
      encodeVolBlur(enc);

      let pass = enc.beginRenderPass({
        colorAttachments: [{ view: RT.v.volColor, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
      });
      pass.setPipeline(frame.mode === 'voxel' ? pipes.voxel : pipes.volume);
      pass.setBindGroup(0, volBG);
      pass.draw(3);
      pass.end();

      pass = enc.beginRenderPass({
        colorAttachments: [{ view: canvasView, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
      });
      pass.setPipeline(pipes.volUpscale);
      pass.setBindGroup(0, RT.volUpBG);
      pass.draw(3);
      if (frame.gizmo) {
        pass.setPipeline(pipes.gizmo);
        pass.setBindGroup(0, renderBGGroup);
        pass.draw(130);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      return;
    }

    if (frame.mode === 'trace') {
      // Progressive path tracer: tent-blur the density grid, extend the
      // accumulation by one frame (ping-ponged rgba32float, running average
      // weighted 1/frame), tone-map into the shared scaled target, upscale.
      // frame.accFrame is the app's accumulation counter; 1 (or absent)
      // means restart — camera/rocks/sim/config changed.
      ensureTraceResources();
      traceUData[0] = Math.max(frame.accFrame || 0, 1);
      device.queue.writeBuffer(bufTraceU, 0, traceUData);

      encodeVolBlur(enc);

      let pass = enc.beginRenderPass({
        colorAttachments: [{ view: traceRT.views[traceFlip], loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(pipes.trace);
      pass.setBindGroup(0, traceBGs.trace[traceFlip]);
      pass.draw(3);
      pass.end();

      pass = enc.beginRenderPass({
        colorAttachments: [{ view: RT.v.volColor, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
      });
      pass.setPipeline(pipes.traceBlit);
      pass.setBindGroup(0, traceBGs.blit[traceFlip]);
      pass.draw(3);
      pass.end();

      pass = enc.beginRenderPass({
        colorAttachments: [{ view: canvasView, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
      });
      pass.setPipeline(pipes.volUpscale);
      pass.setBindGroup(0, RT.volUpBG);
      pass.draw(3);
      if (frame.gizmo) {
        pass.setPipeline(pipes.gizmo);
        pass.setBindGroup(0, renderBGGroup);
        pass.draw(130);
      }
      pass.end();
      traceFlip ^= 1;
      device.queue.submit([enc.finish()]);
      return;
    }

    if (frame.mode === 'mesh') {
      // Marching-cubes isosurface: reuse the volume renderer's tent-blurred
      // density, emit a triangle mesh (atomic slot reservation into a capped
      // vertex pool + indirect args), then draw it depth-tested in the same
      // pass as the raytraced background — real occlusion both ways.
      ensureMeshResources();
      device.queue.writeBuffer(bufMCArgs, 0, mcArgsReset);
      encodeVolBlur(enc);
      const cpass = enc.beginComputePass();
      cpass.setPipeline(pipes.mcEmit);
      cpass.setBindGroup(0, mcBG);
      cpass.dispatchWorkgroups(Math.ceil(NCELL / 64));
      cpass.setPipeline(pipes.mcFinalize);
      cpass.dispatchWorkgroups(1);
      cpass.end();

      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: canvasView, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
        depthStencilAttachment: {
          view: RT.v.sceneDepth, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1,
        },
      });
      pass.setBindGroup(0, renderBGGroup);
      pass.setPipeline(pipes.bgCanvas);
      pass.draw(3);
      pass.setPipeline(pipes.mesh);
      pass.setBindGroup(0, meshBG);
      pass.drawIndirect(bufMCArgs, 0);
      if (frame.gizmo) {
        pass.setPipeline(pipes.gizmoDepth);
        pass.setBindGroup(0, renderBGGroup);
        pass.draw(130);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      return;
    }

    if (frame.mode === 'points') {
      // Legacy view: background + shaded impostors straight to the canvas.
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: canvasView, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
        depthStencilAttachment: {
          view: RT.v.sceneDepth, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1,
        },
      });
      pass.setBindGroup(0, renderBGGroup);
      pass.setPipeline(pipes.bgCanvas);
      pass.draw(3);
      pass.setPipeline(pipes.points);
      pass.draw(4, cfg.N);
      if (frame.gizmo) { pass.setPipeline(pipes.gizmoDepth); pass.draw(130); }
      pass.end();
      device.queue.submit([enc.finish()]);
      return;
    }

    // Screen-space fluid path ('ssf' and 'aniso' — aniso swaps the depth and
    // thickness splat pipelines for ellipsoid variants, everything else is
    // identical).
    const aniso = frame.mode === 'aniso';

    // 1. scene (cube walls + rocks) into offscreen color + depth
    let pass = enc.beginRenderPass({
      colorAttachments: [{ view: RT.v.sceneColor, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
      depthStencilAttachment: {
        view: RT.v.sceneDepth, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1,
      },
    });
    pass.setBindGroup(0, renderBGGroup);
    pass.setPipeline(pipes.bg);
    pass.draw(3);
    pass.end();

    // 2. water surface depth (z-tested against the shared scene depth)
    pass = enc.beginRenderPass({
      colorAttachments: [{ view: RT.v.waterDepth, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      depthStencilAttachment: {
        view: RT.v.sceneDepth, depthLoadOp: 'load', depthStoreOp: 'store',
      },
    });
    pass.setBindGroup(0, renderBGGroup);
    pass.setPipeline(aniso ? pipes.pointDepthAniso : pipes.pointDepth);
    pass.draw(4, cfg.N);
    pass.end();

    // 3. depth-aware separable blur, two iterations
    const scalePx = frame.h * frame.proj[5];
    device.queue.writeBuffer(bufBlurX, 0, new Float32Array([1, 0, scalePx, 0]));
    device.queue.writeBuffer(bufBlurY, 0, new Float32Array([0, 1, scalePx, 0]));
    for (const [outView, bg] of [
      [RT.v.blurA, RT.blurBGs[0]], [RT.v.blurB, RT.blurBGs[1]],
      [RT.v.blurA, RT.blurBGs[2]], [RT.v.blurB, RT.blurBGs[1]],
    ]) {
      pass = enc.beginRenderPass({
        colorAttachments: [{ view: outView, loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(pipes.blur);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // 4. thickness + foam, half resolution, additive
    pass = enc.beginRenderPass({
      colorAttachments: [{ view: RT.v.thick, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    });
    pass.setBindGroup(0, thickBGGroup);
    pass.setPipeline(aniso ? pipes.thickAniso : pipes.thick);
    pass.draw(4, cfg.N);
    pass.end();

    // 5. composite to the canvas (+ gizmo overlay)
    pass = enc.beginRenderPass({
      colorAttachments: [{ view: canvasView, loadOp: 'clear', storeOp: 'store', clearValue: clearCol }],
    });
    pass.setPipeline(pipes.composite);
    pass.setBindGroup(0, RT.compBG);
    pass.draw(3);
    if (frame.gizmo) {
      pass.setPipeline(pipes.gizmo);
      pass.setBindGroup(0, renderBGGroup);
      pass.draw(130);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  async function readParticles() {
    const size = cfg.N * 16;
    const staging = device.createBuffer({
      size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bufPos, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  function dispose() {
    for (const b of simBuffers) b.destroy();
    simBuffers = [];
    if (RT) for (const t of RT.textures) t.destroy();
    device.destroy();
  }

  // effectiveMode: this backend implements every render mode as selected.
  return {
    name: 'webgpu', init, substep, render, readParticles, dispose,
    effectiveMode: (mode) => mode,
  };
}
