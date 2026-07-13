// WebGPU backend — MLS-MPM simulation in compute shaders.
//
// Same pipeline as the WebGL2 backend, without its workarounds: P2G scatter
// is fixed-point atomicAdd into a flat 3D grid storage buffer (no 27x point
// amplification, no float-blend), particle state is SoA storage buffers
// updated in place (no ping-pong MRT), and each substep is six dispatches
// recorded in a single compute pass.

import { ROCKS } from './shaders.js';
import { makeWGSL } from './wgsl.js';

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
  // Config-dependent state, rebuilt by init().

  let cfg = null;
  let NCELL = 0;
  let bufPos, bufVel, bufC, bufAux, bufGridA, bufGridV, bufSimU;
  let simPipes = null; // { clear, p2g1, density, p2g2, grid, g2p }
  let simBG = null;
  let renderPipes = null; // { bg, points }
  let bufRenderU, renderBGGroup;
  let substepCount = 0;
  let allBuffers = [];

  // Sim uniform staging: rocks (16f) + rockVel (16f, vec3 padded) + frame.
  const simUData = new ArrayBuffer(144);
  const simUF32 = new Float32Array(simUData);
  const simUU32 = new Uint32Array(simUData);

  // Render uniform staging (layout must match struct RenderU in wgsl.js).
  const renderUData = new ArrayBuffer((16 * 3 + 4 * 9 + 4 * NROCK + 4 * 3) * 4);
  const renderUF32 = new Float32Array(renderUData);

  function init(config) {
    for (const b of allBuffers) b.destroy();
    allBuffers = [];
    cfg = config;
    NCELL = cfg.GRID ** 3;

    const S = makeWGSL({ GRID: cfg.GRID, LIFE: cfg.LIFE, N: cfg.N });
    const simModule = device.createShaderModule({ code: S.sim });
    const renderModule = device.createShaderModule({ code: S.render });

    const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    bufPos = buf(cfg.N * 16, ST | GPUBufferUsage.COPY_SRC, cfg.initialData);
    bufVel = buf(cfg.N * 16, ST, new Float32Array(cfg.N * 4));
    bufC = buf(cfg.N * 48, ST, new Float32Array(cfg.N * 12));
    bufAux = buf(cfg.N * 16, ST);
    bufGridA = buf(NCELL * 16, ST);
    bufGridV = buf(NCELL * 16, ST);
    bufSimU = buf(144, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    bufRenderU = buf(renderUData.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    allBuffers = [bufPos, bufVel, bufC, bufAux, bufGridA, bufGridV, bufSimU, bufRenderU];

    // One bind group layout shared by all sim kernels.
    const simLayout = device.createBindGroupLayout({
      entries: [
        ...[0, 1, 2, 3, 4, 5].map((i) => ({
          binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' },
        })),
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const simPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
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

    // Render pipelines: raytraced background + instanced-quad impostors.
    const renderLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const renderPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] });
    const depthState = {
      format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less',
    };
    renderPipes = {
      bg: device.createRenderPipeline({
        layout: renderPipeLayout,
        vertex: { module: renderModule, entryPoint: 'vsFull' },
        fragment: { module: renderModule, entryPoint: 'fsBackground', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: depthState,
      }),
      points: device.createRenderPipeline({
        layout: renderPipeLayout,
        vertex: { module: renderModule, entryPoint: 'vsPoints' },
        fragment: { module: renderModule, entryPoint: 'fsPoints', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
        depthStencil: depthState,
      }),
    };
    renderBGGroup = device.createBindGroup({
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: bufRenderU } },
        { binding: 1, resource: { buffer: bufPos } },
        { binding: 2, resource: { buffer: bufVel } },
      ],
    });

    substepCount = 0;
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
  // Rendering

  let depthTex = null, depthView = null, depthW = 0, depthH = 0;

  function ensureTargets(w, h) {
    if (depthTex && depthW === w && depthH === h) return;
    if (depthTex) depthTex.destroy();
    depthTex = device.createTexture({
      size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    depthView = depthTex.createView();
    depthW = w; depthH = h;
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
    f.set([frame.h * frame.proj[5], Math.ceil(frame.h / 2) * frame.proj[5], 0, 0], 76);
    f.set(cfg.rockData, 80);
    const g = frame.gizmo;
    f.set(g ? [...g.a, 1, ...g.b, g.r, 0, 0, 0, 0] : new Array(12).fill(0), 80 + NROCK * 4);
    device.queue.writeBuffer(bufRenderU, 0, renderUData);
  }

  function render(frame) {
    ensureTargets(frame.w, frame.h);
    writeRenderU(frame);

    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0.01, g: 0.015, b: 0.02, a: 1 },
      }],
      depthStencilAttachment: {
        view: depthView, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1,
      },
    });
    pass.setBindGroup(0, renderBGGroup);
    pass.setPipeline(renderPipes.bg);
    pass.draw(3);
    // SSF pipeline lands next; until then both modes render shaded impostors.
    pass.setPipeline(renderPipes.points);
    pass.draw(4, cfg.N);
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
    for (const b of allBuffers) b.destroy();
    allBuffers = [];
    if (depthTex) depthTex.destroy();
    device.destroy();
  }

  return { name: 'webgpu', init, substep, render, readParticles, dispose };
}
