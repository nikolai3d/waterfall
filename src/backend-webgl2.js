// WebGL2 backend — MLS-MPM simulation in float textures.
//
// Pipeline per substep (all on GPU, particle/grid state in float textures):
//   1. P2G-1  scatter mass + momentum to the grid (additive point rendering)
//   2. density  per-particle density/pressure via weakly compressible EOS
//   3. P2G-2  scatter pressure + viscosity forces to the grid
//   4. grid   momentum -> velocity, gravity, boundary conditions
//   5. G2P    gather velocity + affine matrix, advect, spawn/recycle
//
// Scatter is emulated by rendering one GL point per (particle, cell) pair
// with additive blending; the 3D grid is tiled into a 2D texture.

import { makeShaders, gridLayout } from './shaders.js';

export async function createBackend({ canvas, fail }) {
  const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, depth: true });
  if (!gl) fail('WebGL2 is not available in this browser.');
  if (!gl.getExtension('EXT_color_buffer_float')) fail('Missing EXT_color_buffer_float (cannot render to float textures).');
  if (!gl.getExtension('EXT_float_blend')) fail('Missing EXT_float_blend (cannot blend into float textures).');
  if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < 5) fail('Need at least 5 draw buffers.');

  // -------------------------------------------------------------------------
  // GL helpers

  function compile(vsSrc, fsSrc, label) {
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        fail(`${label} ${type === gl.VERTEX_SHADER ? 'VS' : 'FS'}: ${gl.getShaderInfoLog(s)}`);
      }
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      fail(`${label} link: ${gl.getProgramInfoLog(p)}`);
    }
    return p;
  }

  const uniformCache = new Map();
  function u(prog, name) {
    let m = uniformCache.get(prog);
    if (!m) { m = new Map(); uniformCache.set(prog, m); }
    if (!m.has(name)) m.set(name, gl.getUniformLocation(prog, name));
    return m.get(name);
  }

  function createTex(w, h, data = null, internal = gl.RGBA32F, filter = gl.NEAREST) {
    const fmt = {
      [gl.RGBA32F]: [gl.RGBA, gl.FLOAT],
      [gl.R32F]: [gl.RED, gl.FLOAT],
      [gl.RGBA16F]: [gl.RGBA, gl.HALF_FLOAT],
      [gl.RGBA8]: [gl.RGBA, gl.UNSIGNED_BYTE],
    }[internal];
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, fmt[0], fmt[1], data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  function createDepthTex(w, h) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  function createFBO(textures, depthTex = null) {
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    const bufs = textures.map((t, i) => {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
      return gl.COLOR_ATTACHMENT0 + i;
    });
    gl.drawBuffers(bufs);
    if (depthTex) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depthTex, 0);
    }
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      fail('Framebuffer incomplete (float render targets unsupported?).');
    }
    return f;
  }

  function bindTex(unit, tex, prog, name) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(u(prog, name), unit);
  }

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  // -------------------------------------------------------------------------
  // Simulation state (programs + textures bake GRID/PTEX/LIFE in, so all of
  // it is torn down and rebuilt by init when the panel changes a parameter).

  let cfg = null;
  let GTEX = 0;
  let progP2G1, progP2G2, progDensity, progGrid, progG2P, progBG, progPoints,
    progPointDepth, progThick, progBlur, progComposite, progBlit, progGizmo,
    progVolBlur, progVolume, progVoxel, progVolUpscale,
    progPointDepthAniso, progThickAniso;
  let programs = [];
  let cur, nxt, gridA, gridB, gridAFBO, gridBFBO, densTex, densFBO,
    volDens, volDensFBO;
  let substepCount = 0;

  function makeParticleSet(posData) {
    const pos = createTex(cfg.PTEX, cfg.PTEX, posData);
    const vel = createTex(cfg.PTEX, cfg.PTEX);
    const c0 = createTex(cfg.PTEX, cfg.PTEX);
    const c1 = createTex(cfg.PTEX, cfg.PTEX);
    const c2 = createTex(cfg.PTEX, cfg.PTEX);
    const fbo = createFBO([pos, vel, c0, c1, c2]);
    return { pos, vel, c0, c1, c2, fbo };
  }

  function deleteParticleSet(set) {
    gl.deleteFramebuffer(set.fbo);
    for (const t of [set.pos, set.vel, set.c0, set.c1, set.c2]) gl.deleteTexture(t);
  }

  function teardownSim() {
    if (!programs.length) return;
    for (const p of programs) gl.deleteProgram(p);
    programs = [];
    uniformCache.clear();
    deleteParticleSet(cur);
    deleteParticleSet(nxt);
    for (const t of [gridA, gridB, densTex, volDens]) gl.deleteTexture(t);
    for (const f of [gridAFBO, gridBFBO, densFBO, volDensFBO]) gl.deleteFramebuffer(f);
  }

  function init(config) {
    teardownSim();
    cfg = config;
    GTEX = gridLayout(cfg.GRID).GTEX;

    const S = makeShaders({ GRID: cfg.GRID, PTEX: cfg.PTEX, LIFE: cfg.LIFE, ISO: cfg.ISO, K: cfg.K });
    progP2G1 = compile(S.vsP2G1, S.fsScatter, 'p2g1');
    progP2G2 = compile(S.vsP2G2, S.fsScatter, 'p2g2');
    progDensity = compile(S.vsQuad, S.fsDensity, 'density');
    progGrid = compile(S.vsQuad, S.fsGrid, 'grid');
    progG2P = compile(S.vsQuad, S.fsG2P, 'g2p');
    progBG = compile(S.vsQuad, S.fsBackground, 'background');
    progPoints = compile(S.vsPoint, S.fsPoint, 'points');
    progPointDepth = compile(S.vsPoint, S.fsPointDepth, 'pointDepth');
    progThick = compile(S.vsThick, S.fsThick, 'thickness');
    progBlur = compile(S.vsQuad, S.fsBlur, 'blur');
    progComposite = compile(S.vsQuad, S.fsComposite, 'composite');
    progBlit = compile(S.vsQuad, S.fsBlit, 'blit');
    progGizmo = compile(S.vsGizmo, S.fsGizmo, 'gizmo');
    progVolBlur = compile(S.vsQuad, S.fsVolBlur, 'volBlur');
    progVolume = compile(S.vsQuad, S.fsVolume, 'volume');
    progVoxel = compile(S.vsQuad, S.fsVoxel, 'voxel');
    progVolUpscale = compile(S.vsQuad, S.fsVolUpscale, 'volUpscale');
    progPointDepthAniso = compile(S.vsPointDepthAniso, S.fsPointDepthAniso, 'pointDepthAniso');
    progThickAniso = compile(S.vsThickAniso, S.fsThickAniso, 'thicknessAniso');
    programs = [progP2G1, progP2G2, progDensity, progGrid, progG2P, progBG,
      progPoints, progPointDepth, progThick, progBlur, progComposite, progBlit,
      progGizmo, progVolBlur, progVolume, progVoxel, progVolUpscale,
      progPointDepthAniso, progThickAniso];

    cur = makeParticleSet(cfg.initialData);
    nxt = makeParticleSet(cfg.initialData);

    gridA = createTex(GTEX, GTEX); // scatter target (momentum, mass)
    gridB = createTex(GTEX, GTEX); // updated velocities
    gridAFBO = createFBO([gridA]);
    gridBFBO = createFBO([gridB]);

    densTex = createTex(cfg.PTEX, cfg.PTEX);
    densFBO = createFBO([densTex]);

    // Blurred (mass, |momentum|) grid for the volume renderer, tiled like
    // the sim grid (half-float is plenty for mass ~0-30).
    volDens = createTex(GTEX, GTEX, null, gl.RGBA16F);
    volDensFBO = createFBO([volDens]);

    substepCount = 0;
  }

  // Screen-space fluid rendering targets (recreated on resize).
  let RT = null;

  function createTargets(w, h) {
    if (RT) {
      for (const t of RT.textures) gl.deleteTexture(t);
      for (const f of RT.fbos) gl.deleteFramebuffer(f);
    }
    const hw = Math.max(1, Math.ceil(w / 2));
    const hh = Math.max(1, Math.ceil(h / 2));
    const sceneColor = createTex(w, h, null, gl.RGBA8, gl.LINEAR);
    const depthTex = createDepthTex(w, h);
    const waterDepth = createTex(w, h, null, gl.R32F);
    const blurA = createTex(w, h, null, gl.R32F);
    const blurB = createTex(w, h, null, gl.R32F);
    const thick = createTex(hw, hh, null, gl.RGBA16F, gl.LINEAR);
    const volW = Math.max(1, Math.round(w * cfg.RSCALE));
    const volH = Math.max(1, Math.round(h * cfg.RSCALE));
    const volColor = createTex(volW, volH, null, gl.RGBA8, gl.LINEAR);
    RT = {
      w, h, hw, hh, volW, volH,
      sceneColor, waterDepth, blurA, blurB, thick, volColor,
      sceneFBO: createFBO([sceneColor], depthTex),
      waterFBO: createFBO([waterDepth], depthTex), // shares the scene depth
      blurAFBO: createFBO([blurA]),
      blurBFBO: createFBO([blurB]),
      thickFBO: createFBO([thick]),
      volFBO: createFBO([volColor]),
      textures: [sceneColor, depthTex, waterDepth, blurA, blurB, thick, volColor],
      fbos: [],
    };
    RT.fbos = [RT.sceneFBO, RT.waterFBO, RT.blurAFBO, RT.blurBFBO, RT.thickFBO, RT.volFBO];
  }

  // -------------------------------------------------------------------------
  // Simulation substep

  function substep() {
    gl.disable(gl.DEPTH_TEST);

    // 1. clear grid + P2G-1
    gl.bindFramebuffer(gl.FRAMEBUFFER, gridAFBO);
    gl.viewport(0, 0, GTEX, GTEX);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(progP2G1);
    bindTex(0, cur.pos, progP2G1, 'uPos');
    bindTex(1, cur.vel, progP2G1, 'uVel');
    bindTex(2, cur.c0, progP2G1, 'uC0');
    bindTex(3, cur.c1, progP2G1, 'uC1');
    bindTex(4, cur.c2, progP2G1, 'uC2');
    gl.drawArrays(gl.POINTS, 0, cfg.N * 27);
    gl.disable(gl.BLEND);

    // 2. density / pressure
    gl.bindFramebuffer(gl.FRAMEBUFFER, densFBO);
    gl.viewport(0, 0, cfg.PTEX, cfg.PTEX);
    gl.useProgram(progDensity);
    bindTex(0, cur.pos, progDensity, 'uPos');
    bindTex(1, gridA, progDensity, 'uGrid');
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 3. P2G-2 (forces, additive into the same grid)
    gl.bindFramebuffer(gl.FRAMEBUFFER, gridAFBO);
    gl.viewport(0, 0, GTEX, GTEX);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(progP2G2);
    bindTex(0, cur.pos, progP2G2, 'uPos');
    bindTex(1, cur.c0, progP2G2, 'uC0');
    bindTex(2, cur.c1, progP2G2, 'uC1');
    bindTex(3, cur.c2, progP2G2, 'uC2');
    bindTex(4, densTex, progP2G2, 'uAux');
    gl.drawArrays(gl.POINTS, 0, cfg.N * 27);
    gl.disable(gl.BLEND);

    // 4. grid update
    gl.bindFramebuffer(gl.FRAMEBUFFER, gridBFBO);
    gl.viewport(0, 0, GTEX, GTEX);
    gl.useProgram(progGrid);
    bindTex(0, gridA, progGrid, 'uGrid');
    gl.uniform4fv(u(progGrid, 'uRocks'), cfg.rockData);
    gl.uniform3fv(u(progGrid, 'uRockVel'), cfg.rockVel);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // 5. G2P + advect
    gl.bindFramebuffer(gl.FRAMEBUFFER, nxt.fbo);
    gl.viewport(0, 0, cfg.PTEX, cfg.PTEX);
    gl.useProgram(progG2P);
    bindTex(0, cur.pos, progG2P, 'uPos');
    bindTex(1, gridB, progG2P, 'uGrid');
    gl.uniform4fv(u(progG2P, 'uRocks'), cfg.rockData);
    gl.uniform1f(u(progG2P, 'uFrame'), substepCount);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    [cur, nxt] = [nxt, cur];
    substepCount++;
  }

  // -------------------------------------------------------------------------
  // Rendering

  // Wireframe drag gizmo overlay: ghost circle at the grab origin, line to
  // the current center, circle there. Drawn last, no depth test.
  function drawGizmo(frame) {
    if (!frame.gizmo) return;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(progGizmo);
    gl.uniformMatrix4fv(u(progGizmo, 'uPV'), false, frame.pv);
    gl.uniform3fv(u(progGizmo, 'uA'), frame.gizmo.a);
    gl.uniform3fv(u(progGizmo, 'uB'), frame.gizmo.b);
    gl.uniform3fv(u(progGizmo, 'uCamR'), frame.right);
    gl.uniform3fv(u(progGizmo, 'uCamU'), frame.up);
    gl.uniform1f(u(progGizmo, 'uR'), frame.gizmo.r);
    gl.drawArrays(gl.LINES, 0, 130);
    gl.disable(gl.BLEND);
  }

  function render(frame) {
    const { w, h, proj, view, pv, aspect, lightV } = frame;
    if (!RT || RT.w !== w || RT.h !== h) createTargets(w, h);

    if (frame.mode === 'volume' || frame.mode === 'voxel') {
      // Volumetric raymarch / voxel DDA: tent-blur the grid density, trace
      // it into a scaled offscreen target, then upscale to the canvas.
      gl.disable(gl.DEPTH_TEST);
      gl.bindFramebuffer(gl.FRAMEBUFFER, volDensFBO);
      gl.viewport(0, 0, GTEX, GTEX);
      gl.useProgram(progVolBlur);
      bindTex(0, gridB, progVolBlur, 'uGrid');
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      const prog = frame.mode === 'voxel' ? progVoxel : progVolume;
      gl.bindFramebuffer(gl.FRAMEBUFFER, RT.volFBO);
      gl.viewport(0, 0, RT.volW, RT.volH);
      gl.useProgram(prog);
      bindTex(0, volDens, prog, 'uDens');
      gl.uniform3fv(u(prog, 'uCamPos'), frame.eye);
      gl.uniform3fv(u(prog, 'uCamR'), frame.right);
      gl.uniform3fv(u(prog, 'uCamU'), frame.up);
      gl.uniform3fv(u(prog, 'uCamF'), frame.fwd);
      gl.uniform2f(u(prog, 'uRes'), RT.volW, RT.volH);
      gl.uniform1f(u(prog, 'uTanF'), frame.tanF);
      gl.uniform1f(u(prog, 'uAspect'), aspect);
      gl.uniform3fv(u(prog, 'uLightW'), frame.lightW);
      gl.uniform4fv(u(prog, 'uRocks'), cfg.rockData);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, w, h);
      gl.useProgram(progVolUpscale);
      bindTex(0, RT.volColor, progVolUpscale, 'uScene');
      gl.uniform2f(u(progVolUpscale, 'uRes'), w, h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      drawGizmo(frame);
      return;
    }

    // 1. scene (cube walls + rocks) into offscreen color + depth
    gl.bindFramebuffer(gl.FRAMEBUFFER, RT.sceneFBO);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.01, 0.015, 0.02, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.useProgram(progBG);
    gl.uniform3fv(u(progBG, 'uCamPos'), frame.eye);
    gl.uniform3fv(u(progBG, 'uCamR'), frame.right);
    gl.uniform3fv(u(progBG, 'uCamU'), frame.up);
    gl.uniform3fv(u(progBG, 'uCamF'), frame.fwd);
    gl.uniform2f(u(progBG, 'uRes'), w, h);
    gl.uniform1f(u(progBG, 'uTanF'), frame.tanF);
    gl.uniform1f(u(progBG, 'uAspect'), aspect);
    gl.uniformMatrix4fv(u(progBG, 'uPV'), false, pv);
    gl.uniform3fv(u(progBG, 'uLightW'), frame.lightW);
    gl.uniform4fv(u(progBG, 'uRocks'), cfg.rockData);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    if (frame.mode === 'points') {
      // Legacy view: shaded impostors straight into the scene, then blit.
      gl.useProgram(progPoints);
      bindTex(0, cur.pos, progPoints, 'uPos');
      bindTex(1, cur.vel, progPoints, 'uVel');
      gl.uniformMatrix4fv(u(progPoints, 'uProj'), false, proj);
      gl.uniformMatrix4fv(u(progPoints, 'uView'), false, view);
      gl.uniform1f(u(progPoints, 'uPointScale'), h * proj[5]);
      gl.uniform3fv(u(progPoints, 'uLightV'), lightV);
      gl.drawArrays(gl.POINTS, 0, cfg.N);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, w, h);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(progBlit);
      bindTex(0, RT.sceneColor, progBlit, 'uScene');
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      drawGizmo(frame);
      return;
    }

    // Screen-space fluid path ('ssf' and 'aniso' — aniso swaps the depth and
    // thickness splat programs for ellipsoid variants, everything else is
    // identical).
    const aniso = frame.mode === 'aniso';

    // 2. water surface depth (z-tested against the shared scene depth)
    const pd = aniso ? progPointDepthAniso : progPointDepth;
    gl.bindFramebuffer(gl.FRAMEBUFFER, RT.waterFBO);
    gl.clearColor(0, 0, 0, 0); // 0 = no water
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(pd);
    bindTex(0, cur.pos, pd, 'uPos');
    bindTex(1, cur.vel, pd, 'uVel');
    gl.uniformMatrix4fv(u(pd, 'uProj'), false, proj);
    gl.uniformMatrix4fv(u(pd, 'uView'), false, view);
    gl.uniform1f(u(pd, 'uPointScale'), h * proj[5]);
    gl.drawArrays(gl.POINTS, 0, cfg.N);
    gl.disable(gl.DEPTH_TEST);

    // 3. depth-aware separable blur, two iterations
    const scalePx = h * proj[5];
    let src = RT.waterDepth;
    for (const [fbo, tex, dx, dy] of [
      [RT.blurAFBO, RT.blurA, 1, 0], [RT.blurBFBO, RT.blurB, 0, 1],
      [RT.blurAFBO, RT.blurA, 1, 0], [RT.blurBFBO, RT.blurB, 0, 1],
    ]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.useProgram(progBlur);
      bindTex(0, src, progBlur, 'uDepth');
      gl.uniform2f(u(progBlur, 'uDir'), dx, dy);
      gl.uniform1f(u(progBlur, 'uScalePx'), scalePx);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      src = tex;
    }

    // 4. thickness + foam, half resolution, additive
    gl.bindFramebuffer(gl.FRAMEBUFFER, RT.thickFBO);
    gl.viewport(0, 0, RT.hw, RT.hh);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const tk = aniso ? progThickAniso : progThick;
    gl.useProgram(tk);
    bindTex(0, cur.pos, tk, 'uPos');
    bindTex(1, cur.vel, tk, 'uVel');
    // Occlusion vs the RAW water depth: at true stream pixels this culls the
    // pool behind, while pixels between sparse droplets keep their pool
    // contribution (the blurred depth's near-clamped halo would fake holes).
    bindTex(2, RT.waterDepth, tk, 'uFront');
    gl.uniformMatrix4fv(u(tk, 'uProj'), false, proj);
    gl.uniformMatrix4fv(u(tk, 'uView'), false, view);
    gl.uniform1f(u(tk, 'uPointScale'), RT.hh * proj[5]);
    gl.drawArrays(gl.POINTS, 0, cfg.N);
    gl.disable(gl.BLEND);

    // 5. composite to the canvas
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.useProgram(progComposite);
    bindTex(0, RT.sceneColor, progComposite, 'uScene');
    bindTex(1, src, progComposite, 'uDepthS');
    bindTex(2, RT.thick, progComposite, 'uThick');
    gl.uniform2f(u(progComposite, 'uRes'), w, h);
    gl.uniform1f(u(progComposite, 'uTanF'), frame.tanF);
    gl.uniform1f(u(progComposite, 'uAspect'), aspect);
    gl.uniform3fv(u(progComposite, 'uLightV'), lightV);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    drawGizmo(frame);
  }

  function readParticles() {
    const P = cfg.PTEX;
    const buf = new Float32Array(P * P * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, cur.fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, P, P, gl.RGBA, gl.FLOAT, buf);
    return buf;
  }

  function dispose() {
    teardownSim();
    if (RT) {
      for (const t of RT.textures) gl.deleteTexture(t);
      for (const f of RT.fbos) gl.deleteFramebuffer(f);
      RT = null;
    }
    gl.deleteVertexArray(vao);
  }

  return { name: 'webgl2', init, substep, render, readParticles, dispose };
}
