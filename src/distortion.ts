import { createProgram, getUniformLocations } from "./gl";
import vertex from "./shaders/fullscreen.vert?raw";
import fragment from "./shaders/distortion.frag?raw";

export class Distortion {
  enabled = false;
  mode = 0;
  amount = 0.65;
  scale = 4;
  color = 0.35;
  persistence = 0.65;
  sensitivity = 1.5;
  private program: WebGLProgram;
  private copyProgram: WebGLProgram;
  private copyUniforms;
  private uniforms;
  private targets: { texture: WebGLTexture; fbo: WebGLFramebuffer }[] = [];
  private width = 0;
  private height = 0;
  private index = 0;
  private lastTime = -1;
  private spectrum: WebGLTexture;

  constructor(private gl: WebGL2RenderingContext, private vao: WebGLVertexArrayObject) {
    this.program = createProgram(gl, vertex, fragment);
    this.copyProgram = createProgram(gl, vertex, `#version 300 es
      precision highp float;
      uniform sampler2D uImage;
      uniform vec2 uResolution;
      out vec4 color;
      void main() { color = texture(uImage, gl_FragCoord.xy / uResolution); }
    `);
    this.copyUniforms = getUniformLocations(gl, this.copyProgram, ["uImage", "uResolution"]);
    this.uniforms = getUniformLocations(gl, this.program, ["uScene", "uHistory", "uSpectrum", "uResolution", "uTime", "uAmount", "uScale", "uColor", "uPersistence", "uSensitivity", "uMode", "uHasAudio", "uDelta"]);
    this.spectrum = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.spectrum);
    this.configureTexture();
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 64, 1, 0, gl.RED, gl.FLOAT, new Float32Array(64));
  }

  private configureTexture() {
    const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  reset() { this.lastTime = -1; }

  begin(width: number, height: number): WebGLFramebuffer | null {
    if (!this.enabled || this.amount === 0) { this.reset(); return null; }
    const gl = this.gl;
    if (this.width !== width || this.height !== height) {
      for (const target of this.targets) { gl.deleteTexture(target.texture); gl.deleteFramebuffer(target.fbo); }
      this.targets = [];
      this.width = width; this.height = height;
      for (let i = 0; i < 3; i++) {
        const texture = gl.createTexture()!, fbo = gl.createFramebuffer()!;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        this.configureTexture();
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("Unable to allocate distortion render target.");
        this.targets.push({ texture, fbo });
      }
      this.reset();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.targets[0].fbo);
    return this.targets[0].fbo;
  }

  finish(time: number, bands: Float32Array, hasAudio: boolean) {
    const gl = this.gl, u = this.uniforms;
    const delta = this.lastTime < 0 ? 1 / 60 : Math.max(0, time - this.lastTime);
    if (this.lastTime < 0 || delta > 0.25 || time < this.lastTime) {
      for (const target of this.targets.slice(1)) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }
    this.lastTime = time;
    const output = this.targets[1 + this.index], history = this.targets[2 - this.index];
    gl.bindFramebuffer(gl.FRAMEBUFFER, output.fbo);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.program); gl.bindVertexArray(this.vao);
    [this.targets[0].texture, history.texture, this.spectrum].forEach((texture, i) => {
      gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, texture);
    });
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 64, 1, gl.RED, gl.FLOAT, bands);
    gl.uniform1i(u.uScene, 0); gl.uniform1i(u.uHistory, 1); gl.uniform1i(u.uSpectrum, 2);
    gl.uniform2f(u.uResolution, this.width, this.height);
    gl.uniform1f(u.uTime, time); gl.uniform1f(u.uAmount, this.amount);
    gl.uniform1f(u.uScale, this.scale); gl.uniform1f(u.uColor, this.color);
    gl.uniform1f(u.uPersistence, this.persistence); gl.uniform1f(u.uSensitivity, this.sensitivity);
    gl.uniform1f(u.uDelta, Math.min(delta, 0.1));
    gl.uniform1i(u.uMode, this.mode); gl.uniform1i(u.uHasAudio, hasAudio ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // Draw into the antialiased default framebuffer; a single-sample blit into
    // a multisampled canvas is not supported by WebGL2.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(this.copyProgram);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, output.texture);
    gl.uniform1i(this.copyUniforms.uImage, 0);
    gl.uniform2f(this.copyUniforms.uResolution, this.width, this.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (delta > 0) this.index = 1 - this.index;
  }
}
