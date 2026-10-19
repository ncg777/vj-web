var Pn=Object.defineProperty;var Dn=(e,t,n)=>t in e?Pn(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n;var R=(e,t,n)=>Dn(e,typeof t!="symbol"?t+"":t,n);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))r(a);new MutationObserver(a=>{for(const o of a)if(o.type==="childList")for(const l of o.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&r(l)}).observe(document,{childList:!0,subtree:!0});function n(a){const o={};return a.integrity&&(o.integrity=a.integrity),a.referrerPolicy&&(o.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?o.credentials="include":a.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function r(a){if(a.ep)return;a.ep=!0;const o=n(a);fetch(a.href,o)}})();class En{constructor(){R(this,"player",new Audio);R(this,"buffer");R(this,"url");R(this,"frames",new Float32Array);R(this,"generation",0);R(this,"context");R(this,"destination");R(this,"gain");R(this,"bands",new Float32Array(64));this.player.loop=!0}async load(t){const n=++this.generation,r=this.context??(this.context=new AudioContext),a=await r.decodeAudioData(await t.arrayBuffer());if(a.duration>3600)throw new Error("Choose an audio file no longer than one hour.");const o=await Mn(a);if(n!==this.generation)return!1;if(this.player.pause(),this.url&&URL.revokeObjectURL(this.url),this.url=URL.createObjectURL(t),this.player.src=this.url,this.buffer=a,this.frames=o,!this.destination){const l=r.createMediaElementSource(this.player);this.destination=r.createMediaStreamDestination(),this.gain=r.createGain(),l.connect(this.gain),this.gain.connect(r.destination),l.connect(this.destination)}return!0}clear(){this.generation++,this.player.pause(),this.player.removeAttribute("src"),this.player.load(),this.url&&URL.revokeObjectURL(this.url),this.url=void 0,this.buffer=void 0,this.frames=new Float32Array}async play(){var t;await((t=this.context)==null?void 0:t.resume()),await this.player.play()}set volume(t){this.gain&&(this.gain.gain.value=t)}get stream(){var t;return(t=this.destination)==null?void 0:t.stream}sample(t){if(this.bands.fill(0),!this.buffer)return this.bands;const n=this.buffer.duration;if(t>=n&&!this.player.loop)return this.bands;const r=(t%n+n)%n*60,a=this.frames.length/64,o=Math.min(a-1,Math.floor(r)),l=Math.min(a-1,o+1),s=r-Math.floor(r);for(let c=0;c<64;c++)this.bands[c]=this.frames[o*64+c]*(1-s)+this.frames[l*64+c]*s;return this.bands}}async function Mn(e){const n=Math.ceil(e.duration*60),r=new Float32Array(n*64),a=Array.from({length:e.numberOfChannels},(s,c)=>e.getChannelData(c)),o=new Float32Array(2048),l=new Float32Array(2048);for(let s=0;s<n;s++){const c=Math.round(s*e.sampleRate/60)-1024;l.fill(0);for(let f=0;f<2048;f++){let m=0;for(const d of a)m+=d[c+f]??0;o[f]=m/a.length*(.5-.5*Math.cos(2*Math.PI*f/2047))}for(let f=1,m=0;f<2048;f++){let d=1024;for(;m&d;d>>=1)m^=d;m^=d,f<m&&([o[f],o[m]]=[o[m],o[f]])}for(let f=2;f<=2048;f<<=1){const m=-2*Math.PI/f,d=Math.cos(m),g=Math.sin(m);for(let h=0;h<2048;h+=f){let x=1,k=0;for(let M=0;M<f/2;M++){const H=h+M,W=H+f/2,T=x*o[W]-k*l[W],L=x*l[W]+k*o[W];o[W]=o[H]-T,l[W]=l[H]-L,o[H]+=T,l[H]+=L;const he=x*d-k*g;k=x*g+k*d,x=he}}}for(let f=0;f<64;f++){const m=Math.max(1,Math.floor(30*Math.pow(600,f/64)*2048/e.sampleRate)),d=Math.min(2048/2,Math.max(m+1,Math.ceil(30*Math.pow(600,(f+1)/64)*2048/e.sampleRate)));let g=0;for(let x=m;x<d;x++)g=Math.max(g,Math.hypot(o[x],l[x])*4/2048);const h=Math.min(1,Math.log1p(g*30)/Math.log(31));r[s*64+f]=Math.max(h,s?r[(s-1)*64+f]*.88:0)}s%120===0&&await new Promise(f=>setTimeout(f,0))}return r}function Xt(e,t,n){const r=e.createShader(t);if(!r)throw new Error("Failed to create shader");if(e.shaderSource(r,n),e.compileShader(r),!e.getShaderParameter(r,e.COMPILE_STATUS)){const a=e.getShaderInfoLog(r)||"Unknown shader error";throw e.deleteShader(r),new Error(a)}return r}function ft(e,t,n){const r=Xt(e,e.VERTEX_SHADER,t),a=Xt(e,e.FRAGMENT_SHADER,n),o=e.createProgram();if(!o)throw new Error("Failed to create program");if(e.attachShader(o,r),e.attachShader(o,a),e.linkProgram(o),e.deleteShader(r),e.deleteShader(a),!e.getProgramParameter(o,e.LINK_STATUS)){const l=e.getProgramInfoLog(o)||"Unknown program error";throw e.deleteProgram(o),new Error(l)}return o}function mt(e,t,n){const r={};for(const a of n)r[a]=e.getUniformLocation(t,a);return r}function zn(e,t=2){const n=Math.min(window.devicePixelRatio||1,t),r=Math.max(1,Math.floor(e.clientWidth*n)),a=Math.max(1,Math.floor(e.clientHeight*n));return(e.width!==r||e.height!==a)&&(e.width=r,e.height=a),{width:r,height:a,dpr:n}}function Gn(e){const t=e.createVertexArray();if(!t)throw new Error("Failed to create VAO");return e.bindVertexArray(t),t}const pt=`#version 300 es
precision highp float;

const vec2 verts[3] = vec2[](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
);

void main() {
  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
`,In=`#version 300 es
precision highp float;
uniform sampler2D uScene, uHistory, uSpectrum;
uniform vec2 uResolution;
uniform float uTime, uAmount, uScale, uColor, uPersistence, uSensitivity, uDelta;
uniform int uMode, uHasAudio;
out vec4 fragColor;
float band(float x) { return min(1.5, texture(uSpectrum, vec2(x, 0.5)).r * uSensitivity); }
vec2 mirrorUV(vec2 uv) { return 1.0 - abs(mod(uv, 2.0) - 1.0); }
void main() {
  vec2 uv = gl_FragCoord.xy / uResolution;
  vec2 aspect = vec2(uResolution.x / uResolution.y, 1.0);
  vec2 p = (uv - 0.5) * aspect;
  float radius = length(p), angle = atan(p.y, p.x);
  float bass = (band(0.08) + band(0.18)) * 0.5;
  float energy = (bass + band(0.45) + band(0.75)) / 3.0;
  float drive = uHasAudio == 1 ? energy : 0.45;
  float spectral = uHasAudio == 1 ? band(fract(radius * 0.6 + angle / 6.28318)) : 0.3;
  float t = uTime;
  vec2 flow = vec2(sin(p.y * uScale + t * 0.7 + spectral * 5.0), cos(p.x * uScale - t * 0.6 + bass * 4.0));
  float ripple = sin(radius * uScale * 8.0 - t * 3.0 + spectral * 8.0);
  vec2 warped = uv + (flow * 0.055 * drive + normalize(p + 0.0001) * ripple * 0.025 * spectral) * uAmount / aspect;
  if (uMode == 1) {
    float sectors = max(3.0, floor(uScale * 2.0));
    float wedge = 6.28318 / sectors;
    float a = abs(mod(angle + t * 0.08 + bass * 0.4, wedge) - wedge * 0.5);
    vec2 folded = vec2(cos(a), sin(a)) * radius * (1.0 + drive * 0.15);
    warped = mix(warped, folded / aspect + 0.5, uAmount);
  }
  vec2 split = (flow + p) * uColor * uAmount * (0.002 + drive * 0.018) / aspect;
  vec3 color = vec3(texture(uScene, mirrorUV(warped + split)).r, texture(uScene, mirrorUV(warped)).g, texture(uScene, mirrorUV(warped - split)).b);
  if (uMode == 2) {
    float rotation = uDelta * (0.12 + bass * 0.3);
    mat2 rot = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
    vec2 historyUV = rot * p * (1.0 - uDelta * (0.12 + drive * 0.2)) / aspect + 0.5;
    vec4 history = texture(uHistory, mirrorUV(historyUV));
    float retention = pow(uPersistence * 0.97, max(0.001, uDelta) * 60.0);
    color = mix(color, history.rgb, retention * uAmount * history.a);
  }
  fragColor = vec4(color, 1.0);
}
`;class Un{constructor(t,n){R(this,"enabled",!1);R(this,"mode",0);R(this,"amount",.65);R(this,"scale",4);R(this,"color",.35);R(this,"persistence",.65);R(this,"sensitivity",1.5);R(this,"program");R(this,"copyProgram");R(this,"copyUniforms");R(this,"uniforms");R(this,"targets",[]);R(this,"width",0);R(this,"height",0);R(this,"index",0);R(this,"lastTime",-1);R(this,"spectrum");this.gl=t,this.vao=n,this.program=ft(t,pt,In),this.copyProgram=ft(t,pt,`#version 300 es
      precision highp float;
      uniform sampler2D uImage;
      uniform vec2 uResolution;
      out vec4 color;
      void main() { color = texture(uImage, gl_FragCoord.xy / uResolution); }
    `),this.copyUniforms=mt(t,this.copyProgram,["uImage","uResolution"]),this.uniforms=mt(t,this.program,["uScene","uHistory","uSpectrum","uResolution","uTime","uAmount","uScale","uColor","uPersistence","uSensitivity","uMode","uHasAudio","uDelta"]),this.spectrum=t.createTexture(),t.bindTexture(t.TEXTURE_2D,this.spectrum),this.configureTexture(),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MIN_FILTER,t.NEAREST),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MAG_FILTER,t.NEAREST),t.texImage2D(t.TEXTURE_2D,0,t.R32F,64,1,0,t.RED,t.FLOAT,new Float32Array(64))}configureTexture(){const t=this.gl;t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MIN_FILTER,t.LINEAR),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_MAG_FILTER,t.LINEAR),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_S,t.CLAMP_TO_EDGE),t.texParameteri(t.TEXTURE_2D,t.TEXTURE_WRAP_T,t.CLAMP_TO_EDGE)}reset(){this.lastTime=-1}begin(t,n){if(!this.enabled||this.amount===0)return this.reset(),null;const r=this.gl;if(this.width!==t||this.height!==n){for(const a of this.targets)r.deleteTexture(a.texture),r.deleteFramebuffer(a.fbo);this.targets=[],this.width=t,this.height=n;for(let a=0;a<3;a++){const o=r.createTexture(),l=r.createFramebuffer();if(r.bindTexture(r.TEXTURE_2D,o),this.configureTexture(),r.texImage2D(r.TEXTURE_2D,0,r.RGBA8,t,n,0,r.RGBA,r.UNSIGNED_BYTE,null),r.bindFramebuffer(r.FRAMEBUFFER,l),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,o,0),r.checkFramebufferStatus(r.FRAMEBUFFER)!==r.FRAMEBUFFER_COMPLETE)throw new Error("Unable to allocate distortion render target.");this.targets.push({texture:o,fbo:l})}this.reset()}return r.bindFramebuffer(r.FRAMEBUFFER,this.targets[0].fbo),this.targets[0].fbo}finish(t,n,r){const a=this.gl,o=this.uniforms,l=this.lastTime<0?1/60:Math.max(0,t-this.lastTime);if(this.lastTime<0||l>.25||t<this.lastTime)for(const f of this.targets.slice(1))a.bindFramebuffer(a.FRAMEBUFFER,f.fbo),a.clearColor(0,0,0,0),a.clear(a.COLOR_BUFFER_BIT);this.lastTime=t;const s=this.targets[1+this.index],c=this.targets[2-this.index];a.bindFramebuffer(a.FRAMEBUFFER,s.fbo),a.viewport(0,0,this.width,this.height),a.useProgram(this.program),a.bindVertexArray(this.vao),[this.targets[0].texture,c.texture,this.spectrum].forEach((f,m)=>{a.activeTexture(a.TEXTURE0+m),a.bindTexture(a.TEXTURE_2D,f)}),a.texSubImage2D(a.TEXTURE_2D,0,0,0,64,1,a.RED,a.FLOAT,n),a.uniform1i(o.uScene,0),a.uniform1i(o.uHistory,1),a.uniform1i(o.uSpectrum,2),a.uniform2f(o.uResolution,this.width,this.height),a.uniform1f(o.uTime,t),a.uniform1f(o.uAmount,this.amount),a.uniform1f(o.uScale,this.scale),a.uniform1f(o.uColor,this.color),a.uniform1f(o.uPersistence,this.persistence),a.uniform1f(o.uSensitivity,this.sensitivity),a.uniform1f(o.uDelta,Math.min(l,.1)),a.uniform1i(o.uMode,this.mode),a.uniform1i(o.uHasAudio,r?1:0),a.drawArrays(a.TRIANGLES,0,3),a.bindFramebuffer(a.FRAMEBUFFER,null),a.useProgram(this.copyProgram),a.activeTexture(a.TEXTURE0),a.bindTexture(a.TEXTURE_2D,s.texture),a.uniform1i(this.copyUniforms.uImage,0),a.uniform2f(this.copyUniforms.uResolution,this.width,this.height),a.drawArrays(a.TRIANGLES,0,3),l>0&&(this.index=1-this.index)}}function Wn(e,t,n){const r=document.createElement("div");r.className="panel-section audio-panel",r.innerHTML=`
    <div class="section-title">Psychedelic distortion</div>
    <label class="audio-check"><input id="distortion-enabled" type="checkbox"> Enable distortion</label>
    <div id="distortion-options" hidden>
    <label>Style <select id="distortion-mode" class="scene-select">
      <option value="0">Liquid warp</option><option value="1">Spectral kaleidoscope</option><option value="2">Psychedelic feedback</option>
    </select></label>
    <div id="distortion-sliders"></div>
    </div>
    <details id="soundtrack-controls"><summary class="section-title">Soundtrack · optional</summary>
    <input id="audio-file" type="file" accept="audio/*" aria-label="Load soundtrack">
    <div id="audio-status" role="status">Works without audio. Files stay on this device.</div>
    <div class="panel-actions"><button id="audio-play" class="ghost small" disabled>Play</button><button id="audio-clear" class="ghost small" disabled>Remove</button></div>
    <input id="audio-seek" type="range" min="0" max="1" step="0.01" value="0" aria-label="Soundtrack position" disabled>
    <output id="audio-time">0:00 / 0:00</output>
    <label class="audio-check"><input id="audio-loop" type="checkbox" checked> Loop soundtrack</label>
    <label>Preview volume <input id="audio-volume" type="range" min="0" max="1" step="0.01" value="1"></label>
    <label class="audio-check"><input id="audio-include" type="checkbox" checked> Include soundtrack in video</label>
    <button id="audio-duration" class="ghost small" disabled>Use track length for export</button>
    <p class="audio-note">Generate exports from the start of the track at its original volume. Short tracks loop when Loop is enabled; otherwise the remainder is silent. Seeking restarts fluid and feedback history.</p>
    </details>
  `,document.querySelector(".panel").insertBefore(r,document.querySelector(".panel").children[1]);const a=h=>r.querySelector(`#${h}`),o=h=>r.querySelector(`#${h}`),l=r.querySelector("#audio-status"),s=o("audio-play"),c=o("audio-clear"),f=o("audio-duration"),m=a("audio-seek");let d=!1;for(const[h,x,k,M,H]of[["amount","Intensity / mix",0,1,.01],["scale","Pattern scale",1,12,.1],["color","Color separation",0,1,.01],["persistence","Trail persistence",0,.98,.01],["sensitivity","Audio sensitivity",0,4,.05]]){const W=document.createElement("label");W.textContent=x;const T=document.createElement("input");T.id=`distortion-${h}`,T.type="range",T.min=String(k),T.max=String(M),T.step=String(H),T.value=String(t[h]);const L=document.createElement("output");L.textContent=T.value,T.oninput=()=>{t[h]=Number(T.value),L.textContent=T.value},W.append(T,L),r.querySelector("#distortion-sliders").append(W)}a("distortion-enabled").onchange=()=>{t.enabled=a("distortion-enabled").checked,r.querySelector("#distortion-options").hidden=!t.enabled,t.reset()},r.querySelector("#distortion-mode").onchange=h=>{t.mode=Number(h.target.value),t.reset()},a("audio-file").onchange=async()=>{var x;const h=(x=a("audio-file").files)==null?void 0:x[0];if(h){d=!0,s.disabled=c.disabled=f.disabled=m.disabled=!0,a("audio-file").disabled=!0,l.textContent="Analyzing soundtrack…";try{await e.load(h)&&(m.max=String(e.buffer.duration),e.volume=Number(a("audio-volume").value),l.textContent=h.name,n())}catch(k){l.textContent=`Could not load audio: ${k instanceof Error?k.message:k}`}finally{d=!1,a("audio-file").disabled=!1,s.disabled=c.disabled=f.disabled=m.disabled=!e.buffer}}},s.onclick=async()=>{try{e.player.paused?await e.play():e.player.pause()}catch(h){l.textContent=`Playback failed: ${h instanceof Error?h.message:h}`}},c.onclick=()=>{e.clear(),a("audio-file").value="",s.disabled=c.disabled=f.disabled=m.disabled=!0,m.value="0",l.textContent="Works without audio. Files stay on this device.",n()},m.oninput=()=>{e.player.currentTime=Number(m.value),n()},a("audio-loop").onchange=()=>{e.player.loop=a("audio-loop").checked},a("audio-volume").oninput=()=>{e.volume=Number(a("audio-volume").value)},f.onclick=()=>{e.buffer&&(document.querySelector("#offline-duration").value=String(Math.min(3600,Math.max(1,e.buffer.duration))))};const g=h=>`${Math.floor(h/60)}:${String(Math.floor(h%60)).padStart(2,"0")}`;return{get loading(){return d},get includeAudio(){return a("audio-include").checked},update(){var h;s.textContent=e.player.paused?"Play":"Pause",e.buffer&&(m.value=String(e.player.currentTime)),r.querySelector("#audio-time").textContent=`${g(e.player.currentTime)} / ${g(((h=e.buffer)==null?void 0:h.duration)??0)}`}}}const Nn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;
uniform vec2 uResolution;
uniform vec2 uGridSize;
uniform sampler2D uState;
uniform int uPass;
uniform int uStyle;
uniform int uSymmetry;
uniform int uDetail;
uniform float uTime;
uniform float uSpeed;
uniform float uTurbulence;
uniform float uScale;
uniform float uTwist;
uniform float uPersistence;
uniform float uInjection;
uniform float uHue;
uniform float uGlow;
uniform float uSeed;

const float TAU = 6.28318530718;
mat2 rotate(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

vec3 palette(float h) {
  return 0.52 + 0.48 * cos(TAU * (h + vec3(0.0, 0.33, 0.67)));
}

// A periodic multiscale stream function. Its perpendicular gradient gives
// divergence-free stirring without a separate pressure texture/solver.
float stream(vec2 p, float t) {
  float field = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  float seed = mod(uSeed, 997.0) * 0.031;
  for (int i = 0; i < 5; ++i) {
    float f = float(i);
    vec2 q = TAU * p * frequency;
    field += amplitude * sin(q.x + sin(t * 0.31 + f + seed))
      * cos(q.y + cos(t * 0.27 - f + seed));
    frequency *= 2.0;
    amplitude *= 0.32;
  }
  if (uStyle == 1) field += 0.7 * sin(TAU * p.y + t * 0.4);
  if (uStyle == 2) field += 0.45 * sin(TAU * p.x * 3.0) * sin(TAU * p.y * 3.0);
  return field;
}

vec2 velocity(vec2 p, float t) {
  float e = 0.002;
  vec2 gradient = vec2(stream(p + vec2(e, 0), t) - stream(p - vec2(e, 0), t),
    stream(p + vec2(0, e), t) - stream(p - vec2(0, e), t)) / (2.0 * e);
  vec2 flow = vec2(gradient.y, -gradient.x) * (0.0015 * uTurbulence);
  if (uStyle == 0 || uStyle == 3) {
    vec2 d = p - 0.5;
    flow += vec2(-d.y, d.x) * (uStyle == 3 ? 0.08 : 0.035)
      * exp(-dot(d, d) * 5.0);
  }
  return flow;
}

// Float targets use nearest filtering; interpolate explicitly for smooth
// semi-Lagrangian dye transport and full-resolution display.
vec4 sampleDye(vec2 uv) {
  vec2 p = uv * uGridSize - 0.5;
  vec2 base = floor(p);
  vec2 f = fract(p);
  vec2 texel = 1.0 / uGridSize;
  vec2 a = (base + 0.5) * texel;
  return mix(mix(texture(uState, a), texture(uState, a + vec2(texel.x, 0)), f.x),
    mix(texture(uState, a + vec2(0, texel.y)), texture(uState, a + texel), f.x), f.y);
}

vec4 source(vec2 p, float t) {
  vec4 dye = vec4(0);
  float seed = mod(uSeed, 997.0) * 0.017;
  for (int i = 0; i < 6; ++i) {
    float f = float(i);
    float angle = TAU * f / 6.0 + t * 0.19 + seed;
    float radius = 0.25 + 0.08 * sin(t * 0.37 + f * 2.0);
    vec2 center = 0.5 + radius * vec2(cos(angle), sin(angle));
    if (uStyle == 1) center = vec2(fract(f / 6.0 + t * 0.025), 0.5 + 0.28 * sin(f + t * 0.3));
    vec2 d = p - center;
    d -= floor(d + 0.5);
    float ink = exp(-dot(d, d) / (uStyle == 2 ? 0.0018 : 0.0035));
    dye += vec4(palette(f / 6.0 + t * 0.025 + seed), 1.0) * ink;
  }
  return dye;
}

void main() {
  float t = uTime * uSpeed;
  if (uPass == 2) {
    vec2 p = gl_FragCoord.xy / uGridSize;
    vec3 base = palette(p.x + p.y + mod(uSeed, 997.0) * 0.01);
    outColor = vec4(base * 0.18, 0.18) + source(p, t) * 0.7;
    return;
  }
  if (uPass == 0) {
    vec2 p = gl_FragCoord.xy / uGridSize;
    // Fixed simulation step matches the application's one update per frame.
    float dt = uSpeed / 60.0;
    vec2 v = velocity(p, t);
    vec2 midpoint = p - 0.5 * dt * v;
    vec2 back = p - dt * velocity(midpoint, t);
    vec4 dye = sampleDye(back);
    vec2 pixel = 1.0 / uGridSize;
    vec4 neighbors = (sampleDye(back + vec2(pixel.x, 0)) + sampleDye(back - vec2(pixel.x, 0))
      + sampleDye(back + vec2(0, pixel.y)) + sampleDye(back - vec2(0, pixel.y))) * 0.25;
    dye = mix(dye, neighbors, 0.035 * min(uSpeed, 1.0));
    dye *= pow(uPersistence, uSpeed);
    dye += source(p, t) * (0.045 * uInjection * uSpeed);
    outColor = clamp(dye, 0.0, 4.0);
    return;
  }

  vec2 p = (2.0 * gl_FragCoord.xy - uResolution) / min(uResolution.x, uResolution.y);
  float radius = length(p);
  float angle = atan(p.y, p.x) + t * 0.045 + uTwist * radius;
  float wedge = TAU / float(uSymmetry);
  angle = abs(mod(angle + 0.5 * wedge, wedge) - 0.5 * wedge);
  vec2 q = radius * vec2(cos(angle), sin(angle)) * uScale;
  // Recursive folds make nested fluid petals while retaining mirror symmetry.
  float weight = 0.3;
  for (int i = 0; i < 5; ++i) {
    if (i >= uDetail) break;
    q += weight * sin(rotate(float(i) * 0.73) * q * 3.1 + vec2(t * 0.08, -t * 0.06));
    q = abs(q) - vec2(0.28, 0.19);
    weight *= 0.55;
  }
  vec4 dye = sampleDye(q * 0.37 + 0.5);
  vec3 pigment = dye.rgb / max(dye.a, 0.08);
  // Rotate RGB around its neutral axis for a continuous hue control.
  vec3 axis = normalize(vec3(1));
  float hue = TAU * uHue;
  pigment = pigment * cos(hue) + cross(axis, pigment) * sin(hue)
    + axis * dot(axis, pigment) * (1.0 - cos(hue));
  pigment = max(pigment, 0.0);
  float density = 1.0 - exp(-dye.a * 2.3);
  vec3 color = pigment * (0.2 + 1.2 * density);
  float veins = pow(0.5 + 0.5 * sin(dye.a * 22.0 + dot(dye.rgb, vec3(8, 13, 21))), 12.0);
  color += uGlow * veins * pigment * density * 0.4;
  color = 1.0 - exp(-color * 1.7);
  outColor = vec4(pow(color, vec3(0.85)), 1.0);
}
`,On=[{id:"fluid-vortex-mandala",name:"Fluid Vortex Mandala",description:"Rainbow dye swept through turbulent vortices and nested mirrored petals.",symmetry:8,detail:3,twist:.7,scale:1.4,turbulence:1.8},{id:"fluid-prismatic-ribbons",name:"Fluid Prismatic Ribbons",description:"Silky streams of prismatic ink stretched into recursive kaleidoscopic ribbons.",symmetry:6,detail:2,twist:-.5,scale:1.9,turbulence:1.2},{id:"fluid-cellular-jewel",name:"Fluid Cellular Jewel",description:"Colorful cellular eddies mixing inside an intricate fractal jewel.",symmetry:12,detail:4,twist:.2,scale:2.3,turbulence:2.5},{id:"fluid-spiral-bloom",name:"Fluid Spiral Bloom",description:"Swirling liquid pigments blooming along twisting fractal spiral arms.",symmetry:5,detail:3,twist:2.4,scale:1.2,turbulence:2}];function j(e,t,n,r,a,o,l,s="float"){return{id:e,label:t,uniform:n,type:s,value:r,min:a,max:o,step:l}}const Ln=On.map((e,t)=>({id:e.id,name:e.name,description:e.description,fragment:Nn.replace("uniform int uStyle;",`const int uStyle = ${t};`),resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[j("speed","Flow Speed","uSpeed",1,0,3,.05),j("turbulence","Turbulence","uTurbulence",e.turbulence,0,5,.05),j("symmetry","Kaleidoscope Sectors","uSymmetry",e.symmetry,2,24,1,"int"),j("detail","Fractal Folds","uDetail",e.detail,0,5,1,"int"),j("scale","Pattern Scale","uScale",e.scale,.3,4,.05),j("twist","Spiral Twist","uTwist",e.twist,-4,4,.05),j("persistence","Dye Persistence","uPersistence",.992,.95,.999,.001),j("injection","Dye Injection","uInjection",1,0,3,.05),j("hue","Hue Shift","uHue",0,0,1,.01),j("glow","Liquid Vein Glow","uGlow",.7,0,3,.05),{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]})),_n=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;
uniform vec2 uResolution;
uniform vec2 uGridSize;
uniform sampler2D uState;
uniform int uPass;
uniform int uStyle;
uniform float uTime;
uniform float uSpeed;
uniform float uTurbulence;
uniform float uEddyScale;
uniform float uBuoyancy;
uniform float uWind;
uniform float uEmission;
uniform float uWidth;
uniform float uPersistence;
uniform float uDensity;
uniform float uHue;
uniform float uGlow;
uniform float uSeed;

const float TAU = 6.28318530718;

// Saturated emissive pigments; RGB is transported premultiplied by density.
vec3 palette(float h) {
  vec3 c = 0.5 + 0.5 * cos(TAU * (h + uHue + vec3(0.0, 0.333, 0.667)));
  return 0.025 + 0.975 * c * c;
}

// Analytic curl of a multiscale stream function produces smooth eddies.
// Prescribed stirring plus density buoyancy is a visual gas approximation,
// not a pressure-projected Navier-Stokes solver.
vec2 velocity(vec2 p, float t, float density) {
  vec2 curl = vec2(0);
  float frequency = 1.0;
  float amplitude = 1.0;
  float seed = mod(uSeed, 997.0) * 0.071;
  for (int i = 0; i < 5; ++i) {
    float f = float(i);
    float a = f * 1.618 + seed;
    vec2 k = vec2(cos(a), sin(a));
    vec2 l = vec2(-k.y, k.x);
    vec2 q = vec2(dot(p, k), dot(p, l)) * (TAU * frequency * uEddyScale);
    q += vec2(sin(t * 0.29 + f + seed), cos(t * 0.23 - f));
    vec2 gradient = k * cos(q.x) * cos(q.y) - l * sin(q.x) * sin(q.y);
    curl += amplitude * vec2(gradient.y, -gradient.x);
    frequency *= 2.03;
    amplitude *= 0.56;
  }
  vec2 v = curl * (0.055 * uTurbulence);
  v += vec2(uWind * 0.12, uBuoyancy * (0.065 + 0.055 * min(density, 2.0)));
  vec2 d = p - 0.5;
  if (uStyle == 1) {
    v.x += -0.24 * tanh(d.x * 7.0) * exp(-d.y * d.y * 16.0);
    v.y += 0.16 * tanh(d.y * 10.0) * exp(-d.x * d.x * 22.0);
  }
  if (uStyle == 2) {
    v += vec2(-d.y, d.x) * 0.65 * exp(-dot(d, d) * 3.0) - d * 0.025;
  }
  if (uStyle == 3) {
    v.x += 0.13 + 0.055 * sin(p.y * 18.0 + t * 0.3);
    v.y += 0.025 * sin(p.x * 12.0 - t * 0.5);
  }
  if (uStyle == 4) v += vec2(d.y, -d.x) * 0.12;
  return v;
}

// Manual bilinear interpolation supports nearest-filtered floating targets.
// Open boundaries let smoke escape instead of wrapping to the other edge.
vec4 sampleGas(vec2 uv) {
  if (any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)))) return vec4(0);
  uv = clamp(uv, 0.5 / uGridSize, 1.0 - 0.5 / uGridSize);
  vec2 p = uv * uGridSize - 0.5;
  vec2 f = fract(p);
  vec2 pixel = 1.0 / uGridSize;
  vec2 a = (floor(p) + 0.5) * pixel;
  return mix(mix(texture(uState, a), texture(uState, a + vec2(pixel.x, 0)), f.x),
    mix(texture(uState, a + vec2(0, pixel.y)), texture(uState, a + pixel), f.x), f.y);
}

vec4 source(vec2 p, float t) {
  vec4 gas = vec4(0);
  float seed = mod(uSeed, 997.0) * 0.017;
  for (int i = 0; i < 6; ++i) {
    float f = float(i);
    float angle = TAU * f / 6.0 + seed;
    vec2 center = vec2(0.12 + f * 0.152 + 0.015 * sin(t + f), 0.085);
    vec2 size = vec2(0.022, 0.027);
    if (uStyle == 1) {
      center = vec2(i < 3 ? 0.08 : 0.92, 0.3 + mod(f, 3.0) * 0.2);
      size = vec2(0.028, 0.036);
    }
    if (uStyle == 2) {
      center = 0.5 + 0.32 * vec2(cos(angle + t * 0.08), sin(angle + t * 0.08));
      size = vec2(0.038);
    }
    if (uStyle == 3) {
      center = vec2(0.07 + f * 0.15, 0.16 + 0.055 * sin(t * 0.5 + f));
      size = vec2(0.05, 0.025);
    }
    if (uStyle == 4) {
      angle += t * 0.35;
      center = 0.5 + (0.23 + 0.07 * sin(t * 0.4 + f)) * vec2(cos(angle), sin(angle));
      size = vec2(0.018);
    }
    vec2 d = (p - center) / (size * uWidth);
    float amount = exp(-dot(d, d) * 0.5) * (0.75 + 0.25 * sin(t * 1.7 + f * 2.3));
    gas += vec4(palette(f / 6.0 + seed + t * 0.012), 1) * amount;
  }
  return gas * uEmission;
}

void main() {
  float t = uTime * uSpeed;
  if (uPass == 2) {
    // Small emitter puffs appear immediately; the rest of the domain is clear.
    outColor = source(gl_FragCoord.xy / uGridSize, t) * 0.8;
    return;
  }
  if (uPass == 0) {
    vec2 p = gl_FragCoord.xy / uGridSize;
    float dt = uSpeed / 60.0;
    float density = texture(uState, p).a;
    vec2 midpoint = p - 0.5 * dt * velocity(p, t, density);
    vec2 back = p - dt * velocity(midpoint, t, density);
    vec4 gas = sampleGas(back);
    vec2 pixel = 1.0 / uGridSize;
    vec4 neighbors = (sampleGas(back + vec2(pixel.x, 0)) + sampleGas(back - vec2(pixel.x, 0))
      + sampleGas(back + vec2(0, pixel.y)) + sampleGas(back - vec2(0, pixel.y))) * 0.25;
    gas = mix(gas, neighbors, 0.025 * min(uSpeed, 1.0));
    gas *= pow(uPersistence, uSpeed);
    gas += source(p, t) * (0.065 * uSpeed);
    outColor = clamp(gas, 0.0, 6.0);
    return;
  }

  // Fill the screen; rendering and transport use the same normalized domain.
  vec2 uv = gl_FragCoord.xy / uResolution;
  vec4 gas = sampleGas(uv);
  vec3 pigment = gas.rgb / max(gas.a, 0.0001);
  vec2 pixel = 1.5 / uGridSize;
  float dx = sampleGas(uv + vec2(pixel.x, 0)).a - sampleGas(uv - vec2(pixel.x, 0)).a;
  float dy = sampleGas(uv + vec2(0, pixel.y)).a - sampleGas(uv - vec2(0, pixel.y)).a;
  vec3 normal = normalize(vec3(-dx * 5.0, -dy * 5.0, 0.65));
  float light = 0.55 + 0.65 * max(dot(normal, normalize(vec3(-0.5, 0.7, 1))), 0.0);
  float opacity = 1.0 - exp(-gas.a * uDensity * 1.7);
  float wisps = pow(max(gas.a, 0.0), 0.55) * exp(-gas.a * 0.8);
  vec3 color = pigment * (opacity * light * 1.7 + wisps * uGlow * 0.65);
  vec3 background = vec3(0.006, 0.009, 0.022);
  color += background * (1.0 - opacity);
  color = 1.0 - exp(-color * 1.4);
  outColor = vec4(pow(color, vec3(0.85)), 1);
}
`,Hn=[{id:"smoke-rainbow-plumes",name:"Smoke Rainbow Plumes",description:"Hot pink, cyan and golden smoke billowing upward from rainbow vents.",turbulence:1.2,buoyancy:1,lifetime:.993},{id:"gas-chromatic-collision",name:"Gas Chromatic Collision",description:"Opposing multicolored gas jets collide and roll into turbulent mixing clouds.",turbulence:1.5,buoyancy:.15,lifetime:.992},{id:"smoke-nebula-vortex",name:"Smoke Nebula Vortex",description:"Luminous violet, turquoise and orange gases spiral into a cosmic vortex.",turbulence:1,buoyancy:0,lifetime:.996},{id:"gas-neon-fog",name:"Gas Neon Fog",description:"Low banks of electric rainbow fog tumble in rolling crosswinds.",turbulence:1.8,buoyancy:.25,lifetime:.995},{id:"smoke-orbital-trails",name:"Smoke Orbital Trails",description:"Orbiting emitters leave curling ribbons of vivid smoke that disperse into wisps.",turbulence:1.3,buoyancy:.1,lifetime:.991}];function q(e,t,n,r,a,o,l){return{id:e,label:t,uniform:n,type:"float",value:r,min:a,max:o,step:l}}const qn=Hn.map((e,t)=>({id:e.id,name:e.name,description:e.description,fragment:_n.replace("uniform int uStyle;",`const int uStyle = ${t};`),resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[q("speed","Flow Speed","uSpeed",1,0,3,.05),q("turbulence","Turbulence","uTurbulence",e.turbulence,0,4,.05),q("eddyScale","Eddy Scale","uEddyScale",1,.4,3,.05),q("buoyancy","Buoyancy","uBuoyancy",e.buoyancy,-1,2,.05),q("wind","Crosswind","uWind",0,-1,1,.05),q("emission","Gas Emission","uEmission",1,0,3,.05),q("width","Emitter Width","uWidth",1,.4,2.5,.05),q("persistence","Smoke Persistence","uPersistence",e.lifetime,.97,.999,.001),q("density","Optical Density","uDensity",1.3,.2,3,.05),q("hue","Hue Shift","uHue",0,0,1,.01),q("glow","Scattering Glow","uGlow",.8,0,2,.05),{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]})),Vn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uPhase;
uniform int uComponents;
uniform int uIsoBands;
uniform float uLineThickness;
uniform float uNoiseAmount;
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

float randI(int i, float s, float ch) {
  return hash11(float(i) * 17.0 + s * 251.0 + ch * 0.61803);
}

int randint(int i, float s, float ch, int lo, int hiInclusive) {
  float r = randI(i, s, ch);
  return lo + int(floor(r * float(hiInclusive - lo + 1)));
}

float randUniform(int i, float s, float ch, float lo, float hi) {
  float r = randI(i, s, ch);
  return mix(lo, hi, r);
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

void main() {
  float minDim = min(uResolution.x, uResolution.y);
  vec2 p = (2.0 * gl_FragCoord.xy - uResolution) / minDim;

  float rn = length(p) / sqrt(2.0);
  float th = atan(p.y, p.x);

  int fMin = 1;
  int fMax = 4;
  float aMin = 0.20;
  float aMax = 0.85;
  int tCyclesMin = 0;
  int tCyclesMax = 2;
  int nThetaMin = 1;
  int nThetaMax = 4;
  int nTimeMin = 1;
  int nTimeMax = 2;
  int nRadMin = 0;
  int nRadMax = 2;

  float s = 0.0;
  float ampSum = 0.0;
  float twoPI = 6.283185307179586;

  for (int i = 0; i < 64; ++i) {
    if (i >= uComponents) {
      break;
    }

    int f = randint(i, uSeed, 11.0, fMin, fMax);
    float amp = randUniform(i, uSeed, 13.0, aMin, aMax);
    float phi0 = randUniform(i, uSeed, 17.0, 0.0, twoPI);
    int tCyc = randint(i, uSeed, 19.0, tCyclesMin, tCyclesMax);
    int nTh = randint(i, uSeed, 23.0, nThetaMin, nThetaMax);
    int nTi = randint(i, uSeed, 29.0, nTimeMin, nTimeMax);
    int nRa = randint(i, uSeed, 31.0, nRadMin, nRadMax);
    float nPhi = randUniform(i, uSeed, 37.0, 0.0, twoPI);

    amp *= 1.0 / max(1.0, sqrt(float(uComponents)));
    ampSum += abs(amp);

    float tTerm = twoPI * (float(tCyc) * uPhase);
    float angNoise = sin(float(nTh) * th + twoPI * float(nTi) * uPhase + nPhi);
    float radNoise = sin(twoPI * (float(nRa) * rn + float(nTi + 1) * uPhase) + 0.37 * nPhi);
    float noise = uNoiseAmount * (rn * angNoise + 0.4 * radNoise);

    s += amp * sin(twoPI * (float(f) * rn) + phi0 + tTerm + noise);
  }

  float ampNorm = (ampSum > 1e-9) ? ampSum : 1.0;
  float v = s / ampNorm;

  float line = abs(sin(3.141592653589793 * float(uIsoBands) * v));
  float lt = clamp(uLineThickness, 0.01, 0.75);

  float core = pow(max(0.0, 1.0 - (line / lt)), 1.5);
  float glow = pow(max(0.0, 1.0 - (line / (lt * 2.8))), 2.2);
  float intensity = min(1.0, core + 0.45 * glow);

  if (rn > 0.985) {
    float t = clamp((rn - 0.985) / (1.0 - 0.985), 0.0, 1.0);
    intensity *= (1.0 - t);
  }

  float t1 = sin(twoPI * (1.0 * uPhase));
  float t2 = cos(twoPI * (2.0 * uPhase));
  float hue = 0.24 * rn
    + 0.18 * v
    + 0.12 * sin(th)
    + 0.08 * cos(2.0 * th)
    + 0.06 * sin(3.0 * th)
    + 0.07 * t1
    + 0.05 * t2
    + 0.06 * sin(twoPI * (0.25 * rn + 1.0 * uPhase));
  hue = hue - floor(hue);

  float sat = min(1.0, 0.9 + 0.1 * intensity);
  float bri = min(1.0, 0.95 * intensity + 0.35 * glow);

  vec3 rgb = hsv2rgb(vec3(hue, sat, bri));
  outColor = vec4(rgb, 1.0);
}
`,jn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uPhase;
uniform float uScale;
uniform int uOctaves;
uniform float uLacunarity;
uniform float uGain;
uniform int uIsoBands;
uniform float uLineThickness;
uniform float uSeed;
uniform float uBubbleAmp;
uniform float uBubbleFreq;
uniform float uBubbleDetail;

const float PI = 3.14159265358979323846;
const float TAU = 6.28318530717958647692;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

float h11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec2 h21(float n) {
  return vec2(h11(n * 19.0 + 0.73), h11(n * 23.0 + 1.91));
}

float fbm(vec2 p, int octaves, float lac, float gain) {
  float sum = 0.0;
  float amp = 0.5;
  float norm = 0.0;
  vec2 pp = p;
  for (int i = 0; i < 12; ++i) {
    if (i >= octaves) {
      break;
    }
    sum += amp * noise(pp);
    norm += amp;
    pp *= lac;
    amp *= gain;
  }
  return (norm > 1e-6) ? sum / norm : 0.0;
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

void main() {
  float minDim = min(uResolution.x, uResolution.y);
  vec2 p = (gl_FragCoord.xy - 0.5 * uResolution) / minDim;

  vec2 seedShift = (h21(uSeed * 0.137) - 0.5) * 1024.0;
  vec2 timeShift = vec2(cos(TAU * uPhase), sin(TAU * uPhase)) * (0.75 * uScale);
  vec2 world = p * uScale + seedShift + timeShift;

  vec2 warpOff = vec2(cos(TAU * (uPhase + 0.27)), sin(TAU * (uPhase + 0.27))) * (0.33 * uScale);
  float base0 = fbm(world + warpOff, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));
  float signed0 = base0 * 2.0 - 1.0;
  float tanh0 = tanh(1.35 * signed0);

  vec2 swirl = vec2(-p.y, p.x);
  vec2 warp = (0.18 * uScale) * (swirl * tanh0)
    + (0.12 * uScale) * vec2(sin(world.y * 0.8), cos(world.x * 0.8)) * tanh0;
  vec2 world2 = world + warp;

  float base1 = fbm(world2 + warpOff * 0.6, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));
  float signed1 = base1 * 2.0 - 1.0;
  float h = 0.5 + 0.5 * tanh(1.25 * signed1);
  float hCurve = h * h * (3.0 - 2.0 * h);
  float hFinal = mix(h, hCurve, 0.6);

  float bubbleDet = max(0.25, uBubbleDetail);
  vec2 bubbleTimeShift = vec2(cos(TAU * (uPhase + 0.43)), sin(TAU * (uPhase + 0.43))) * (0.55 * bubbleDet);
  float bubbleNoise = fbm(world2 * bubbleDet + bubbleTimeShift, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));
  float bubbleWave = sin(TAU * (uBubbleFreq * uPhase) + bubbleNoise * PI + 1.5 * tanh0);
  float hBubbled = hFinal + uBubbleAmp * bubbleWave * (0.35 + 0.65 * bubbleNoise);
  hBubbled = clamp(hBubbled, 0.0, 1.0);

  float e = 1.25 / minDim;
  float hx = fbm(world2 + vec2(e, 0.0), uOctaves, uLacunarity, uGain)
    - fbm(world2 - vec2(e, 0.0), uOctaves, uLacunarity, uGain);
  float hy = fbm(world2 + vec2(0.0, e), uOctaves, uLacunarity, uGain)
    - fbm(world2 - vec2(0.0, e), uOctaves, uLacunarity, uGain);
  float slope = length(vec2(hx, hy));

  int bands = max(1, uIsoBands);
  float line = abs(sin(PI * float(bands) * hBubbled));
  float lt = clamp(uLineThickness, 0.02, 0.75);

  float core = pow(max(0.0, 1.0 - (line / lt)), 1.35);
  float glow = pow(max(0.0, 1.0 - (line / (lt * 3.0))), 2.2);
  float intensity = clamp(core + 0.5 * glow, 0.0, 1.0);

  float r = length(p) / 0.9;
  float vignette = smoothstep(1.0, 0.6, r);
  intensity *= vignette;

  float hue = fract(0.62 * hBubbled + 0.18 * slope + 0.1 * sin(TAU * uPhase));
  float sat = mix(0.65, 1.0, intensity);
  float bri = mix(0.12, 1.0, intensity);
  hue = fract(hue + 0.05 * tanh0 + 0.04 * sin(TAU * (uPhase + hBubbled)));

  vec3 rgb = hsv2rgb(vec3(hue, sat, bri));
  outColor = vec4(rgb, 1.0);
}
`,Xn=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 iResolution;\r
uniform float iTime;\r
\r
uniform float uLoopDuration;\r
uniform float uSpeed;\r
uniform float uTwist;\r
uniform float uNoiseScale;\r
uniform float uNoiseAmp;\r
uniform float uColorCycle;\r
uniform float uFogDensity;\r
uniform vec3 uBaseColor;\r
\r
const float TAU = 6.28318530718;\r
\r
float hash(vec2 p) {\r
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);\r
}\r
\r
float noise(vec2 p) {\r
  vec2 i = floor(p);\r
  vec2 f = fract(p);\r
  vec2 u = f * f * (3.0 - 2.0 * f);\r
  return mix(\r
    mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),\r
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),\r
    u.y\r
  );\r
}\r
\r
float fbm(vec2 p) {\r
  float sum = 0.0;\r
  float amp = 0.5;\r
  for (int i = 0; i < 5; i++) {\r
    sum += amp * noise(p);\r
    p *= 2.0;\r
    amp *= 0.5;\r
  }\r
  return sum;\r
}\r
\r
vec3 tunnelPalette(float t) {\r
  return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.17, 0.36)));\r
}\r
\r
void main() {\r
  vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;\r
\r
  float phase = mod(iTime, uLoopDuration) / max(uLoopDuration, 0.001);\r
  float theta = phase * TAU;\r
  vec2 loopA = vec2(cos(theta), sin(theta));\r
  vec2 loopB = vec2(cos(2.0 * theta + 0.7), sin(3.0 * theta - 0.5));\r
\r
  float baseR = length(uv);\r
  float a = atan(uv.y, uv.x);\r
  float twist = uTwist * (0.65 + 0.35 * loopA.x);\r
  a += twist * baseR + 0.25 * loopB.y;\r
\r
  vec2 dir = vec2(cos(a), sin(a));\r
  vec2 flow = loopA * (0.7 + 0.3 * uSpeed) + loopB * (0.25 + 0.2 * uSpeed);\r
  vec2 np = dir * (0.9 * uNoiseScale) + uv * (2.2 * uNoiseScale) + flow * (0.75 * uNoiseScale);\r
  float n = fbm(np);\r
  float n2 = fbm(np * 1.9 + loopB * 3.0 + loopA.yx * 1.7);\r
\r
  float r = baseR + uNoiseAmp * ((n - 0.5) * 1.4 + (n2 - 0.5) * 0.8);\r
  float lane = 0.5 + 0.5 * sin(a * 9.0 + n * 4.0 + 2.8 * loopB.x);\r
  float rings = 0.5 + 0.5 * cos(r * 22.0 - n2 * 5.5 + 2.2 * loopA.y);\r
  float tunnel = smoothstep(0.28, 0.92, lane * 0.75 + rings * 0.95);\r
\r
  float huePhase = a / TAU + 0.25 * n + 0.12 * n2 + 0.12 * uColorCycle * loopA.y;\r
  vec3 dynamicHue = tunnelPalette(huePhase);\r
  vec3 oilHue = tunnelPalette(huePhase + 0.12 * loopB.x + 0.08 * loopA.y);\r
  vec3 col = mix(uBaseColor, dynamicHue, 0.45);\r
  col = mix(col, oilHue, 0.45 + 0.25 * lane);\r
  col = mix(col, vec3(1.0), 0.55 * tunnel);\r
\r
  float fogBase = exp(-baseR * uFogDensity);\r
  float glowBase = pow(fogBase, 1.8);\r
  float e = 0.003 * max(0.5, uNoiseScale);\r
  vec2 grad;\r
  grad.x = fbm(np + vec2(e, 0.0)) - fbm(np - vec2(e, 0.0));\r
  grad.y = fbm(np + vec2(0.0, e)) - fbm(np - vec2(0.0, e));\r
  vec2 normal2D = normalize(grad + vec2(1e-6));\r
  vec2 uvR = uv + normal2D * (0.028 + 0.02 * glowBase);\r
\r
  float rR = length(uvR);\r
  float aR = atan(uvR.y, uvR.x) + twist * rR + 0.25 * loopB.x;\r
  vec2 dirR = vec2(cos(aR), sin(aR));\r
  vec2 npR = dirR * (0.9 * uNoiseScale) + uvR * (2.2 * uNoiseScale) + flow * (0.75 * uNoiseScale);\r
  float nR = fbm(npR);\r
  float nR2 = fbm(npR * 1.9 + loopB * 3.0 + loopA.yx * 1.7);\r
  float laneR = 0.5 + 0.5 * sin(aR * 9.0 + nR * 4.0 + 2.8 * loopA.x);\r
  float ringsR = 0.5 + 0.5 * cos(rR * 22.0 - nR2 * 5.5 + 2.2 * loopB.y);\r
  vec3 colR = mix(uBaseColor, tunnelPalette(aR / TAU + 0.25 * nR + 0.12 * nR2 + 0.12 * uColorCycle * loopB.x), 0.45);\r
  colR = mix(colR, tunnelPalette(aR / TAU + 0.14 * loopA.x + 0.08 * nR), 0.45 + 0.25 * laneR);\r
  colR = mix(colR, vec3(1.0), 0.45 * smoothstep(0.28, 0.92, laneR * 0.75 + ringsR * 0.95));\r
\r
  col = mix(col, colR, 0.58);\r
  col *= mix(0.55, 1.7, fogBase);\r
  col += glowBase * 0.32 * (0.5 * dynamicHue + 0.5 * oilHue);\r
\r
  col = clamp(col, 0.0, 1.0);\r
  outColor = vec4(col, 1.0);\r
}\r
`,Zn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform int uIterations;
uniform float uScale;
uniform float uRotation;
uniform float uGlowIntensity;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;

float sdSegment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

vec2 rotate(vec2 p, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

float kochSegmentDistanceIter(vec2 p, vec2 a, vec2 b, int iterations) {
  vec2 ab = b - a;
  float len = max(length(ab), 1e-6);
  vec2 dir = ab / len;
  vec2 nrm = vec2(-dir.y, dir.x);
  vec2 pl = vec2(dot(p - a, dir) / len, dot(p - a, nrm) / len);

  const float c60 = 0.5;
  const float s60 = 0.8660254037844386;
  mat2 invRotPlus = mat2(c60, s60, -s60, c60);
  mat2 invRotMinus = mat2(c60, -s60, s60, c60);

  const int MAX_ITERS = 8;
  int it = min(iterations, MAX_ITERS);
  float scaleAccum = 1.0;

  for (int i = 0; i < MAX_ITERS; ++i) {
    if (i >= it) {
      break;
    }
    pl *= 3.0;
    float region = floor(pl.x);

    if (region == 1.0) {
      vec2 c = vec2(1.5, 0.0);
      vec2 pr = pl - c;
      vec2 pr1 = invRotPlus * pr;
      vec2 pr2 = invRotMinus * pr;
      vec2 p1 = pr1 + c;
      vec2 p2 = pr2 + c;
      pl = (abs(p1.y) < abs(p2.y)) ? p1 : p2;
    }

    pl.x -= region;
    scaleAccum *= (1.0 / 3.0);
  }

  float dLocal = sdSegment(pl, vec2(0.0, 0.0), vec2(1.0, 0.0));
  return dLocal * len * scaleAccum;
}

float kochSnowflakeDistance(vec2 p, float size, int iterations) {
  float h = size * sqrt(3.0) / 2.0;
  vec2 v1 = vec2(0.0, h * 2.0 / 3.0);
  vec2 v2 = vec2(-size / 2.0, -h / 3.0);
  vec2 v3 = vec2(size / 2.0, -h / 3.0);

  float d1 = kochSegmentDistanceIter(p, v1, v2, iterations);
  float d2 = kochSegmentDistanceIter(p, v2, v3, iterations);
  float d3 = kochSegmentDistanceIter(p, v3, v1, iterations);
  return min(min(d1, d2), d3);
}

float trianglePerimeterDistance(vec2 p, float size) {
  float h = size * sqrt(3.0) / 2.0;
  vec2 v1 = vec2(0.0, h * 2.0 / 3.0);
  vec2 v2 = vec2(-size / 2.0, -h / 3.0);
  vec2 v3 = vec2(size / 2.0, -h / 3.0);
  float d1 = sdSegment(p, v1, v2);
  float d2 = sdSegment(p, v2, v3);
  float d3 = sdSegment(p, v3, v1);
  return min(min(d1, d2), d3);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / min(uResolution.x, uResolution.y);
  float angle = uTime * uRotation;
  uv = rotate(uv, angle);

  float distKoch = kochSnowflakeDistance(uv, uScale, uIterations);
  float distTri = trianglePerimeterDistance(uv, uScale);
  float dist = min(distKoch, distTri * 0.75);

  const float lineWidth = 0.004;
  const float lineOuterMult = 1.5;
  const float lineInnerMult = 0.5;
  const float distanceScale = 15.0;
  const float timeScale = 2.0;
  const float glowMix = 0.4;
  const float edgeGlowMult = 0.3;

  float line = smoothstep(lineWidth * lineOuterMult, lineWidth * lineInnerMult, dist);
  float glow = exp(-dist * distanceScale * uGlowIntensity);
  float colorMix = sin(dist * distanceScale - uTime * timeScale) * 0.5 + 0.5;
  vec3 color = mix(uColorPrimary, uColorSecondary, colorMix);

  vec3 finalColor = color * (line + glow * glowMix);
  vec3 edgeGlowColor = vec3(0.2, 0.3, 0.5);
  finalColor += edgeGlowColor * glow * uGlowIntensity * edgeGlowMult;

  outColor = vec4(finalColor, 1.0);
}
`,$n=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform int uIterations;
uniform float uScale;
uniform float uRotation;
uniform float uGlowIntensity;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;

mat2 rot2(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat2(c, -s, s, c);
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

float quasi(vec2 p, int waves) {
  float A = 0.0;
  for (int i = 0; i < 16; ++i) {
    if (i >= waves) {
      break;
    }
    float ang = 6.2831853 * float(i) * 0.5 * (sqrt(5.0) - 1.0);
    vec2 k = vec2(cos(ang), sin(ang));
    A += cos(dot(k, p) * 3.0);
  }
  return A / float(max(1, waves));
}

float kochSegmentIter(vec2 p, vec2 a, vec2 b, int it) {
  vec2 ex = normalize(b - a);
  vec2 ey = vec2(-ex.y, ex.x);
  float L = length(b - a);

  vec2 v = vec2(dot(p - a, ex), dot(p - a, ey));
  vec2 w = v / L;

  float s = 1.0;
  for (int k = 0; k < 8; ++k) {
    if (k >= it) {
      break;
    }
    w *= 3.0;
    s /= 3.0;
    if (w.x > 1.0 && w.x < 2.0) {
      w = rot2(-3.14159265 / 3.0) * (w - vec2(1.0, 0.0));
    } else if (w.x >= 2.0) {
      w.x -= 2.0;
    }
  }

  float d = sdSegment(w, vec2(0.0), vec2(1.0, 0.0));
  return d * L * s;
}

float kochSnowflakeDist(vec2 p, float size, int it) {
  float r = size;
  vec2 v0 = r * vec2(cos(0.0), sin(0.0));
  vec2 v1 = r * vec2(cos(2.094395102), sin(2.094395102));
  vec2 v2 = r * vec2(cos(4.188790205), sin(4.188790205));

  float d0 = kochSegmentIter(p, v0, v1, it);
  float d1 = kochSegmentIter(p, v1, v2, it);
  float d2 = kochSegmentIter(p, v2, v0, it);
  return min(d0, min(d1, d2));
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;

  float r = length(uv);
  float vig = smoothstep(1.2, 0.2, r);
  vec3 bg = mix(uColorSecondary * 0.06, uColorSecondary * 0.22, vig);

  vec2 p = uv * uScale;
  p *= rot2(uRotation);
  float q1 = quasi(p * 2.8 + 0.3 * vec2(cos(uTime * 0.17), sin(uTime * 0.21)), 9);
  float q2 = quasi(p.yx * 3.1 + 0.2 * vec2(sin(uTime * 0.13), cos(uTime * 0.19)), 7);
  float warpAmp = 0.06 + 0.045 * (0.5 + 0.5 * sin(uTime * 0.57));
  vec2 pWarp = p + warpAmp * vec2(q1, q2);

  float maxIt = float(clamp(uIterations, 1, 8));
  float minIt = max(1.0, maxIt - 3.0);
  float iAnim = mix(minIt, maxIt, 0.5 + 0.5 * sin(uTime * 0.27));
  int i0 = int(floor(iAnim));
  int i1 = min(i0 + 1, 8);
  float itMix = fract(iAnim);

  float radius = 0.70 + 0.12 * sin(uTime * 0.41);
  float d0 = kochSnowflakeDist(pWarp, radius, i0);
  float d1 = kochSnowflakeDist(pWarp, radius, i1);
  float d = mix(d0, d1, itMix);

  float lineWidth = 0.0035 + 0.0015 * (0.5 + 0.5 * sin(uTime * 0.77));
  float edge = smoothstep(lineWidth, 0.0, d);
  float glow = exp(-14.0 * d) * uGlowIntensity;

  vec3 snow = mix(uColorSecondary, uColorPrimary, edge) + glow * uColorPrimary;
  vec3 col = bg + snow;

  outColor = vec4(col, 1.0);
}
`,Yn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform vec2 uGridSize;
uniform sampler2D uState;
uniform int uPass;
uniform float uTime;
uniform float uSelfWeight;
uniform float uNeighborWeight;
uniform float uDecay;
uniform float uRotate;
uniform float uInjectAmp;
uniform float uInjectRadius;
uniform float uValueGain;
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

vec2 sampleState(vec2 uv) {
  vec2 gridUV = uv * uGridSize - 0.5;
  vec2 base = floor(gridUV);
  vec2 f = fract(gridUV);
  vec2 invGrid = 1.0 / uGridSize;
  vec2 p00 = (base + vec2(0.5, 0.5)) * invGrid;
  vec2 p10 = (base + vec2(1.5, 0.5)) * invGrid;
  vec2 p01 = (base + vec2(0.5, 1.5)) * invGrid;
  vec2 p11 = (base + vec2(1.5, 1.5)) * invGrid;
  vec2 v00 = texture(uState, p00).rg;
  vec2 v10 = texture(uState, p10).rg;
  vec2 v01 = texture(uState, p01).rg;
  vec2 v11 = texture(uState, p11).rg;
  vec2 v0 = mix(v00, v10, f.x);
  vec2 v1 = mix(v01, v11, f.x);
  return mix(v0, v1, f.y);
}

vec2 diffuseVec2(vec2 uv, vec2 texel) {
  vec2 c = texture(uState, uv).rg;
  vec2 sum = c * uSelfWeight;
  sum += texture(uState, uv + vec2(texel.x, 0.0)).rg * uNeighborWeight;
  sum += texture(uState, uv - vec2(texel.x, 0.0)).rg * uNeighborWeight;
  sum += texture(uState, uv + vec2(0.0, texel.y)).rg * uNeighborWeight;
  sum += texture(uState, uv - vec2(0.0, texel.y)).rg * uNeighborWeight;
  float norm = max(1e-4, uSelfWeight + 4.0 * uNeighborWeight);
  return sum / norm;
}

void main() {
  if (uPass == 2) {
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    vec2 ab = vec2(0.0);
    float seed = uSeed * 0.001;
    float radius = 0.08;
    for (int i = 0; i < 3; ++i) {
      float fi = float(i);
      vec2 pos = vec2(hash11(seed + fi * 3.1 + 1.0), hash11(seed + fi * 4.7 + 2.0));
      float ang = hash11(seed + fi * 5.3 + 3.0) * 6.2831853;
      float d = distance(uv, pos);
      float g = exp(-d * d / (radius * radius));
      ab += g * vec2(cos(ang), sin(ang));
    }
    outColor = vec4(ab, 0.0, 1.0);
    return;
  }

  if (uPass == 0) {
    vec2 texel = 1.0 / uGridSize;
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    vec2 ab = diffuseVec2(uv, texel);

    float ang = uRotate;
    float ca = cos(ang);
    float sa = sin(ang);
    ab = mat2(ca, -sa, sa, ca) * ab;
    ab *= uDecay;

    float seed = uSeed * 0.001;
    float t = uTime * 0.6 + seed * 3.0;
    vec2 pos = 0.5 + 0.32 * vec2(sin(t * 1.1 + seed), cos(t * 1.4 + seed * 1.7));
    float injectAng = t * 1.7 + seed * 5.0;
    float dist = distance(uv, pos);
    float sigma = max(1e-4, uInjectRadius);
    float g = exp(-dist * dist / (sigma * sigma));
    ab += uInjectAmp * g * vec2(cos(injectAng), sin(injectAng));

    outColor = vec4(ab, 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  vec2 ab = sampleState(uv);
  float angle = atan(ab.y, ab.x);
  float mag = length(ab);
  float hue = (angle + 3.14159265) / 6.2831853;
  float value = clamp(mag * uValueGain, 0.0, 1.0);
  vec3 rgb = hsv2rgb(vec3(hue, 1.0, value));
  outColor = vec4(rgb, 1.0);
}
`,Kn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform vec2 uGridSize;
uniform sampler2D uState;
uniform int uPass;
uniform float uTime;
uniform float uSelfWeight;
uniform float uNeighborWeight;
uniform float uDecay;
uniform float uBlobAmp;
uniform float uBlobRadius;
uniform float uSpeed;
uniform float uFlowGain;
uniform float uFlowThreshold;
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

float sampleState(vec2 uv) {
  vec2 gridUV = uv * uGridSize - 0.5;
  vec2 base = floor(gridUV);
  vec2 f = fract(gridUV);
  vec2 invGrid = 1.0 / uGridSize;
  vec2 p00 = (base + vec2(0.5, 0.5)) * invGrid;
  vec2 p10 = (base + vec2(1.5, 0.5)) * invGrid;
  vec2 p01 = (base + vec2(0.5, 1.5)) * invGrid;
  vec2 p11 = (base + vec2(1.5, 1.5)) * invGrid;
  float v00 = texture(uState, p00).r;
  float v10 = texture(uState, p10).r;
  float v01 = texture(uState, p01).r;
  float v11 = texture(uState, p11).r;
  float v0 = mix(v00, v10, f.x);
  float v1 = mix(v01, v11, f.x);
  return mix(v0, v1, f.y);
}

float diffuseScalar(vec2 uv, vec2 texel) {
  float c = texture(uState, uv).r;
  float sum = c * uSelfWeight;
  sum += texture(uState, uv + vec2(texel.x, 0.0)).r * uNeighborWeight;
  sum += texture(uState, uv - vec2(texel.x, 0.0)).r * uNeighborWeight;
  sum += texture(uState, uv + vec2(0.0, texel.y)).r * uNeighborWeight;
  sum += texture(uState, uv - vec2(0.0, texel.y)).r * uNeighborWeight;
  float norm = max(1e-4, uSelfWeight + 4.0 * uNeighborWeight);
  return sum / norm;
}

void main() {
  if (uPass == 2) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  if (uPass == 0) {
    vec2 texel = 1.0 / uGridSize;
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    float x = diffuseScalar(uv, texel);
    x *= uDecay;

    float seed = uSeed * 0.001;
    float t = uTime * uSpeed + seed * 2.0;
    vec2 c1 = 0.5 + 0.34 * vec2(sin(t * 1.2 + seed), cos(t * 1.6 + seed * 1.3));
    vec2 c2 = 0.5 + 0.30 * vec2(sin(t * 0.8 + seed * 2.1), cos(t * 1.1 + seed * 0.7));
    float sigma = max(1e-4, uBlobRadius);
    float g1 = exp(-distance(uv, c1) * distance(uv, c1) / (sigma * sigma));
    float g2 = exp(-distance(uv, c2) * distance(uv, c2) / (sigma * sigma));
    x += uBlobAmp * (g1 + 0.8 * g2);

    x = clamp(x, 0.0, 1.0);
    outColor = vec4(x, 0.0, 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  vec2 texel = 1.0 / uGridSize;
  float xL = sampleState(uv - vec2(texel.x, 0.0));
  float xR = sampleState(uv + vec2(texel.x, 0.0));
  float xD = sampleState(uv - vec2(0.0, texel.y));
  float xU = sampleState(uv + vec2(0.0, texel.y));

  vec2 grad = vec2(xR - xL, xU - xD);
  float mag = length(grad) * uFlowGain;
  float threshold = max(0.0, uFlowThreshold);
  float edge = smoothstep(threshold, threshold + 0.05, mag);

  float hue = (atan(grad.y, grad.x) + 3.14159265) / 6.2831853;
  float value = clamp(mag, 0.0, 1.0) * edge;
  vec3 rgb = hsv2rgb(vec3(hue, 0.9, value));
  outColor = vec4(rgb, 1.0);
}
`,Qn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform vec2 uGridSize;
uniform sampler2D uState;
uniform int uPass;
uniform float uTime;
uniform float uSelfWeight;
uniform float uNeighborWeight;
uniform float uDecay;
uniform float uThreshold;
uniform float uSharpness;
uniform float uNoiseAmp;
uniform float uTurbulence;
uniform float uInjectAmp;
uniform float uInjectRadius;
uniform float uSpeed;
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float diffuseScalar(vec2 uv, vec2 texel) {
  float c = texture(uState, uv).r;
  float sum = c * uSelfWeight;
  sum += texture(uState, uv + vec2(texel.x, 0.0)).r * uNeighborWeight;
  sum += texture(uState, uv - vec2(texel.x, 0.0)).r * uNeighborWeight;
  sum += texture(uState, uv + vec2(0.0, texel.y)).r * uNeighborWeight;
  sum += texture(uState, uv - vec2(0.0, texel.y)).r * uNeighborWeight;
  float norm = max(1e-4, uSelfWeight + 4.0 * uNeighborWeight);
  return sum / norm;
}

float sigmoid(float z) {
  return 1.0 / (1.0 + exp(-z));
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

float sampleState(vec2 uv) {
  vec2 gridUV = uv * uGridSize - 0.5;
  vec2 base = floor(gridUV);
  vec2 f = fract(gridUV);
  vec2 invGrid = 1.0 / uGridSize;
  vec2 p00 = (base + vec2(0.5, 0.5)) * invGrid;
  vec2 p10 = (base + vec2(1.5, 0.5)) * invGrid;
  vec2 p01 = (base + vec2(0.5, 1.5)) * invGrid;
  vec2 p11 = (base + vec2(1.5, 1.5)) * invGrid;
  float x00 = texture(uState, p00).r;
  float x10 = texture(uState, p10).r;
  float x01 = texture(uState, p01).r;
  float x11 = texture(uState, p11).r;
  float x0 = mix(x00, x10, f.x);
  float x1 = mix(x01, x11, f.x);
  return mix(x0, x1, f.y);
}

vec2 flowField(vec2 uv, float t) {
  float s1 = sin(uv.y * 6.0 + t);
  float s2 = cos(uv.x * 6.0 - t * 1.1);
  float n1 = hash21(uv * uGridSize + t * 0.7);
  float n2 = hash21(uv * uGridSize + t * 0.7 + vec2(12.3, 45.6));
  vec2 flow = vec2(s1 + (n1 - 0.5) * 1.2, s2 + (n2 - 0.5) * 1.2);
  return normalize(flow + vec2(1e-3));
}

void main() {
  if (uPass == 2) {
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    float seed = uSeed * 0.001;
    float x = hash21(uv * uGridSize + seed) * 0.25;
    outColor = vec4(x, 0.0, 0.0, 1.0);
    return;
  }

  if (uPass == 0) {
    vec2 texel = 1.0 / uGridSize;
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    float t = uTime * uSpeed + uSeed * 0.001;
    float advect = 0.02 + 0.04 * clamp(uNoiseAmp, 0.0, 1.0);
    vec2 flow = flowField(uv, t);
    vec2 uvAdv = uv + flow * advect * max(0.0, uTurbulence);
    float x = diffuseScalar(uvAdv, texel);

    x = sigmoid(uSharpness * (x - uThreshold));
    x *= uDecay;

    float seed = uSeed * 0.001;
    float injectT = uTime * uSpeed + seed * 4.0;
    vec2 pos = 0.5 + 0.33 * vec2(sin(injectT * 1.1 + seed), cos(injectT * 1.4 + seed * 1.9));
    float sigma = max(1e-4, uInjectRadius);
    float g = exp(-distance(uv, pos) * distance(uv, pos) / (sigma * sigma));
    x += uInjectAmp * g;

    float noise = (hash21(uv * uGridSize + uTime * 2.0 + seed) - 0.5) * uNoiseAmp;
    x = clamp(x + noise, 0.0, 1.0);
    outColor = vec4(x, 0.0, 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  float x = sampleState(uv);
  float hue = fract(0.6 + 0.1 * sin(uTime * 0.25) + x * 1.2);
  float sat = clamp(0.5 + x * 0.8, 0.0, 1.0);
  float val = clamp(0.15 + x * 1.1, 0.0, 1.0);
  vec3 rgb = hsv2rgb(vec3(hue, sat, val));
  outColor = vec4(rgb, 1.0);
}
`,Jn=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform float uTime;\r
uniform vec2 uResolution;\r
uniform int uIterations;\r
uniform float uRotateSpeed;\r
uniform float uFoldOffset;\r
uniform float uStepScale;\r
uniform float uGlow;\r
uniform float uCameraDistance;\r
uniform float uCameraSpin;\r
uniform vec3 uColorPrimary;\r
uniform vec3 uColorSecondary;\r
uniform float uColorMix;\r
uniform float uAlphaGain;\r
\r
vec3 palette(float d) {\r
    vec3 base = mix(uColorPrimary, uColorSecondary, clamp(d, 0.0, 1.0));\r
    return mix(base, base * base, uColorMix);\r
}\r
\r
vec2 rotate2D(vec2 p, float a) {\r
    float c = cos(a);\r
    float s = sin(a);\r
    return mat2(c, s, -s, c) * p;\r
}\r
\r
float mapFunc(vec3 p) {\r
    float t = uTime * uRotateSpeed;\r
    for (int i = 0; i < 64; ++i) {\r
        if (i >= uIterations) break;\r
        p.xz = rotate2D(p.xz, t);\r
        p.xy = rotate2D(p.xy, t * 1.89);\r
        p.xz = abs(p.xz);\r
        p.xz -= vec2(uFoldOffset);\r
    }\r
    return dot(sign(p), p) / uStepScale;\r
}\r
\r
vec4 rm(vec3 ro, vec3 rd) {\r
    float t = 0.0;\r
    vec3 col = vec3(0.0);\r
    float d = 1.0;\r
\r
    for (int i = 0; i < 72; ++i) {\r
        vec3 p = ro + rd * t;\r
        d = mapFunc(p) * 0.5;\r
\r
        if (d < 0.02) break;\r
        if (d > 120.0) break;\r
\r
        float shade = length(p) * 0.08;\r
        col += palette(shade) * uGlow / (400.0 * d);\r
        t += d;\r
    }\r
\r
    float alpha = 1.0 / (max(d, 0.01) * 100.0);\r
    return vec4(col, clamp(alpha * uAlphaGain, 0.0, 1.0));\r
}\r
\r
void main() {\r
    vec2 fragCoord = gl_FragCoord.xy;\r
    vec2 uv = (fragCoord - (uResolution * 0.5)) / uResolution.x;\r
\r
    vec3 ro = vec3(0.0, 0.0, -uCameraDistance);\r
    ro.xz = rotate2D(ro.xz, uTime * uCameraSpin);\r
\r
    vec3 cf = normalize(-ro);\r
    vec3 cs = normalize(cross(cf, vec3(0.0, 1.0, 0.0)));\r
    vec3 cu = normalize(cross(cf, cs));\r
\r
    vec3 uuv = ro + cf * 3.0 + uv.x * cs + uv.y * cu;\r
    vec3 rd = normalize(uuv - ro);\r
\r
    outColor = rm(ro, rd);\r
}\r
`,er=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeaHeight;
uniform float uSeaChoppy;
uniform float uSeaSpeed;
uniform float uSeaFreq;
uniform float uCamHeight;
uniform float uCamDistance;
uniform float uCamYaw;
uniform float uCamPitch;
uniform float uSkyBoost;
uniform float uWaterBrightness;
uniform vec3 uWaterTint;

const int NUM_STEPS = 32;
const int ITER_GEOMETRY = 3;
const int ITER_FRAGMENT = 5;
const float PI = 3.141592;
const float EPSILON = 1e-3;
#define EPSILON_NRM (0.1 / uResolution.x)
const mat2 octave_m = mat2(1.6, 1.2, -1.2, 1.6);

mat3 fromEuler(vec3 ang) {
  vec2 a1 = vec2(sin(ang.x), cos(ang.x));
  vec2 a2 = vec2(sin(ang.y), cos(ang.y));
  vec2 a3 = vec2(sin(ang.z), cos(ang.z));
  mat3 m;
  m[0] = vec3(a1.y * a3.y + a1.x * a2.x * a3.x, a1.y * a2.x * a3.x + a3.y * a1.x, -a2.y * a3.x);
  m[1] = vec3(-a2.y * a1.x, a1.y * a2.y, a2.x);
  m[2] = vec3(a3.y * a1.x * a2.x + a1.y * a3.x, a1.x * a3.x - a1.y * a3.y * a2.x, a2.y * a3.y);
  return m;
}

float hash(vec2 p) {
  float h = dot(p, vec2(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return -1.0 + 2.0 * mix(
    mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

float diffuse(vec3 n, vec3 l, float p) {
  return pow(dot(n, l) * 0.4 + 0.6, p);
}

float specular(vec3 n, vec3 l, vec3 e, float s) {
  float nrm = (s + 8.0) / (PI * 8.0);
  return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;
}

vec3 getSkyColor(vec3 e) {
  e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;
  return vec3(pow(1.0 - e.y, 2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4) * uSkyBoost;
}

float sea_octave(vec2 uv, float choppy) {
  uv += noise(uv);
  vec2 wv = 1.0 - abs(sin(uv));
  vec2 swv = abs(cos(uv));
  wv = mix(wv, swv, wv);
  return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

float map(vec3 p, float seaTime) {
  float freq = uSeaFreq;
  float amp = uSeaHeight;
  float choppy = uSeaChoppy;
  vec2 uv = p.xz;
  uv.x *= 0.75;

  float d;
  float h = 0.0;
  for (int i = 0; i < ITER_GEOMETRY; i++) {
    d = sea_octave((uv + seaTime) * freq, choppy);
    d += sea_octave((uv - seaTime) * freq, choppy);
    h += d * amp;
    uv *= octave_m;
    freq *= 1.9;
    amp *= 0.22;
    choppy = mix(choppy, 1.0, 0.2);
  }
  return p.y - h;
}

float map_detailed(vec3 p, float seaTime) {
  float freq = uSeaFreq;
  float amp = uSeaHeight;
  float choppy = uSeaChoppy;
  vec2 uv = p.xz;
  uv.x *= 0.75;

  float d;
  float h = 0.0;
  for (int i = 0; i < ITER_FRAGMENT; i++) {
    d = sea_octave((uv + seaTime) * freq, choppy);
    d += sea_octave((uv - seaTime) * freq, choppy);
    h += d * amp;
    uv *= octave_m;
    freq *= 1.9;
    amp *= 0.22;
    choppy = mix(choppy, 1.0, 0.2);
  }
  return p.y - h;
}

vec3 getSeaColor(vec3 p, vec3 n, vec3 l, vec3 eye, vec3 dist) {
  float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);
  fresnel = min(fresnel * fresnel * fresnel, 0.5);

  vec3 seaBase = uWaterTint * 0.2;
  vec3 seaWater = mix(vec3(0.8, 0.9, 0.6), uWaterTint, 0.5) * uWaterBrightness;

  vec3 reflected = getSkyColor(reflect(eye, n));
  vec3 refracted = seaBase + diffuse(n, l, 80.0) * seaWater * 0.12;

  vec3 color = mix(refracted, reflected, fresnel);

  float atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);
  color += seaWater * (p.y - uSeaHeight) * 0.18 * atten;

  color += specular(n, l, eye, 600.0 * inversesqrt(dot(dist, dist)));

  return color;
}

vec3 getNormal(vec3 p, float eps, float seaTime) {
  vec3 n;
  n.y = map_detailed(p, seaTime);
  n.x = map_detailed(vec3(p.x + eps, p.y, p.z), seaTime) - n.y;
  n.z = map_detailed(vec3(p.x, p.y, p.z + eps), seaTime) - n.y;
  n.y = eps;
  return normalize(n);
}

float heightMapTracing(vec3 ori, vec3 dir, out vec3 p, float seaTime) {
  float tm = 0.0;
  float tx = 1000.0;
  float hx = map(ori + dir * tx, seaTime);
  if (hx > 0.0) {
    p = ori + dir * tx;
    return tx;
  }
  float hm = map(ori, seaTime);
  for (int i = 0; i < NUM_STEPS; i++) {
    float tmid = mix(tm, tx, hm / (hm - hx));
    p = ori + dir * tmid;
    float hmid = map(p, seaTime);
    if (hmid < 0.0) {
      tx = tmid;
      hx = hmid;
    } else {
      tm = tmid;
      hm = hmid;
    }
    if (abs(hmid) < EPSILON) break;
  }
  return mix(tm, tx, hm / (hm - hx));
}

vec3 getPixel(vec2 coord, float time, float seaTime) {
  vec2 uv = coord / uResolution.xy;
  uv = uv * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;

  vec3 ang = vec3(sin(time * 3.0) * 0.1 + uCamPitch, sin(time) * 0.2 + 0.3, time + uCamYaw);
  vec3 ori = vec3(0.0, uCamHeight, time * uCamDistance);
  vec3 dir = normalize(vec3(uv.xy, -2.0));
  dir.z += length(uv) * 0.14;
  dir = normalize(dir) * fromEuler(ang);

  vec3 p;
  heightMapTracing(ori, dir, p, seaTime);
  vec3 dist = p - ori;
  vec3 n = getNormal(p, dot(dist, dist) * EPSILON_NRM, seaTime);
  vec3 light = normalize(vec3(0.0, 1.0, 0.8));

  return mix(
    getSkyColor(dir),
    getSeaColor(p, n, light, dir, dist),
    pow(smoothstep(0.0, -0.02, dir.y), 0.2)
  );
}

void main() {
  vec2 fragCoord = gl_FragCoord.xy;
  float time = uTime * uTimeScale;
  float seaTime = 1.0 + time * uSeaSpeed;

  vec3 color = getPixel(fragCoord, time, seaTime);
  outColor = vec4(pow(color, vec3(0.65)), 1.0);
}\r
`,tr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uSeaHeight;\r
uniform float uSeaChoppy;\r
uniform float uSeaSpeed;\r
uniform float uSeaFreq;\r
uniform float uCamHeight;\r
uniform float uCamDistance;\r
uniform float uCamYaw;\r
uniform float uCamPitch;\r
uniform float uSkyBoost;\r
uniform float uWaterBrightness;\r
uniform float uFilmThickness;\r
uniform float uDiffraction;\r
uniform float uOilScale;\r
uniform float uOilWarp;\r
uniform float uPlasmaDensity;\r
uniform float uDischargeSpeed;\r
uniform float uSpectralContrast;\r
uniform float uPlasmaGlow;\r
uniform float uAcidSky;\r
uniform float uHueShift;\r
uniform float uSeed;\r
\r
const int NUM_STEPS = 32;\r
const int ITER_GEOMETRY = 3;\r
const int ITER_FRAGMENT = 5;\r
const float PI = 3.141592;\r
const float TAU = 6.28318530718;\r
const float EPSILON = 1e-3;\r
#define EPSILON_NRM (0.1 / uResolution.x)\r
const mat2 octave_m = mat2(1.6, 1.2, -1.2, 1.6);\r
\r
mat2 rotate2d(float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return mat2(c, -s, s, c);\r
}\r
\r
mat3 fromEuler(vec3 angle) {\r
  vec2 a1 = vec2(sin(angle.x), cos(angle.x));\r
  vec2 a2 = vec2(sin(angle.y), cos(angle.y));\r
  vec2 a3 = vec2(sin(angle.z), cos(angle.z));\r
  mat3 matrix;\r
  matrix[0] = vec3(a1.y * a3.y + a1.x * a2.x * a3.x, a1.y * a2.x * a3.x + a3.y * a1.x, -a2.y * a3.x);\r
  matrix[1] = vec3(-a2.y * a1.x, a1.y * a2.y, a2.x);\r
  matrix[2] = vec3(a3.y * a1.x * a2.x + a1.y * a3.x, a1.x * a3.x - a1.y * a3.y * a2.x, a2.y * a3.y);\r
  return matrix;\r
}\r
\r
float hash(vec2 point) {\r
  float value = dot(point + fract(uSeed * 0.000001) * 19.17, vec2(127.1, 311.7));\r
  return fract(sin(value) * 43758.5453123);\r
}\r
\r
float signedNoise(vec2 point) {\r
  vec2 cell = floor(point);\r
  vec2 local = fract(point);\r
  vec2 curve = local * local * (3.0 - 2.0 * local);\r
  return -1.0 + 2.0 * mix(\r
    mix(hash(cell), hash(cell + vec2(1.0, 0.0)), curve.x),\r
    mix(hash(cell + vec2(0.0, 1.0)), hash(cell + vec2(1.0)), curve.x),\r
    curve.y\r
  );\r
}\r
\r
float fbm(vec2 point) {\r
  float value = 0.0;\r
  float amplitude = 0.52;\r
  for (int octave = 0; octave < 5; octave++) {\r
    value += amplitude * signedNoise(point);\r
    point = rotate2d(0.57) * point * 2.03 + vec2(1.7, -2.4);\r
    amplitude *= 0.49;\r
  }\r
  return value;\r
}\r
\r
float diffuse(vec3 normal, vec3 light, float power) {\r
  return pow(dot(normal, light) * 0.4 + 0.6, power);\r
}\r
\r
float specular(vec3 normal, vec3 light, vec3 eye, float shininess) {\r
  float normalization = (shininess + 8.0) / (PI * 8.0);\r
  return pow(max(dot(reflect(eye, normal), light), 0.0), shininess) * normalization;\r
}\r
\r
vec3 spectralPalette(float phase) {\r
  return 0.5 + 0.5 * cos(TAU * (\r
    phase * vec3(1.0, 1.29, 1.63)\r
      + vec3(0.02, 0.31, 0.68)\r
      + uHueShift * vec3(1.0, 0.83, 1.17)\r
  ));\r
}\r
\r
vec3 getSkyColor(vec3 eye) {\r
  eye.y = (max(eye.y, 0.0) * 0.8 + 0.2) * 0.8;\r
  float horizon = 1.0 - eye.y;\r
  vec3 naturalSky = vec3(horizon * horizon, horizon, 0.6 + horizon * 0.4);\r
  vec3 acidSky = mix(\r
    spectralPalette(eye.y * 0.35 + 0.08),\r
    spectralPalette(0.62 - eye.y * 0.22),\r
    horizon\r
  ) * vec3(0.62, 0.72, 0.82);\r
  return mix(naturalSky, acidSky, uAcidSky) * uSkyBoost;\r
}\r
\r
float seaOctave(vec2 uv, float choppy) {\r
  uv += signedNoise(uv);\r
  vec2 wave = 1.0 - abs(sin(uv));\r
  vec2 smoothWave = abs(cos(uv));\r
  wave = mix(wave, smoothWave, wave);\r
  return pow(1.0 - pow(wave.x * wave.y, 0.65), choppy);\r
}\r
\r
float map(vec3 point, float seaTime) {\r
  float frequency = uSeaFreq;\r
  float amplitude = uSeaHeight;\r
  float choppy = uSeaChoppy;\r
  vec2 uv = point.xz;\r
  uv.x *= 0.75;\r
  float height = 0.0;\r
\r
  for (int octave = 0; octave < ITER_GEOMETRY; octave++) {\r
    float wave = seaOctave((uv + seaTime) * frequency, choppy);\r
    wave += seaOctave((uv - seaTime) * frequency, choppy);\r
    height += wave * amplitude;\r
    uv *= octave_m;\r
    frequency *= 1.9;\r
    amplitude *= 0.22;\r
    choppy = mix(choppy, 1.0, 0.2);\r
  }\r
  return point.y - height;\r
}\r
\r
float mapDetailed(vec3 point, float seaTime) {\r
  float frequency = uSeaFreq;\r
  float amplitude = uSeaHeight;\r
  float choppy = uSeaChoppy;\r
  vec2 uv = point.xz;\r
  uv.x *= 0.75;\r
  float height = 0.0;\r
\r
  for (int octave = 0; octave < ITER_FRAGMENT; octave++) {\r
    float wave = seaOctave((uv + seaTime) * frequency, choppy);\r
    wave += seaOctave((uv - seaTime) * frequency, choppy);\r
    height += wave * amplitude;\r
    uv *= octave_m;\r
    frequency *= 1.9;\r
    amplitude *= 0.22;\r
    choppy = mix(choppy, 1.0, 0.2);\r
  }\r
  return point.y - height;\r
}\r
\r
vec3 getAcidSeaColor(\r
  vec3 point,\r
  vec3 normal,\r
  vec3 light,\r
  vec3 eye,\r
  vec3 distance,\r
  float time\r
) {\r
  float fresnel = clamp(1.0 - dot(normal, -eye), 0.0, 1.0);\r
  fresnel = min(fresnel * fresnel * fresnel, 0.58);\r
\r
  vec2 oilPoint = point.xz * uOilScale;\r
  vec2 drift = vec2(time * 0.045, -time * 0.037);\r
  vec2 warpA = vec2(\r
    fbm(oilPoint * 0.72 + drift),\r
    fbm(oilPoint * 0.72 + vec2(5.2, -3.4) - drift.yx)\r
  );\r
  vec2 warpB = vec2(\r
    fbm(oilPoint * 1.65 + warpA * 1.4 - drift * 0.6),\r
    fbm(oilPoint * 1.65 - warpA.yx * 1.3 + drift * 0.5 + 8.7)\r
  );\r
  vec2 fluidPoint = oilPoint + (warpA * 0.72 + warpB * 0.28) * uOilWarp;\r
\r
  float broadFilm = fbm(fluidPoint * 0.85 - drift * 0.7);\r
  float fineFilm = fbm(fluidPoint * 2.6 + warpB * 1.8 + drift);\r
  float crestFilm = dot(normal.xz, vec2(0.7, -0.5)) * 0.65;\r
  float thickness = uFilmThickness * (\r
    broadFilm * 2.5 + fineFilm * 0.72 + crestFilm + point.y * 0.22\r
  );\r
\r
  vec3 film = spectralPalette(thickness * uDiffraction);\r
  film = smoothstep(vec3(0.06), vec3(0.94), film);\r
  film = pow(max(film, 0.0), vec3(max(uSpectralContrast, 0.1)));\r
  vec3 shiftedFilm = spectralPalette((thickness + fineFilm * 0.34) * uDiffraction + 0.17);\r
\r
  float diffractionRidge = pow(\r
    1.0 - abs(sin((thickness + broadFilm * 0.28) * uDiffraction * TAU)),\r
    8.0\r
  );\r
  float plasmaPhase = fineFilm * uPlasmaDensity * 7.5\r
    + broadFilm * 4.0\r
    + dot(normal.xz, vec2(2.7, -2.1))\r
    - time * uDischargeSpeed;\r
  float plasmaVein = pow(1.0 - abs(sin(plasmaPhase)), 13.0);\r
  float crest = pow(max(0.0, 1.0 - normal.y), 2.2);\r
  float travelingCharge = pow(\r
    1.0 - abs(sin(point.x * 0.11 + point.z * 0.08 - time * uDischargeSpeed * 1.6)),\r
    16.0\r
  );\r
\r
  vec3 reflected = getSkyColor(reflect(eye, normal));\r
  vec3 acidBase = film * (0.14 + 0.26 * uWaterBrightness);\r
  acidBase += shiftedFilm * diffractionRidge * (0.16 + 0.18 * uWaterBrightness);\r
  acidBase += diffuse(normal, light, 70.0) * film * 0.09;\r
  vec3 color = mix(acidBase, reflected * (0.7 + film * 0.45), fresnel);\r
\r
  float attenuation = max(1.0 - dot(distance, distance) * 0.001, 0.0);\r
  color += film * (point.y - uSeaHeight) * 0.12 * attenuation;\r
  color += mix(film, vec3(0.78, 0.9, 1.0), 0.68)\r
    * plasmaVein * uPlasmaGlow * (0.34 + crest * 0.9);\r
  color += shiftedFilm * travelingCharge * plasmaVein * uPlasmaGlow * 0.75;\r
  color += vec3(0.92, 0.98, 1.0)\r
    * specular(normal, light, eye, 520.0 * inversesqrt(dot(distance, distance)));\r
  return color;\r
}\r
\r
vec3 getNormal(vec3 point, float epsilon, float seaTime) {\r
  vec3 normal;\r
  normal.y = mapDetailed(point, seaTime);\r
  normal.x = mapDetailed(vec3(point.x + epsilon, point.y, point.z), seaTime) - normal.y;\r
  normal.z = mapDetailed(vec3(point.x, point.y, point.z + epsilon), seaTime) - normal.y;\r
  normal.y = epsilon;\r
  return normalize(normal);\r
}\r
\r
float heightMapTracing(vec3 origin, vec3 direction, out vec3 point, float seaTime) {\r
  float nearDistance = 0.0;\r
  float farDistance = 1000.0;\r
  float farHeight = map(origin + direction * farDistance, seaTime);\r
  if (farHeight > 0.0) {\r
    point = origin + direction * farDistance;\r
    return farDistance;\r
  }\r
  float nearHeight = map(origin, seaTime);\r
  for (int stepIndex = 0; stepIndex < NUM_STEPS; stepIndex++) {\r
    float midpoint = mix(nearDistance, farDistance, nearHeight / (nearHeight - farHeight));\r
    point = origin + direction * midpoint;\r
    float midpointHeight = map(point, seaTime);\r
    if (midpointHeight < 0.0) {\r
      farDistance = midpoint;\r
      farHeight = midpointHeight;\r
    } else {\r
      nearDistance = midpoint;\r
      nearHeight = midpointHeight;\r
    }\r
    if (abs(midpointHeight) < EPSILON) {\r
      break;\r
    }\r
  }\r
  return mix(nearDistance, farDistance, nearHeight / (nearHeight - farHeight));\r
}\r
\r
vec3 getPixel(vec2 coordinate, float time, float seaTime) {\r
  vec2 uv = coordinate / uResolution.xy;\r
  uv = uv * 2.0 - 1.0;\r
  uv.x *= uResolution.x / uResolution.y;\r
\r
  vec3 angle = vec3(sin(time * 3.0) * 0.1 + uCamPitch, sin(time) * 0.2 + 0.3, time + uCamYaw);\r
  vec3 origin = vec3(0.0, uCamHeight, time * uCamDistance);\r
  vec3 direction = normalize(vec3(uv.xy, -2.0));\r
  direction.z += length(uv) * 0.14;\r
  direction = normalize(direction) * fromEuler(angle);\r
\r
  vec3 point;\r
  heightMapTracing(origin, direction, point, seaTime);\r
  vec3 distance = point - origin;\r
  vec3 normal = getNormal(point, dot(distance, distance) * EPSILON_NRM, seaTime);\r
  vec3 light = normalize(vec3(0.0, 1.0, 0.8));\r
\r
  return mix(\r
    getSkyColor(direction),\r
    getAcidSeaColor(point, normal, light, direction, distance, time),\r
    pow(smoothstep(0.0, -0.02, direction.y), 0.2)\r
  );\r
}\r
\r
void main() {\r
  float time = uTime * uTimeScale;\r
  float seaTime = 1.0 + time * uSeaSpeed;\r
  vec3 color = getPixel(gl_FragCoord.xy, time, seaTime);\r
  outColor = vec4(pow(max(color, 0.0), vec3(0.68)), 1.0);\r
}`,nr=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeed;

uniform float uSeaHeight;
uniform float uSeaChoppy;
uniform float uSeaSpeed;
uniform float uSeaFreq;
uniform float uCamHeight;
uniform float uCamDistance;
uniform float uCamYaw;
uniform float uCamPitch;
uniform float uReflection;

uniform float uWaterScale;
uniform float uWaterSpeed;
uniform int uFractalIters;
uniform float uFoldScale;
uniform float uFoldOffset;
uniform float uRotSpeed;
uniform float uDetailLevel;
uniform float uSmoothBlend;
uniform float uWaterHue;
uniform float uHueSpeed;
uniform float uSaturation;
uniform float uWaterBrightness;
uniform float uWaterContrast;
uniform float uWaterGlow;

uniform float uSkyScale;
uniform float uSkySpeed;
uniform float uSkyBoost;
uniform float uFilmThickness;
uniform float uDiffraction;
uniform float uFluidWarp;
uniform float uDropletScale;
uniform float uPlasmaDensity;
uniform float uDischargeSpeed;
uniform float uSpectralContrast;
uniform float uPlasmaGlow;
uniform float uSkyHue;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;
const mat2 OCTAVE_MATRIX = mat2(1.6, 1.2, -1.2, 1.6);

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

float seaHash(vec2 point) {
  float value = dot(point + fract(uSeed * 0.000001) * 19.17, vec2(127.1, 311.7));
  return fract(sin(value) * 43758.5453123);
}

float seaNoise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  vec2 curve = local * local * (3.0 - 2.0 * local);
  return -1.0 + 2.0 * mix(
    mix(seaHash(cell), seaHash(cell + vec2(1.0, 0.0)), curve.x),
    mix(seaHash(cell + vec2(0.0, 1.0)), seaHash(cell + vec2(1.0)), curve.x),
    curve.y
  );
}

float seaOctave(vec2 point, float choppy) {
  point += seaNoise(point);
  vec2 wave = 1.0 - abs(sin(point));
  wave = mix(wave, abs(cos(point)), wave);
  return pow(max(1.0 - pow(wave.x * wave.y, 0.65), 0.0), choppy);
}

// The same multiscale wave surface used by Seascape and Acidscape.
float seaMap(vec3 point, float seaTime, bool detailed) {
  vec2 uv = point.xz * vec2(0.75, 1.0);
  float frequency = uSeaFreq;
  float amplitude = uSeaHeight;
  float choppy = uSeaChoppy;
  float height = 0.0;
  for (int octave = 0; octave < 5; octave++) {
    if (!detailed && octave >= 3) break;
    float wave = seaOctave((uv + seaTime) * frequency, choppy);
    wave += seaOctave((uv - seaTime) * frequency, choppy);
    height += wave * amplitude;
    uv *= OCTAVE_MATRIX;
    frequency *= 1.9;
    amplitude *= 0.22;
    choppy = mix(choppy, 1.0, 0.2);
  }
  return point.y - height;
}

bool traceSea(vec3 origin, vec3 direction, float seaTime, out vec3 point) {
  float nearDistance = 0.0;
  float farDistance = 450.0;
  point = origin + direction * farDistance;
  float nearHeight = seaMap(origin, seaTime, false);
  float farHeight = seaMap(point, seaTime, false);
  if (farHeight > 0.0) return false;

  for (int stepIndex = 0; stepIndex < 32; stepIndex++) {
    float fraction = nearHeight / max(nearHeight - farHeight, 0.00001);
    float midpoint = mix(nearDistance, farDistance, fraction);
    point = origin + direction * midpoint;
    float midpointHeight = seaMap(point, seaTime, false);
    if (abs(midpointHeight) < 0.001) return true;
    if (midpointHeight < 0.0) {
      farDistance = midpoint;
      farHeight = midpointHeight;
    } else {
      nearDistance = midpoint;
      nearHeight = midpointHeight;
    }
  }
  return true;
}

vec3 seaNormal(vec3 point, float distanceSquared, float seaTime) {
  float epsilon = clamp(distanceSquared * 0.1 / uResolution.x, 0.002, 35.0);
  float height = seaMap(point, seaTime, true);
  return normalize(vec3(
    seaMap(point + vec3(epsilon, 0.0, 0.0), seaTime, true) - height,
    epsilon,
    seaMap(point + vec3(0.0, 0.0, epsilon), seaTime, true) - height
  ));
}

// Fractal Reverie's noise, nested domain warps, and four color palettes.
float reverieHash(vec2 point) {
  point = fract(point * vec2(234.34, 435.45));
  point += dot(point, point + 34.23 + fract(uSeed * 0.000001));
  return fract(point.x * point.y);
}

float reverieNoise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  vec2 curve = local * local * local * (local * (local * 6.0 - 15.0) + 10.0);
  return mix(
    mix(reverieHash(cell), reverieHash(cell + vec2(1.0, 0.0)), curve.x),
    mix(reverieHash(cell + vec2(0.0, 1.0)), reverieHash(cell + vec2(1.0)), curve.x),
    curve.y
  );
}

float reverieFbm(vec2 point, float footprint) {
  float value = 0.0;
  float amplitude = 0.55;
  float normalization = 0.0;
  int octaves = clamp(uFractalIters + 3, 3, 8);
  for (int octave = 0; octave < 8; octave++) {
    if (octave >= octaves) break;
    // Blend unresolved octaves to their mean instead of aliasing near the horizon.
    float detail = 1.0 - smoothstep(0.25, 0.8, footprint);
    value += amplitude * mix(0.5, reverieNoise(point), detail);
    normalization += amplitude;
    point = rotate2d(0.45 + 0.03 * float(octave)) * point * 2.02 + vec2(0.31, -0.27);
    footprint *= 2.02;
    amplitude *= mix(0.46, 0.60, uSmoothBlend);
  }
  return value / max(normalization, 0.0001);
}

vec3 palNeon(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(1.0, 0.8, 0.6) + vec3(0.00, 0.33, 0.67)));
}

vec3 palLava(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.7, 0.9, 1.3) + vec3(0.00, 0.18, 0.55)));
}

vec3 palFire(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(1.3, 0.5, 0.8) + vec3(0.25, 0.00, 0.55)));
}

vec3 palIce(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.6, 0.9, 1.2) + vec3(0.55, 0.70, 0.85)));
}

vec2 reverieWarp(vec2 point, float time, float footprint) {
  float drift = time * (0.05 + uRotSpeed * 0.12);
  float scale = 0.75 + uFoldScale * 0.35;
  vec2 q = vec2(
    reverieFbm(point * scale + vec2(0.0, drift), footprint * scale),
    reverieFbm(point * scale + vec2(4.8, -drift * 0.8), footprint * scale)
  );
  vec2 r = vec2(
    reverieFbm(point * (scale + 0.6) + (q - 0.5) * 2.0 + vec2(1.7, -2.6) + drift * 0.4, footprint * (scale + 0.6)),
    reverieFbm(point * (scale + 0.4) + (q - 0.5) * 2.0 + vec2(-3.1, 0.9) - drift * 0.3, footprint * (scale + 0.4))
  );
  return point + (q + r - 1.0) * (0.22 + uFoldOffset * 0.25);
}

vec3 reverieWater(vec3 point, vec3 normal, float time, float footprint) {
  vec2 uv = point.xz * uWaterScale;
  vec2 p = uv * 2.174 + vec2(time * 0.0108, 0.066);
  // A small normal offset lets the color fields bend over the wave crests.
  p += normal.xz * 0.12;
  vec2 q = reverieWarp(p, time, footprint);
  vec2 r = reverieWarp(q * 1.35 + vec2(2.4, -1.8), time * 0.7, footprint * 1.35);
  float field = reverieFbm(q, footprint);
  float filaments = reverieFbm(r * 1.7 + vec2(time * 0.03, -time * 0.025), footprint * 2.295);
  float mist = reverieFbm(p * 0.65 - vec2(time * 0.02, time * 0.015), footprint * 0.65);
  float ridgeScale = 1.4 + 0.2 * uDetailLevel;
  float ridges = 1.0 - abs(2.0 * reverieFbm(q * ridgeScale, footprint * ridgeScale) - 1.0);
  float glowMask = smoothstep(0.30, 0.88, mix(field, ridges, 0.55));
  float veil = smoothstep(0.18, 0.82, mix(mist, filaments, 0.4));
  float colorTime = uWaterHue + time * uHueSpeed * 0.25 + fract(uSeed * 0.000001);
  float horizon = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
  // Bounded radial variation avoids rapid palette flicker in distant water.
  float radial = length(uv) / (1.0 + length(uv));
  vec3 cool = mix(
    palIce(colorTime + field * 0.45 + mist * 0.2),
    palNeon(colorTime + filaments * 0.35 + horizon * 0.15),
    0.45 + 0.25 * horizon
  );
  vec3 warm = mix(
    palLava(colorTime + ridges * 0.50 + 0.08 * radial),
    palFire(colorTime + glowMask * 0.65 + filaments * 0.15),
    glowMask
  );
  vec3 color = mix(cool * (0.22 + 0.18 * horizon), warm, veil);
  color += palFire(colorTime + filaments + radial * 0.1) * glowMask * glowMask
    * (0.10 + 0.05 * uWaterGlow);
  color += palNeon(colorTime + mist * 0.6 + 0.12 * radial)
    * pow(max(1.0 - radial * 0.75, 0.0), 2.0) * 0.095;
  float haze = smoothstep(0.25, 1.35, radial + (1.0 - horizon) * 0.35);
  color = mix(color, cool * 0.35, haze * 0.35);
  color *= 0.75 + 0.25 * uWaterBrightness;
  float luma = dot(color, vec3(0.299, 0.587, 0.114));
  color = mix(vec3(luma), color, uSaturation);
  color = max(mix(vec3(0.5), color, uWaterContrast), 0.0);
  return color / (color + 0.35);
}

// Plasma Oil Diffraction projected onto a sky dome, also sampled by reflections.
float oilHash(vec2 point) {
  point += fract(uSeed * 0.000001) * 17.73;
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

vec2 oilHash22(vec2 point) {
  float value = oilHash(point);
  return vec2(value, oilHash(point + value + 19.19));
}

float oilNoise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  local = local * local * (3.0 - 2.0 * local);
  return mix(
    mix(oilHash(cell), oilHash(cell + vec2(1.0, 0.0)), local.x),
    mix(oilHash(cell + vec2(0.0, 1.0)), oilHash(cell + vec2(1.0)), local.x),
    local.y
  );
}

float oilFbm(vec2 point) {
  float value = 0.0;
  float amplitude = 0.52;
  for (int octave = 0; octave < 6; octave++) {
    value += amplitude * oilNoise(point);
    point = rotate2d(0.58) * point * 2.03 + vec2(1.7, -2.4);
    amplitude *= 0.49;
  }
  return value;
}

float cellularEdge(vec2 point, float time, out float cellDistance) {
  vec2 base = floor(point);
  vec2 local = fract(point);
  float nearest = 10.0;
  float secondNearest = 10.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 offset = vec2(float(x), float(y));
      vec2 random = oilHash22(base + offset);
      vec2 center = offset + 0.5
        + 0.34 * sin(time * vec2(0.23, 0.19) + random * TAU) - local;
      float distanceSquared = dot(center, center);
      if (distanceSquared < nearest) {
        secondNearest = nearest;
        nearest = distanceSquared;
      } else if (distanceSquared < secondNearest) {
        secondNearest = distanceSquared;
      }
    }
  }
  cellDistance = sqrt(nearest);
  return sqrt(secondNearest) - sqrt(nearest);
}

vec3 thinFilm(float thickness, float contrast) {
  vec3 phase = TAU * (thickness * vec3(1.0, 1.29, 1.63) * uDiffraction
    + uSkyHue * vec3(1.0, 0.83, 1.17));
  vec3 reflected = 0.5 + 0.5 * cos(phase + vec3(0.0, 0.45, 0.9));
  reflected = smoothstep(vec3(0.08), vec3(0.92), reflected);
  return pow(max(reflected, 0.0), vec3(max(contrast, 0.1)));
}

vec3 plasmaSky(vec3 direction, float time) {
  vec2 point = direction.xz / (0.65 + max(direction.y, 0.0));
  point *= uSkyScale;
  vec2 drift = vec2(time * 0.055, -time * 0.041);
  vec2 warpA = vec2(
    oilFbm(point * 1.15 + drift),
    oilFbm(point * 1.15 + vec2(5.4, -3.1) - drift.yx)
  ) - 0.5;
  vec2 warpB = vec2(
    oilFbm(point * 2.05 + warpA * 1.7 - drift * 0.6),
    oilFbm(point * 2.05 - warpA.yx * 1.5 + drift * 0.45 + 8.2)
  ) - 0.5;
  vec2 fluidPoint = point + (warpA * 0.75 + warpB * 0.35) * uFluidWarp;
  float cellDistance;
  float cellGap = cellularEdge(fluidPoint * uDropletScale, time, cellDistance);
  float oilBoundary = exp(-cellGap * 38.0);
  float dropletInterior = 1.0 - smoothstep(0.08, 0.62, cellDistance);
  float broadFilm = oilFbm(fluidPoint * 1.7 - drift * 0.8);
  float fineFilm = oilFbm(fluidPoint * 4.6 + warpB * 2.1 + drift);
  float radialFilm = sin(length(fluidPoint + warpA * 0.3) * 8.0 - time * 0.36) * 0.5 + 0.5;
  float thickness = uFilmThickness * (
    broadFilm * 2.8 + fineFilm * 0.85 + radialFilm * 0.4 + cellDistance * 1.25
  );
  vec3 diffractionColor = thinFilm(thickness, uSpectralContrast);
  vec3 shiftedFilm = thinFilm(thickness + cellGap * 2.5 + 0.11, uSpectralContrast * 0.85);
  float diffractionRings = pow(
    1.0 - abs(sin((thickness + fineFilm * 0.3) * uDiffraction * TAU)), 7.0
  );
  float gradientField = oilFbm(fluidPoint * 3.2 + warpA * 2.0 - time * 0.09);
  float plasmaPhase = gradientField * uPlasmaDensity * 8.0
    + broadFilm * 5.0 + cellGap * 7.0 - time * uDischargeSpeed;
  float plasma = pow(1.0 - abs(sin(plasmaPhase)), 12.0);
  float boundaryPlasma = oilBoundary * pow(
    1.0 - abs(sin(thickness * 4.0 - time * uDischargeSpeed * 1.3)), 5.0
  );
  float sparks = pow(
    1.0 - abs(sin((fluidPoint.x - fluidPoint.y) * 11.0 + warpB.x * 9.0 - time * 1.7)), 18.0
  ) * plasma;
  vec3 color = vec3(0.004, 0.006, 0.012);
  color += diffractionColor * (0.13 + dropletInterior * 0.24);
  color += shiftedFilm * oilBoundary * 0.42;
  color += diffractionColor * diffractionRings * (0.2 + uPlasmaGlow * 0.18);
  color += mix(diffractionColor, vec3(0.72, 0.88, 1.0), 0.62) * plasma * uPlasmaGlow * 0.92;
  color += mix(shiftedFilm, vec3(1.0), 0.7) * boundaryPlasma * uPlasmaGlow * 1.25;
  color += vec3(0.72, 0.86, 1.0) * sparks * uPlasmaGlow * 1.4;
  float oilySpecular = pow(max(0.0, 1.0 - length(warpB) * 1.35), 6.0);
  color += shiftedFilm * oilySpecular * dropletInterior * 0.24;
  return (1.0 - exp(-color * 1.42)) * uSkyBoost;
}

void main() {
  float time = uTime * uTimeScale;
  float seaTime = 1.0 + time * uSeaSpeed;
  float skyTime = time * uSkySpeed;
  vec2 uv = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  vec3 origin = vec3(0.0, max(uCamHeight, uSeaHeight * 2.6 + 0.2), time * uCamDistance);
  vec3 direction = normalize(vec3(uv, -2.0));
  direction.z += length(uv) * 0.14;
  direction.yz = rotate2d(uCamPitch + sin(time * 0.16) * 0.018) * direction.yz;
  direction.xy = rotate2d(sin(time * 0.11) * 0.012) * direction.xy;
  direction.xz = rotate2d(uCamYaw + time * 0.035) * direction.xz;
  direction = normalize(direction);
  vec3 sky = plasmaSky(direction, skyTime);
  vec3 color = sky;
  vec3 point;
  bool hitSea = traceSea(origin, direction, seaTime, point);
  // Evaluate derivatives before the branch so neighboring sky pixels stay defined.
  float footprint = max(length(dFdx(point.xz)), length(dFdy(point.xz))) * uWaterScale * 2.174;
  if (hitSea) {
    vec3 distance = point - origin;
    float distanceSquared = max(dot(distance, distance), 0.001);
    vec3 normal = seaNormal(point, distanceSquared, seaTime);
    vec3 light = normalize(vec3(0.0, 1.0, 0.8));
    vec3 water = reverieWater(point, normal, time * uWaterSpeed, footprint);
    water *= 0.55 + 0.45 * max(dot(normal, light), 0.0);
    float fresnel = pow(clamp(1.0 - dot(normal, -direction), 0.0, 1.0), 3.0);
    vec3 reflected = plasmaSky(reflect(direction, normal), skyTime);
    water = mix(water, reflected, fresnel * uReflection);
    float crest = pow(max(1.0 - normal.y, 0.0), 2.0);
    water += palNeon(uWaterHue + point.y * 0.2) * crest * uWaterGlow * 0.045;
    float shininess = 520.0 * inversesqrt(distanceSquared);
    float specular = pow(max(dot(reflect(direction, normal), light), 0.0), shininess);
    water += vec3(0.86, 0.94, 1.0) * specular * (shininess + 8.0) / (PI * 8.0) * 0.35;
    float haze = 1.0 - exp(-sqrt(distanceSquared) * 0.012);
    water = mix(water, sky, haze * 0.72);
    float horizonBlend = pow(1.0 - smoothstep(-0.025, 0.0, direction.y), 0.2);
    color = mix(sky, water, horizonBlend);
  }
  outColor = vec4(pow(max(color, 0.0), vec3(0.85)), 1.0);
}
`,rr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uTileScale;\r
uniform float uIntensity;\r
uniform float uContrast;\r
uniform float uWaveShift;\r
uniform vec3 uTint;\r
uniform float uSwirlStrength;\r
uniform float uSwirlGridScale;\r
uniform float uSwirlRadius;\r
uniform float uSwirlWobble;\r
uniform float uSwirlWobbleSpeed;\r
uniform float uSwirlSpin;\r
uniform float uSwirlDesync;\r
uniform float uSwirlPulse;\r
uniform float uSwirlColorTwist;\r
uniform float uRainbowStrength;\r
uniform float uRainbowScale;\r
uniform float uRainbowSpeed;\r
uniform float uRainbowContrast;\r
\r
const float TAU = 6.28318530718;\r
const int MAX_ITER = 5;\r
\r
vec2 hash22(vec2 p) {\r
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p3 += dot(p3, p3.yzx + 33.33);\r
  return fract((p3.xx + p3.yz) * p3.zy);\r
}\r
\r
mat2 rot(float a) {\r
  float c = cos(a);\r
  float s = sin(a);\r
  return mat2(c, -s, s, c);\r
}\r
\r
// Loose wobbly lattice of out-of-sync whirlpools: each cell hosts a\r
// breathing vortex whose radius pulses and whose spin waxes and wanes on\r
// its own clock. Warps \`uv\` in place and returns the net swirl amount.\r
float whirlpoolField(inout vec2 uv, float time) {\r
  float grid = max(uSwirlGridScale, 0.01);\r
  vec2 gp = uv * grid;\r
  vec2 cell = floor(gp);\r
  float swirlSum = 0.0;\r
  vec2 warped = uv;\r
  for (int oy = -1; oy <= 1; oy++) {\r
    for (int ox = -1; ox <= 1; ox++) {\r
      vec2 id = cell + vec2(float(ox), float(oy));\r
      vec2 rnd = hash22(id + 41.7);\r
      float phase = (rnd.x + rnd.y) * TAU * uSwirlDesync;\r
      vec2 wobble = vec2(\r
        cos(time * uSwirlWobbleSpeed * (0.5 + rnd.y) + phase),\r
        sin(time * uSwirlWobbleSpeed * (0.6 + rnd.x) + phase * 2.3)\r
      ) * uSwirlWobble;\r
      vec2 center = (id + 0.5 + (rnd - 0.5) * 0.9 + wobble) / grid;\r
      vec2 d = warped - center;\r
      float dist = length(d);\r
      // Breathing radius: each whirlpool inhales and exhales out of sync.\r
      float breath = 1.0 + uSwirlPulse * sin(time * (0.4 + rnd.x * 0.7) + phase);\r
      float radius = max(uSwirlRadius, 0.01) * breath / grid;\r
      float falloff = smoothstep(radius, 0.0, dist);\r
      falloff *= falloff * (3.0 - 2.0 * falloff);\r
      // Alternate spin direction on a checkerboard for counter-rotation.\r
      float dir = mod(id.x + id.y, 2.0) < 1.0 ? 1.0 : -1.0;\r
      float spin = sin(time * uSwirlSpin * (0.8 + rnd.y * 0.5) + phase) * 0.6 + 0.7;\r
      float angle = uSwirlStrength * dir * spin * falloff;\r
      warped = center + rot(angle) * (warped - center);\r
      swirlSum += angle;\r
    }\r
  }\r
  uv = warped;\r
  return swirlSum;\r
}\r
\r
void main() {\r
  float time = uTime * uTimeScale + 23.0;\r
  vec2 uv = gl_FragCoord.xy / uResolution.xy;\r
\r
  float swirl = whirlpoolField(uv, time);\r
\r
  vec2 p = mod(uv * TAU * uTileScale, TAU) - 250.0;\r
  vec2 i = p;\r
  float c = 1.0;\r
  float inten = 0.005;\r
\r
  for (int n = 0; n < MAX_ITER; n++) {\r
    float t = time * (1.0 - (3.5 / float(n + 1))) + uWaveShift;\r
    i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));\r
    vec2 denom = vec2(p.x / (sin(i.x + t) / inten), p.y / (cos(i.y + t) / inten));\r
    c += 1.0 / length(denom);\r
  }\r
\r
  c /= float(MAX_ITER);\r
  c = 1.17 - pow(c, 1.4);\r
  float cAdj = pow(clamp(c, 0.0, 1.0), max(0.1, uContrast));\r
  vec3 color = vec3(pow(abs(cAdj), 8.0)) * uIntensity;\r
  color = clamp(color + uTint, 0.0, 1.0);\r
\r
  // Twist hue channels inside the whirlpools.\r
  if (abs(uSwirlColorTwist) > 0.0001) {\r
    float twist = swirl * uSwirlColorTwist;\r
    color.rg = rot(twist) * color.rg;\r
    color.gb = rot(twist * 0.7) * color.gb;\r
    color = abs(color);\r
  }\r
\r
  // Oil-slick interference sheen: film thickness rides the water caustic\r
  // brightness and the swirl field, refracting into a spectral gradient.\r
  if (uRainbowStrength > 0.0001) {\r
    float thickness = cAdj * uRainbowScale + swirl * 2.0 + time * uRainbowSpeed * 0.1;\r
    thickness += sin(uv.x * 9.0 + time * 0.3) * 0.3 + cos(uv.y * 7.0 - time * 0.23) * 0.3;\r
    vec3 sheen = 0.5 + 0.5 * cos(TAU * (thickness * vec3(1.0, 1.3, 1.7) + vec3(0.0, 0.33, 0.67)));\r
    sheen = pow(sheen, vec3(max(uRainbowContrast, 0.1)));\r
    float lum = clamp(dot(color, vec3(0.299, 0.587, 0.114)) * 1.8, 0.0, 1.0);\r
    color = mix(color, color * (0.4 + 1.2 * sheen) + sheen * lum * 0.5, uRainbowStrength);\r
  }\r
\r
  outColor = vec4(clamp(color, 0.0, 1.0), 1.0);\r
}\r
`,ar=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uTurbulence;
uniform float uCloudHeight;
uniform float uStepBase;
uniform float uStepScale;
uniform float uHueShift;
uniform float uHueSpeed;
uniform float uIntensity;

void main() {
  vec2 I = gl_FragCoord.xy;
  float t = uTime * uTimeScale;
  float i = 0.0;
  float z = 0.0;
  float d = 0.0;
  float s = 0.0;
  vec4 O = vec4(0.0);

  for (O *= i; i++ < 100.0;) {
    vec3 p = z * normalize(vec3(I + I, 0.0) - uResolution.xyy);

    for (d = 5.0; d < 200.0; d += d) {
      p += uTurbulence * 0.6 * sin(p.yzx * d - 0.2 * t) / d;
    }

    float height = max(0.05, uCloudHeight);
    s = height - abs(p.y);
    z += d = uStepBase + max(s, -s * 0.2) / uStepScale;

    vec4 phase = vec4(3.0, 4.0, 5.0, 0.0) + uHueShift + uHueSpeed * t;
    O += (cos(s / 0.07 + p.x + 0.5 * t - phase) + 1.5) * exp(s / 0.1) / d;
  }

  O = tanh(O * O / 4e8);
  outColor = vec4(O.rgb * uIntensity, 1.0);
}
`,or=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2  uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uTurbulence;\r
uniform float uCloudHeight;\r
uniform float uStepBase;\r
uniform float uStepScale;\r
uniform float uHueShift;\r
uniform float uHueSpeed;\r
uniform float uIntensity;\r
\r
// --- new degrees of freedom ---\r
uniform float uWarpAmp;       // amplitude of horizon sine warp\r
uniform float uWarpFreq;      // spatial frequency of the warp\r
uniform float uWarpSpeed;     // temporal speed of the warp animation\r
uniform float uWarpHarmonics; // add harmonic overtones for richer shape\r
uniform float uOrbitRadius;   // radius of circular horizon orbit\r
uniform float uOrbitSpeed;    // angular speed of the orbit\r
uniform float uOrbitEcc;      // eccentricity (ellipse stretch)\r
uniform float uTiltAngle;     // tilt the horizon plane\r
uniform float uCloudDensity;  // per-step density multiplier\r
uniform float uFogFalloff;    // controls exponential fog rolloff\r
uniform float uColorSep;      // chromatic separation between RGB channels\r
\r
/*\r
   Smooth periodic horizon function.\r
   The horizon is no longer a single flat plane at y=0.\r
   Instead it follows a warped surface whose height varies with (x,z)\r
   and is additionally orbited in the (y, z) plane over time.\r
*/\r
float horizonHeight(vec3 p, float t) {\r
    // Base warp: sum of harmonics of a spatial sine wave\r
    float h = 0.0;\r
    float amp = uWarpAmp;\r
    float freq = uWarpFreq;\r
    for (float k = 1.0; k <= 5.0; k += 1.0) {\r
        if (k > uWarpHarmonics) break;\r
        // phase varies with time, direction alternates each harmonic\r
        float phase = uWarpSpeed * t * (0.7 + 0.3 * k) + k * 1.37;\r
        h += amp * sin(freq * p.x + phase)\r
           * cos(freq * 0.6 * p.z + phase * 0.8);\r
        amp  *= 0.55;   // each harmonic is weaker\r
        freq *= 1.8;    // and higher frequency\r
    }\r
\r
    // Orbit: translate the center of the horizon along an elliptical path\r
    float orbitAngle = uOrbitSpeed * t;\r
    float oy = uOrbitRadius * sin(orbitAngle);\r
    float oz = uOrbitRadius * uOrbitEcc * cos(orbitAngle);\r
\r
    // Tilt: rotate horizon normal by uTiltAngle around the x-axis\r
    float ct = cos(uTiltAngle);\r
    float st = sin(uTiltAngle);\r
    float tiltedY = ct * (p.y - oy) - st * (p.z - oz);\r
\r
    return h - tiltedY;   // positive = inside clouds\r
}\r
\r
void main() {\r
    vec2 I = gl_FragCoord.xy;\r
    float t = uTime * uTimeScale;\r
    float z = 0.0;\r
    float d = 0.0;\r
    float s = 0.0;\r
    vec4 O = vec4(0.0);\r
\r
    vec3 rd = normalize(vec3(I + I, 0.0) - uResolution.xyy);\r
\r
    for (float i = 0.0; i < 100.0; i++) {\r
        vec3 p = z * rd;\r
\r
        // Volumetric turbulence (same family as sunset_plus)\r
        for (d = 5.0; d < 200.0; d += d) {\r
            p += uTurbulence * 0.6 * sin(p.yzx * d - 0.2 * t) / d;\r
        }\r
\r
        float height = max(0.05, uCloudHeight);\r
\r
        // Sample the periodic horizon surface\r
        s = horizonHeight(p, t);\r
        s = height - abs(s);            // cloud shell thickness\r
\r
        // Adaptive ray step\r
        z += d = uStepBase + max(s, -s * 0.2) / uStepScale;\r
\r
        // Colour with per-channel separation for richer sunsets\r
        vec4 phaseR = vec4(3.0, 4.0, 5.0, 0.0) + uHueShift + uHueSpeed * t;\r
        vec4 phaseG = phaseR + uColorSep;\r
        vec4 phaseB = phaseR - uColorSep;\r
\r
        float envelope = exp(s / 0.1) / d;\r
        float density  = uCloudDensity * envelope;\r
\r
        float fogAtt = exp(-z * uFogFalloff);\r
\r
        vec4 cR = (cos(s / 0.07 + p.x + 0.5 * t - phaseR) + 1.5) * density;\r
        vec4 cG = (cos(s / 0.07 + p.x + 0.5 * t - phaseG) + 1.5) * density;\r
        vec4 cB = (cos(s / 0.07 + p.x + 0.5 * t - phaseB) + 1.5) * density;\r
\r
        O.r += cR.r * fogAtt;\r
        O.g += cG.g * fogAtt;\r
        O.b += cB.b * fogAtt;\r
    }\r
\r
    O = tanh(O * O / 4e8);\r
    outColor = vec4(O.rgb * uIntensity, 1.0);\r
}\r
`,ir=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uZoom;
uniform float uTimeScale;
uniform float uTwist;
uniform float uWarp;
uniform float uPulse;
uniform float uIterLimit;
uniform float uGlow;
uniform float uOffsetX;
uniform float uOffsetY;
uniform float uColorShift;

void main() {
  vec2 frag = gl_FragCoord.xy;
  vec2 res = uResolution.xy;
  vec2 uv = frag;
  vec2 v = res;
  vec2 offset = vec2(uOffsetX, uOffsetY) * res;

  uv = uZoom * (uv + uv - v + offset) / v.y;

  vec4 z = vec4(1.0, 2.0, 3.0, 0.0);
  vec4 o = z;
  float a = 0.5;
  float t = uTime * uTimeScale;

  for (int i = 0; i < 19; ++i) {
    float fi = float(i) + 1.0;
    float mask = step(fi, uIterLimit);
    float denom = length(
      (1.0 + fi * dot(v, v))
        * sin(1.5 * uv / (0.5 - dot(uv, uv)) - uTwist * 9.0 * uv.yx + t)
    );
    o += mask * (1.0 + cos(z + t + uColorShift)) / max(1e-3, denom);

    a += 0.03;
    float ap = pow(a, fi);
    t += 1.0;
    v = cos(t - uPulse * 7.0 * uv * ap) - 5.0 * uv;

    uv *= mat2(cos(fi + 0.02 * t - vec4(0.0, 11.0, 33.0, 0.0)));
    vec2 warp = tanh(uWarp * 40.0 * dot(uv, uv) * cos(100.0 * uv.yx + t)) / 200.0;
    uv += warp
      + 0.2 * a * uv
      + cos(4.0 / exp(dot(o, o) / 100.0) + t) / 300.0;
  }

  vec4 mapped = (25.6 * uGlow) / (min(o, 13.0) + 164.0 / o);
  mapped -= dot(uv, uv) / 250.0;
  outColor = vec4(mapped.rgb, 1.0);
}
`,lr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2  uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform float uCloudScale;\r
uniform float uCloudSpeed;\r
uniform float uCloudDensity;\r
uniform float uCloudDetail;\r
uniform float uBoltLengthMin;\r
uniform float uBoltLengthMax;\r
uniform float uBoltWidth;\r
uniform float uBoltWiggle;\r
uniform float uBoltNoiseScale;\r
uniform float uBoltNoiseSpeed;\r
uniform float uBoltBranching;\r
uniform float uBoltIntensity;\r
uniform float uFlickerSpeed;\r
uniform float uCloudIllumination;\r
uniform float uSeed;\r
uniform int   uBoltCount;\r
uniform int   uNoiseOctaves;\r
uniform vec3  uCloudColor;\r
uniform vec3  uLightningColor;\r
\r
const float TAU = 6.28318530718;\r
\r
float hash(vec2 point) {\r
  vec3 hashed = fract(vec3(point.xyx) * 0.1031);\r
  hashed += dot(hashed, hashed.yzx + 33.33);\r
  return fract((hashed.x + hashed.y) * hashed.z);\r
}\r
\r
float noise(vec2 point) {\r
  vec2 cell = floor(point);\r
  vec2 blend = fract(point);\r
  blend = blend * blend * (3.0 - 2.0 * blend);\r
  return mix(mix(hash(cell), hash(cell + vec2(1.0, 0.0)), blend.x),\r
             mix(hash(cell + vec2(0.0, 1.0)), hash(cell + 1.0), blend.x), blend.y);\r
}\r
\r
float fbm(vec2 point, int octaves) {\r
  float value = 0.0;\r
  float amplitude = 0.5;\r
  float total = 0.0;\r
  for (int octave = 0; octave < 8; octave++) {\r
    if (octave >= octaves) break;\r
    value += noise(point) * amplitude;\r
    total += amplitude;\r
    point = mat2(1.6, 1.2, -1.2, 1.6) * point + vec2(1.7, 9.2);\r
    amplitude *= 0.5;\r
  }\r
  return value / max(total, 0.001);\r
}\r
\r
float segmentDistance(vec2 point, vec2 start, vec2 end) {\r
  vec2 direction = end - start;\r
  float along = clamp(dot(point - start, direction) / max(dot(direction, direction), 1e-10), 0.0, 1.0);\r
  return length(point - start - direction * along);\r
}\r
\r
float jaggedNoise(float position, float seed) {\r
  float cell = floor(position);\r
  return mix(hash(vec2(cell, seed)), hash(vec2(cell + 1.0, seed)), fract(position)) * 2.0 - 1.0;\r
}\r
\r
vec2 channelPoint(vec2 start, vec2 end, float progress, float seed) {\r
  vec2 direction = end - start;\r
  vec2 perpendicular = vec2(-direction.y, direction.x);\r
  float frequency = uBoltNoiseScale;\r
  float displacement = jaggedNoise(progress * 3.0, seed) * 0.55\r
                     + jaggedNoise(progress * frequency, seed + 13.0) * 0.30\r
                     + jaggedNoise(progress * frequency * 2.7, seed + 29.0) * 0.15;\r
  float envelope = pow(max(sin(progress * 3.14159265), 0.0), 0.45);\r
  return mix(start, end, progress) + perpendicular * displacement * uBoltWiggle * 4.0 * envelope;\r
}\r
\r
vec3 channelLight(float distanceToChannel, float width, float energy, float pixelSize) {\r
  float core = (1.0 - smoothstep(width * 0.35, width + pixelSize, distanceToChannel))\r
             * width / max(width, pixelSize * 0.65);\r
  float corona = exp(-distanceToChannel / max(width * 3.0, pixelSize)) * 0.32;\r
  float bloom = exp(-distanceToChannel / max(width * 18.0, pixelSize * 2.0)) * 0.035;\r
  vec3 hotColor = mix(uLightningColor, vec3(max(max(uLightningColor.r, uLightningColor.g), uLightningColor.b)), 0.88);\r
  return energy * (hotColor * core * 4.5 + uLightningColor * (corona + bloom));\r
}\r
\r
vec3 discharge(vec2 point, vec2 start, vec2 end, float seed, float growth,\r
               float energy, float branchEnergy, float pixelSize, out float channelDistance) {\r
  channelDistance = 100.0;\r
  float nearestProgress = 0.0;\r
  vec2 previous = start;\r
  for (int segment = 1; segment <= 8; segment++) {\r
    float progress = min(float(segment) / 8.0, growth);\r
    vec2 current = channelPoint(start, end, progress, seed);\r
    channelDistance = min(channelDistance, segmentDistance(point, previous, current));\r
    previous = current;\r
    if (float(segment) / 8.0 >= growth) break;\r
  }\r
\r
  vec3 light = vec3(0.0);\r
  vec2 direction = end - start;\r
  vec2 perpendicular = vec2(-direction.y, direction.x);\r
  float haloRadius = max(uBoltWidth * 120.0, pixelSize * 12.0);\r
  if (segmentDistance(point, start, end) < length(direction) * uBoltWiggle * 4.0 + haloRadius) {\r
    float coreDistance = 100.0;\r
    previous = start;\r
    for (int segment = 1; segment <= 64; segment++) {\r
      float progress = min(float(segment) / 64.0, growth);\r
      vec2 current = channelPoint(start, end, progress, seed);\r
      float distanceToSegment = segmentDistance(point, previous, current);\r
      if (distanceToSegment < coreDistance) {\r
        coreDistance = distanceToSegment;\r
        nearestProgress = progress;\r
      }\r
      previous = current;\r
      if (float(segment) / 64.0 >= growth) break;\r
    }\r
    float width = uBoltWidth * mix(1.15, 0.55, nearestProgress);\r
    light = channelLight(coreDistance, width, energy, pixelSize);\r
  }\r
\r
  for (int branch = 0; branch < 7; branch++) {\r
    float branchIndex = float(branch);\r
    float branchSeed = seed + branchIndex * 17.31 + 71.0;\r
    if (hash(vec2(branchSeed, 4.0)) >= uBoltBranching) continue;\r
    float attachment = 0.12 + branchIndex * 0.105 + hash(vec2(branchSeed, 5.0)) * 0.06;\r
    attachment = floor(attachment * 64.0) / 64.0;\r
    if (growth <= attachment) continue;\r
    float side = hash(vec2(branchSeed, 6.0)) < 0.5 ? -1.0 : 1.0;\r
    float extent = mix(0.16, 0.44, hash(vec2(branchSeed, 7.0))) * (1.0 - attachment * 0.5);\r
    vec2 branchStart = channelPoint(start, end, attachment, seed);\r
    vec2 branchEnd = branchStart + (direction + perpendicular * side * mix(0.5, 1.5, hash(vec2(branchSeed, 8.0)))) * extent;\r
    float branchGrowth = clamp((growth - attachment) / extent, 0.0, 1.0);\r
    vec2 forkStart = channelPoint(branchStart, branchEnd, 0.5, branchSeed);\r
    vec2 forkEnd = forkStart + (direction * 0.5 - perpendicular * side * 0.7) * extent * 0.5;\r
    float branchBound = length(branchEnd - branchStart) * uBoltWiggle * 4.0 + haloRadius;\r
    float forkBound = length(forkEnd - forkStart) * uBoltWiggle * 4.0 + haloRadius;\r
    if (segmentDistance(point, branchStart, branchEnd) > branchBound\r
      && segmentDistance(point, forkStart, forkEnd) > forkBound) continue;\r
    float branchDistance = 100.0;\r
    float branchProgress = 0.0;\r
    previous = branchStart;\r
    for (int segment = 1; segment <= 16; segment++) {\r
      float progress = min(float(segment) / 16.0, branchGrowth);\r
      vec2 current = channelPoint(branchStart, branchEnd, progress, branchSeed);\r
      float distanceToSegment = segmentDistance(point, previous, current);\r
      if (distanceToSegment < branchDistance) {\r
        branchDistance = distanceToSegment;\r
        branchProgress = progress;\r
      }\r
      previous = current;\r
      if (float(segment) / 16.0 >= branchGrowth) break;\r
    }\r
    float branchWidth = uBoltWidth * mix(0.55, 0.08, branchProgress);\r
    light += channelLight(branchDistance, branchWidth, branchEnergy * (1.0 - branchProgress * 0.75), pixelSize);\r
\r
    if (hash(vec2(branchSeed, 9.0)) < uBoltBranching && branchGrowth > 0.5) {\r
      float forkGrowth = clamp((branchGrowth - 0.5) * 2.5, 0.0, 1.0);\r
      float forkDistance = 100.0;\r
      previous = forkStart;\r
      for (int segment = 1; segment <= 8; segment++) {\r
        float progress = min(float(segment) / 8.0, forkGrowth);\r
        vec2 current = channelPoint(forkStart, forkEnd, progress, branchSeed + 53.0);\r
        forkDistance = min(forkDistance, segmentDistance(point, previous, current));\r
        previous = current;\r
        if (float(segment) / 8.0 >= forkGrowth) break;\r
      }\r
      light += channelLight(forkDistance, uBoltWidth * 0.18, branchEnergy * 0.22, pixelSize);\r
    }\r
  }\r
  return light;\r
}\r
\r
float strokePulse(float age, float onset, float decay) {\r
  float elapsed = age - onset;\r
  return smoothstep(0.0, 0.002, elapsed) * exp(-max(elapsed, 0.0) / decay);\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y * uZoom;\r
  float pixelSize = uZoom / uResolution.y;\r
  float aspect = uResolution.x / uResolution.y;\r
  float time = uTime * uTimeScale;\r
  vec2 cloudUV = point * uCloudScale + time * uCloudSpeed * vec2(0.03, 0.02);\r
  float cloudBase = fbm(cloudUV, uNoiseOctaves);\r
  float cloudDetail = fbm(cloudUV * 3.1 + cloudBase * 2.0, max(uNoiseOctaves - 1, 1));\r
  float cloudMass = mix(cloudBase, cloudBase * 0.65 + cloudDetail * 0.35, uCloudDetail);\r
  float clouds = smoothstep(0.68 - uCloudDensity * 0.48, 0.88 - uCloudDensity * 0.30, cloudMass);\r
  float relief = fbm(cloudUV + vec2(0.07, 0.11), max(uNoiseOctaves - 1, 1)) - cloudBase;\r
  vec3 color = uCloudColor * (0.10 + clouds * 0.65) * clamp(0.75 + relief * 3.0, 0.3, 1.2);\r
  vec3 lightning = vec3(0.0);\r
\r
  float rate = max(0.08, uFlickerSpeed * 0.6);\r
  float baseEvent = floor(time * rate);\r
  for (int lane = 0; lane < 12; lane++) {\r
    if (lane >= uBoltCount) break;\r
    float laneSeed = float(lane) * 19.19 + mod(uSeed, 10000.0) * 0.137;\r
    for (int candidate = 0; candidate < 4; candidate++) {\r
      float event = baseEvent - float(candidate);\r
      float eventSeed = hash(vec2(event, laneSeed + 11.0)) * 937.0;\r
      float eventTime = (event + hash(vec2(event, laneSeed + 23.0))) / rate;\r
      float age = time - eventTime;\r
      if (age < 0.0 || age > 0.52 || hash(vec2(event, laneSeed + 37.0)) < 0.52) continue;\r
\r
      float leaderDuration = mix(0.018, 0.045, hash(vec2(eventSeed, 1.0)));\r
      float strokeAge = age - leaderDuration;\r
      float spacing = mix(0.11, 0.042, clamp(uBoltNoiseSpeed / 8.0, 0.0, 1.0));\r
      float primary = strokePulse(strokeAge, 0.0, 0.018);\r
      float secondary = strokePulse(strokeAge, spacing, 0.012) * 0.65;\r
      float tertiary = strokePulse(strokeAge, spacing * 2.3, 0.016) * 0.4;\r
      secondary *= step(0.3, hash(vec2(eventSeed, 2.0)));\r
      tertiary *= step(0.55, hash(vec2(eventSeed, 3.0)));\r
      float afterglow = strokePulse(strokeAge, 0.0, 0.065) * 0.08;\r
      float leader = strokeAge < 0.0 ? 0.045 : 0.0;\r
      float flash = primary + secondary + tertiary + afterglow;\r
      float energy = (flash + leader) * uBoltIntensity * 4.0;\r
      if (energy < 0.001) continue;\r
\r
      vec2 start = (vec2(hash(vec2(eventSeed, 21.0)), hash(vec2(eventSeed, 25.0))) - 0.5)\r
                 * vec2(aspect * 0.85, 0.85);\r
      float channelLength = mix(min(uBoltLengthMin, uBoltLengthMax), max(uBoltLengthMin, uBoltLengthMax), hash(vec2(eventSeed, 37.0)));\r
      float angle = hash(vec2(eventSeed, 29.0)) * TAU;\r
      vec2 end = start + vec2(cos(angle), sin(angle)) * channelLength;\r
      float growth = strokeAge < 0.0 ? floor(clamp(age / leaderDuration, 0.0, 1.0) * 24.0) / 24.0 : 1.0;\r
      float branchEnergy = (primary + secondary * 0.22 + tertiary * 0.12 + afterglow + leader) * uBoltIntensity * 3.2;\r
      float channelDistance;\r
      vec3 channel = discharge(point, start, end, eventSeed, growth, energy, branchEnergy, pixelSize, channelDistance);\r
      float veil = smoothstep(0.40, 0.78, cloudMass) * 0.55;\r
      lightning += channel * (1.0 - veil);\r
      float scatter = exp(-channelDistance / (0.045 + clouds * 0.12));\r
      color += mix(uLightningColor, vec3(1.0), 0.3) * scatter * flash * uCloudIllumination\r
             * uBoltIntensity * (0.12 + clouds * 1.8) * (0.65 + cloudDetail * 0.7);\r
    }\r
  }\r
\r
  color += lightning;\r
  color = vec3(1.0) - exp(-color);\r
  outColor = vec4(pow(max(color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);\r
}`,sr=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uPower;
uniform float uBulbSpin;
uniform float uMaxRayLength;
uniform float uTolerance;
uniform float uNormOffset;
uniform float uInitStep;
uniform float uRotSpeedX;
uniform float uRotSpeedY;
uniform float uCamDistance;
uniform float uCamHeight;
uniform float uFov;
uniform float uSkyBoost;
uniform float uGlowBoost;
uniform float uGlowFalloff;
uniform float uDiffuseBoost;
uniform float uMatTransmit;
uniform float uMatReflect;
uniform float uRefractIndex;
uniform float uHueShift;
uniform float uGlowHueOffset;
uniform float uNebulaMix;
uniform float uNebulaHueShift;
uniform float uNebulaSat;
uniform float uNebulaVal;
uniform float uNebulaGlowHue;
uniform float uNebulaGlowBoost;
uniform float uSkySat;
uniform float uSkyVal;
uniform float uGlowSat;
uniform float uGlowVal;
uniform float uDiffuseSat;
uniform float uDiffuseVal;
uniform vec3 uBeerColor;
uniform vec3 uLightPos;
uniform int uLoops;
uniform int uRayMarches;
uniform int uBounces;

const float PI = 3.141592654;
const float TAU = 6.28318530718;

mat3 g_rot = mat3(1.0);

const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}

vec3 sRGB(vec3 t) {
  return mix(1.055 * pow(t, vec3(1.0 / 2.4)) - 0.055, 12.92 * t, step(t, vec3(0.0031308)));
}

vec3 aces_approx(vec3 v) {
  v = max(v, 0.0);
  v *= 0.6;
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0);
}

float boxSDF(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float rayPlane(vec3 ro, vec3 rd, vec4 p) {
  return -(dot(ro, p.xyz) + p.w) / dot(rd, p.xyz);
}

float mandelBulb(vec3 p, float time) {
  vec3 z = p;
  float r = 0.0;
  float dr = 1.0;

  for (int i = 0; i < 6; ++i) {
    if (i >= uLoops) {
      break;
    }
    r = length(z);
    if (r > 2.0) {
      break;
    }
    r = max(r, 1e-6);
    float theta = atan(z.y, z.x);
    float phi = asin(clamp(z.z / r, -1.0, 1.0)) + time * uBulbSpin;

    dr = pow(r, uPower - 1.0) * dr * uPower + 1.0;
    r = pow(r, uPower);
    theta *= uPower;
    phi *= uPower;
    z = r * vec3(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)) + p;
  }

  return 0.5 * log(max(r, 1e-6)) * r / dr;
}

mat3 rot_z(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      c, s, 0.0,
     -s, c, 0.0,
      0.0, 0.0, 1.0
    );
}

mat3 rot_y(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      c, 0.0, s,
      0.0, 1.0, 0.0,
     -s, 0.0, c
    );
}

mat3 rot_x(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      1.0, 0.0, 0.0,
      0.0, c, s,
      0.0, -s, c
    );
}

vec3 skyColor(vec3 ro, vec3 rd, vec3 skyCol) {
  vec3 col = clamp(vec3(0.0025 / abs(rd.y)) * skyCol, 0.0, 1.0);

  float tp0 = rayPlane(ro, rd, vec4(vec3(0.0, 1.0, 0.0), 4.0));
  float tp1 = rayPlane(ro, rd, vec4(vec3(0.0, -1.0, 0.0), 6.0));
  float tp = max(tp0, tp1);
  if (tp > 0.0) {
    vec3 pos = ro + tp * rd;
    vec2 pp = pos.xz;
    float db = boxSDF(pp, vec2(6.0, 9.0)) - 1.0;
    col += vec3(4.0) * skyCol * rd.y * rd.y * smoothstep(0.25, 0.0, db);
    col += vec3(0.8) * skyCol * exp(-0.5 * max(db, 0.0));
  }

  if (tp0 > 0.0) {
    vec3 pos = ro + tp0 * rd;
    vec2 pp = pos.xz;
    float ds = length(pp) - 0.5;
    col += vec3(0.25) * skyCol * exp(-0.5 * max(ds, 0.0));
  }

  return clamp(col, 0.0, 10.0);
}

float df(vec3 p, float time) {
  p *= g_rot;
  const float z1 = 2.0;
  return mandelBulb(p / z1, time) * z1;
}

vec3 normal(vec3 pos, float time) {
  vec2 eps = vec2(uNormOffset, 0.0);
  vec3 nor;
  nor.x = df(pos + eps.xyy, time) - df(pos - eps.xyy, time);
  nor.y = df(pos + eps.yxy, time) - df(pos - eps.yxy, time);
  nor.z = df(pos + eps.yyx, time) - df(pos - eps.yyx, time);
  return normalize(nor);
}

float rayMarch(vec3 ro, vec3 rd, float dfactor, float time, out int ii) {
  float t = 0.0;
  float tol = dfactor * uTolerance;
  ii = uRayMarches;
  for (int i = 0; i < 96; ++i) {
    if (i >= uRayMarches) {
      break;
    }
    if (t > uMaxRayLength) {
      t = uMaxRayLength;
      break;
    }
    float d = dfactor * df(ro + rd * t, time);
    if (d < tol) {
      ii = i;
      break;
    }
    t += d;
  }
  return t;
}

vec3 render(vec3 ro, vec3 rd, float time) {
  vec3 agg = vec3(0.0);
  vec3 ragg = vec3(1.0);

  bool isInside = df(ro, time) < 0.0;

  vec3 baseSky = hsv2rgb(vec3(uHueShift + 0.6, uSkySat, uSkyVal)) * uSkyBoost;
  vec3 baseGlow = hsv2rgb(vec3(uHueShift + uGlowHueOffset, uGlowSat, uGlowVal)) * uGlowBoost;
  vec3 baseDiffuse = hsv2rgb(vec3(uHueShift + 0.6, uDiffuseSat, uDiffuseVal)) * uDiffuseBoost;

  vec3 nebulaSky = hsv2rgb(vec3(uNebulaHueShift + 0.18, uNebulaSat, uNebulaVal)) * (uSkyBoost * 0.9);
  vec3 nebulaGlow = hsv2rgb(vec3(uNebulaGlowHue, uNebulaSat, uNebulaVal * 1.5)) * uNebulaGlowBoost;
  vec3 nebulaDiffuse = hsv2rgb(vec3(uNebulaHueShift + 0.55, uNebulaSat * 0.8, uNebulaVal)) * uDiffuseBoost;

  float nebulaMix = clamp(uNebulaMix, 0.0, 1.0);
  vec3 skyCol = mix(baseSky, nebulaSky, nebulaMix);
  vec3 glowCol = mix(baseGlow, nebulaGlow, nebulaMix);
  vec3 diffuseCol = mix(baseDiffuse, nebulaDiffuse, nebulaMix);

  for (int bounce = 0; bounce < 5; ++bounce) {
    if (bounce >= uBounces) {
      break;
    }
    float dfactor = isInside ? -1.0 : 1.0;
    float mragg = max(max(ragg.x, ragg.y), ragg.z);
    if (mragg < 0.025) {
      break;
    }
    int iter;
    float st = rayMarch(ro, rd, dfactor, time, iter);
    if (st >= uMaxRayLength) {
      agg += ragg * skyColor(ro, rd, skyCol);
      break;
    }

    vec3 sp = ro + rd * st;
    vec3 sn = dfactor * normal(sp, time);

    float fre = 1.0 + dot(rd, sn);
    fre *= fre;
    fre = mix(0.1, 1.0, fre);

    vec3 ld = normalize(uLightPos - sp);
    float dif = max(dot(ld, sn), 0.0);
    vec3 ref = reflect(rd, sn);
    float re = uRefractIndex;
    float ire = 1.0 / re;
    vec3 refr = refract(rd, sn, !isInside ? re : ire);
    vec3 rsky = skyColor(sp, ref, skyCol);

    vec3 col = vec3(0.0);
    col += diffuseCol * dif * dif * (1.0 - uMatTransmit);
    float edge = smoothstep(1.0, 0.9, fre);
    col += rsky * uMatReflect * edge;
    col += glowCol * exp(-float(iter) * uGlowFalloff);

    if (isInside) {
      ragg *= exp(-(st + uInitStep) * uBeerColor);
    }
    agg += ragg * col;

    if (refr == vec3(0.0)) {
      rd = ref;
    } else {
      ragg *= uMatTransmit;
      isInside = !isInside;
      rd = refr;
    }
    ro = sp + uInitStep * rd;
  }

  return agg;
}

vec3 effect(vec2 p, float time) {
  g_rot = rot_x(uRotSpeedX * time) * rot_y(uRotSpeedY * time);
  vec3 ro = vec3(0.0, uCamHeight, uCamDistance);
  const vec3 la = vec3(0.0);
  const vec3 up = vec3(0.0, 1.0, 0.0);

  vec3 ww = normalize(la - ro);
  vec3 uu = normalize(cross(up, ww));
  vec3 vv = cross(ww, uu);
  float fov = tan(uFov);
  vec3 rd = normalize(-p.x * uu + p.y * vv + fov * ww);

  return render(ro, rd, time);
}

void main() {
  vec2 q = gl_FragCoord.xy / uResolution.xy;
  vec2 p = -1.0 + 2.0 * q;
  p.x *= uResolution.x / uResolution.y;
  float time = uTime * uTimeScale;
  vec3 col = effect(p, time);
  col = aces_approx(col);
  col = sRGB(col);
  outColor = vec4(col, 1.0);
}
`,ur=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uAuroraSpeed;\r
uniform float uAuroraScale;\r
uniform float uAuroraWarp;\r
uniform float uAuroraBase;\r
uniform float uAuroraStride;\r
uniform float uAuroraCurve;\r
uniform float uAuroraIntensity;\r
uniform float uTrailBlend;\r
uniform float uTrailFalloff;\r
uniform float uTrailFade;\r
uniform float uDitherStrength;\r
uniform float uHorizonFade;\r
uniform float uCamYaw;\r
uniform float uCamPitch;\r
uniform float uCamWobble;\r
uniform float uCamDistance;\r
uniform float uCamHeight;\r
uniform float uSkyStrength;\r
uniform float uStarDensity;\r
uniform float uStarIntensity;\r
uniform float uReflectionStrength;\r
uniform float uReflectionTint;\r
uniform float uReflectionFog;\r
uniform float uColorBand;\r
uniform float uColorSpeed;\r
uniform vec3 uAuroraColorA;\r
uniform vec3 uAuroraColorB;\r
uniform vec3 uAuroraColorC;\r
uniform vec3 uBgColorA;\r
uniform vec3 uBgColorB;\r
uniform int uAuroraSteps;\r
uniform float uVortexStrength;\r
uniform float uVortexGridScale;\r
uniform float uVortexRadius;\r
uniform float uVortexWobble;\r
uniform float uVortexWobbleSpeed;\r
uniform float uVortexSpin;\r
uniform float uVortexDesync;\r
uniform float uVortexDrift;\r
uniform float uVortexColorShift;\r
uniform float uOilStrength;\r
uniform float uOilScale;\r
uniform float uOilSpeed;\r
uniform float uOilContrast;\r
\r
const float TAU = 6.28318530718;\r
const float VORTEX_FADE_START = 1.0;\r
const float VORTEX_FADE_END = 1.5;\r
\r
mat2 mm2(float a) {\r
  float c = cos(a);\r
  float s = sin(a);\r
  return mat2(c, s, -s, c);\r
}\r
\r
mat2 m2 = mat2(0.95534, 0.29552, -0.29552, 0.95534);\r
\r
float tri(float x) {\r
  return clamp(abs(fract(x) - 0.5), 0.01, 0.49);\r
}\r
\r
vec2 tri2(vec2 p) {\r
  return vec2(tri(p.x) + tri(p.y), tri(p.y + tri(p.x)));\r
}\r
\r
float triNoise2d(vec2 p, float spd, float time) {\r
  float z = 1.8;\r
  float z2 = 2.5;\r
  float rz = 0.0;\r
  p *= mm2(p.x * 0.06);\r
  vec2 bp = p;\r
  for (float i = 0.0; i < 5.0; i++) {\r
    vec2 dg = tri2(bp * 1.85) * 0.75;\r
    dg *= mm2(time * spd);\r
    p -= dg / z2;\r
\r
    bp *= 1.3;\r
    z2 *= 0.45;\r
    z *= 0.42;\r
    p *= 1.21 + (rz - 1.0) * 0.02;\r
\r
    rz += tri(p.x + tri(p.y)) * z;\r
    p *= -m2;\r
  }\r
  return clamp(1.0 / pow(rz * 29.0, 1.3), 0.0, 0.55);\r
}\r
\r
float hash21(vec2 n) {\r
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);\r
}\r
\r
vec4 aurora(vec3 ro, vec3 rd, float time) {\r
  vec4 col = vec4(0.0);\r
  vec4 avgCol = vec4(0.0);\r
  int steps = max(uAuroraSteps, 1);\r
\r
  for (int i = 0; i < 64; i++) {\r
    if (i >= steps) {\r
      break;\r
    }\r
    float fi = float(i);\r
    float of = uDitherStrength * hash21(gl_FragCoord.xy) * smoothstep(0.0, 15.0, fi);\r
    float pt = ((uAuroraBase + pow(fi, uAuroraCurve) * uAuroraStride) - ro.y) / (rd.y * 2.0 + 0.4);\r
    pt -= of;\r
    vec3 bpos = ro + pt * rd;\r
    vec2 p = bpos.zx;\r
    float rzt = triNoise2d(p * uAuroraScale, uAuroraSpeed, time);\r
    rzt = mix(rzt, pow(rzt, 1.0 + uAuroraWarp), uAuroraWarp);\r
\r
    vec3 wave = sin(vec3(0.0, 2.1, 4.2) + fi * uColorBand + time * uColorSpeed);\r
    vec3 palette = mix(uAuroraColorA, uAuroraColorB, 0.5 + 0.5 * wave);\r
    palette = mix(palette, uAuroraColorC, rzt);\r
\r
    vec4 col2 = vec4(palette * rzt * uAuroraIntensity, rzt);\r
    avgCol = mix(avgCol, col2, uTrailBlend);\r
    col += avgCol * exp2(-fi * uTrailFalloff - uTrailFade) * smoothstep(0.0, 5.0, fi);\r
  }\r
\r
  col *= clamp(rd.y * 15.0 + 0.4, 0.0, 1.0);\r
  return col;\r
}\r
\r
vec3 nmzHash33(vec3 q) {\r
  uvec3 p = uvec3(ivec3(q));\r
  p = p * uvec3(374761393U, 1103515245U, 668265263U) + p.zxy + p.yzx;\r
  p = p.yzx * (p.zxy ^ (p >> 3U));\r
  return vec3(p ^ (p >> 16U)) * (1.0 / vec3(0xffffffffU));\r
}\r
\r
vec3 stars(vec3 p) {\r
  vec3 c = vec3(0.0);\r
  float res = uResolution.x * 1.0;\r
\r
  for (float i = 0.0; i < 4.0; i++) {\r
    vec3 q = fract(p * (0.15 * res)) - 0.5;\r
    vec3 id = floor(p * (0.15 * res));\r
    vec2 rn = nmzHash33(id).xy;\r
    float c2 = 1.0 - smoothstep(0.0, 0.6, length(q));\r
    c2 *= step(rn.x, uStarDensity + i * i * 0.001);\r
    c += c2 * (mix(vec3(1.0, 0.49, 0.1), vec3(0.75, 0.9, 1.0), rn.y) * 0.1 + 0.9);\r
    p *= 1.3;\r
  }\r
  return c * c * uStarIntensity;\r
}\r
\r
vec2 hash22(vec2 p) {\r
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p3 += dot(p3, p3.yzx + 33.33);\r
  return fract((p3.xx + p3.yz) * p3.zy);\r
}\r
\r
// Loose wobbly grid of out-of-sync vortices. Warps \`p\` in place and\r
// returns the accumulated signed swirl amount for color effects.\r
float vortexField(inout vec2 p, float time) {\r
  float grid = max(uVortexGridScale, 0.01);\r
  vec2 gp = p * grid;\r
  vec2 cell = floor(gp);\r
  float swirlSum = 0.0;\r
  vec2 warped = p;\r
  for (int oy = -1; oy <= 1; oy++) {\r
    for (int ox = -1; ox <= 1; ox++) {\r
      vec2 id = cell + vec2(float(ox), float(oy));\r
      vec2 rnd = hash22(id + 17.31);\r
      // Each vortex wobbles around its jittered lattice point, out of sync.\r
      float phase = rnd.x * TAU * uVortexDesync;\r
      vec2 wobble = vec2(\r
        sin(time * uVortexWobbleSpeed * (0.6 + rnd.x * 0.8) + phase),\r
        cos(time * uVortexWobbleSpeed * (0.5 + rnd.y * 0.9) + phase * 1.7)\r
      ) * uVortexWobble;\r
      vec2 center = (id + 0.5 + (rnd - 0.5) * 0.8 + wobble) / grid;\r
      vec2 d = p - center;\r
      float dist = length(d);\r
      float radius = max(uVortexRadius, 0.01) / grid;\r
      vec2 cellDistance = abs(gp - (id + 0.5));\r
      vec2 cellFade = 1.0 - smoothstep(vec2(VORTEX_FADE_START), vec2(VORTEX_FADE_END), cellDistance);\r
      float falloff = exp(-(dist * dist) / (radius * radius)) * cellFade.x * cellFade.y;\r
      float dir = rnd.y > 0.5 ? 1.0 : -1.0;\r
      float spin = sin(time * uVortexSpin * (0.7 + rnd.y * 0.6) + phase) * 0.5 + 0.75;\r
      float angle = uVortexStrength * dir * spin * falloff;\r
      angle += uVortexDrift * dir * falloff;\r
      warped = center + mm2(angle) * (warped - center);\r
      swirlSum += angle;\r
    }\r
  }\r
  p = warped;\r
  return swirlSum;\r
}\r
\r
// Thin-film "oil on water" interference rainbow driven by film thickness.\r
vec3 oilFilm(vec2 p, float swirl, float time) {\r
  float thickness = triNoise2d(p * uOilScale + vec2(time * uOilSpeed * 0.1, -time * uOilSpeed * 0.07), 0.03, time);\r
  thickness = thickness * 4.0 + swirl * 1.5;\r
  vec3 rainbow = 0.5 + 0.5 * cos(TAU * (thickness * vec3(1.0, 1.35, 1.8) + vec3(0.0, 0.33, 0.67)));\r
  return pow(rainbow, vec3(max(uOilContrast, 0.1)));\r
}\r
\r
vec3 bg(vec3 rd) {\r
  float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd) * 0.5 + 0.5;\r
  sd = pow(sd, 5.0);\r
  vec3 col = mix(uBgColorA, uBgColorB, sd);\r
  return col * uSkyStrength;\r
}\r
\r
void main() {\r
  vec2 q = gl_FragCoord.xy / uResolution.xy;\r
  vec2 p = q - 0.5;\r
  p.x *= uResolution.x / uResolution.y;\r
\r
  float time = uTime * uTimeScale;\r
\r
  float swirl = vortexField(p, time);\r
\r
  vec3 ro = vec3(0.0, uCamHeight, -uCamDistance);\r
  vec3 rd = normalize(vec3(p, 1.3));\r
  rd.yz *= mm2(uCamPitch + sin(time * 0.05) * uCamWobble);\r
  rd.xz *= mm2(uCamYaw + sin(time * 0.05) * uCamWobble);\r
\r
  vec3 col = vec3(0.0);\r
  float fade = smoothstep(0.0, uHorizonFade, abs(rd.y)) * 0.1 + 0.9;\r
  col = bg(rd) * fade;\r
\r
  if (rd.y > 0.0) {\r
    vec4 aur = smoothstep(0.0, 1.5, aurora(ro, rd, time)) * fade;\r
    col += stars(rd);\r
    col = col * (1.0 - aur.a) + aur.rgb;\r
  } else {\r
    rd.y = abs(rd.y);\r
    col = bg(rd) * fade * uReflectionStrength;\r
    vec4 aur = smoothstep(0.0, 2.5, aurora(ro, rd, time));\r
    col += stars(rd) * 0.1;\r
    col = col * (1.0 - aur.a) + aur.rgb;\r
    vec3 pos = ro + ((0.5 - ro.y) / rd.y) * rd;\r
    float nz2 = triNoise2d(pos.xz * vec2(0.5, 0.7), 0.0, time);\r
    vec3 waterTint = mix(vec3(0.2, 0.25, 0.5) * 0.08, vec3(0.3, 0.3, 0.5) * 0.7, nz2 * 0.4);\r
    col += waterTint * uReflectionTint;\r
    col *= mix(1.0, exp(-abs(rd.y) * uReflectionFog), uReflectionStrength);\r
  }\r
\r
  // Twist the colors in the vortex shapes, then wash them with an\r
  // iridescent oil-in-water interference film.\r
  if (abs(uVortexColorShift) > 0.0001) {\r
    float hueTwist = swirl * uVortexColorShift;\r
    vec3 twist = vec3(\r
      dot(col, vec3(0.6, 0.3, 0.1)),\r
      dot(col, vec3(0.1, 0.6, 0.3)),\r
      dot(col, vec3(0.3, 0.1, 0.6))\r
    );\r
    col = mix(col, twist, clamp(abs(hueTwist), 0.0, 1.0));\r
    col.rb = mm2(hueTwist) * col.rb;\r
    col = abs(col);\r
  }\r
  if (uOilStrength > 0.0001) {\r
    vec3 film = oilFilm(p, swirl, time);\r
    float lum = clamp(dot(col, vec3(0.299, 0.587, 0.114)) * 1.6, 0.0, 1.0);\r
    col = mix(col, col * (0.35 + 1.3 * film) + film * lum * 0.55, uOilStrength);\r
  }\r
\r
  outColor = vec4(col, 1.0);\r
}\r
`,cr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
// --- UNIFORMS ---\r
uniform float uTime;\r
uniform vec2 uResolution;\r
uniform float uZoom;\r
uniform float uColorShift;\r
uniform int uIterations;\r
uniform float uDistort;\r
uniform float uRotateSpeed;\r
uniform float uMaxSteps;\r
\r
// --- MATH HELPERS ---\r
\r
// Rotation Matrix\r
mat2 rot(float a) {\r
    float s = sin(a), c = cos(a);\r
    return mat2(c, -s, s, c);\r
}\r
\r
// Palette function for coloring (IQ style)\r
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {\r
    return a + b * cos(6.28318 * (c * t + d));\r
}\r
\r
// The SDF (Signed Distance Function) - The Core Math\r
// This function returns the distance from point 'p' to the fractal surface\r
float map(vec3 p, float time, float distort) {\r
    float scale = 1.0;\r
    float offset = 1.0;\r
    \r
    // Recursive Folding Loop - Menger-like fractal\r
    for (int i = 0; i < 8; i++) {\r
        if (i >= uIterations) break;\r
        \r
        // Rotate space\r
        p.xy *= rot(time);\r
        p.yz *= rot(time * 0.7);\r
        \r
        // Folding - creates symmetry\r
        p = abs(p);\r
        \r
        // Menger fold\r
        if (p.x < p.y) p.xy = p.yx;\r
        if (p.x < p.z) p.xz = p.zx;\r
        if (p.y < p.z) p.yz = p.zy;\r
        \r
        // Scale and translate\r
        p = p * distort - offset * (distort - 1.0);\r
        scale *= distort;\r
    }\r
    \r
    // Return distance to a box, scaled back\r
    float d = (length(p) - 1.5) / scale;\r
    return d;\r
}\r
\r
void main() {\r
    vec2 fragCoord = gl_FragCoord.xy;\r
    \r
    // 1. Setup Camera\r
    vec2 uv = (fragCoord - uResolution * 0.5) / min(uResolution.x, uResolution.y);\r
    uv *= uZoom;\r
\r
    vec3 ro = vec3(0.0, 0.0, -3.0); // Ray Origin\r
    vec3 rd = normalize(vec3(uv, 1.0)); // Ray Direction\r
\r
    float time = uTime * uRotateSpeed;\r
\r
    // 2. Raymarching Loop\r
    float t = 0.0; // Total distance traveled\r
    float d = 0.0; // Distance to surface\r
    int maxSteps = int(uMaxSteps);\r
\r
    vec3 col = vec3(0.0);\r
    vec3 p = ro;\r
    float glow = 0.0;\r
\r
    for (int i = 0; i < 200; i++) {\r
        if (i >= maxSteps) break;\r
\r
        p = ro + rd * t;\r
        d = map(p, time, uDistort); // Get distance to fractal\r
\r
        // Accumulate glow based on proximity\r
        glow += 0.02 / (0.1 + abs(d));\r
\r
        // If we hit the surface\r
        if (abs(d) < 0.001) {\r
            // Calculate Normal\r
            vec2 e = vec2(0.001, 0.0);\r
            vec3 n = normalize(vec3(\r
                map(p + e.xyy, time, uDistort) - map(p - e.xyy, time, uDistort),\r
                map(p + e.yxy, time, uDistort) - map(p - e.yxy, time, uDistort),\r
                map(p + e.yyx, time, uDistort) - map(p - e.yyx, time, uDistort)\r
            ));\r
\r
            // Lighting\r
            vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));\r
            float diff = max(dot(n, lightDir), 0.0);\r
            float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 16.0);\r
\r
            // Coloring based on position and normal\r
            float fresnel = pow(1.0 + dot(rd, n), 3.0);\r
            \r
            // Dynamic Palette\r
            vec3 paletteColor = palette(\r
                length(p) * 0.4 + uTime * 0.1 + uColorShift, \r
                vec3(0.5), \r
                vec3(0.5), \r
                vec3(1.0), \r
                vec3(0.263, 0.416, 0.557)\r
            );\r
\r
            col = paletteColor * (diff * 0.8 + 0.2) + vec3(1.0) * spec * 0.5;\r
            col = mix(col, vec3(1.0), fresnel * 0.3);\r
            break;\r
        }\r
\r
        // Move ray forward\r
        t += d * 0.5; // Use smaller steps for safety\r
        \r
        // Stop if too far\r
        if (t > 20.0) break;\r
    }\r
\r
    // Add glow effect for missed rays\r
    col += glow * 0.02 * palette(\r
        uTime * 0.05 + uColorShift,\r
        vec3(0.5), \r
        vec3(0.5), \r
        vec3(1.0), \r
        vec3(0.263, 0.416, 0.557)\r
    );\r
\r
    // 3. Post-Processing (Vignette)\r
    vec2 vUv = fragCoord / uResolution;\r
    col *= 1.0 - length(vUv - 0.5) * 0.5;\r
\r
    outColor = vec4(col, 1.0);\r
}\r
`,fr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2  uResolution;\r
uniform float uPhase;\r
\r
// Structure\r
uniform int   uOrder;       // Hadamard order: matrix size = 2^order\r
uniform float uRotSpeed;    // rotations per loop cycle\r
uniform float uZoom;        // disk scale\r
uniform float uRadialPow;   // power curve on radial mapping\r
uniform float uSpiral;      // spiral twist amount\r
\r
// Cell appearance\r
uniform float uSmooth;      // smoothing between cells (0 = crisp)\r
uniform float uGap;         // visible gap between cells\r
uniform float uFadeStart;   // edge fade begin (fraction of radius)\r
uniform float uFadeWidth;   // edge fade extent\r
uniform float uPulse;       // radial breathing amount\r
\r
// Color\r
uniform float uBaseR;\r
uniform float uBaseG;\r
uniform float uBaseB;\r
uniform float uAmpR;        // red modulation amplitude (cos)\r
uniform float uAmpG;        // green modulation amplitude (sin)\r
uniform float uAmpB;        // blue modulation amplitude (sin)\r
uniform float uFreqR;       // red oscillation cycles per loop\r
uniform float uFreqG;       // green oscillation cycles per loop\r
uniform float uFreqB;       // blue oscillation cycles per loop\r
\r
// Extra\r
uniform float uGlow;        // glow halo intensity\r
uniform float uBgBright;    // background brightness\r
uniform float uSeed;        // random angular offset\r
\r
#define PI  3.14159265359\r
#define TAU 6.28318530718\r
\r
/* ── Hadamard value (Sylvester / natural order) ────────────────\r
   H[row][col] = (-1)^popcount(row & col)\r
   Returns 1.0 for +1 entries, 0.0 for -1 entries. */\r
int popcount(int x) {\r
    int c = 0;\r
    int v = x;\r
    for (int i = 0; i < 16; i++) {\r
        c += v & 1;\r
        v >>= 1;\r
        if (v == 0) break;\r
    }\r
    return c;\r
}\r
\r
float hadamard(int row, int col) {\r
    return float(1 - (popcount(row & col) & 1));\r
}\r
\r
/* Simple hash for seed-based offset */\r
float hash(float n) {\r
    return fract(sin(n * 127.1) * 43758.5453);\r
}\r
\r
void main() {\r
    float minDim = min(uResolution.x, uResolution.y);\r
    vec2 uv = (2.0 * gl_FragCoord.xy - uResolution) / minDim;\r
\r
    // Zoom\r
    uv /= max(uZoom, 0.01);\r
\r
    float r     = length(uv);\r
    float theta = atan(uv.y, uv.x);\r
    float t     = uPhase;                       // 0 → 1 per loop\r
\r
    // Seed-based static angular offset\r
    theta += hash(uSeed * 13.37) * TAU;\r
\r
    // Rotation\r
    theta += t * uRotSpeed * TAU;\r
\r
    // Spiral warp: angle shifts proportionally to radius\r
    theta += uSpiral * r * TAU;\r
\r
    // Radial pulse (breathing)\r
    float rAdj = r + uPulse * 0.08 * sin(t * TAU);\r
\r
    // Matrix dimensions\r
    int   size  = 1 << clamp(uOrder, 1, 10);\r
    float fSize = float(size);\r
\r
    // ── Map polar coords → matrix indices ─────────────────────\r
    // Radius  → row  (center = row 0, edge = last row)\r
    float rN   = clamp(rAdj, 0.0, 0.9999);\r
    rN         = pow(rN, uRadialPow);\r
    float rowF = rN * fSize;\r
    int   row  = clamp(int(floor(rowF)), 0, size - 1);\r
\r
    // Angle → column (wraps)\r
    float aN   = fract(theta / TAU);\r
    float colF = aN * fSize;\r
    int   col  = clamp(int(floor(colF)), 0, size - 1);\r
\r
    // ── Compute cell value ────────────────────────────────────\r
    float v;\r
    if (uSmooth > 0.001) {\r
        // Bilinear interpolation between neighbouring cells\r
        int row2 = min(row + 1, size - 1);\r
        int col2 = (col + 1) % size;          // wrap angularly\r
\r
        float fr = fract(rowF);\r
        float fc = fract(colF);\r
\r
        // Smoothstep the fractional parts for a tuneable blend\r
        float sf = uSmooth;\r
        fr = smoothstep(0.5 - 0.5 * sf, 0.5 + 0.5 * sf, fr);\r
        fc = smoothstep(0.5 - 0.5 * sf, 0.5 + 0.5 * sf, fc);\r
\r
        v = mix(\r
            mix(hadamard(row, col),  hadamard(row2, col),  fr),\r
            mix(hadamard(row, col2), hadamard(row2, col2), fr),\r
            fc\r
        );\r
    } else {\r
        v = hadamard(row, col);\r
    }\r
\r
    // ── Cell gap ──────────────────────────────────────────────\r
    if (uGap > 0.001) {\r
        float edgeR = min(fract(rowF), 1.0 - fract(rowF));\r
        float edgeC = min(fract(colF), 1.0 - fract(colF));\r
        float edge  = min(edgeR, edgeC);\r
        v *= smoothstep(0.0, uGap * 0.5 + 0.005, edge);\r
    }\r
\r
    // ── Time-varying colour (matches original Java formula) ───\r
    float cr = (uBaseR + uAmpR * cos(uFreqR * TAU * t)) * v;\r
    float cg = (uBaseG + uAmpG * sin(uFreqG * TAU * t)) * v;\r
    float cb = (uBaseB + uAmpB * sin(uFreqB * TAU * t)) * v;\r
\r
    // ── Edge fade ─────────────────────────────────────────────\r
    float fadeEnd = uFadeStart + max(uFadeWidth, 0.001);\r
    float alpha   = smoothstep(fadeEnd, uFadeStart, r);\r
\r
    // ── Glow halo around disk edge ────────────────────────────\r
    vec3 glowCol = vec3(\r
        uBaseR + uAmpR * cos(uFreqR * TAU * t),\r
        uBaseG + uAmpG * sin(uFreqG * TAU * t),\r
        uBaseB + uAmpB * sin(uFreqB * TAU * t)\r
    );\r
    float glowFactor = uGlow * 0.35 * exp(-10.0 * max(r - uFadeStart, 0.0));\r
\r
    // ── Compose ───────────────────────────────────────────────\r
    vec3 fg    = clamp(vec3(cr, cg, cb), 0.0, 1.0);\r
    vec3 bg    = vec3(uBgBright);\r
    vec3 color = mix(bg, fg, alpha) + glowFactor * glowCol;\r
\r
    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);\r
}\r
`,mr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uSeed;\r
\r
uniform int uFractalIters;\r
uniform float uFoldScale;\r
uniform float uFoldOffset;\r
uniform float uRotSpeed;\r
\r
uniform float uDetailLevel;\r
\r
uniform float uLightIntensity;\r
\r
uniform float uHueShift;\r
uniform float uHueSpeed;\r
uniform float uSaturation;\r
uniform float uBrightness;\r
uniform float uContrast;\r
uniform float uGlowIntensity;\r
uniform float uChromaShift;\r
uniform float uSmoothBlend;\r
\r
uniform float uZoom;\r
uniform float uCamHeight;\r
uniform float uCamOrbit;\r
\r
const float TAU = 6.28318530718;\r
const vec2 HASH_SCALE = vec2(234.34, 435.45);\r
const float HASH_BIAS = 34.23;\r
\r
mat2 rot2(float a) {\r
    float s = sin(a);\r
    float c = cos(a);\r
    return mat2(c, -s, s, c);\r
}\r
\r
float hash21(vec2 p) {\r
    p = fract(p * HASH_SCALE);\r
    p += dot(p, p + HASH_BIAS + fract(uSeed * 0.000001));\r
    return fract(p.x * p.y);\r
}\r
\r
float noise(vec2 p) {\r
    vec2 i = floor(p);\r
    vec2 f = fract(p);\r
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);\r
\r
    float a = hash21(i);\r
    float b = hash21(i + vec2(1.0, 0.0));\r
    float c = hash21(i + vec2(0.0, 1.0));\r
    float d = hash21(i + vec2(1.0, 1.0));\r
\r
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);\r
}\r
\r
float fbm(vec2 p) {\r
    float sum = 0.0;\r
    float amp = 0.55;\r
    float norm = 0.0;\r
    int octaves = clamp(uFractalIters + 3, 3, 8);\r
    float persistence = mix(0.46, 0.60, uSmoothBlend);\r
\r
    for (int i = 0; i < 8; i++) {\r
        if (i >= octaves) {\r
            break;\r
        }\r
        sum += amp * noise(p);\r
        norm += amp;\r
        p = rot2(0.45 + 0.03 * float(i)) * p * 2.02 + vec2(0.31, -0.27);\r
        amp *= persistence;\r
    }\r
\r
    return sum / max(norm, 0.0001);\r
}\r
\r
vec3 palNeon(float t) {\r
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.0, 0.8, 0.6) + vec3(0.00, 0.33, 0.67)));\r
}\r
\r
vec3 palLava(float t) {\r
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.7, 0.9, 1.3) + vec3(0.00, 0.18, 0.55)));\r
}\r
\r
vec3 palFire(float t) {\r
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.3, 0.5, 0.8) + vec3(0.25, 0.00, 0.55)));\r
}\r
\r
vec3 palIce(float t) {\r
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.6, 0.9, 1.2) + vec3(0.55, 0.70, 0.85)));\r
}\r
\r
vec2 warp(vec2 p, float time) {\r
    float drift = time * (0.05 + uRotSpeed * 0.12);\r
    float warpScale = 0.75 + uFoldScale * 0.35;\r
    float warpGain = 0.22 + uFoldOffset * 0.25;\r
\r
    vec2 q = vec2(\r
        fbm(p * warpScale + vec2(0.0, drift)),\r
        fbm(p * warpScale + vec2(4.8, -drift * 0.8))\r
    );\r
\r
    vec2 r = vec2(\r
        fbm(p * (warpScale + 0.6) + (q - 0.5) * 2.0 + vec2(1.7, -2.6) + drift * 0.4),\r
        fbm(p * (warpScale + 0.4) + (q - 0.5) * 2.0 + vec2(-3.1, 0.9) - drift * 0.3)\r
    );\r
\r
    return p + (q + r - 1.0) * warpGain;\r
}\r
\r
vec3 background(vec2 uv, vec3 rd, float time) {\r
    float colorT = uHueShift + time * uHueSpeed * 0.25 + fract(uSeed * 0.000001);\r
    vec2 p = uv * (1.8 + uZoom * 0.55);\r
    p += vec2(time * uCamOrbit * 0.6, uCamHeight * 0.22);\r
\r
    vec2 q = warp(p, time);\r
    vec2 r = warp(q * 1.35 + vec2(2.4, -1.8), time * 0.7);\r
\r
    float field = fbm(q);\r
    float filaments = fbm(r * 1.7 + vec2(time * 0.03, -time * 0.025));\r
    float mist = fbm(p * 0.65 - vec2(time * 0.02, time * 0.015));\r
    float ridges = 1.0 - abs(2.0 * fbm(q * (1.4 + 0.2 * uDetailLevel)) - 1.0);\r
    float glowMask = smoothstep(0.30, 0.88, mix(field, ridges, 0.55));\r
    float veil = smoothstep(0.18, 0.82, mix(mist, filaments, 0.4));\r
\r
    float horizon = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);\r
    float radial = length(uv);\r
\r
    vec3 cool = mix(\r
        palIce(colorT + field * 0.45 + mist * 0.2),\r
        palNeon(colorT + filaments * 0.35 + horizon * 0.15),\r
        0.45 + 0.25 * horizon\r
    );\r
\r
    vec3 warm = mix(\r
        palLava(colorT + ridges * 0.50 + 0.08 * radial),\r
        palFire(colorT + glowMask * 0.65 + filaments * 0.15),\r
        glowMask\r
    );\r
\r
    vec3 col = mix(cool * (0.22 + 0.18 * horizon), warm, veil);\r
    col += palFire(colorT + filaments + radial * 0.1) * glowMask * glowMask\r
        * (0.10 + 0.05 * uGlowIntensity);\r
    col += palNeon(colorT + mist * 0.6 + 0.12 * radial)\r
        * pow(max(1.0 - radial * 0.75, 0.0), 2.0)\r
        * (0.06 + 0.03 * uLightIntensity);\r
\r
    float haze = smoothstep(0.25, 1.35, radial + (1.0 - horizon) * 0.35);\r
    col = mix(col, cool * 0.35, haze * 0.35);\r
\r
    return col * (0.75 + 0.25 * uBrightness);\r
}\r
\r
void main() {\r
    vec2 uv = (gl_FragCoord.xy - uResolution * 0.5) / min(uResolution.x, uResolution.y);\r
    float time = uTime * uTimeScale;\r
    vec3 rd = normalize(vec3(uv, 1.9 + 0.25 * uZoom));\r
\r
    vec3 col = background(uv, rd, time);\r
\r
    float luma = dot(col, vec3(0.299, 0.587, 0.114));\r
    col = mix(vec3(luma), col, uSaturation);\r
\r
    float ca = uChromaShift * 0.003;\r
    float r2 = dot(uv, uv);\r
    col.r *= 1.0 + ca * r2 * 1.7;\r
    col.b *= 1.0 - ca * r2 * 1.7;\r
\r
    col = mix(vec3(0.5), col, uContrast);\r
    col = max(col, 0.0);\r
    col = col / (col + 0.35);\r
\r
    vec2 vUv = gl_FragCoord.xy / uResolution;\r
    col *= 1.0 - length(vUv - 0.5) * 0.5;\r
    col = pow(max(col, 0.0), vec3(0.85));\r
\r
    outColor = vec4(col, 1.0);\r
}\r
`,pr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2  uResolution;     // pixels\r
uniform float uPhase;          // [0,1] loop time\r
uniform int   uSymmetry;       // rotational symmetry copies\r
uniform int   uSubdivisions;   // curve sample points per copy\r
uniform float uScale;          // overall radius scale (0.0 - 1.0)\r
\r
uniform float uSinAmp;         // amplitude of the outer sine\r
uniform float uBaseFreq;       // base frequency term\r
uniform float uModAmp;         // amplitude modulator\r
uniform float uModFreq;        // inner modulation frequency\r
uniform float uModDiv;         // inner modulation divisor\r
uniform float uThetaScale;     // scale of theta\r
\r
uniform float uLineWidth;      // antialiased line width in pixels\r
uniform float uHueCycles;      // hue cycles per loop\r
\r
uniform float uSeed;           // random seed for variation\r
\r
const float PI  = 3.14159265358979323846;\r
const float TAU = 6.28318530717958647692;\r
\r
float dot2(vec2 v) { return dot(v, v); }\r
\r
// Exact SDF to quadratic Bezier (Inigo Quilez), returns vec2(distance, closest_t)\r
// where closest_t in [0,1] is the parameter on the Bezier nearest to pos.\r
vec2 sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C) {\r
  vec2 a = B - A;\r
  vec2 b = A - 2.0*B + C;\r
  if (dot(b, b) < 1e-10) {\r
    // Degenerate: line fallback — compute t along segment\r
    vec2 pa = pos - A, ba = C - A;\r
    float ht = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);\r
    return vec2(length(pa - ba * ht), ht);\r
  }\r
  vec2 c = a * 2.0;\r
  vec2 d = A - pos;\r
  float kk = 1.0 / dot(b, b);\r
  float kx = kk * dot(a, b);\r
  float ky = kk * (2.0*dot(a,a) + dot(d,b)) / 3.0;\r
  float kz = kk * dot(d, a);\r
  float p  = ky - kx*kx;\r
  float p3 = p*p*p;\r
  float q  = kx*(2.0*kx*kx - 3.0*ky) + kz;\r
  float h  = q*q + 4.0*p3;\r
  float res;\r
  float bestT;\r
  if (h >= 0.0) {\r
    h = sqrt(h);\r
    vec2 x = (vec2(h, -h) - q) / 2.0;\r
    vec2 uv = sign(x) * pow(abs(x), vec2(1.0/3.0));\r
    bestT = clamp(uv.x + uv.y - kx, 0.0, 1.0);\r
    res = dot2(d + (c + b*bestT)*bestT);\r
  } else {\r
    float z = sqrt(-p);\r
    float v = acos(clamp(q/(p*z*2.0), -1.0, 1.0)) / 3.0;\r
    float m = cos(v);\r
    float n = sin(v) * 1.732050808;\r
    vec3  t = clamp(vec3(m+m, -n-m, n-m)*z - kx, 0.0, 1.0);\r
    float d0 = dot2(d + (c + b*t.x)*t.x);\r
    float d1 = dot2(d + (c + b*t.y)*t.y);\r
    if (d0 < d1) { res = d0; bestT = t.x; }\r
    else         { res = d1; bestT = t.y; }\r
  }\r
  return vec2(sqrt(res), bestT);\r
}\r
\r
// The transformation function\r
float transformation(float t) {\r
  float inner = sin(t * uModFreq * PI);\r
  float outer = uBaseFreq + uModAmp * sin(TAU * inner / uModDiv);\r
  return (1.0 + uSinAmp * sin(outer * t * uThetaScale * PI)) * PI;\r
}\r
\r
// Evaluate a point on the polar rose curve with symmetry rotation\r
vec2 curvePoint(float t, float cosR, float sinR) {\r
  float theta = transformation(t);\r
  float r = sin(t * TAU + uPhase * TAU) * uScale * 0.5;\r
  vec2 c = vec2(r * cos(theta), r * sin(theta));\r
  return vec2(cosR * c.x - sinR * c.y, sinR * c.x + cosR * c.y);\r
}\r
\r
void main() {\r
  float minDim = min(uResolution.x, uResolution.y);\r
  vec2 px = (gl_FragCoord.xy - 0.5 * uResolution) / minDim;\r
\r
  float dMin = 1e9;\r
  float closestT = 0.0;\r
\r
  int N   = max(3, uSubdivisions);\r
  int sym = max(1, uSymmetry);\r
\r
  // Bounding-box margin covers the glow radius\r
  float margin = uLineWidth * 3.0 / minDim;\r
\r
  for (int s = 0; s < 128; ++s) {\r
    if (s >= sym) break;\r
    float rotAngle = float(s) * TAU / float(sym);\r
    float cosR = cos(rotAngle);\r
    float sinR = sin(rotAngle);\r
\r
    // Closed-loop Catmull-Rom Bezier: N sample points, indices wrap mod N\r
    // so the curve forms a seamless loop with no dangling endpoints.\r
    vec2 Pprev = curvePoint(float(N - 1) / float(N), cosR, sinR);\r
    vec2 Pcurr = curvePoint(0.0, cosR, sinR);\r
\r
    for (int i = 0; i < 8192; ++i) {\r
      if (i >= N) break;\r
\r
      vec2 Pnext = curvePoint(float((i + 1) % N) / float(N), cosR, sinR);\r
\r
      // Catmull-Rom midpoint Bezier: mid(prev,curr) → curr → mid(curr,next)\r
      // Gives C1-continuous joins — no sharp corners anywhere.\r
      vec2 A = 0.5 * (Pprev + Pcurr);\r
      vec2 B = Pcurr;\r
      vec2 C = 0.5 * (Pcurr + Pnext);\r
\r
      // Bounding-box culling: skip the expensive sdBezier when far away.\r
      vec2 lo = min(A, min(B, C)) - margin;\r
      vec2 hi = max(A, max(B, C)) + margin;\r
\r
      if (px.x >= lo.x && px.x <= hi.x && px.y >= lo.y && px.y <= hi.y) {\r
        vec2 db = sdBezier(px, A, B, C);  // .x = distance, .y = bezier param [0,1]\r
        if (db.x < dMin) {\r
          dMin = db.x;\r
          // Continuous curve parameter: segment i, interpolated by Bezier parameter\r
          closestT = (float(i) + db.y) / float(N);\r
        }\r
      }\r
\r
      Pprev = Pcurr;\r
      Pcurr = Pnext;\r
    }\r
  }\r
\r
  // --- Sharp core + soft bloom glow ---\r
  float lineHalf = uLineWidth * 0.5 / minDim;\r
\r
  // Sharp antialiased core\r
  float core = 1.0 - smoothstep(lineHalf * 0.35, lineHalf, dMin);\r
\r
  // Soft bloom glow extending beyond the core\r
  float glowR = lineHalf * 5.0;\r
  float glow  = exp(-dMin * dMin / (glowR * glowR * 0.18)) * 0.3;\r
\r
  float alpha = max(core, glow);\r
\r
  // Hue from curve parameter\r
  float hue = 0.5 + 0.5 * sin(closestT * TAU * uHueCycles);\r
  hue = fract(hue + 0.1 * sin(uPhase * TAU));\r
\r
  // Glow is slightly desaturated; core is vivid\r
  float sat = mix(0.55, 0.9, core);\r
  float val = 0.99;\r
\r
  // HSV → RGB\r
  vec3 rgb;\r
  {\r
    vec3 cv = vec3(hue, sat, val);\r
    vec3 q  = abs(fract(cv.xxx + vec3(0.0, 2.0/6.0, 4.0/6.0)) * 6.0 - 3.0);\r
    rgb = cv.z * mix(vec3(1.0), clamp(q - 1.0, 0.0, 1.0), cv.y);\r
  }\r
\r
  outColor = vec4(rgb * alpha, alpha);\r
}\r
`,dr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
\r
uniform int uLoopCount;\r
uniform float uCycleMin;\r
uniform float uCycleMax;\r
uniform float uRadiusMin;\r
uniform float uRadiusMax;\r
uniform float uCenterDrift;\r
uniform float uCenterDriftSpeed;\r
uniform int uHarmonicCount;\r
uniform float uWobbleAmount;\r
uniform float uWobbleFalloff;\r
uniform float uMorphSpeed;\r
uniform float uVibrationAmount;\r
uniform int uVibrationFreq;\r
uniform float uVibrationSpeed;\r
uniform float uStrokeWidthMin;\r
uniform float uStrokeWidthMax;\r
uniform float uWidthMod;\r
uniform float uInkTexture;\r
uniform float uDrawSoftness;\r
uniform float uBleedStrength;\r
uniform float uBleedSpread;\r
uniform float uSoakStrength;\r
uniform float uDryMix;\r
uniform float uWetMix;\r
uniform float uAbsorbStrength;\r
uniform float uHueShift;\r
uniform float uSaturationBoost;\r
uniform float uValueBoost;\r
uniform float uPastelMix;\r
uniform float uPaperGrainScale;\r
uniform float uPaperGrainAmount;\r
uniform float uPaperPulpAmount;\r
uniform float uPaperRingDensity;\r
uniform float uPaperRingWobble;\r
uniform float uPaperRingAmount;\r
uniform float uPaperFleckAmount;\r
uniform float uPaperBlotchAmount;\r
uniform float uVignetteStrength;\r
\r
const float TAU = 6.28318530718;\r
const float EPSILON = 0.001;\r
const int MAX_LOOP_COUNT = 24;\r
const int MAX_HARMONICS = 8;\r
\r
float hash11(float p) {\r
  p = fract(p * 0.1031);\r
  p *= p + 33.33;\r
  p *= p + p;\r
  return fract(p);\r
}\r
\r
float hash21(vec2 p) {\r
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);\r
  p3 += dot(p3, p3.yzx + 33.33);\r
  return fract((p3.x + p3.y) * p3.z);\r
}\r
\r
mat2 rot(float a) {\r
  float s = sin(a);\r
  float c = cos(a);\r
  return mat2(c, -s, s, c);\r
}\r
\r
float noise(vec2 p) {\r
  vec2 i = floor(p);\r
  vec2 f = fract(p);\r
  f = f * f * (3.0 - 2.0 * f);\r
\r
  float a = hash21(i);\r
  float b = hash21(i + vec2(1.0, 0.0));\r
  float c = hash21(i + vec2(0.0, 1.0));\r
  float d = hash21(i + vec2(1.0, 1.0));\r
\r
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);\r
}\r
\r
float fbm(vec2 p) {\r
  float value = 0.0;\r
  float amplitude = 0.5;\r
  for (int i = 0; i < 5; i++) {\r
    value += amplitude * noise(p);\r
    p = rot(0.35) * p * 2.02 + vec2(11.7, 4.3);\r
    amplitude *= 0.52;\r
  }\r
  return value;\r
}\r
\r
// Sample fbm in a domain that is continuous under theta -> theta + TAU by\r
// embedding the angle on a circle. angScale is the linear arc length of the\r
// full loop in noise-space (matching a former thetaNorm * angScale span).\r
float polarFbm(float theta, float radial, float angScale, float radScale, vec2 seed) {\r
  float r = angScale / TAU;\r
  return fbm(vec2(cos(theta), sin(theta)) * r + vec2(radial * radScale, 0.0) + seed);\r
}\r
\r
// Smooth brush window along a 1D stroke parameter: fade in at the tail,\r
// hold, then fade out at the head. edge is clamped so short strokes still\r
// taper instead of inverting.\r
float strokeWindow(float u, float extent, float edge) {\r
  float e = min(max(edge, EPSILON), max(extent * 0.49, EPSILON));\r
  float fadeIn = smoothstep(0.0, e, u);\r
  float fadeOut = 1.0 - smoothstep(extent - e, extent, u);\r
  return clamp(fadeIn * fadeOut, 0.0, 1.0);\r
}\r
\r
// Growing loop stroke on the unit circle [0,1). Both ends taper radially\r
// (angularly) so the origin is never a hard cut. Near closure the head\r
// overshoots past a full turn and overlaps the faded tail; max() of the\r
// two laps blends the join without a seam. When fully drawn the solid\r
// middles overlap and the ring is uniform.\r
float loopArcReveal(float thetaNorm, float arcLen, float soft) {\r
  float t = fract(thetaNorm);\r
  float edge = max(soft, EPSILON);\r
  float len = clamp(arcLen, 0.0, 1.0);\r
\r
  // Early on: a little head runway for a soft brush tip.\r
  // Near the end: enough overshoot that the head's solid body covers the\r
  // tail fade, then the head fade settles across the join.\r
  float close = smoothstep(0.55, 0.98, len);\r
  float extent = len + edge * mix(0.55, 2.15, close);\r
\r
  float reveal = strokeWindow(t, extent, edge);\r
  // Second lap: head wrapping over the origin / tail.\r
  reveal = max(reveal, strokeWindow(t + 1.0, extent, edge));\r
  return reveal;\r
}\r
\r
vec3 hsv2rgb(vec3 c) {\r
  vec3 p = abs(fract(c.xxx + vec3(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0);\r
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);\r
}\r
\r
// Smooth wobbly closed-loop radius built from integer harmonics of the\r
// polar angle. Because every term is a whole multiple of theta the curve\r
// always closes perfectly and stays C-infinity smooth, while stacked\r
// harmonics plus a high-frequency vibration term keep it rich in detail.\r
float loopRadius(float theta, float fi, float time) {\r
  float r = 1.0;\r
  int harmonics = clamp(uHarmonicCount, 0, MAX_HARMONICS);\r
  for (int k = 0; k < MAX_HARMONICS; k++) {\r
    if (k >= harmonics) {\r
      break;\r
    }\r
    float n = float(k + 2);\r
    float amp = uWobbleAmount * (hash11(fi * 7.77 + n * 13.13) - 0.5) * 2.0 / pow(n, max(uWobbleFalloff, 0.1));\r
    float phase = hash11(fi * 3.33 + n * 21.7) * TAU;\r
    float drift = time * uMorphSpeed * (0.3 + hash11(fi * 5.11 + n * 9.19));\r
    r += amp * sin(n * theta + phase + drift);\r
  }\r
  int vibFreq = max(uVibrationFreq, 1);\r
  float vibPhase = hash11(fi * 91.3) * TAU;\r
  r += uVibrationAmount * sin(float(vibFreq) * theta + vibPhase + time * uVibrationSpeed);\r
  r += uVibrationAmount * 0.45 * sin(float(vibFreq * 2 + 1) * theta - vibPhase * 1.7 + time * uVibrationSpeed * 1.31);\r
  return r;\r
}\r
\r
// Wobbly closed-ring paper texture: concentric harmonic loops around a\r
// handful of scattered centers replace the old straight linear fibers.\r
float paperRings(vec2 uv, float aspect) {\r
  float rings = 0.0;\r
  for (int i = 0; i < 3; i++) {\r
    float fi = float(i) + 1.0;\r
    vec2 center = vec2(hash11(fi * 27.7), hash11(fi * 63.1));\r
    vec2 d = (uv - center) * vec2(aspect, 1.0);\r
    float theta = atan(d.y, d.x);\r
    float wobble = uPaperRingWobble * (\r
      sin(theta * 3.0 + fi * 2.3) * 0.5 +\r
      sin(theta * 7.0 - fi * 4.1) * 0.3 +\r
      sin(theta * 13.0 + fi * 7.9) * 0.2\r
    );\r
    float band = sin((length(d) * (1.0 + wobble)) * uPaperRingDensity * TAU + fi * 11.0);\r
    rings += smoothstep(0.55, 0.95, band) / 3.0;\r
  }\r
  return rings;\r
}\r
\r
void main() {\r
  vec2 uv = gl_FragCoord.xy / uResolution.xy;\r
  float aspect = uResolution.x / max(uResolution.y, 1.0);\r
  float time = uTime * uTimeScale;\r
\r
  float grain = fbm(uv * vec2(18.0, 24.0) * max(uPaperGrainScale, 0.05));\r
  float pulp = fbm(uv * vec2(7.0, 10.0) + vec2(3.1, 8.7));\r
  float rings = paperRings(uv, aspect);\r
  float flecks = fbm(uv * vec2(90.0, 115.0) + vec2(12.0, 4.0));\r
  float blotches = fbm(uv * 4.0 + vec2(0.0, 7.3));\r
  float vignette = smoothstep(1.15, 0.15, length((uv - 0.5) * vec2(1.1, 0.95)));\r
\r
  vec3 paper = vec3(0.955, 0.925, 0.875);\r
  paper += uPaperGrainAmount * (grain - 0.5);\r
  paper += vec3(0.035, 0.032, 0.026) * rings * uPaperRingAmount;\r
  paper += uPaperPulpAmount * (pulp - 0.5);\r
  paper -= uPaperFleckAmount * smoothstep(0.7, 0.92, flecks);\r
  paper -= uPaperBlotchAmount * smoothstep(0.45, 0.95, blotches);\r
  paper *= 1.0 - uVignetteStrength + uVignetteStrength * vignette;\r
\r
  vec3 color = paper;\r
\r
  int loopCount = clamp(uLoopCount, 1, MAX_LOOP_COUNT);\r
\r
  for (int i = 0; i < MAX_LOOP_COUNT; i++) {\r
    if (i >= loopCount) {\r
      break;\r
    }\r
\r
    float fi = float(i) + 1.0;\r
    vec2 center = vec2(\r
      mix(0.14, 0.86, hash11(fi * 7.13)),\r
      mix(0.14, 0.86, hash11(fi * 11.9))\r
    );\r
    float driftPhase = hash11(fi * 17.3) * TAU;\r
    center += uCenterDrift * vec2(\r
      sin(time * uCenterDriftSpeed * (0.5 + hash11(fi * 19.7) * 0.8) + driftPhase),\r
      cos(time * uCenterDriftSpeed * (0.4 + hash11(fi * 23.1) * 0.9) + driftPhase * 1.9)\r
    );\r
\r
    vec2 d = (uv - center) * vec2(aspect, 1.0);\r
    float dist = length(d);\r
    float theta = atan(d.y, d.x);\r
\r
    float baseR = mix(uRadiusMin, uRadiusMax, hash11(fi * 29.7));\r
    float radius = baseR * loopRadius(theta, fi, time);\r
    float signedDist = dist - radius;\r
\r
    // Progressive painting: each stroke grows around its loop with faded\r
    // tail/head, overlaps itself as it closes, lingers as a solid ring, then\r
    // washes out before repaint. Start angle is hashed per loop.\r
    float cycle = mix(uCycleMin, uCycleMax, hash11(fi * 2.71));\r
    float phase = fract(time / max(cycle, EPSILON) + hash11(fi * 41.3));\r
    float paintOn = smoothstep(0.0, 0.06, phase) * (1.0 - smoothstep(0.86, 0.99, phase));\r
    float travel = smoothstep(0.02, 0.62, phase);\r
    float thetaNorm = fract(theta / TAU + hash11(fi * 47.9));\r
    float soft = max(uDrawSoftness, EPSILON);\r
    float reveal = loopArcReveal(thetaNorm, travel, soft);\r
\r
    float width = baseR * mix(uStrokeWidthMin, uStrokeWidthMax, hash11(fi * 31.7));\r
    width *= 1.0 + uWidthMod * sin(theta * 3.0 + hash11(fi * 53.3) * TAU + time * uMorphSpeed * 0.4);\r
    width = max(width, EPSILON);\r
\r
    // Angle must enter noise via cos/sin so ink/bleed are seamless on the loop.\r
    // Use the same per-loop angle origin as reveal so texture lines up with paint.\r
    float thetaStroke = thetaNorm * TAU;\r
    float ink = mix(\r
      1.0 - uInkTexture,\r
      1.0,\r
      polarFbm(thetaStroke, signedDist, 6.0, 18.0, vec2(fi * 0.7, fi * 1.3))\r
    );\r
    float strokeCore = smoothstep(width * 1.5, width * 0.35, abs(signedDist));\r
    float strokeMask = strokeCore * reveal * paintOn * ink;\r
\r
    float bleedSpread = max(width * uBleedSpread, EPSILON);\r
    float bleed =\r
      exp(-pow(signedDist / bleedSpread, 2.0)) *\r
      (0.45 + 0.55 * polarFbm(thetaStroke, dist, 2.2 * TAU, 9.0, vec2(fi * 0.7, 3.1))) *\r
      reveal * paintOn * uBleedStrength;\r
\r
    float soak =\r
      smoothstep(0.0, -baseR * 0.6, signedDist) *\r
      (0.35 + 0.65 * grain + 0.18 * pulp) *\r
      (0.3 + 0.7 * paintOn) *\r
      uSoakStrength;\r
\r
    float dryGhost = strokeCore * ink * 0.6 * (1.0 - paintOn);\r
\r
    float dryMask = clamp(dryGhost + bleed * 0.18 + soak * 0.2, 0.0, 1.0);\r
    float wetMask = clamp(strokeMask + bleed * 0.4 + soak * 0.5, 0.0, 1.0);\r
\r
    float hue = fract(hash11(fi * 13.7) * 0.9 + 0.08 * sin(fi * 1.7) + uHueShift);\r
    float saturation = clamp(mix(0.45, 0.78, hash11(fi * 53.9)) + uSaturationBoost, 0.0, 1.0);\r
    float value = clamp(mix(0.56, 0.84, hash11(fi * 59.2)) + uValueBoost, 0.0, 1.0);\r
    vec3 pigment = hsv2rgb(vec3(hue, saturation, value));\r
    pigment = mix(pigment, vec3(0.98, 0.96, 0.93), clamp(uPastelMix, 0.0, 1.0));\r
\r
    float pooling = clamp(strokeMask * 0.55 + soak * 0.5 + dryMask * 0.24, 0.0, 1.0);\r
    float absorb = (0.72 + 0.6 * grain + 0.18 * pulp) * uAbsorbStrength;\r
    vec3 dryTint = mix(pigment, paper, 0.54 + 0.16 * blotches + 0.06 * grain);\r
    vec3 edgeTint = mix(pigment, paper, 0.3 + 0.22 * blotches + 0.1 * grain);\r
    vec3 seepTint = mix(pigment, paper, 0.62 + 0.16 * grain);\r
    vec3 dryWash = mix(dryTint, pigment * 0.72, clamp(dryMask * 0.62 + bleed * 0.14, 0.0, 1.0));\r
    vec3 soakWash = mix(seepTint, pigment * 0.8, clamp(soak * 0.72 + bleed * 0.18, 0.0, 1.0));\r
    vec3 wash = mix(edgeTint, pigment * 0.9, pooling);\r
    color = mix(color, dryWash, dryMask * absorb * 0.5 * uDryMix);\r
    color = mix(color, soakWash, clamp(soak, 0.0, 1.0) * absorb * 0.32 * uWetMix);\r
    color = mix(color, wash, wetMask * absorb * 0.78 * uWetMix);\r
  }\r
\r
  color *= 0.98 + 0.02 * vignette;\r
  outColor = vec4(clamp(color, 0.0, 1.0), 1.0);\r
}\r
`,hr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform float uPlasmaScale;\r
uniform float uWarp;\r
uniform int uFold;\r
uniform float uMorph;\r
uniform int uShapeCount;\r
uniform float uOrbit;\r
uniform float uFilament;\r
uniform float uGlow;\r
uniform float uColorCycle;\r
uniform float uSeed;\r
\r
const float TAU = 6.28318530718;\r
\r
mat2 rotate2d(float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return mat2(c, -s, s, c);\r
}\r
\r
float hash11(float value) {\r
  return fract(sin(value * 127.1 + uSeed * 0.000017) * 43758.5453123);\r
}\r
\r
float hash21(vec2 point) {\r
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p += dot(p, p.yzx + 33.33 + fract(uSeed * 0.000001));\r
  return fract((p.x + p.y) * p.z);\r
}\r
\r
float noise(vec2 point) {\r
  vec2 cell = floor(point);\r
  vec2 local = fract(point);\r
  local = local * local * (3.0 - 2.0 * local);\r
  float a = hash21(cell);\r
  float b = hash21(cell + vec2(1.0, 0.0));\r
  float c = hash21(cell + vec2(0.0, 1.0));\r
  float d = hash21(cell + vec2(1.0));\r
  return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);\r
}\r
\r
float fbm(vec2 point) {\r
  float value = 0.0;\r
  float amplitude = 0.5;\r
  mat2 turn = rotate2d(0.57);\r
  for (int octave = 0; octave < 5; octave++) {\r
    value += amplitude * noise(point);\r
    point = turn * point * 2.03 + vec2(1.7, -2.4);\r
    amplitude *= 0.5;\r
  }\r
  return value;\r
}\r
\r
vec2 kaleidoscope(vec2 point, float folds) {\r
  float radius = length(point);\r
  float sector = TAU / max(1.0, floor(folds + 0.5));\r
  float angle = abs(mod(atan(point.y, point.x) + sector * 0.5, sector) - sector * 0.5);\r
  return radius * vec2(cos(angle), sin(angle));\r
}\r
\r
vec3 plasmaPalette(float phase) {\r
  vec3 base = vec3(0.50, 0.48, 0.52);\r
  vec3 amplitude = vec3(0.50, 0.46, 0.48);\r
  vec3 frequency = vec3(1.0, 0.72, 0.47);\r
  vec3 offset = vec3(0.02, 0.24, 0.58);\r
  return base + amplitude * cos(TAU * (frequency * phase + offset));\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;\r
  point *= uZoom;\r
  float time = uTime * uTimeScale;\r
\r
  vec2 folded = kaleidoscope(point, float(uFold));\r
  vec2 noiseDrift = vec2(time * 0.08, -time * 0.055);\r
  vec2 warpField = vec2(\r
    fbm(folded * 1.35 + noiseDrift),\r
    fbm(folded * 1.35 + vec2(5.2, -3.7) - noiseDrift.yx)\r
  ) - 0.5;\r
  vec2 plasmaPoint = mix(point, folded, 0.28) + warpField * uWarp;\r
\r
  float plasma = sin(plasmaPoint.x * uPlasmaScale + time * 1.3);\r
  plasma += sin(plasmaPoint.y * uPlasmaScale * 1.17 - time * 1.05);\r
  plasma += sin((plasmaPoint.x + plasmaPoint.y) * uPlasmaScale * 0.73 + time * 0.61);\r
  plasma += sin(length(plasmaPoint + warpField) * uPlasmaScale * 1.85 - time * 1.7);\r
  plasma *= 0.25;\r
\r
  float filamentFrequency = mix(3.0, 12.0, clamp(uFilament, 0.0, 1.0));\r
  float filaments = pow(1.0 - abs(sin(plasma * filamentFrequency)), 5.0);\r
  float cloud = smoothstep(-0.7, 0.9, plasma + (fbm(plasmaPoint * 2.1 - time * 0.04) - 0.5));\r
\r
  float shapeBody = 0.0;\r
  float shapeEdge = 0.0;\r
  float eyes = 0.0;\r
  float tendrils = 0.0;\r
  float shapePhase = 0.0;\r
\r
  for (int index = 0; index < 8; index++) {\r
    if (index >= uShapeCount) {\r
      break;\r
    }\r
\r
    float fi = float(index);\r
    float randomA = hash11(fi + 2.0);\r
    float randomB = hash11(fi + 19.0);\r
    float phase = randomA * TAU;\r
    float speed = 0.23 + randomB * 0.24;\r
    vec2 center = vec2(\r
      sin(time * speed + phase),\r
      cos(time * speed * 0.73 + phase * 1.61)\r
    ) * uOrbit * vec2(0.72, 0.46);\r
\r
    vec2 creature = point - center;\r
    creature = rotate2d(-phase * 0.35 - sin(time * 0.19 + phase) * 0.8) * creature;\r
    creature += uMorph * 0.045 * vec2(\r
      sin(creature.y * 9.0 + time * 1.2 + phase),\r
      cos(creature.x * 8.0 - time * 0.9 + phase)\r
    );\r
\r
    float angle = atan(creature.y, creature.x);\r
    float radius = length(creature);\r
    float lobes = 3.0 + floor(randomA * 5.0);\r
    float membrane = 0.13 + randomB * 0.075;\r
    membrane += uMorph * 0.035 * sin(angle * lobes + time * (0.8 + randomA) + phase);\r
    membrane += uMorph * 0.018 * sin(angle * (lobes + 3.0) - time * 1.6);\r
    float signedShape = radius - membrane;\r
    float antialias = max(fwidth(signedShape), 0.001);\r
    float body = 1.0 - smoothstep(-antialias, antialias, signedShape);\r
    float edge = exp(-max(abs(signedShape) - antialias, 0.0) * mix(70.0, 24.0, uGlow));\r
\r
    vec2 eyePoint = creature;\r
    eyePoint.x += membrane * 0.15;\r
    float eyeDistance = length(eyePoint * vec2(0.8, 1.9));\r
    float eyeRing = exp(-abs(eyeDistance - membrane * 0.34) * 95.0) * body;\r
    float pupil = exp(-eyeDistance * 75.0) * body;\r
\r
    float tailGate = smoothstep(0.06, membrane * 1.4, creature.x)\r
      * (1.0 - smoothstep(membrane * 1.4, membrane * 4.0, creature.x));\r
    float tailPath = abs(\r
      creature.y - sin(creature.x * (12.0 + randomA * 8.0) - time * 2.0 + phase)\r
      * (0.025 + uMorph * 0.035)\r
    );\r
    float tail = exp(-tailPath * 100.0) * tailGate;\r
\r
    shapeBody = max(shapeBody, body * (0.35 + 0.65 * sin(angle * lobes + phase) * 0.5 + 0.325));\r
    shapeEdge += edge;\r
    eyes += eyeRing + pupil * 1.8;\r
    tendrils += tail;\r
    shapePhase += (edge + eyeRing) * (randomA - 0.5);\r
  }\r
\r
  float pulse = 0.82 + 0.18 * sin(time * 3.0 + plasma * 4.0);\r
  vec3 background = vec3(0.004, 0.006, 0.018);\r
  vec3 plasmaColor = plasmaPalette(plasma * 0.26 + time * uColorCycle * 0.08);\r
  vec3 creatureColor = plasmaPalette(0.62 + shapePhase * 0.16 - time * uColorCycle * 0.055);\r
  vec3 eyeColor = plasmaPalette(0.12 - time * 0.025);\r
\r
  vec3 color = background;\r
  color += plasmaColor * (filaments * 0.72 + cloud * 0.12);\r
  color += creatureColor * shapeBody * 0.22;\r
  color += creatureColor * shapeEdge * uGlow * 1.35 * pulse;\r
  color += eyeColor * eyes * uGlow * 1.8;\r
  color += plasmaPalette(0.88 + plasma * 0.1) * tendrils * uGlow;\r
\r
  float bloom = filaments * shapeEdge + eyes * 0.45 + tendrils * 0.35;\r
  color += vec3(0.55, 0.72, 1.0) * bloom * uGlow * 0.45;\r
  color *= 1.0 - 0.28 * smoothstep(0.45, 1.45, length(point));\r
  color = 1.0 - exp(-color * (0.9 + uGlow * 0.5));\r
  color = pow(max(color, 0.0), vec3(0.82));\r
\r
  outColor = vec4(color, 1.0);\r
}`,vr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform int uBranches;\r
uniform float uGrowth;\r
uniform float uCurl;\r
uniform float uPulse;\r
uniform float uArcDensity;\r
uniform float uGlow;\r
uniform float uHue;\r
uniform float uSeed;\r
\r
const float TAU = 6.28318530718;\r
\r
mat2 rotate2d(float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return mat2(c, -s, s, c);\r
}\r
\r
float hash11(float value) {\r
  return fract(sin(value * 127.17 + uSeed * 0.000013) * 43758.5453);\r
}\r
\r
float hash21(vec2 point) {\r
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p += dot(p, p.yzx + 33.33 + fract(uSeed * 0.000001));\r
  return fract((p.x + p.y) * p.z);\r
}\r
\r
float noise(vec2 point) {\r
  vec2 cell = floor(point);\r
  vec2 local = fract(point);\r
  local = local * local * (3.0 - 2.0 * local);\r
  return mix(\r
    mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), local.x),\r
    mix(hash21(cell + vec2(0.0, 1.0)), hash21(cell + vec2(1.0)), local.x),\r
    local.y\r
  );\r
}\r
\r
float fbm(vec2 point) {\r
  float value = 0.0;\r
  float amplitude = 0.52;\r
  for (int octave = 0; octave < 5; octave++) {\r
    value += amplitude * noise(point);\r
    point = rotate2d(0.68) * point * 2.04 + vec2(2.3, -1.6);\r
    amplitude *= 0.49;\r
  }\r
  return value;\r
}\r
\r
vec3 palette(float phase) {\r
  return 0.48 + 0.52 * cos(TAU * (phase * vec3(1.0, 0.78, 0.55) + vec3(0.02, 0.28, 0.62)));\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;\r
  point *= uZoom;\r
  float time = uTime * uTimeScale;\r
\r
  vec2 smokePoint = point * 2.2;\r
  smokePoint += vec2(\r
    fbm(point * 1.4 + vec2(time * 0.08, 0.0)),\r
    fbm(point * 1.4 - vec2(0.0, time * 0.07))\r
  ) * 0.8;\r
  float smoke = fbm(smokePoint - time * 0.035);\r
  float arcField = 1.0 - abs(sin((smoke + point.x * 0.35 - point.y * 0.22) * uArcDensity * 8.0));\r
  arcField = pow(arcField, 7.0);\r
\r
  float coral = 0.0;\r
  float buds = 0.0;\r
  float membranes = 0.0;\r
  float choirPhase = 0.0;\r
  float branchCount = float(uBranches);\r
\r
  for (int colony = 0; colony < 4; colony++) {\r
    float fi = float(colony);\r
    float randomA = hash11(fi + 5.0);\r
    float randomB = hash11(fi + 31.0);\r
    float phase = randomA * TAU;\r
    vec2 center = vec2(\r
      sin(time * (0.11 + randomA * 0.08) + phase),\r
      cos(time * (0.09 + randomB * 0.07) + phase * 1.4)\r
    ) * vec2(0.64, 0.42);\r
    vec2 local = rotate2d(phase + time * (randomA - 0.5) * 0.16) * (point - center);\r
    float radius = length(local);\r
    float angle = atan(local.y, local.x);\r
\r
    float curledAngle = angle * branchCount\r
      + uCurl * sin(radius * (8.0 + 5.0 * randomB) - time * 1.2 + phase)\r
      + sin(radius * 21.0 - time * 0.7) * 0.35;\r
    float branchDistance = abs(sin(curledAngle));\r
    float trunk = exp(-branchDistance * (19.0 - min(uGrowth, 2.0) * 3.0));\r
    float reach = 1.0 - smoothstep(0.08, 0.48 + uGrowth * 0.16, radius);\r
    float hollow = smoothstep(0.018, 0.075, radius);\r
    float forkPulse = 0.58 + 0.42 * sin(radius * 34.0 - time * uPulse * 3.0 + phase);\r
    float branch = trunk * reach * hollow * (0.48 + 0.52 * forkPulse);\r
\r
    float shellRadius = 0.12 + randomB * 0.06 + sin(time * uPulse + phase) * 0.018;\r
    float shell = exp(-abs(radius - shellRadius) * 75.0);\r
    float budBand = abs(fract(radius * (4.0 + uGrowth * 2.0) - time * uPulse * 0.23 + randomA) - 0.5);\r
    float bud = exp(-budBand * 24.0) * pow(trunk, 2.0) * reach;\r
\r
    coral += branch;\r
    membranes += shell;\r
    buds += bud;\r
    choirPhase += (branch + shell) * (randomA - 0.5);\r
  }\r
\r
  float heartbeat = 0.78 + 0.22 * sin(time * uPulse * 2.0 + smoke * 5.0);\r
  vec3 color = vec3(0.004, 0.007, 0.015);\r
  vec3 smokeColor = palette(uHue + smoke * 0.24 + time * 0.018);\r
  vec3 coralColor = palette(uHue + 0.42 + choirPhase * 0.035 - time * 0.012);\r
  vec3 budColor = palette(uHue + 0.78 - smoke * 0.12);\r
\r
  color += smokeColor * smoke * 0.10;\r
  color += smokeColor * arcField * (0.18 + uGlow * 0.24);\r
  color += coralColor * coral * uGlow * 0.92 * heartbeat;\r
  color += budColor * buds * uGlow * 1.35;\r
  color += mix(coralColor, vec3(0.85, 0.95, 1.0), 0.55) * membranes * uGlow;\r
  color += vec3(0.25, 0.55, 1.0) * coral * arcField * uGlow * 0.8;\r
\r
  color *= 1.0 - smoothstep(0.55, 1.5, length(point)) * 0.45;\r
  color = 1.0 - exp(-color * 1.45);\r
  color = pow(max(color, 0.0), vec3(0.86));\r
  outColor = vec4(color, 1.0);\r
}`,yr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform float uCellScale;\r
uniform float uJitter;\r
uniform float uFracture;\r
uniform float uCoreRadius;\r
uniform float uSpin;\r
uniform float uShockwave;\r
uniform float uGlow;\r
uniform float uHue;\r
uniform float uSeed;\r
\r
const float TAU = 6.28318530718;\r
\r
mat2 rotate2d(float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return mat2(c, -s, s, c);\r
}\r
\r
vec2 hash22(vec2 point) {\r
  point += fract(uSeed * 0.000001) * 19.17;\r
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p += dot(p, p.yzx + 33.33);\r
  return fract((p.xx + p.yz) * p.zy);\r
}\r
\r
void voronoi(vec2 point, float time, out float nearest, out float secondNearest, out vec2 cellId) {\r
  vec2 base = floor(point);\r
  vec2 local = fract(point);\r
  nearest = 10.0;\r
  secondNearest = 10.0;\r
  cellId = vec2(0.0);\r
\r
  for (int y = -1; y <= 1; y++) {\r
    for (int x = -1; x <= 1; x++) {\r
      vec2 offset = vec2(float(x), float(y));\r
      vec2 id = base + offset;\r
      vec2 random = hash22(id);\r
      vec2 animated = 0.5 + uJitter * 0.44 * sin(time * (0.35 + random.x * 0.4) + TAU * random);\r
      vec2 delta = offset + animated - local;\r
      float distanceSquared = dot(delta, delta);\r
      if (distanceSquared < nearest) {\r
        secondNearest = nearest;\r
        nearest = distanceSquared;\r
        cellId = id;\r
      } else if (distanceSquared < secondNearest) {\r
        secondNearest = distanceSquared;\r
      }\r
    }\r
  }\r
}\r
\r
vec3 palette(float phase) {\r
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.85, 1.0, 0.63) + vec3(0.08, 0.34, 0.67)));\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;\r
  point *= uZoom;\r
  float time = uTime * uTimeScale;\r
  point = rotate2d(time * uSpin * 0.08) * point;\r
\r
  float radius = length(point);\r
  float lens = 0.1 / max(0.16, radius);\r
  vec2 crystalPoint = point + normalize(point + vec2(0.0001)) * lens * sin(time * 0.4 + radius * 8.0);\r
\r
  float nearest;\r
  float secondNearest;\r
  vec2 cellId;\r
  voronoi(crystalPoint * uCellScale, time, nearest, secondNearest, cellId);\r
  float edgeGap = sqrt(secondNearest) - sqrt(nearest);\r
  float fractures = exp(-edgeGap * (18.0 + uFracture * 42.0));\r
  float shardPulse = 0.5 + 0.5 * sin(time * 1.4 + dot(cellId, vec2(1.7, 2.3)) + sqrt(nearest) * 12.0);\r
  float facets = pow(max(0.0, 1.0 - sqrt(nearest)), 3.0) * (0.18 + 0.42 * shardPulse);\r
\r
  float angle = atan(point.y, point.x) + time * uSpin * 0.24;\r
  float sides = 6.0 + floor(fract(uSeed * 0.00001) * 4.0);\r
  float coreShape = uCoreRadius + 0.035 * cos(angle * sides + sin(time * 0.7) * 0.7);\r
  float coreEdge = exp(-abs(radius - coreShape) * 95.0);\r
  float coreFill = 1.0 - smoothstep(coreShape * 0.25, coreShape, radius);\r
\r
  float wavePhase = radius * max(uShockwave, 0.1) * 8.0 - time * 2.1;\r
  float shock = pow(1.0 - abs(sin(wavePhase)), 12.0);\r
  shock *= smoothstep(coreShape * 0.7, coreShape * 3.8, radius)\r
    * (1.0 - smoothstep(coreShape * 3.8, coreShape * 6.5, radius));\r
  shock *= 0.55 + 0.45 * cos(angle * sides * 0.5 + time);\r
\r
  float flare = exp(-radius * 3.7) * (0.6 + 0.4 * sin(angle * sides - time * 1.8));\r
  vec3 fractureColor = palette(uHue + dot(cellId, vec2(0.037, 0.051)) + time * 0.018);\r
  vec3 coreColor = palette(uHue + 0.42 - time * 0.03);\r
  vec3 shockColor = palette(uHue + 0.78 + radius * 0.12);\r
\r
  vec3 color = vec3(0.003, 0.006, 0.016);\r
  color += fractureColor * fractures * uGlow * (0.48 + facets);\r
  color += fractureColor * facets * 0.16;\r
  color += coreColor * coreEdge * uGlow * 1.8;\r
  color += mix(coreColor, vec3(0.75, 0.9, 1.0), 0.65) * coreFill * (0.25 + flare);\r
  color += shockColor * shock * uGlow * 0.95;\r
  color += vec3(0.35, 0.55, 1.0) * fractures * shock * uGlow;\r
\r
  color *= 1.0 - smoothstep(0.7, 1.7, radius) * 0.42;\r
  color = 1.0 - exp(-color * 1.35);\r
  color = pow(max(color, 0.0), vec3(0.84));\r
  outColor = vec4(color, 1.0);\r
}`,gr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform int uPoleCount;\r
uniform float uFieldLines;\r
uniform float uViscosity;\r
uniform float uBlobSize;\r
uniform float uPoleOrbit;\r
uniform float uEyeStrength;\r
uniform float uGlow;\r
uniform float uHue;\r
uniform float uSeed;\r
\r
const float TAU = 6.28318530718;\r
\r
float hash11(float value) {\r
  return fract(sin(value * 117.13 + uSeed * 0.000019) * 43758.5453);\r
}\r
\r
vec3 palette(float phase) {\r
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.67, 0.91, 1.13) + vec3(0.03, 0.26, 0.59)));\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;\r
  point *= uZoom;\r
  float time = uTime * uTimeScale;\r
\r
  vec2 field = vec2(0.0);\r
  float potential = 0.0;\r
  float fluid = 0.0;\r
  float eyes = 0.0;\r
  float poleGlow = 0.0;\r
\r
  for (int index = 0; index < 7; index++) {\r
    if (index >= uPoleCount) {\r
      break;\r
    }\r
    float fi = float(index);\r
    float randomA = hash11(fi + 4.0);\r
    float randomB = hash11(fi + 29.0);\r
    float phase = randomA * TAU;\r
    float polarity = mod(fi, 2.0) < 1.0 ? 1.0 : -1.0;\r
    vec2 pole = vec2(\r
      sin(time * (0.13 + randomA * 0.12) + phase),\r
      cos(time * (0.11 + randomB * 0.1) + phase * 1.31)\r
    ) * uPoleOrbit * vec2(0.7, 0.48);\r
    vec2 delta = point - pole;\r
    float distanceSquared = dot(delta, delta) + 0.018;\r
    float inverseDistance = inversesqrt(distanceSquared);\r
    field += polarity * delta / distanceSquared;\r
    potential += polarity * log(distanceSquared) * 0.5;\r
\r
    float blobRadius = uBlobSize * (0.11 + randomB * 0.055);\r
    float wobble = 1.0 + uViscosity * 0.16 * sin(atan(delta.y, delta.x) * (4.0 + floor(randomA * 5.0)) + time + phase);\r
    float signedBlob = length(delta) - blobRadius * wobble;\r
    fluid += 1.0 - smoothstep(-0.01, 0.025, signedBlob);\r
    poleGlow += exp(-abs(signedBlob) * 70.0);\r
\r
    vec2 eyePoint = delta * vec2(0.8, 1.65);\r
    float eyeRadius = length(eyePoint);\r
    float eye = exp(-abs(eyeRadius - blobRadius * 0.34) * 105.0);\r
    float pupil = exp(-eyeRadius * 80.0);\r
    eyes += (eye + pupil * 1.6) * (1.0 - smoothstep(blobRadius, blobRadius * 1.3, length(delta)));\r
    fluid += inverseDistance * 0.006 * uViscosity;\r
  }\r
\r
  float fieldAngle = atan(field.y, field.x);\r
  float linePhase = potential * uFieldLines * 3.2 + fieldAngle * 1.7 - time * 0.36;\r
  float magneticLines = pow(1.0 - abs(sin(linePhase)), 10.0);\r
  magneticLines *= smoothstep(0.25, 2.2, length(field));\r
\r
  float ridge = pow(1.0 - abs(sin(length(field) * 0.18 - time * 0.7)), 8.0);\r
  ridge *= smoothstep(0.3, 5.0, length(field)) * 0.55;\r
  float silhouette = smoothstep(0.15, 1.15, fluid);\r
  float oily = 0.5 + 0.5 * sin(potential * 5.0 + time * 0.5);\r
\r
  vec3 lineColor = palette(uHue + potential * 0.08 + time * 0.016);\r
  vec3 fluidColor = palette(uHue + 0.46 + oily * 0.18);\r
  vec3 eyeColor = palette(uHue + 0.82 - time * 0.025);\r
\r
  vec3 color = vec3(0.003, 0.006, 0.012);\r
  color += lineColor * magneticLines * uGlow * 1.05;\r
  color += lineColor * ridge * uGlow * 0.62;\r
  color += fluidColor * silhouette * (0.12 + oily * 0.2);\r
  color += fluidColor * poleGlow * uGlow * 0.95;\r
  color += eyeColor * eyes * uEyeStrength * uGlow * 1.4;\r
  color += vec3(0.55, 0.72, 1.0) * magneticLines * poleGlow * uGlow;\r
\r
  color *= 1.0 - smoothstep(0.65, 1.65, length(point)) * 0.38;\r
  color = 1.0 - exp(-color * 1.5);\r
  color = pow(max(color, 0.0), vec3(0.83));\r
  outColor = vec4(color, 1.0);\r
}`,br=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform int uColumns;\r
uniform int uGlyphComplexity;\r
uniform float uProcession;\r
uniform float uPortalBend;\r
uniform float uSignalNoise;\r
uniform float uScanRate;\r
uniform float uGlow;\r
uniform float uHue;\r
uniform float uSeed;\r
\r
const float TAU = 6.28318530718;\r
\r
mat2 rotate2d(float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return mat2(c, -s, s, c);\r
}\r
\r
float hash21(vec2 point) {\r
  point += fract(uSeed * 0.000001) * 13.71;\r
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p += dot(p, p.yzx + 33.33);\r
  return fract((p.x + p.y) * p.z);\r
}\r
\r
float noise(vec2 point) {\r
  vec2 cell = floor(point);\r
  vec2 local = fract(point);\r
  local = local * local * (3.0 - 2.0 * local);\r
  return mix(\r
    mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), local.x),\r
    mix(hash21(cell + vec2(0.0, 1.0)), hash21(cell + vec2(1.0)), local.x),\r
    local.y\r
  );\r
}\r
\r
vec3 palette(float phase) {\r
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(1.0, 0.73, 0.51) + vec3(0.0, 0.31, 0.64)));\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;\r
  point *= uZoom;\r
  float time = uTime * uTimeScale;\r
\r
  float columnCount = float(uColumns);\r
  vec2 gridPoint = point * columnCount;\r
  gridPoint.y += time * uProcession * 0.22;\r
  gridPoint.x += sin(gridPoint.y * 0.7 + time) * uPortalBend * 0.12;\r
  vec2 cellId = floor(gridPoint);\r
  vec2 local = fract(gridPoint) - 0.5;\r
  float random = hash21(cellId);\r
  float phase = random * TAU;\r
\r
  local = rotate2d(sin(time * 0.37 + phase) * 0.65 + phase) * local;\r
  local += vec2(\r
    noise(cellId + time * 0.13),\r
    noise(cellId - time * 0.11)\r
  ) * uSignalNoise * 0.12 - uSignalNoise * 0.06;\r
\r
  float radius = length(local);\r
  float angle = atan(local.y, local.x);\r
  float complexity = float(uGlyphComplexity);\r
  float runeRadius = 0.2 + 0.055 * sin(angle * complexity + time * (0.7 + random) + phase);\r
  float rune = exp(-abs(radius - runeRadius) * 85.0);\r
  float spokes = exp(-abs(sin(angle * complexity + phase)) * 24.0)\r
    * smoothstep(0.06, 0.17, radius)\r
    * (1.0 - smoothstep(0.17, 0.42, radius));\r
  float orbit = exp(-abs(radius - 0.34) * 95.0)\r
    * (0.4 + 0.6 * pow(1.0 - abs(sin(angle * 3.0 - time + phase)), 8.0));\r
\r
  vec2 archPoint = vec2(local.x, local.y + 0.16);\r
  float arch = exp(-abs(length(archPoint) - 0.29) * 75.0) * smoothstep(-0.08, 0.04, local.y);\r
  float pillars = exp(-abs(abs(local.x) - 0.29) * 85.0)\r
    * smoothstep(-0.45, -0.05, local.y)\r
    * (1.0 - smoothstep(-0.05, 0.14, local.y));\r
\r
  float scan = pow(1.0 - abs(sin((gridPoint.y + random) * TAU - time * uScanRate * 2.0)), 18.0);\r
  float glyph = rune + spokes + orbit + arch + pillars;\r
  glyph *= 0.72 + 0.28 * sin(time * 2.0 + phase + radius * 14.0);\r
\r
  vec2 gridEdgeDistance = abs(fract(gridPoint) - 0.5);\r
  float gridEdge = exp(-min(0.5 - gridEdgeDistance.x, 0.5 - gridEdgeDistance.y) * 48.0);\r
  float transmission = pow(1.0 - abs(sin(\r
    point.x * columnCount * 1.7\r
      + noise(point * 4.0 + time * 0.08) * uSignalNoise * 5.0\r
      - time * uScanRate\r
  )), 12.0);\r
\r
  float globalRadius = length(point);\r
  float portalRadius = 0.64 + uPortalBend * 0.08 * sin(atan(point.y, point.x) * 6.0 - time);\r
  float portal = exp(-abs(globalRadius - portalRadius) * 65.0);\r
  float portalSpokes = pow(1.0 - abs(sin(atan(point.y, point.x) * 12.0 + time)), 16.0)\r
    * smoothstep(0.24, 0.66, globalRadius)\r
    * (1.0 - smoothstep(0.66, 1.0, globalRadius));\r
\r
  vec3 glyphColor = palette(uHue + random * 0.35 + time * 0.015);\r
  vec3 signalColor = palette(uHue + 0.58 - time * 0.022 + point.y * 0.08);\r
  vec3 portalColor = palette(uHue + 0.84 + time * 0.018);\r
\r
  vec3 color = vec3(0.003, 0.005, 0.014);\r
  color += glyphColor * glyph * uGlow * 0.82;\r
  color += glyphColor * scan * rune * uGlow * 1.35;\r
  color += signalColor * transmission * (0.2 + gridEdge * 0.45) * uGlow;\r
  color += signalColor * gridEdge * 0.11;\r
  color += portalColor * (portal + portalSpokes) * uGlow * 0.9;\r
  color += vec3(0.55, 0.78, 1.0) * portal * glyph * uGlow;\r
\r
  color *= 1.0 - smoothstep(0.7, 1.65, globalRadius) * 0.35;\r
  color = 1.0 - exp(-color * 1.45);\r
  color = pow(max(color, 0.0), vec3(0.84));\r
  outColor = vec4(color, 1.0);\r
}`,Sr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform int uMedusaCount;\r
uniform int uBellRibs;\r
uniform int uTentacles;\r
uniform float uBellSize;\r
uniform float uPulse;\r
uniform float uRiseSpeed;\r
uniform float uTentacleLength;\r
uniform float uTentacleSway;\r
uniform float uTransparency;\r
uniform float uGlow;\r
uniform float uMarineSnow;\r
uniform float uHue;\r
uniform float uSeed;\r
\r
const float TAU = 6.28318530718;\r
\r
mat2 rotate2d(float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return mat2(c, -s, s, c);\r
}\r
\r
float hash11(float value) {\r
  return fract(sin(value * 127.17 + uSeed * 0.000019) * 43758.5453);\r
}\r
\r
float hash21(vec2 point) {\r
  point += fract(uSeed * 0.000001) * 17.31;\r
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p += dot(p, p.yzx + 33.33);\r
  return fract((p.x + p.y) * p.z);\r
}\r
\r
float noise(vec2 point) {\r
  vec2 cell = floor(point);\r
  vec2 local = fract(point);\r
  local = local * local * (3.0 - 2.0 * local);\r
  return mix(\r
    mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), local.x),\r
    mix(hash21(cell + vec2(0.0, 1.0)), hash21(cell + vec2(1.0)), local.x),\r
    local.y\r
  );\r
}\r
\r
float fbm(vec2 point) {\r
  float value = 0.0;\r
  float amplitude = 0.52;\r
  for (int octave = 0; octave < 5; octave++) {\r
    value += amplitude * noise(point);\r
    point = rotate2d(0.61) * point * 2.03 + vec2(1.7, -2.1);\r
    amplitude *= 0.49;\r
  }\r
  return value;\r
}\r
\r
vec3 medusaPalette(float phase) {\r
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.82, 1.0, 0.61) + vec3(0.03, 0.3, 0.66)));\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;\r
  point *= uZoom;\r
  float time = uTime * uTimeScale;\r
\r
  float depthNoise = fbm(point * 1.25 + vec2(time * 0.018, -time * 0.025));\r
  float verticalDepth = clamp(gl_FragCoord.y / uResolution.y, 0.0, 1.0);\r
  vec3 color = mix(vec3(0.001, 0.004, 0.016), vec3(0.004, 0.032, 0.055), verticalDepth);\r
  color += vec3(0.005, 0.025, 0.04) * depthNoise;\r
\r
  float causticA = pow(1.0 - abs(sin(point.x * 2.8 + depthNoise * 5.0 + time * 0.12)), 10.0);\r
  float causticB = pow(1.0 - abs(sin(point.x * 4.1 - depthNoise * 4.0 - time * 0.09)), 12.0);\r
  float surfaceFade = smoothstep(-0.7, 1.25, point.y);\r
  color += vec3(0.02, 0.12, 0.16) * (causticA + causticB) * surfaceFade * 0.28;\r
\r
  vec3 medusaLight = vec3(0.0);\r
  float medusaMask = 0.0;\r
  float electricContact = 0.0;\r
\r
  for (int medusaIndex = 0; medusaIndex < 8; medusaIndex++) {\r
    if (medusaIndex >= uMedusaCount) {\r
      break;\r
    }\r
\r
    float fi = float(medusaIndex);\r
    float randomA = hash11(fi + 3.0);\r
    float randomB = hash11(fi + 27.0);\r
    float randomC = hash11(fi + 61.0);\r
    float phase = randomA * TAU;\r
    float depth = 0.55 + randomC * 0.65;\r
    float travel = 2.9 + uTentacleLength * 0.35;\r
    float rise = time * uRiseSpeed * (0.075 + randomB * 0.055);\r
    vec2 center = vec2(\r
      (randomA - 0.5) * 1.85 + sin(time * 0.12 + phase) * 0.12,\r
      mod(randomB * travel + rise + travel * 0.5, travel) - travel * 0.5\r
    );\r
    float scale = uBellSize * depth;\r
    vec2 local = (point - center) / max(scale, 0.05);\r
\r
    float breathing = sin(time * (0.9 + randomB * 0.55) * uPulse + phase);\r
    local.x *= 1.0 + breathing * 0.08;\r
    local.y *= 1.0 - breathing * 0.1;\r
    local = rotate2d(sin(time * 0.19 + phase) * 0.08) * local;\r
\r
    float radius = length(vec2(local.x, (local.y - 0.025) * 0.9));\r
    float skirtY = -0.075\r
      + 0.018 * cos(local.x * 68.0 + phase)\r
      + breathing * 0.012;\r
    float domeDistance = radius - 0.235;\r
    float antialias = max(fwidth(domeDistance), 0.0015);\r
    float dome = 1.0 - smoothstep(-antialias, antialias, domeDistance);\r
    float aboveSkirt = smoothstep(skirtY - antialias, skirtY + antialias, local.y);\r
    float bellBody = dome * aboveSkirt;\r
    float domeEdge = exp(-max(abs(domeDistance) - antialias, 0.0) * 80.0) * aboveSkirt;\r
    float skirtEdge = exp(-abs(local.y - skirtY) * 105.0)\r
      * (1.0 - smoothstep(0.19, 0.245, abs(local.x)));\r
\r
    float bellAngle = atan(local.y - 0.015, local.x);\r
    float ribs = pow(1.0 - abs(sin(bellAngle * float(uBellRibs) + phase)), 18.0);\r
    ribs *= bellBody * smoothstep(0.035, 0.22, radius);\r
    float organRing = exp(-abs(length(local * vec2(0.82, 1.55)) - 0.082) * 95.0) * bellBody;\r
    float organCore = exp(-length(local * vec2(0.75, 1.5)) * 52.0) * bellBody;\r
\r
    float tentacleGlow = 0.0;\r
    float oralArms = 0.0;\r
    float tentacleCount = float(uTentacles);\r
    for (int tentacleIndex = 0; tentacleIndex < 10; tentacleIndex++) {\r
      if (tentacleIndex >= uTentacles) {\r
        break;\r
      }\r
      float fj = float(tentacleIndex);\r
      float anchor = (fj / max(tentacleCount - 1.0, 1.0) - 0.5) * 0.34;\r
      float tailProgress = clamp((-local.y - 0.06) / max(uTentacleLength - 0.06, 0.05), 0.0, 1.0);\r
      float strandPhase = phase + fj * 1.73;\r
      float strandX = anchor\r
        + sin(local.y * (8.0 + randomC * 5.0) - time * (1.1 + randomA) + strandPhase)\r
          * uTentacleSway * (0.014 + tailProgress * 0.058)\r
        + sin(local.y * 21.0 + time * 0.45 + fj) * uTentacleSway * 0.008;\r
      float verticalMask = smoothstep(-uTentacleLength, -uTentacleLength + 0.12, local.y)\r
        * (1.0 - smoothstep(-0.055, -0.015, local.y));\r
      float strand = exp(-abs(local.x - strandX) * (105.0 - tailProgress * 32.0));\r
      strand *= verticalMask * (1.0 - tailProgress * 0.55);\r
      float travelingCharge = 0.55 + 0.45 * pow(\r
        1.0 - abs(sin(local.y * 18.0 - time * 2.4 + strandPhase)),\r
        5.0\r
      );\r
      tentacleGlow += strand * travelingCharge;\r
\r
      if (tentacleIndex < 3) {\r
        float armAnchor = (fj - 1.0) * 0.055;\r
        float armX = armAnchor + sin(local.y * 6.5 - time + strandPhase) * uTentacleSway * 0.045;\r
        oralArms += exp(-abs(local.x - armX) * 48.0) * verticalMask\r
          * (1.0 - smoothstep(0.15, 0.72, tailProgress));\r
      }\r
    }\r
\r
    float intensity = mix(0.5, 1.15, randomC);\r
    vec3 bellColor = medusaPalette(uHue + randomA * 0.42 + time * 0.012);\r
    vec3 organColor = medusaPalette(uHue + 0.55 + randomB * 0.23 - time * 0.018);\r
    float membrane = bellBody * (0.08 + depthNoise * 0.12) * uTransparency;\r
    float brightEdge = domeEdge + skirtEdge + ribs * 0.62;\r
\r
    medusaLight += bellColor * membrane * intensity;\r
    medusaLight += bellColor * brightEdge * uGlow * intensity;\r
    medusaLight += organColor * (organRing + organCore * 1.5) * uGlow * intensity;\r
    medusaLight += mix(bellColor, organColor, 0.45) * tentacleGlow * uGlow * 0.78 * intensity;\r
    medusaLight += organColor * oralArms * uGlow * 0.42 * intensity;\r
    medusaMask += bellBody * 0.18 + tentacleGlow * 0.06;\r
    electricContact += brightEdge * tentacleGlow;\r
  }\r
\r
  vec2 snowPoint = point * vec2(15.0, 11.0) + vec2(0.0, time * (0.18 + uRiseSpeed * 0.12));\r
  vec2 snowCell = floor(snowPoint);\r
  vec2 snowLocal = fract(snowPoint) - 0.5;\r
  vec2 snowOffset = vec2(hash21(snowCell), hash21(snowCell + 7.13)) - 0.5;\r
  float snowDistance = length(snowLocal - snowOffset * 0.72);\r
  float snow = exp(-snowDistance * 48.0) * step(0.58, hash21(snowCell + 19.7));\r
  snow *= 0.55 + 0.45 * sin(time * 1.7 + hash21(snowCell) * TAU);\r
\r
  color *= 1.0 - clamp(medusaMask, 0.0, 0.42);\r
  color += medusaLight;\r
  color += vec3(0.42, 0.78, 1.0) * snow * uMarineSnow * 0.65;\r
  color += vec3(0.75, 0.88, 1.0) * electricContact * uGlow * 0.55;\r
  color *= 1.0 - smoothstep(0.65, 1.75, length(point)) * 0.34;\r
  color = 1.0 - exp(-color * 1.3);\r
  color = pow(max(color, 0.0), vec3(0.86));\r
  outColor = vec4(color, 1.0);\r
}`,xr=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform float uFilmThickness;\r
uniform float uDiffraction;\r
uniform float uFluidWarp;\r
uniform float uDropletScale;\r
uniform float uPlasmaDensity;\r
uniform float uDischargeSpeed;\r
uniform float uSpectralContrast;\r
uniform float uGlow;\r
uniform float uHueShift;\r
uniform float uSeed;\r
\r
const float TAU = 6.28318530718;\r
\r
mat2 rotate2d(float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return mat2(c, -s, s, c);\r
}\r
\r
float hash21(vec2 point) {\r
  point += fract(uSeed * 0.000001) * 17.73;\r
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));\r
  p += dot(p, p.yzx + 33.33);\r
  return fract((p.x + p.y) * p.z);\r
}\r
\r
vec2 hash22(vec2 point) {\r
  float value = hash21(point);\r
  return vec2(value, hash21(point + value + 19.19));\r
}\r
\r
float noise(vec2 point) {\r
  vec2 cell = floor(point);\r
  vec2 local = fract(point);\r
  local = local * local * (3.0 - 2.0 * local);\r
  return mix(\r
    mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), local.x),\r
    mix(hash21(cell + vec2(0.0, 1.0)), hash21(cell + vec2(1.0)), local.x),\r
    local.y\r
  );\r
}\r
\r
float fbm(vec2 point) {\r
  float value = 0.0;\r
  float amplitude = 0.52;\r
  for (int octave = 0; octave < 6; octave++) {\r
    value += amplitude * noise(point);\r
    point = rotate2d(0.58) * point * 2.03 + vec2(1.7, -2.4);\r
    amplitude *= 0.49;\r
  }\r
  return value;\r
}\r
\r
float cellularEdge(vec2 point, float time, out float cellDistance) {\r
  vec2 base = floor(point);\r
  vec2 local = fract(point);\r
  float nearest = 10.0;\r
  float secondNearest = 10.0;\r
\r
  for (int y = -1; y <= 1; y++) {\r
    for (int x = -1; x <= 1; x++) {\r
      vec2 offset = vec2(float(x), float(y));\r
      vec2 random = hash22(base + offset);\r
      vec2 center = offset + 0.5\r
        + 0.34 * sin(time * vec2(0.23, 0.19) + random * TAU) - local;\r
      float distanceSquared = dot(center, center);\r
      if (distanceSquared < nearest) {\r
        secondNearest = nearest;\r
        nearest = distanceSquared;\r
      } else if (distanceSquared < secondNearest) {\r
        secondNearest = distanceSquared;\r
      }\r
    }\r
  }\r
\r
  cellDistance = sqrt(nearest);\r
  return sqrt(secondNearest) - sqrt(nearest);\r
}\r
\r
vec3 thinFilm(float thickness, float contrast) {\r
  vec3 wavelengths = vec3(1.0, 1.29, 1.63);\r
  vec3 phase = TAU * (thickness * wavelengths * uDiffraction + uHueShift * vec3(1.0, 0.83, 1.17));\r
  vec3 reflected = 0.5 + 0.5 * cos(phase + vec3(0.0, 0.45, 0.9));\r
  reflected = smoothstep(vec3(0.08), vec3(0.92), reflected);\r
  return pow(max(reflected, 0.0), vec3(max(contrast, 0.1)));\r
}\r
\r
void main() {\r
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;\r
  point *= uZoom;\r
  float time = uTime * uTimeScale;\r
\r
  vec2 drift = vec2(time * 0.055, -time * 0.041);\r
  vec2 warpA = vec2(\r
    fbm(point * 1.15 + drift),\r
    fbm(point * 1.15 + vec2(5.4, -3.1) - drift.yx)\r
  ) - 0.5;\r
  vec2 warpB = vec2(\r
    fbm(point * 2.05 + warpA * 1.7 - drift * 0.6),\r
    fbm(point * 2.05 - warpA.yx * 1.5 + drift * 0.45 + 8.2)\r
  ) - 0.5;\r
  vec2 fluidPoint = point + (warpA * 0.75 + warpB * 0.35) * uFluidWarp;\r
\r
  float cellDistance;\r
  float cellGap = cellularEdge(fluidPoint * uDropletScale, time, cellDistance);\r
  float oilBoundary = exp(-cellGap * 38.0);\r
  float dropletInterior = smoothstep(0.62, 0.08, cellDistance);\r
\r
  float broadFilm = fbm(fluidPoint * 1.7 - drift * 0.8);\r
  float fineFilm = fbm(fluidPoint * 4.6 + warpB * 2.1 + drift);\r
  float radialFilm = sin(length(fluidPoint + warpA * 0.3) * 8.0 - time * 0.36) * 0.5 + 0.5;\r
  float thickness = uFilmThickness * (\r
    broadFilm * 2.8\r
      + fineFilm * 0.85\r
      + radialFilm * 0.4\r
      + cellDistance * 1.25\r
  );\r
\r
  vec3 diffractionColor = thinFilm(thickness, uSpectralContrast);\r
  vec3 shiftedFilm = thinFilm(thickness + cellGap * 2.5 + 0.11, uSpectralContrast * 0.85);\r
  float diffractionRings = pow(\r
    1.0 - abs(sin((thickness + fineFilm * 0.3) * uDiffraction * TAU)),\r
    7.0\r
  );\r
\r
  float gradientField = fbm(fluidPoint * 3.2 + warpA * 2.0 - time * 0.09);\r
  float plasmaPhase = gradientField * uPlasmaDensity * 8.0\r
    + broadFilm * 5.0\r
    + cellGap * 7.0\r
    - time * uDischargeSpeed;\r
  float plasma = pow(1.0 - abs(sin(plasmaPhase)), 12.0);\r
  float boundaryPlasma = oilBoundary * pow(\r
    1.0 - abs(sin(thickness * 4.0 - time * uDischargeSpeed * 1.3)),\r
    5.0\r
  );\r
  float sparks = pow(\r
    1.0 - abs(sin((fluidPoint.x - fluidPoint.y) * 11.0 + warpB.x * 9.0 - time * 1.7)),\r
    18.0\r
  ) * plasma;\r
\r
  vec3 color = vec3(0.004, 0.006, 0.012);\r
  color += diffractionColor * (0.13 + dropletInterior * 0.24);\r
  color += shiftedFilm * oilBoundary * 0.42;\r
  color += diffractionColor * diffractionRings * (0.2 + uGlow * 0.18);\r
  color += mix(diffractionColor, vec3(0.72, 0.88, 1.0), 0.62) * plasma * uGlow * 0.92;\r
  color += mix(shiftedFilm, vec3(1.0), 0.7) * boundaryPlasma * uGlow * 1.25;\r
  color += vec3(0.72, 0.86, 1.0) * sparks * uGlow * 1.4;\r
\r
  float oilySpecular = pow(max(0.0, 1.0 - length(warpB) * 1.35), 6.0);\r
  color += shiftedFilm * oilySpecular * dropletInterior * 0.24;\r
  color *= 1.0 - smoothstep(0.65, 1.7, length(point)) * 0.34;\r
  color = 1.0 - exp(-color * 1.42);\r
  color = pow(max(color, 0.0), vec3(0.84));\r
  outColor = vec4(color, 1.0);\r
}`,pe=[...qn,...Ln,{id:"neon",name:"Neon Isoclines",description:"Electric contour bands driven by seeded radial harmonics.",fragment:Vn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"components",label:"Components",uniform:"uComponents",type:"int",value:64,min:1,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:10}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:8,min:1,max:64,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:10}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.25,min:.01,max:.75,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.05}},{id:"noiseAmount",label:"Noise Amount",uniform:"uNoiseAmount",type:"float",value:2.5,min:0,max:5,step:.05,key:{inc:"r",dec:"f",step:.1,shiftStep:.25}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tanh-terrain",name:"Tanh Terrain Isoclines",description:"Tanh warped contours with bubbling noise and topo glow.",fragment:jn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:2.1,min:.1,max:6,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"octaves",label:"Octaves",uniform:"uOctaves",type:"int",value:4,min:1,max:12,step:1,key:{inc:"3",dec:"4",step:1}},{id:"lacunarity",label:"Lacunarity",uniform:"uLacunarity",type:"float",value:1.4,min:1.01,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"gain",label:"Gain",uniform:"uGain",type:"float",value:.5,min:.01,max:.99,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:16,min:1,max:96,step:1,key:{inc:"q",dec:"a",step:4,shiftStep:12}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.2,min:.02,max:.75,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"bubbleAmp",label:"Bubble Amp",uniform:"uBubbleAmp",type:"float",value:.26,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.08}},{id:"bubbleFreq",label:"Bubble Freq",uniform:"uBubbleFreq",type:"float",value:2,min:0,max:6,step:.05,key:{inc:"r",dec:"f",step:.25,shiftStep:.75}},{id:"bubbleDetail",label:"Bubble Detail",uniform:"uBubbleDetail",type:"float",value:1.2,min:.1,max:3,step:.05,key:{inc:"t",dec:"g",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tunnel",name:"Brownian Loop Tunnel",description:"Looped tunnel with Brownian noise, fog, and hue spin.",fragment:Xn,resolutionUniform:"iResolution",timeUniform:"iTime",timeMode:"looped",loopDuration:18,loopUniform:"uLoopDuration",params:[{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:4,min:0,max:10,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"noiseScale",label:"Noise Scale",uniform:"uNoiseScale",type:"float",value:1.9,min:.1,max:4,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.5,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"fogDensity",label:"Fog Density",uniform:"uFogDensity",type:"float",value:2,min:.1,max:6,step:.05,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"baseRed",label:"Base Red",uniform:"uBaseColor",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:0},{id:"baseGreen",label:"Base Green",uniform:"uBaseColor",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:1},{id:"baseBlue",label:"Base Blue",uniform:"uBaseColor",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:2}]},{id:"prismatic-fold",name:"Prismatic Fold Raymarch",description:"Rotating folded planes with prismatic glow and controllable depth.",fragment:Jn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:24,step:1,key:{inc:"1",dec:"2",step:1,shiftStep:4}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.2,min:-1.5,max:1.5,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.5,min:.1,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:5,min:1.5,max:10,step:.1,key:{inc:"7",dec:"8",step:.2,shiftStep:.6}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"cameraDistance",label:"Camera Distance",uniform:"uCameraDistance",type:"float",value:50,min:10,max:120,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:5}},{id:"cameraSpin",label:"Camera Spin",uniform:"uCameraSpin",type:"float",value:1,min:-3,max:3,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.4}},{id:"colorMix",label:"Color Mix",uniform:"uColorMix",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"alphaGain",label:"Alpha Gain",uniform:"uAlphaGain",type:"float",value:1,min:.3,max:2,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"[",dec:"]",step:.05},component:2}]},{id:"koch",name:"Koch Snowflake",description:"Iterative snowflake edges with neon glow mixing.",fragment:Zn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.8,min:.1,max:2,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:.2,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:2,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.3,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:2}]},{id:"quasi",name:"Quasi Snowflake",description:"Quasicrystal warp with a drifting snowflake outline.",fragment:$n,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:6,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:1.1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.8,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.02,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.03,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.02},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.05,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.02},component:2}]},{id:"watercolor-drips",name:"Watercolor Drips",description:"Textured paper painted with wobbly closed-loop watercolor strokes built from vibrating harmonic curves.",fragment:dr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:4,step:.02},{id:"loopCount",label:"Loop Count",uniform:"uLoopCount",type:"int",value:12,min:1,max:24,step:1},{id:"cycleMin",label:"Cycle Min",uniform:"uCycleMin",type:"float",value:14,min:2,max:40,step:.25},{id:"cycleMax",label:"Cycle Max",uniform:"uCycleMax",type:"float",value:26,min:4,max:60,step:.25},{id:"radiusMin",label:"Radius Min",uniform:"uRadiusMin",type:"float",value:.06,min:.01,max:.4,step:.005},{id:"radiusMax",label:"Radius Max",uniform:"uRadiusMax",type:"float",value:.18,min:.02,max:.6,step:.005},{id:"centerDrift",label:"Center Drift",uniform:"uCenterDrift",type:"float",value:.03,min:0,max:.25,step:.005},{id:"centerDriftSpeed",label:"Center Drift Speed",uniform:"uCenterDriftSpeed",type:"float",value:.4,min:0,max:4,step:.05},{id:"harmonicCount",label:"Harmonic Count",uniform:"uHarmonicCount",type:"int",value:6,min:0,max:8,step:1},{id:"wobbleAmount",label:"Wobble Amount",uniform:"uWobbleAmount",type:"float",value:.22,min:0,max:.8,step:.01},{id:"wobbleFalloff",label:"Wobble Falloff",uniform:"uWobbleFalloff",type:"float",value:1.1,min:.1,max:3,step:.05},{id:"morphSpeed",label:"Morph Speed",uniform:"uMorphSpeed",type:"float",value:.5,min:-4,max:4,step:.05},{id:"vibrationAmount",label:"Vibration Amount",uniform:"uVibrationAmount",type:"float",value:.035,min:0,max:.3,step:.005},{id:"vibrationFreq",label:"Vibration Freq",uniform:"uVibrationFreq",type:"int",value:14,min:1,max:48,step:1},{id:"vibrationSpeed",label:"Vibration Speed",uniform:"uVibrationSpeed",type:"float",value:1.6,min:-8,max:8,step:.05},{id:"strokeWidthMin",label:"Stroke Width Min",uniform:"uStrokeWidthMin",type:"float",value:.08,min:.01,max:.8,step:.01},{id:"strokeWidthMax",label:"Stroke Width Max",uniform:"uStrokeWidthMax",type:"float",value:.22,min:.02,max:1.2,step:.01},{id:"widthMod",label:"Width Modulation",uniform:"uWidthMod",type:"float",value:.4,min:0,max:1,step:.02},{id:"inkTexture",label:"Ink Texture",uniform:"uInkTexture",type:"float",value:.55,min:0,max:1,step:.02},{id:"drawSoftness",label:"Draw Softness",uniform:"uDrawSoftness",type:"float",value:.16,min:.02,max:.55,step:.01},{id:"bleedStrength",label:"Bleed Strength",uniform:"uBleedStrength",type:"float",value:1.1,min:0,max:3,step:.02},{id:"bleedSpread",label:"Bleed Spread",uniform:"uBleedSpread",type:"float",value:3.2,min:.5,max:8,step:.1},{id:"soakStrength",label:"Soak Strength",uniform:"uSoakStrength",type:"float",value:.5,min:0,max:2,step:.02},{id:"dryMix",label:"Dry Mix",uniform:"uDryMix",type:"float",value:1.1,min:0,max:3,step:.02},{id:"wetMix",label:"Wet Mix",uniform:"uWetMix",type:"float",value:1.4,min:0,max:3,step:.02},{id:"absorbStrength",label:"Absorb Strength",uniform:"uAbsorbStrength",type:"float",value:1.5,min:0,max:3,step:.02},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"saturationBoost",label:"Saturation Boost",uniform:"uSaturationBoost",type:"float",value:0,min:-1,max:1,step:.01},{id:"valueBoost",label:"Value Boost",uniform:"uValueBoost",type:"float",value:0,min:-1,max:1,step:.01},{id:"pastelMix",label:"Pastel Mix",uniform:"uPastelMix",type:"float",value:.14,min:0,max:1,step:.01},{id:"paperGrainScale",label:"Paper Grain Scale",uniform:"uPaperGrainScale",type:"float",value:1,min:.05,max:4,step:.05},{id:"paperGrainAmount",label:"Paper Grain Amount",uniform:"uPaperGrainAmount",type:"float",value:.08,min:0,max:.5,step:.005},{id:"paperPulpAmount",label:"Paper Pulp Amount",uniform:"uPaperPulpAmount",type:"float",value:.05,min:0,max:.4,step:.005},{id:"paperRingDensity",label:"Paper Ring Density",uniform:"uPaperRingDensity",type:"float",value:22,min:2,max:80,step:.5},{id:"paperRingWobble",label:"Paper Ring Wobble",uniform:"uPaperRingWobble",type:"float",value:.12,min:0,max:.6,step:.01},{id:"paperRingAmount",label:"Paper Ring Amount",uniform:"uPaperRingAmount",type:"float",value:1,min:0,max:3,step:.02},{id:"paperFleckAmount",label:"Paper Fleck Amount",uniform:"uPaperFleckAmount",type:"float",value:.018,min:0,max:.25,step:.002},{id:"paperBlotchAmount",label:"Paper Blotch Amount",uniform:"uPaperBlotchAmount",type:"float",value:.06,min:0,max:.4,step:.005},{id:"vignetteStrength",label:"Vignette Strength",uniform:"uVignetteStrength",type:"float",value:.07,min:0,max:1,step:.01}]},{id:"tileable-water-plus",name:"Tileable Water Plus",description:"Tileable water ripples with tunable speed, scale, and tint.",fragment:rr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"tileScale",label:"Tile Scale",uniform:"uTileScale",type:"float",value:1,min:.5,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.2,max:2.5,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.2,min:.3,max:2.5,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"waveShift",label:"Wave Shift",uniform:"uWaveShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"tintRed",label:"Tint Red",uniform:"uTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02},component:0},{id:"tintGreen",label:"Tint Green",uniform:"uTint",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02},component:1},{id:"tintBlue",label:"Tint Blue",uniform:"uTint",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:2},{id:"swirlStrength",label:"Swirl Strength",uniform:"uSwirlStrength",type:"float",value:1.2,min:-6,max:6,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"swirlGridScale",label:"Swirl Grid Scale",uniform:"uSwirlGridScale",type:"float",value:3,min:.5,max:12,step:.1,key:{inc:"y",dec:"h",step:.1,shiftStep:.5}},{id:"swirlRadius",label:"Swirl Radius",uniform:"uSwirlRadius",type:"float",value:.55,min:.05,max:1.5,step:.01},{id:"swirlWobble",label:"Swirl Wobble",uniform:"uSwirlWobble",type:"float",value:.3,min:0,max:1.5,step:.02},{id:"swirlWobbleSpeed",label:"Swirl Wobble Speed",uniform:"uSwirlWobbleSpeed",type:"float",value:.6,min:0,max:4,step:.02},{id:"swirlSpin",label:"Swirl Spin",uniform:"uSwirlSpin",type:"float",value:.5,min:-4,max:4,step:.02},{id:"swirlDesync",label:"Swirl Desync",uniform:"uSwirlDesync",type:"float",value:1,min:0,max:3,step:.05},{id:"swirlPulse",label:"Swirl Pulse",uniform:"uSwirlPulse",type:"float",value:.35,min:0,max:1,step:.02},{id:"swirlColorTwist",label:"Swirl Color Twist",uniform:"uSwirlColorTwist",type:"float",value:.4,min:-2,max:2,step:.02},{id:"rainbowStrength",label:"Rainbow Strength",uniform:"uRainbowStrength",type:"float",value:.45,min:0,max:1,step:.02},{id:"rainbowScale",label:"Rainbow Scale",uniform:"uRainbowScale",type:"float",value:1.4,min:.1,max:6,step:.05},{id:"rainbowSpeed",label:"Rainbow Speed",uniform:"uRainbowSpeed",type:"float",value:.5,min:-4,max:4,step:.05},{id:"rainbowContrast",label:"Rainbow Contrast",uniform:"uRainbowContrast",type:"float",value:1.2,min:.1,max:4,step:.05}]},{id:"seascape",name:"Seascape Plus",description:"Raymarched ocean with tunable swell and camera drift.",fragment:er,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Freq",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:1,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:0,min:-1,max:1,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:0,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1.1,min:.6,max:1.6,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"waterBrightness",label:"Water Bright",uniform:"uWaterBrightness",type:"float",value:.6,min:.2,max:1.2,step:.02,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"waterRed",label:"Water Red",uniform:"uWaterTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02},component:0},{id:"waterGreen",label:"Water Green",uniform:"uWaterTint",type:"float",value:.09,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02},component:1},{id:"waterBlue",label:"Water Blue",uniform:"uWaterTint",type:"float",value:.18,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.02},component:2}]},{id:"acidscape",name:"Acidscape",description:"A raymarched ocean skinned with flowing oil diffraction, spectral wave crests, and racing plasma veins.",fragment:tr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Frequency",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Camera Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Camera Distance",uniform:"uCamDistance",type:"float",value:5,min:1,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Camera Yaw",uniform:"uCamYaw",type:"float",value:0,min:-1,max:1,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Camera Pitch",uniform:"uCamPitch",type:"float",value:0,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1,min:.4,max:1.8,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"waterBrightness",label:"Film Brightness",uniform:"uWaterBrightness",type:"float",value:.75,min:.1,max:1.8,step:.02,key:{inc:"u",dec:"j",step:.03,shiftStep:.1}},{id:"filmThickness",label:"Film Thickness",uniform:"uFilmThickness",type:"float",value:1,min:.15,max:3.5,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"diffraction",label:"Diffraction Bands",uniform:"uDiffraction",type:"float",value:1.35,min:.2,max:5,step:.03,key:{inc:"o",dec:"l",step:.08,shiftStep:.3}},{id:"oilScale",label:"Oil Scale",uniform:"uOilScale",type:"float",value:.35,min:.05,max:1.5,step:.01,key:{inc:"p",dec:";",step:.03,shiftStep:.1}},{id:"oilWarp",label:"Oil Warp",uniform:"uOilWarp",type:"float",value:1,min:0,max:3,step:.02},{id:"plasmaDensity",label:"Plasma Density",uniform:"uPlasmaDensity",type:"float",value:1,min:.2,max:4,step:.03},{id:"dischargeSpeed",label:"Discharge Speed",uniform:"uDischargeSpeed",type:"float",value:1,min:-4,max:4,step:.03},{id:"spectralContrast",label:"Spectral Contrast",uniform:"uSpectralContrast",type:"float",value:1.1,min:.2,max:3,step:.02},{id:"plasmaGlow",label:"Plasma Glow",uniform:"uPlasmaGlow",type:"float",value:1.2,min:.2,max:3,step:.02},{id:"acidSky",label:"Acid Sky",uniform:"uAcidSky",type:"float",value:.35,min:0,max:1,step:.01},{id:"hueShift",label:"Spectral Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"reverie-seascape",name:"Reverie Seascape",description:"Fractal Reverie's drifting color fields flow over a raymarched ocean beneath a reflected Plasma Oil Diffraction sky.",fragment:nr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Frequency",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Camera Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Camera Travel",uniform:"uCamDistance",type:"float",value:1.2,min:0,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Camera Yaw",uniform:"uCamYaw",type:"float",value:0,min:-3.14,max:3.14,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Camera Pitch",uniform:"uCamPitch",type:"float",value:.08,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"waterScale",label:"Water Fractal Scale",uniform:"uWaterScale",type:"float",value:.16,min:.02,max:.8,step:.005,key:{inc:"y",dec:"h",step:.01,shiftStep:.05}},{id:"waterSpeed",label:"Water Reverie Speed",uniform:"uWaterSpeed",type:"float",value:.16,min:0,max:3,step:.01},{id:"fractalIters",label:"Water Fractal Iters",uniform:"uFractalIters",type:"int",value:2,min:1,max:5,step:1},{id:"foldScale",label:"Water Fold Scale",uniform:"uFoldScale",type:"float",value:1.18,min:1,max:4,step:.01},{id:"foldOffset",label:"Water Fold Offset",uniform:"uFoldOffset",type:"float",value:.42,min:0,max:3,step:.01},{id:"rotSpeed",label:"Water Warp Speed",uniform:"uRotSpeed",type:"float",value:.03,min:0,max:3,step:.01},{id:"detailLevel",label:"Water Detail",uniform:"uDetailLevel",type:"float",value:1,min:.2,max:5,step:.1},{id:"smoothBlend",label:"Water Smooth Blend",uniform:"uSmoothBlend",type:"float",value:.9,min:0,max:1,step:.01},{id:"waterBrightness",label:"Water Brightness",uniform:"uWaterBrightness",type:"float",value:1.8,min:.1,max:4,step:.05,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"saturation",label:"Water Saturation",uniform:"uSaturation",type:"float",value:2,min:0,max:3,step:.01},{id:"waterContrast",label:"Water Contrast",uniform:"uWaterContrast",type:"float",value:1.1,min:.5,max:3,step:.01},{id:"waterGlow",label:"Water Glow",uniform:"uWaterGlow",type:"float",value:4.5,min:0,max:8,step:.1},{id:"waterHue",label:"Water Hue",uniform:"uWaterHue",type:"float",value:0,min:-1,max:1,step:.01},{id:"hueSpeed",label:"Water Hue Drift",uniform:"uHueSpeed",type:"float",value:.05,min:0,max:1,step:.005},{id:"reflection",label:"Sky Reflection",uniform:"uReflection",type:"float",value:.45,min:0,max:1,step:.01},{id:"skyScale",label:"Sky Oil Scale",uniform:"uSkyScale",type:"float",value:1,min:.25,max:3,step:.02},{id:"skySpeed",label:"Sky Flow Speed",uniform:"uSkySpeed",type:"float",value:1,min:0,max:3,step:.02},{id:"skyBoost",label:"Sky Brightness",uniform:"uSkyBoost",type:"float",value:1,min:0,max:1.8,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"filmThickness",label:"Sky Film Thickness",uniform:"uFilmThickness",type:"float",value:1,min:.15,max:3.5,step:.02},{id:"diffraction",label:"Sky Diffraction Bands",uniform:"uDiffraction",type:"float",value:1.35,min:.2,max:5,step:.03,key:{inc:"o",dec:"l",step:.08,shiftStep:.3}},{id:"fluidWarp",label:"Sky Fluid Warp",uniform:"uFluidWarp",type:"float",value:1,min:0,max:3,step:.02},{id:"dropletScale",label:"Sky Droplet Scale",uniform:"uDropletScale",type:"float",value:3.8,min:1,max:10,step:.1},{id:"plasmaDensity",label:"Sky Plasma Density",uniform:"uPlasmaDensity",type:"float",value:1,min:.2,max:4,step:.03},{id:"dischargeSpeed",label:"Sky Discharge Speed",uniform:"uDischargeSpeed",type:"float",value:1,min:-4,max:4,step:.03},{id:"spectralContrast",label:"Sky Spectral Contrast",uniform:"uSpectralContrast",type:"float",value:1.15,min:.2,max:3,step:.02},{id:"plasmaGlow",label:"Sky Plasma Glow",uniform:"uPlasmaGlow",type:"float",value:1.2,min:0,max:3,step:.02},{id:"skyHue",label:"Sky Spectral Shift",uniform:"uSkyHue",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"p",dec:";",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"sunset-plus",name:"Sunset Plus",description:"Volumetric sunset clouds with tunable turbulence and hue drift.",fragment:ar,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}}]},{id:"sunset-orbit",name:"Sunset Orbit",description:"Volumetric sunset with horizon wrapped around an animated periodic orbit.",fragment:or,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"warpAmp",label:"Warp Amplitude",uniform:"uWarpAmp",type:"float",value:.4,min:0,max:2,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.1}},{id:"warpFreq",label:"Warp Frequency",uniform:"uWarpFreq",type:"float",value:1.5,min:.1,max:8,step:.1,key:{inc:"y",dec:"h",step:.1,shiftStep:.5}},{id:"warpSpeed",label:"Warp Speed",uniform:"uWarpSpeed",type:"float",value:.5,min:0,max:3,step:.05,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"warpHarmonics",label:"Warp Harmonics",uniform:"uWarpHarmonics",type:"float",value:3,min:1,max:5,step:1,key:{inc:"i",dec:"k",step:1}},{id:"orbitRadius",label:"Orbit Radius",uniform:"uOrbitRadius",type:"float",value:.3,min:0,max:2,step:.02,key:{inc:"o",dec:"l",step:.02,shiftStep:.1}},{id:"orbitSpeed",label:"Orbit Speed",uniform:"uOrbitSpeed",type:"float",value:.3,min:0,max:3,step:.05,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"orbitEcc",label:"Orbit Eccentricity",uniform:"uOrbitEcc",type:"float",value:.6,min:0,max:2,step:.05},{id:"tiltAngle",label:"Tilt Angle",uniform:"uTiltAngle",type:"float",value:0,min:-1.57,max:1.57,step:.02},{id:"cloudDensity",label:"Cloud Density",uniform:"uCloudDensity",type:"float",value:1,min:.1,max:3,step:.05},{id:"fogFalloff",label:"Fog Falloff",uniform:"uFogFalloff",type:"float",value:.01,min:0,max:.1,step:.002},{id:"colorSep",label:"Color Separation",uniform:"uColorSep",type:"float",value:.15,min:0,max:1,step:.01}]},{id:"diff-chromatic",name:"Chromatic Flow",description:"Two-channel diffusion with hue-as-angle and drifting color pulses.",fragment:Yn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.998,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"rotate",label:"Rotate",uniform:"uRotate",type:"float",value:.02,min:-.2,max:.2,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.03}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.35,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"w",dec:"s",step:.01,shiftStep:.03}},{id:"valueGain",label:"Value Gain",uniform:"uValueGain",type:"float",value:2.2,min:.2,max:6,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"zippy-zaps",name:"Zippy Zaps Plus",description:"Tanh-warped chromatic flow with twistable energy and glow.",fragment:ir,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.2,min:.05,max:.5,step:.005,key:{inc:"1",dec:"2",step:.01,shiftStep:.03}},{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1,min:.2,max:2,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:1,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"iterLimit",label:"Iter Limit",uniform:"uIterLimit",type:"float",value:19,min:4,max:19,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:3}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.4,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"offsetX",label:"Offset X",uniform:"uOffsetX",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"offsetY",label:"Offset Y",uniform:"uOffsetY",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}}]},{id:"plasma-menagerie",name:"Plasma Menagerie",description:"A charged kaleidoscope where luminous amoeba-glyphs orbit, blink, and trail electric tendrils through living plasma.",fragment:hr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"plasmaScale",label:"Plasma Scale",uniform:"uPlasmaScale",type:"float",value:5.5,min:1,max:14,step:.1,key:{inc:"5",dec:"6",step:.2,shiftStep:.8}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:.85,min:0,max:2.5,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"fold",label:"Fold Symmetry",uniform:"uFold",type:"int",value:7,min:1,max:16,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:3}},{id:"morph",label:"Creature Morph",uniform:"uMorph",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"shapeCount",label:"Creature Count",uniform:"uShapeCount",type:"int",value:6,min:1,max:8,step:1,key:{inc:"e",dec:"d",step:1,shiftStep:2}},{id:"orbit",label:"Orbit Reach",uniform:"uOrbit",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"filament",label:"Filament Density",uniform:"uFilament",type:"float",value:.65,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.03,shiftStep:.1}},{id:"glow",label:"Discharge Glow",uniform:"uGlow",type:"float",value:1.1,min:.2,max:2.5,step:.02,key:{inc:"y",dec:"h",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:-3,max:3,step:.02,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-coral-choir",name:"Plasma Coral Choir",description:"Branching electric colonies grow antlers, pulse charged buds, and sing through a smoky plasma reef.",fragment:vr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.4,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"branches",label:"Branch Count",uniform:"uBranches",type:"int",value:7,min:3,max:16,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:3}},{id:"growth",label:"Colony Growth",uniform:"uGrowth",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"curl",label:"Branch Curl",uniform:"uCurl",type:"float",value:1.15,min:0,max:4,step:.03,key:{inc:"q",dec:"a",step:.08,shiftStep:.3}},{id:"pulse",label:"Choir Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:4,step:.03,key:{inc:"w",dec:"s",step:.08,shiftStep:.3}},{id:"arcDensity",label:"Arc Density",uniform:"uArcDensity",type:"float",value:1,min:.2,max:3,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"glow",label:"Bioelectric Glow",uniform:"uGlow",type:"float",value:1.2,min:.2,max:3,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"hue",label:"Reef Hue",uniform:"uHue",type:"float",value:.05,min:-1,max:1,step:.01,key:{inc:"t",dec:"g",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-crystal-reactor",name:"Plasma Crystal Reactor",description:"A polygonal reactor fractures its surroundings into restless crystal cells and ejects prismatic shock rings.",fragment:yr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cellScale",label:"Crystal Scale",uniform:"uCellScale",type:"float",value:5,min:1.5,max:12,step:.1,key:{inc:"5",dec:"6",step:.2,shiftStep:.8}},{id:"jitter",label:"Lattice Jitter",uniform:"uJitter",type:"float",value:.75,min:0,max:1.5,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"fracture",label:"Fracture Sharpness",uniform:"uFracture",type:"float",value:.65,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"coreRadius",label:"Core Radius",uniform:"uCoreRadius",type:"float",value:.2,min:.05,max:.55,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"spin",label:"Reactor Spin",uniform:"uSpin",type:"float",value:1,min:-4,max:4,step:.03,key:{inc:"e",dec:"d",step:.08,shiftStep:.3}},{id:"shockwave",label:"Shockwave Density",uniform:"uShockwave",type:"float",value:1,min:.2,max:3,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"glow",label:"Crystal Glow",uniform:"uGlow",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"hue",label:"Prism Hue",uniform:"uHue",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"y",dec:"h",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-ferrofluid-oracle",name:"Plasma Ferrofluid Oracle",description:"Wandering magnetic eyes pull liquid metal into impossible bodies while luminous field lines reveal their predictions.",fragment:gr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"poleCount",label:"Magnetic Poles",uniform:"uPoleCount",type:"int",value:5,min:2,max:7,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:2}},{id:"fieldLines",label:"Field Line Density",uniform:"uFieldLines",type:"float",value:2,min:.3,max:6,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.4}},{id:"viscosity",label:"Fluid Viscosity",uniform:"uViscosity",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"blobSize",label:"Oracle Size",uniform:"uBlobSize",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"poleOrbit",label:"Pole Orbit",uniform:"uPoleOrbit",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"eyeStrength",label:"Oracle Eyes",uniform:"uEyeStrength",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"glow",label:"Magnetic Glow",uniform:"uGlow",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"hue",label:"Oil Hue",uniform:"uHue",type:"float",value:.05,min:-1,max:1,step:.01,key:{inc:"y",dec:"h",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-signal-cathedral",name:"Plasma Signal Cathedral",description:"An endless procession of alien plasma runes transmits through bending arches and a radiant central portal.",fragment:br,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"columns",label:"Rune Columns",uniform:"uColumns",type:"int",value:4,min:2,max:10,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:2}},{id:"glyphComplexity",label:"Glyph Complexity",uniform:"uGlyphComplexity",type:"int",value:7,min:3,max:16,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:3}},{id:"procession",label:"Rune Procession",uniform:"uProcession",type:"float",value:1,min:-3,max:3,step:.03,key:{inc:"q",dec:"a",step:.08,shiftStep:.3}},{id:"portalBend",label:"Portal Bend",uniform:"uPortalBend",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"signalNoise",label:"Signal Corruption",uniform:"uSignalNoise",type:"float",value:.65,min:0,max:2,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"scanRate",label:"Scan Rate",uniform:"uScanRate",type:"float",value:1,min:-4,max:4,step:.03,key:{inc:"r",dec:"f",step:.08,shiftStep:.3}},{id:"glow",label:"Sanctum Glow",uniform:"uGlow",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"hue",label:"Signal Hue",uniform:"uHue",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"y",dec:"h",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasmic-deep-sea-medusas",name:"Plasmic Deep-Sea Medusas",description:"Translucent abyssal medusas breathe, ascend, and trail charged tendrils through caustic haze and marine snow.",fragment:Sr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"medusaCount",label:"Medusa Count",uniform:"uMedusaCount",type:"int",value:6,min:1,max:8,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:2}},{id:"bellRibs",label:"Bell Ribs",uniform:"uBellRibs",type:"int",value:9,min:3,max:18,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:3}},{id:"tentacles",label:"Tentacles",uniform:"uTentacles",type:"int",value:7,min:3,max:10,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:2}},{id:"bellSize",label:"Bell Size",uniform:"uBellSize",type:"float",value:1,min:.45,max:2,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"pulse",label:"Breathing Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:4,step:.03,key:{inc:"e",dec:"d",step:.08,shiftStep:.3}},{id:"riseSpeed",label:"Ascent Speed",uniform:"uRiseSpeed",type:"float",value:1,min:-2,max:4,step:.03,key:{inc:"r",dec:"f",step:.08,shiftStep:.3}},{id:"tentacleLength",label:"Tentacle Length",uniform:"uTentacleLength",type:"float",value:.82,min:.25,max:1.6,step:.02,key:{inc:"t",dec:"g",step:.04,shiftStep:.15}},{id:"tentacleSway",label:"Tentacle Sway",uniform:"uTentacleSway",type:"float",value:1,min:0,max:3,step:.03,key:{inc:"y",dec:"h",step:.08,shiftStep:.3}},{id:"transparency",label:"Bell Transparency",uniform:"uTransparency",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"glow",label:"Bioluminescence",uniform:"uGlow",type:"float",value:1.2,min:.2,max:3,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"marineSnow",label:"Marine Snow",uniform:"uMarineSnow",type:"float",value:.75,min:0,max:2.5,step:.02,key:{inc:"o",dec:"l",step:.05,shiftStep:.2}},{id:"hue",label:"Abyssal Hue",uniform:"uHue",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"p",dec:";",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-oil-diffraction",name:"Plasma Oil Diffraction",description:"Charged filaments ignite along flowing oil membranes while thin-film diffraction splits every ripple into spectral bands.",fragment:xr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"filmThickness",label:"Film Thickness",uniform:"uFilmThickness",type:"float",value:1,min:.15,max:3.5,step:.02,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"diffraction",label:"Diffraction Bands",uniform:"uDiffraction",type:"float",value:1.35,min:.2,max:5,step:.03,key:{inc:"7",dec:"8",step:.08,shiftStep:.3}},{id:"fluidWarp",label:"Fluid Warp",uniform:"uFluidWarp",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"dropletScale",label:"Droplet Scale",uniform:"uDropletScale",type:"float",value:3.8,min:1,max:10,step:.1,key:{inc:"w",dec:"s",step:.2,shiftStep:.8}},{id:"plasmaDensity",label:"Plasma Density",uniform:"uPlasmaDensity",type:"float",value:1,min:.2,max:4,step:.03,key:{inc:"e",dec:"d",step:.08,shiftStep:.3}},{id:"dischargeSpeed",label:"Discharge Speed",uniform:"uDischargeSpeed",type:"float",value:1,min:-4,max:4,step:.03,key:{inc:"r",dec:"f",step:.08,shiftStep:.3}},{id:"spectralContrast",label:"Spectral Contrast",uniform:"uSpectralContrast",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"glow",label:"Plasma Glow",uniform:"uGlow",type:"float",value:1.2,min:.2,max:3,step:.02,key:{inc:"y",dec:"h",step:.05,shiftStep:.2}},{id:"hueShift",label:"Spectral Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"u",dec:"j",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"thunderstorm",name:"Thunderstorm",description:"Top-down storm clouds with stepped-leader lightning, return strokes, restrikes, and branching plasma channels.",fragment:lr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.2,max:3,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"cloudScale",label:"Cloud Scale",uniform:"uCloudScale",type:"float",value:3,min:.5,max:10,step:.1,key:{inc:"5",dec:"6",step:.2,shiftStep:1}},{id:"cloudSpeed",label:"Cloud Speed",uniform:"uCloudSpeed",type:"float",value:1,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"cloudDensity",label:"Cloud Density",uniform:"uCloudDensity",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.02,shiftStep:.1}},{id:"cloudDetail",label:"Cloud Detail",uniform:"uCloudDetail",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.1}},{id:"boltCount",label:"Bolt Count",uniform:"uBoltCount",type:"int",value:6,min:1,max:12,step:1,key:{inc:"e",dec:"d",step:1,shiftStep:2}},{id:"boltLengthMin",label:"Bolt Length Min",uniform:"uBoltLengthMin",type:"float",value:.25,min:.02,max:.4,step:.01,key:{inc:"r",dec:"f",step:.01,shiftStep:.05}},{id:"boltLengthMax",label:"Bolt Length Max",uniform:"uBoltLengthMax",type:"float",value:.7,min:.1,max:.8,step:.01,key:{inc:"t",dec:"g",step:.01,shiftStep:.05}},{id:"boltWidth",label:"Bolt Width",uniform:"uBoltWidth",type:"float",value:8e-4,min:1e-4,max:.005,step:1e-4,key:{inc:"y",dec:"h",step:2e-4,shiftStep:.001}},{id:"boltWiggle",label:"Bolt Wiggle",uniform:"uBoltWiggle",type:"float",value:.075,min:0,max:.2,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"boltNoiseScale",label:"Bolt Noise Scale",uniform:"uBoltNoiseScale",type:"float",value:15,min:3,max:60,step:.5,key:{inc:"i",dec:"k",step:1,shiftStep:3}},{id:"boltNoiseSpeed",label:"Restrike Speed",uniform:"uBoltNoiseSpeed",type:"float",value:2,min:0,max:8,step:.05,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"boltBranching",label:"Bolt Branching",uniform:"uBoltBranching",type:"float",value:.72,min:0,max:1,step:.02,key:{inc:"p",dec:";",step:.02,shiftStep:.1}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:.65,min:.05,max:1.5,step:.02},{id:"flickerSpeed",label:"Strike Rate",uniform:"uFlickerSpeed",type:"float",value:2,min:0,max:8,step:.1},{id:"cloudIllumination",label:"Cloud Illumination",uniform:"uCloudIllumination",type:"float",value:1,min:0,max:3,step:.05},{id:"noiseOctaves",label:"Noise Octaves",uniform:"uNoiseOctaves",type:"int",value:5,min:1,max:8,step:1},{id:"cloudRed",label:"Cloud Red",uniform:"uCloudColor",type:"float",value:.12,min:0,max:1,step:.01,component:0},{id:"cloudGreen",label:"Cloud Green",uniform:"uCloudColor",type:"float",value:.14,min:0,max:1,step:.01,component:1},{id:"cloudBlue",label:"Cloud Blue",uniform:"uCloudColor",type:"float",value:.16,min:0,max:1,step:.01,component:2},{id:"lightningRed",label:"Lightning Red",uniform:"uLightningColor",type:"float",value:.78,min:0,max:1,step:.01,component:0},{id:"lightningGreen",label:"Lightning Green",uniform:"uLightningColor",type:"float",value:.84,min:0,max:1,step:.01,component:1},{id:"lightningBlue",label:"Lightning Blue",uniform:"uLightningColor",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"mandelbulb-inside-plus",name:"Inside the Mandelbulb Plus",description:"Raymarched mandelbulb interior with tunable optics and palette glow.",fragment:sr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"power",label:"Power",uniform:"uPower",type:"float",value:8,min:2,max:12,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"bulbSpin",label:"Bulb Spin",uniform:"uBulbSpin",type:"float",value:.2,min:0,max:1.5,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"loops",label:"Loops",uniform:"uLoops",type:"int",value:2,min:1,max:6,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:1}},{id:"rayMarches",label:"Ray Marches",uniform:"uRayMarches",type:"int",value:60,min:20,max:96,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:5}},{id:"maxRayLength",label:"Max Ray Length",uniform:"uMaxRayLength",type:"float",value:20,min:5,max:40,step:.5,key:{inc:"w",dec:"s",step:.5,shiftStep:2}},{id:"tolerance",label:"Tolerance",uniform:"uTolerance",type:"float",value:1e-4,min:1e-5,max:.001,step:1e-5,key:{inc:"e",dec:"d",step:2e-5,shiftStep:1e-4}},{id:"normOffset",label:"Normal Offset",uniform:"uNormOffset",type:"float",value:.005,min:.001,max:.02,step:5e-4,key:{inc:"r",dec:"f",step:5e-4,shiftStep:.002}},{id:"bounces",label:"Bounces",uniform:"uBounces",type:"int",value:5,min:1,max:5,step:1,key:{inc:"t",dec:"g",step:1,shiftStep:1}},{id:"initStep",label:"Init Step",uniform:"uInitStep",type:"float",value:.1,min:.01,max:.3,step:.01,key:{inc:"y",dec:"h",step:.01,shiftStep:.05}},{id:"rotSpeedX",label:"Rot Speed X",uniform:"uRotSpeedX",type:"float",value:.2,min:-1,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.1}},{id:"rotSpeedY",label:"Rot Speed Y",uniform:"uRotSpeedY",type:"float",value:.3,min:-1,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:2,max:10,step:.1,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:2,min:.5,max:5,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"fov",label:"FOV",uniform:"uFov",type:"float",value:.523,min:.3,max:1.2,step:.01},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"glowBoost",label:"Glow Boost",uniform:"uGlowBoost",type:"float",value:1.2,min:0,max:4,step:.05},{id:"glowFalloff",label:"Glow Falloff",uniform:"uGlowFalloff",type:"float",value:.06,min:.01,max:.2,step:.005},{id:"diffuseBoost",label:"Diffuse Boost",uniform:"uDiffuseBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"matTransmit",label:"Mat Transmit",uniform:"uMatTransmit",type:"float",value:.8,min:0,max:1,step:.01},{id:"matReflect",label:"Mat Reflect",uniform:"uMatReflect",type:"float",value:.5,min:0,max:1,step:.01},{id:"refractIndex",label:"Refract Index",uniform:"uRefractIndex",type:"float",value:1.05,min:1,max:2,step:.01},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"glowHueOffset",label:"Glow Hue Offset",uniform:"uGlowHueOffset",type:"float",value:.065,min:-.5,max:.5,step:.005},{id:"nebulaMix",label:"Nebula Mix",uniform:"uNebulaMix",type:"float",value:0,min:0,max:1,step:.01},{id:"nebulaHueShift",label:"Nebula Hue",uniform:"uNebulaHueShift",type:"float",value:.12,min:-1,max:1,step:.01},{id:"nebulaSat",label:"Nebula Sat",uniform:"uNebulaSat",type:"float",value:.9,min:0,max:1,step:.01},{id:"nebulaVal",label:"Nebula Val",uniform:"uNebulaVal",type:"float",value:1.6,min:.2,max:3,step:.02},{id:"nebulaGlowHue",label:"Nebula Glow Hue",uniform:"uNebulaGlowHue",type:"float",value:.35,min:-1,max:1,step:.01},{id:"nebulaGlowBoost",label:"Nebula Glow",uniform:"uNebulaGlowBoost",type:"float",value:1.6,min:0,max:4,step:.05},{id:"skySat",label:"Sky Saturation",uniform:"uSkySat",type:"float",value:.86,min:0,max:1,step:.01},{id:"skyVal",label:"Sky Value",uniform:"uSkyVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"glowSat",label:"Glow Saturation",uniform:"uGlowSat",type:"float",value:.8,min:0,max:1,step:.01},{id:"glowVal",label:"Glow Value",uniform:"uGlowVal",type:"float",value:6,min:.5,max:8,step:.1},{id:"diffuseSat",label:"Diffuse Saturation",uniform:"uDiffuseSat",type:"float",value:.85,min:0,max:1,step:.01},{id:"diffuseVal",label:"Diffuse Value",uniform:"uDiffuseVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"beerRed",label:"Beer Red",uniform:"uBeerColor",type:"float",value:.02,min:0,max:.2,step:.005,component:0},{id:"beerGreen",label:"Beer Green",uniform:"uBeerColor",type:"float",value:.08,min:0,max:.2,step:.005,component:1},{id:"beerBlue",label:"Beer Blue",uniform:"uBeerColor",type:"float",value:.12,min:0,max:.2,step:.005,component:2},{id:"lightX",label:"Light X",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:0},{id:"lightY",label:"Light Y",uniform:"uLightPos",type:"float",value:10,min:-5,max:25,step:.5,component:1},{id:"lightZ",label:"Light Z",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:2}]},{id:"auroras-plus",name:"Auroras Plus",description:"Volumetric auroras with tunable trails, palette waves, and sky glare.",fragment:ur,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"auroraSpeed",label:"Aurora Speed",uniform:"uAuroraSpeed",type:"float",value:.06,min:0,max:.2,step:.005,key:{inc:"3",dec:"4",step:.005,shiftStep:.02}},{id:"auroraScale",label:"Aurora Scale",uniform:"uAuroraScale",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"auroraWarp",label:"Aurora Warp",uniform:"uAuroraWarp",type:"float",value:.35,min:0,max:1,step:.02,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"auroraSteps",label:"Aurora Steps",uniform:"uAuroraSteps",type:"int",value:50,min:8,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"auroraBase",label:"Aurora Base",uniform:"uAuroraBase",type:"float",value:.8,min:.2,max:1.6,step:.02,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"auroraStride",label:"Aurora Stride",uniform:"uAuroraStride",type:"float",value:.002,min:2e-4,max:.01,step:2e-4,key:{inc:"e",dec:"d",step:2e-4,shiftStep:.001}},{id:"auroraCurve",label:"Aurora Curve",uniform:"uAuroraCurve",type:"float",value:1.4,min:.8,max:2.2,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"auroraIntensity",label:"Aurora Intensity",uniform:"uAuroraIntensity",type:"float",value:1.8,min:.2,max:4,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"trailBlend",label:"Trail Blend",uniform:"uTrailBlend",type:"float",value:.5,min:.1,max:.9,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"trailFalloff",label:"Trail Falloff",uniform:"uTrailFalloff",type:"float",value:.065,min:.01,max:.15,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"trailFade",label:"Trail Fade",uniform:"uTrailFade",type:"float",value:2.5,min:.5,max:5,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.4}},{id:"ditherStrength",label:"Dither Strength",uniform:"uDitherStrength",type:"float",value:.006,min:0,max:.02,step:5e-4,key:{inc:"o",dec:"l",step:5e-4,shiftStep:.002}},{id:"horizonFade",label:"Horizon Fade",uniform:"uHorizonFade",type:"float",value:.01,min:.001,max:.05,step:.001,key:{inc:"p",dec:";",step:.001,shiftStep:.005}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:-.1,min:-1,max:1,step:.01},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:.1,min:-1,max:1,step:.01},{id:"camWobble",label:"Cam Wobble",uniform:"uCamWobble",type:"float",value:.2,min:0,max:.6,step:.01},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:6.7,min:4,max:12,step:.1},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:0,min:-1,max:2,step:.05},{id:"skyStrength",label:"Sky Strength",uniform:"uSkyStrength",type:"float",value:.63,min:.1,max:2,step:.02},{id:"starDensity",label:"Star Density",uniform:"uStarDensity",type:"float",value:5e-4,min:0,max:.005,step:1e-4},{id:"starIntensity",label:"Star Intensity",uniform:"uStarIntensity",type:"float",value:.8,min:0,max:2,step:.05},{id:"reflectionStrength",label:"Reflection Strength",uniform:"uReflectionStrength",type:"float",value:.6,min:0,max:1.5,step:.05},{id:"reflectionTint",label:"Reflection Tint",uniform:"uReflectionTint",type:"float",value:1,min:0,max:2,step:.05},{id:"reflectionFog",label:"Reflection Fog",uniform:"uReflectionFog",type:"float",value:2,min:0,max:6,step:.1},{id:"colorBand",label:"Color Band",uniform:"uColorBand",type:"float",value:.043,min:0,max:.2,step:.002},{id:"colorSpeed",label:"Color Speed",uniform:"uColorSpeed",type:"float",value:0,min:-1,max:1,step:.01},{id:"vortexStrength",label:"Vortex Strength",uniform:"uVortexStrength",type:"float",value:.8,min:-4,max:4,step:.02},{id:"vortexGridScale",label:"Vortex Grid Scale",uniform:"uVortexGridScale",type:"float",value:2.5,min:.5,max:10,step:.1},{id:"vortexRadius",label:"Vortex Radius",uniform:"uVortexRadius",type:"float",value:.45,min:.05,max:1.5,step:.01},{id:"vortexWobble",label:"Vortex Wobble",uniform:"uVortexWobble",type:"float",value:.35,min:0,max:1.5,step:.02},{id:"vortexWobbleSpeed",label:"Vortex Wobble Speed",uniform:"uVortexWobbleSpeed",type:"float",value:.5,min:0,max:4,step:.02},{id:"vortexSpin",label:"Vortex Spin",uniform:"uVortexSpin",type:"float",value:.4,min:-4,max:4,step:.02},{id:"vortexDesync",label:"Vortex Desync",uniform:"uVortexDesync",type:"float",value:1,min:0,max:3,step:.05},{id:"vortexDrift",label:"Vortex Drift",uniform:"uVortexDrift",type:"float",value:.2,min:-2,max:2,step:.02},{id:"vortexColorShift",label:"Vortex Color Shift",uniform:"uVortexColorShift",type:"float",value:.5,min:-2,max:2,step:.02},{id:"oilStrength",label:"Oil Film Strength",uniform:"uOilStrength",type:"float",value:.35,min:0,max:1,step:.02},{id:"oilScale",label:"Oil Film Scale",uniform:"uOilScale",type:"float",value:1.6,min:.1,max:8,step:.05},{id:"oilSpeed",label:"Oil Film Speed",uniform:"uOilSpeed",type:"float",value:.6,min:-4,max:4,step:.05},{id:"oilContrast",label:"Oil Film Contrast",uniform:"uOilContrast",type:"float",value:1.4,min:.1,max:4,step:.05},{id:"auroraRedA",label:"Aurora Red A",uniform:"uAuroraColorA",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenA",label:"Aurora Green A",uniform:"uAuroraColorA",type:"float",value:.9,min:0,max:1,step:.01,component:1},{id:"auroraBlueA",label:"Aurora Blue A",uniform:"uAuroraColorA",type:"float",value:.6,min:0,max:1,step:.01,component:2},{id:"auroraRedB",label:"Aurora Red B",uniform:"uAuroraColorB",type:"float",value:.6,min:0,max:1,step:.01,component:0},{id:"auroraGreenB",label:"Aurora Green B",uniform:"uAuroraColorB",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"auroraBlueB",label:"Aurora Blue B",uniform:"uAuroraColorB",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"auroraRedC",label:"Aurora Red C",uniform:"uAuroraColorC",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenC",label:"Aurora Green C",uniform:"uAuroraColorC",type:"float",value:.6,min:0,max:1,step:.01,component:1},{id:"auroraBlueC",label:"Aurora Blue C",uniform:"uAuroraColorC",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedA",label:"BG Red A",uniform:"uBgColorA",type:"float",value:.05,min:0,max:1,step:.01,component:0},{id:"bgGreenA",label:"BG Green A",uniform:"uBgColorA",type:"float",value:.1,min:0,max:1,step:.01,component:1},{id:"bgBlueA",label:"BG Blue A",uniform:"uBgColorA",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedB",label:"BG Red B",uniform:"uBgColorB",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"bgGreenB",label:"BG Green B",uniform:"uBgColorB",type:"float",value:.05,min:0,max:1,step:.01,component:1},{id:"bgBlueB",label:"BG Blue B",uniform:"uBgColorB",type:"float",value:.2,min:0,max:1,step:.01,component:2}]},{id:"diff-edge-flow",name:"Edge Flow Vectors",description:"Diffusive scalar field rendered as glowing edge-flow vectors.",fragment:Kn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.996,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"blobAmp",label:"Blob Amp",uniform:"uBlobAmp",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"blobRadius",label:"Blob Radius",uniform:"uBlobRadius",type:"float",value:.07,min:.01,max:.25,step:.005,key:{inc:"q",dec:"a",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.8,min:0,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"flowGain",label:"Flow Gain",uniform:"uFlowGain",type:"float",value:3,min:.2,max:8,step:.1,key:{inc:"e",dec:"d",step:.2,shiftStep:.6}},{id:"flowThreshold",label:"Flow Threshold",uniform:"uFlowThreshold",type:"float",value:.02,min:0,max:.2,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"diff-threshold",name:"Threshold Feedback",description:"Diffusion with nonlinear feedback for digital fungus crackle.",fragment:Qn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:192,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.5,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.995,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"threshold",label:"Threshold",uniform:"uThreshold",type:"float",value:.5,min:.1,max:.9,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.06}},{id:"sharpness",label:"Sharpness",uniform:"uSharpness",type:"float",value:18,min:1,max:40,step:.5,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.08,min:0,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.2,min:0,max:1.5,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractal-fold",name:"Fractal Fold Raymarch",description:"Recursive box-folding fractal with prismatic lighting and IQ palette.",fragment:cr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.5,max:5,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"distort",label:"Distort",uniform:"uDistort",type:"float",value:2.5,min:1.5,max:4,step:.02,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:1,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.15}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.1,min:-.5,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"maxSteps",label:"Max Steps",uniform:"uMaxSteps",type:"float",value:100,min:20,max:200,step:10,key:{inc:"e",dec:"d",step:10,shiftStep:30}}]},{id:"hadamard-disk",name:"Hadamard Disk",description:"Spinning Hadamard matrix mapped onto a disk with animated colour cycling.",fragment:fr,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"order",label:"Order",uniform:"uOrder",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"rotSpeed",label:"Rotation Speed",uniform:"uRotSpeed",type:"float",value:1,min:-4,max:4,step:.1,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.95,min:.2,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"radialPow",label:"Radial Power",uniform:"uRadialPow",type:"float",value:1,min:.1,max:5,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"spiral",label:"Spiral",uniform:"uSpiral",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"smooth",label:"Smooth",uniform:"uSmooth",type:"float",value:0,min:0,max:1,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"gap",label:"Cell Gap",uniform:"uGap",type:"float",value:0,min:0,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.05}},{id:"fadeStart",label:"Fade Start",uniform:"uFadeStart",type:"float",value:.9,min:.3,max:1,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.05}},{id:"fadeWidth",label:"Fade Width",uniform:"uFadeWidth",type:"float",value:.08,min:.01,max:.5,step:.01,key:{inc:"5",dec:"6",step:.01,shiftStep:.05}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:0,min:0,max:2,step:.05,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:0,min:0,max:2,step:.05,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"bgBright",label:"Background",uniform:"uBgBright",type:"float",value:0,min:0,max:.5,step:.01},{id:"baseR",label:"Base Red",uniform:"uBaseR",type:"float",value:.75,min:0,max:1,step:.01},{id:"baseG",label:"Base Green",uniform:"uBaseG",type:"float",value:.75,min:0,max:1,step:.01},{id:"baseB",label:"Base Blue",uniform:"uBaseB",type:"float",value:.75,min:0,max:1,step:.01},{id:"ampR",label:"Amp Red",uniform:"uAmpR",type:"float",value:-.25,min:-1,max:1,step:.01},{id:"ampG",label:"Amp Green",uniform:"uAmpG",type:"float",value:.25,min:-1,max:1,step:.01},{id:"ampB",label:"Amp Blue",uniform:"uAmpB",type:"float",value:.25,min:-1,max:1,step:.01},{id:"freqR",label:"Freq Red",uniform:"uFreqR",type:"float",value:1,min:0,max:16,step:.5},{id:"freqG",label:"Freq Green",uniform:"uFreqG",type:"float",value:2,min:0,max:16,step:.5},{id:"freqB",label:"Freq Blue",uniform:"uFreqB",type:"float",value:4,min:0,max:16,step:.5},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractalReverie",name:"Fractal Reverie",description:"Continuous fractal color fields with smooth drifting motion and a saturated glowing backdrop.",fragment:mr,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.05,min:0,max:3,step:.01,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"fractalIters",label:"Fractal Iters",uniform:"uFractalIters",type:"int",value:2,min:1,max:12,step:1,key:{inc:"w",dec:"s",step:1}},{id:"foldScale",label:"Fold Scale",uniform:"uFoldScale",type:"float",value:1.18,min:1,max:4,step:.01,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.42,min:0,max:3,step:.01,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"rotSpeed",label:"Rot Speed",uniform:"uRotSpeed",type:"float",value:.03,min:0,max:3,step:.01},{id:"detailLevel",label:"Detail Level",uniform:"uDetailLevel",type:"float",value:1,min:.2,max:5,step:.1},{id:"lightIntensity",label:"Light Intensity",uniform:"uLightIntensity",type:"float",value:1.5,min:0,max:5,step:.05,key:{inc:"u",dec:"j",step:.1,shiftStep:.5}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.05,min:0,max:1,step:.005},{id:"saturation",label:"Saturation",uniform:"uSaturation",type:"float",value:2,min:0,max:3,step:.01,key:{inc:"o",dec:"l",step:.05,shiftStep:.2}},{id:"brightness",label:"Brightness",uniform:"uBrightness",type:"float",value:1.8,min:.1,max:4,step:.05},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.1,min:.5,max:3,step:.01,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"glowIntensity",label:"Glow Intensity",uniform:"uGlowIntensity",type:"float",value:4.5,min:0,max:8,step:.1,key:{inc:"[",dec:"]",step:.2,shiftStep:.5}},{id:"chromaShift",label:"Chroma Shift",uniform:"uChromaShift",type:"float",value:.5,min:0,max:20,step:.1},{id:"smoothBlend",label:"Smooth Blend",uniform:"uSmoothBlend",type:"float",value:.9,min:0,max:1,step:.01},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.68,min:.2,max:3,step:.01},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:.3,min:-3,max:3,step:.05},{id:"camOrbit",label:"Cam Orbit",uniform:"uCamOrbit",type:"float",value:.018,min:0,max:1,step:.005},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"polar-rose",name:"Polar Rose",description:"Animated polar rose curves with C1-continuous Bezier segments and chromatic glow.",fragment:pr,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"symmetry",label:"Symmetry",uniform:"uSymmetry",type:"int",value:5,min:1,max:32,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"subdivisions",label:"Subdivisions",uniform:"uSubdivisions",type:"int",value:64,min:16,max:1024,step:16,key:{inc:"w",dec:"s",step:16,shiftStep:64}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.85,min:.05,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.1}},{id:"sinAmp",label:"Sin Amplitude",uniform:"uSinAmp",type:"float",value:.4,min:0,max:2,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.1}},{id:"baseFreq",label:"Base Frequency",uniform:"uBaseFreq",type:"float",value:3,min:.5,max:12,step:.1,key:{inc:"t",dec:"g",step:.1,shiftStep:.5}},{id:"modAmp",label:"Mod Amplitude",uniform:"uModAmp",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"y",dec:"h",step:.05,shiftStep:.25}},{id:"modFreq",label:"Mod Frequency",uniform:"uModFreq",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"u",dec:"j",step:.1,shiftStep:.5}},{id:"modDiv",label:"Mod Divisor",uniform:"uModDiv",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.5}},{id:"thetaScale",label:"Theta Scale",uniform:"uThetaScale",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"o",dec:"l",step:.05,shiftStep:.25}},{id:"lineWidth",label:"Line Width",uniform:"uLineWidth",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"[",dec:"]",step:.1,shiftStep:.5}},{id:"hueCycles",label:"Hue Cycles",uniform:"uHueCycles",type:"float",value:4,min:0,max:12,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]}];pe.sort((e,t)=>e.name.localeCompare(t.name));var Bt=(e,t,n)=>{if(!t.has(e))throw TypeError("Cannot "+n)},i=(e,t,n)=>(Bt(e,t,"read from private field"),n?n.call(e):t.get(e)),b=(e,t,n)=>{if(t.has(e))throw TypeError("Cannot add the same private member more than once");t instanceof WeakSet?t.add(e):t.set(e,n)},E=(e,t,n,r)=>(Bt(e,t,"write to private field"),t.set(e,n),n),wr=(e,t,n,r)=>({set _(a){E(e,t,a)},get _(){return i(e,t,r)}}),S=(e,t,n)=>(Bt(e,t,"access private method"),n),w=new Uint8Array(8),K=new DataView(w.buffer),P=e=>[(e%256+256)%256],C=e=>(K.setUint16(0,e,!1),[w[0],w[1]]),Cr=e=>(K.setInt16(0,e,!1),[w[0],w[1]]),Jt=e=>(K.setUint32(0,e,!1),[w[1],w[2],w[3]]),p=e=>(K.setUint32(0,e,!1),[w[0],w[1],w[2],w[3]]),Tr=e=>(K.setInt32(0,e,!1),[w[0],w[1],w[2],w[3]]),fe=e=>(K.setUint32(0,Math.floor(e/2**32),!1),K.setUint32(4,e,!1),[w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7]]),Ft=e=>(K.setInt16(0,2**8*e,!1),[w[0],w[1]]),Z=e=>(K.setInt32(0,2**16*e,!1),[w[0],w[1],w[2],w[3]]),lt=e=>(K.setInt32(0,2**30*e,!1),[w[0],w[1],w[2],w[3]]),_=(e,t=!1)=>{let n=Array(e.length).fill(null).map((r,a)=>e.charCodeAt(a));return t&&n.push(0),n},Xe=e=>e&&e[e.length-1],Pt=e=>{let t;for(let n of e)(!t||n.presentationTimestamp>t.presentationTimestamp)&&(t=n);return t},$=(e,t,n=!0)=>{let r=e*t;return n?Math.round(r):r},en=e=>{let t=e*(Math.PI/180),n=Math.cos(t),r=Math.sin(t);return[n,r,0,-r,n,0,0,0,1]},tn=en(0),nn=e=>[Z(e[0]),Z(e[1]),lt(e[2]),Z(e[3]),Z(e[4]),lt(e[5]),Z(e[6]),Z(e[7]),lt(e[8])],Be=e=>!e||typeof e!="object"?e:Array.isArray(e)?e.map(Be):Object.fromEntries(Object.entries(e).map(([t,n])=>[t,Be(n)])),Se=e=>e>=0&&e<2**32,F=(e,t,n)=>({type:e,contents:t&&new Uint8Array(t.flat(10)),children:n}),A=(e,t,n,r,a)=>F(e,[P(t),Jt(n),r??[]],a),kr=e=>{let t=512;return e.fragmented?F("ftyp",[_("iso5"),p(t),_("iso5"),_("iso6"),_("mp41")]):F("ftyp",[_("isom"),p(t),_("isom"),e.holdsAvc?_("avc1"):[],_("mp41")])},dt=e=>({type:"mdat",largeSize:e}),Rr=e=>({type:"free",size:e}),Oe=(e,t,n=!1)=>F("moov",null,[Ar(t,e),...e.map(r=>Br(r,t)),n?oa(e):null]),Ar=(e,t)=>{let n=$(Math.max(0,...t.filter(l=>l.samples.length>0).map(l=>{const s=Pt(l.samples);return s.presentationTimestamp+s.duration})),yt),r=Math.max(...t.map(l=>l.id))+1,a=!Se(e)||!Se(n),o=a?fe:p;return A("mvhd",+a,0,[o(e),o(e),p(yt),o(n),Z(1),Ft(1),Array(10).fill(0),nn(tn),Array(24).fill(0),p(r)])},Br=(e,t)=>F("trak",null,[Fr(e,t),Pr(e,t)]),Fr=(e,t)=>{let n=Pt(e.samples),r=$(n?n.presentationTimestamp+n.duration:0,yt),a=!Se(t)||!Se(r),o=a?fe:p,l;return e.info.type==="video"?l=typeof e.info.rotation=="number"?en(e.info.rotation):e.info.rotation:l=tn,A("tkhd",+a,3,[o(t),o(t),p(e.id),p(0),o(r),Array(8).fill(0),C(0),C(0),Ft(e.info.type==="audio"?1:0),C(0),nn(l),Z(e.info.type==="video"?e.info.width:0),Z(e.info.type==="video"?e.info.height:0)])},Pr=(e,t)=>F("mdia",null,[Dr(e,t),Er(e.info.type==="video"?"vide":"soun"),Mr(e)]),Dr=(e,t)=>{let n=Pt(e.samples),r=$(n?n.presentationTimestamp+n.duration:0,e.timescale),a=!Se(t)||!Se(r),o=a?fe:p;return A("mdhd",+a,0,[o(t),o(t),p(e.timescale),o(r),C(21956),C(0)])},Er=e=>A("hdlr",0,0,[_("mhlr"),_(e),p(0),p(0),p(0),_("mp4-muxer-hdlr",!0)]),Mr=e=>F("minf",null,[e.info.type==="video"?zr():Gr(),Ir(),Nr(e)]),zr=()=>A("vmhd",0,1,[C(0),C(0),C(0),C(0)]),Gr=()=>A("smhd",0,0,[C(0),C(0)]),Ir=()=>F("dinf",null,[Ur()]),Ur=()=>A("dref",0,0,[p(1)],[Wr()]),Wr=()=>A("url ",0,1),Nr=e=>{const t=e.compositionTimeOffsetTable.length>1||e.compositionTimeOffsetTable.some(n=>n.sampleCompositionTimeOffset!==0);return F("stbl",null,[Or(e),Jr(e),ea(e),ta(e),na(e),ra(e),t?aa(e):null])},Or=e=>A("stsd",0,0,[p(1)],[e.info.type==="video"?Lr(ha[e.info.codec],e):Yr(ya[e.info.codec],e)]),Lr=(e,t)=>F(e,[Array(6).fill(0),C(1),C(0),C(0),Array(12).fill(0),C(t.info.width),C(t.info.height),p(4718592),p(4718592),p(0),C(1),Array(32).fill(0),C(24),Cr(65535)],[va[t.info.codec](t),t.info.decoderConfig.colorSpace?Vr(t):null]),_r={bt709:1,bt470bg:5,smpte170m:6},Hr={bt709:1,smpte170m:6,"iec61966-2-1":13},qr={rgb:0,bt709:1,bt470bg:5,smpte170m:6},Vr=e=>F("colr",[_("nclx"),C(_r[e.info.decoderConfig.colorSpace.primaries]),C(Hr[e.info.decoderConfig.colorSpace.transfer]),C(qr[e.info.decoderConfig.colorSpace.matrix]),P((e.info.decoderConfig.colorSpace.fullRange?1:0)<<7)]),jr=e=>e.info.decoderConfig&&F("avcC",[...new Uint8Array(e.info.decoderConfig.description)]),Xr=e=>e.info.decoderConfig&&F("hvcC",[...new Uint8Array(e.info.decoderConfig.description)]),Zr=e=>{if(!e.info.decoderConfig)return null;let t=e.info.decoderConfig;if(!t.colorSpace)throw new Error("'colorSpace' is required in the decoder config for VP9.");let n=t.codec.split("."),r=Number(n[1]),a=Number(n[2]),s=(Number(n[3])<<4)+(0<<1)+Number(t.colorSpace.fullRange);return A("vpcC",1,0,[P(r),P(a),P(s),P(2),P(2),P(2),C(0)])},$r=()=>{let n=(1<<7)+1;return F("av1C",[n,0,0,0])},Yr=(e,t)=>F(e,[Array(6).fill(0),C(1),C(0),C(0),p(0),C(t.info.numberOfChannels),C(16),C(0),C(0),Z(t.info.sampleRate)],[ga[t.info.codec](t)]),Kr=e=>{let t=new Uint8Array(e.info.decoderConfig.description);return A("esds",0,0,[p(58753152),P(32+t.byteLength),C(1),P(0),p(75530368),P(18+t.byteLength),P(64),P(21),Jt(0),p(130071),p(130071),p(92307584),P(t.byteLength),...t,p(109084800),P(1),P(2)])},Qr=e=>{var a;let t=3840,n=0;const r=(a=e.info.decoderConfig)==null?void 0:a.description;if(r){if(r.byteLength<18)throw new TypeError("Invalid decoder description provided for Opus; must be at least 18 bytes long.");const o=ArrayBuffer.isView(r)?new DataView(r.buffer,r.byteOffset,r.byteLength):new DataView(r);t=o.getUint16(10,!0),n=o.getInt16(14,!0)}return F("dOps",[P(0),P(e.info.numberOfChannels),C(t),p(e.info.sampleRate),Ft(n),P(0)])},Jr=e=>A("stts",0,0,[p(e.timeToSampleTable.length),e.timeToSampleTable.map(t=>[p(t.sampleCount),p(t.sampleDelta)])]),ea=e=>{if(e.samples.every(n=>n.type==="key"))return null;let t=[...e.samples.entries()].filter(([,n])=>n.type==="key");return A("stss",0,0,[p(t.length),t.map(([n])=>p(n+1))])},ta=e=>A("stsc",0,0,[p(e.compactlyCodedChunkTable.length),e.compactlyCodedChunkTable.map(t=>[p(t.firstChunk),p(t.samplesPerChunk),p(1)])]),na=e=>A("stsz",0,0,[p(0),p(e.samples.length),e.samples.map(t=>p(t.size))]),ra=e=>e.finalizedChunks.length>0&&Xe(e.finalizedChunks).offset>=2**32?A("co64",0,0,[p(e.finalizedChunks.length),e.finalizedChunks.map(t=>fe(t.offset))]):A("stco",0,0,[p(e.finalizedChunks.length),e.finalizedChunks.map(t=>p(t.offset))]),aa=e=>A("ctts",0,0,[p(e.compositionTimeOffsetTable.length),e.compositionTimeOffsetTable.map(t=>[p(t.sampleCount),p(t.sampleCompositionTimeOffset)])]),oa=e=>F("mvex",null,e.map(ia)),ia=e=>A("trex",0,0,[p(e.id),p(1),p(0),p(0),p(0)]),Zt=(e,t)=>F("moof",null,[la(e),...t.map(sa)]),la=e=>A("mfhd",0,0,[p(e)]),rn=e=>{let t=0,n=0,r=0,a=0,o=e.type==="delta";return n|=+o,o?t|=1:t|=2,t<<24|n<<16|r<<8|a},sa=e=>F("traf",null,[ua(e),ca(e),fa(e)]),ua=e=>{let t=0;t|=8,t|=16,t|=32,t|=131072;let n=e.currentChunk.samples[1]??e.currentChunk.samples[0],r={duration:n.timescaleUnitsToNextSample,size:n.size,flags:rn(n)};return A("tfhd",0,t,[p(e.id),p(r.duration),p(r.size),p(r.flags)])},ca=e=>A("tfdt",1,0,[fe($(e.currentChunk.startTimestamp,e.timescale))]),fa=e=>{let t=e.currentChunk.samples.map(k=>k.timescaleUnitsToNextSample),n=e.currentChunk.samples.map(k=>k.size),r=e.currentChunk.samples.map(rn),a=e.currentChunk.samples.map(k=>$(k.presentationTimestamp-k.decodeTimestamp,e.timescale)),o=new Set(t),l=new Set(n),s=new Set(r),c=new Set(a),f=s.size===2&&r[0]!==r[1],m=o.size>1,d=l.size>1,g=!f&&s.size>1,h=c.size>1||[...c].some(k=>k!==0),x=0;return x|=1,x|=4*+f,x|=256*+m,x|=512*+d,x|=1024*+g,x|=2048*+h,A("trun",1,x,[p(e.currentChunk.samples.length),p(e.currentChunk.offset-e.currentChunk.moofOffset||0),f?p(r[0]):[],e.currentChunk.samples.map((k,M)=>[m?p(t[M]):[],d?p(n[M]):[],g?p(r[M]):[],h?Tr(a[M]):[]])])},ma=e=>F("mfra",null,[...e.map(pa),da()]),pa=(e,t)=>A("tfra",1,0,[p(e.id),p(63),p(e.finalizedChunks.length),e.finalizedChunks.map(r=>[fe($(r.startTimestamp,e.timescale)),fe(r.moofOffset),p(t+1),p(1),p(1)])]),da=()=>A("mfro",0,0,[p(0)]),ha={avc:"avc1",hevc:"hvc1",vp9:"vp09",av1:"av01"},va={avc:jr,hevc:Xr,vp9:Zr,av1:$r},ya={aac:"mp4a",opus:"Opus"},ga={aac:Kr,opus:Qr},tt=class{},ba=class extends tt{constructor(){super(...arguments),this.buffer=null}},Dt=class extends tt{constructor(e){if(super(),this.options=e,typeof e!="object")throw new TypeError("StreamTarget requires an options object to be passed to its constructor.");if(e.onData){if(typeof e.onData!="function")throw new TypeError("options.onData, when provided, must be a function.");if(e.onData.length<2)throw new TypeError("options.onData, when provided, must be a function that takes in at least two arguments (data and position). Ignoring the position argument, which specifies the byte offset at which the data is to be written, can lead to broken outputs.")}if(e.chunked!==void 0&&typeof e.chunked!="boolean")throw new TypeError("options.chunked, when provided, must be a boolean.");if(e.chunkSize!==void 0&&(!Number.isInteger(e.chunkSize)||e.chunkSize<1024))throw new TypeError("options.chunkSize, when provided, must be an integer and not smaller than 1024.")}},Sa=class extends tt{constructor(e,t){if(super(),this.stream=e,this.options=t,!(e instanceof FileSystemWritableFileStream))throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");if(t!==void 0&&typeof t!="object")throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");if(t&&t.chunkSize!==void 0&&(!Number.isInteger(t.chunkSize)||t.chunkSize<=0))throw new TypeError("options.chunkSize, when provided, must be a positive integer")}},ae,ve,an=class{constructor(){this.pos=0,b(this,ae,new Uint8Array(8)),b(this,ve,new DataView(i(this,ae).buffer)),this.offsets=new WeakMap}seek(e){this.pos=e}writeU32(e){i(this,ve).setUint32(0,e,!1),this.write(i(this,ae).subarray(0,4))}writeU64(e){i(this,ve).setUint32(0,Math.floor(e/2**32),!1),i(this,ve).setUint32(4,e,!1),this.write(i(this,ae).subarray(0,8))}writeAscii(e){for(let t=0;t<e.length;t++)i(this,ve).setUint8(t%8,e.charCodeAt(t)),t%8===7&&this.write(i(this,ae));e.length%8!==0&&this.write(i(this,ae).subarray(0,e.length%8))}writeBox(e){if(this.offsets.set(e,this.pos),e.contents&&!e.children)this.writeBoxHeader(e,e.size??e.contents.byteLength+8),this.write(e.contents);else{let t=this.pos;if(this.writeBoxHeader(e,0),e.contents&&this.write(e.contents),e.children)for(let a of e.children)a&&this.writeBox(a);let n=this.pos,r=e.size??n-t;this.seek(t),this.writeBoxHeader(e,r),this.seek(n)}}writeBoxHeader(e,t){this.writeU32(e.largeSize?1:t),this.writeAscii(e.type),e.largeSize&&this.writeU64(t)}measureBoxHeader(e){return 8+(e.largeSize?8:0)}patchBox(e){let t=this.pos;this.seek(this.offsets.get(e)),this.writeBox(e),this.seek(t)}measureBox(e){if(e.contents&&!e.children)return this.measureBoxHeader(e)+e.contents.byteLength;{let t=this.measureBoxHeader(e);if(e.contents&&(t+=e.contents.byteLength),e.children)for(let n of e.children)n&&(t+=this.measureBox(n));return t}}};ae=new WeakMap;ve=new WeakMap;var Le,ue,Me,ke,_e,ht,xa=class extends an{constructor(e){super(),b(this,_e),b(this,Le,void 0),b(this,ue,new ArrayBuffer(2**16)),b(this,Me,new Uint8Array(i(this,ue))),b(this,ke,0),E(this,Le,e)}write(e){S(this,_e,ht).call(this,this.pos+e.byteLength),i(this,Me).set(e,this.pos),this.pos+=e.byteLength,E(this,ke,Math.max(i(this,ke),this.pos))}finalize(){S(this,_e,ht).call(this,this.pos),i(this,Le).buffer=i(this,ue).slice(0,Math.max(i(this,ke),this.pos))}};Le=new WeakMap;ue=new WeakMap;Me=new WeakMap;ke=new WeakMap;_e=new WeakSet;ht=function(e){let t=i(this,ue).byteLength;for(;t<e;)t*=2;if(t===i(this,ue).byteLength)return;let n=new ArrayBuffer(t),r=new Uint8Array(n);r.set(i(this,Me),0),E(this,ue,n),E(this,Me,r)};var wa=2**24,Ca=2,Fe,oe,Re,J,O,Ze,vt,Et,on,Mt,ln,Pe,$e,zt=class extends an{constructor(e){var t,n;super(),b(this,Ze),b(this,Et),b(this,Mt),b(this,Pe),b(this,Fe,void 0),b(this,oe,[]),b(this,Re,void 0),b(this,J,void 0),b(this,O,[]),E(this,Fe,e),E(this,Re,((t=e.options)==null?void 0:t.chunked)??!1),E(this,J,((n=e.options)==null?void 0:n.chunkSize)??wa)}write(e){i(this,oe).push({data:e.slice(),start:this.pos}),this.pos+=e.byteLength}flush(){var n,r;if(i(this,oe).length===0)return;let e=[],t=[...i(this,oe)].sort((a,o)=>a.start-o.start);e.push({start:t[0].start,size:t[0].data.byteLength});for(let a=1;a<t.length;a++){let o=e[e.length-1],l=t[a];l.start<=o.start+o.size?o.size=Math.max(o.size,l.start+l.data.byteLength-o.start):e.push({start:l.start,size:l.data.byteLength})}for(let a of e){a.data=new Uint8Array(a.size);for(let o of i(this,oe))a.start<=o.start&&o.start<a.start+a.size&&a.data.set(o.data,o.start-a.start);i(this,Re)?(S(this,Ze,vt).call(this,a.data,a.start),S(this,Pe,$e).call(this)):(r=(n=i(this,Fe).options).onData)==null||r.call(n,a.data,a.start)}i(this,oe).length=0}finalize(){i(this,Re)&&S(this,Pe,$e).call(this,!0)}};Fe=new WeakMap;oe=new WeakMap;Re=new WeakMap;J=new WeakMap;O=new WeakMap;Ze=new WeakSet;vt=function(e,t){let n=i(this,O).findIndex(s=>s.start<=t&&t<s.start+i(this,J));n===-1&&(n=S(this,Mt,ln).call(this,t));let r=i(this,O)[n],a=t-r.start,o=e.subarray(0,Math.min(i(this,J)-a,e.byteLength));r.data.set(o,a);let l={start:a,end:a+o.byteLength};if(S(this,Et,on).call(this,r,l),r.written[0].start===0&&r.written[0].end===i(this,J)&&(r.shouldFlush=!0),i(this,O).length>Ca){for(let s=0;s<i(this,O).length-1;s++)i(this,O)[s].shouldFlush=!0;S(this,Pe,$e).call(this)}o.byteLength<e.byteLength&&S(this,Ze,vt).call(this,e.subarray(o.byteLength),t+o.byteLength)};Et=new WeakSet;on=function(e,t){let n=0,r=e.written.length-1,a=-1;for(;n<=r;){let o=Math.floor(n+(r-n+1)/2);e.written[o].start<=t.start?(n=o+1,a=o):r=o-1}for(e.written.splice(a+1,0,t),(a===-1||e.written[a].end<t.start)&&a++;a<e.written.length-1&&e.written[a].end>=e.written[a+1].start;)e.written[a].end=Math.max(e.written[a].end,e.written[a+1].end),e.written.splice(a+1,1)};Mt=new WeakSet;ln=function(e){let n={start:Math.floor(e/i(this,J))*i(this,J),data:new Uint8Array(i(this,J)),written:[],shouldFlush:!1};return i(this,O).push(n),i(this,O).sort((r,a)=>r.start-a.start),i(this,O).indexOf(n)};Pe=new WeakSet;$e=function(e=!1){var t,n;for(let r=0;r<i(this,O).length;r++){let a=i(this,O)[r];if(!(!a.shouldFlush&&!e)){for(let o of a.written)(n=(t=i(this,Fe).options).onData)==null||n.call(t,a.data.subarray(o.start,o.end),a.start+o.start);i(this,O).splice(r--,1)}}};var Ta=class extends zt{constructor(e){var t;super(new Dt({onData:(n,r)=>e.stream.write({type:"write",data:n,position:r}),chunked:!0,chunkSize:(t=e.options)==null?void 0:t.chunkSize}))}},yt=1e3,ka=["avc","hevc","vp9","av1"],Ra=["aac","opus"],Aa=2082844800,Ba=["strict","offset","cross-track-offset"],v,y,Ye,N,G,D,ye,be,Gt,ie,le,De,gt,sn,bt,un,It,cn,St,fn,Ut,mn,He,xt,X,Q,Wt,pn,Ee,Ke,Qe,Nt,xe,ze,qe,wt,Fa=class{constructor(e){if(b(this,gt),b(this,bt),b(this,It),b(this,St),b(this,Ut),b(this,He),b(this,X),b(this,Wt),b(this,Ee),b(this,Qe),b(this,xe),b(this,qe),b(this,v,void 0),b(this,y,void 0),b(this,Ye,void 0),b(this,N,void 0),b(this,G,null),b(this,D,null),b(this,ye,Math.floor(Date.now()/1e3)+Aa),b(this,be,[]),b(this,Gt,1),b(this,ie,[]),b(this,le,[]),b(this,De,!1),S(this,gt,sn).call(this,e),e.video=Be(e.video),e.audio=Be(e.audio),e.fastStart=Be(e.fastStart),this.target=e.target,E(this,v,{firstTimestampBehavior:"strict",...e}),e.target instanceof ba)E(this,y,new xa(e.target));else if(e.target instanceof Dt)E(this,y,new zt(e.target));else if(e.target instanceof Sa)E(this,y,new Ta(e.target));else throw new Error(`Invalid target: ${e.target}`);S(this,St,fn).call(this),S(this,bt,un).call(this)}addVideoChunk(e,t,n,r){if(!(e instanceof EncodedVideoChunk))throw new TypeError("addVideoChunk's first argument (sample) must be of type EncodedVideoChunk.");if(t&&typeof t!="object")throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");if(n!==void 0&&(!Number.isFinite(n)||n<0))throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");if(r!==void 0&&!Number.isFinite(r))throw new TypeError("addVideoChunk's fourth argument (compositionTimeOffset), when provided, must be a real number.");let a=new Uint8Array(e.byteLength);e.copyTo(a),this.addVideoChunkRaw(a,e.type,n??e.timestamp,e.duration,t,r)}addVideoChunkRaw(e,t,n,r,a,o){if(!(e instanceof Uint8Array))throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");if(t!=="key"&&t!=="delta")throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(n)||n<0)throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(r)||r<0)throw new TypeError("addVideoChunkRaw's fourth argument (duration) must be a non-negative real number.");if(a&&typeof a!="object")throw new TypeError("addVideoChunkRaw's fifth argument (meta), when provided, must be an object.");if(o!==void 0&&!Number.isFinite(o))throw new TypeError("addVideoChunkRaw's sixth argument (compositionTimeOffset), when provided, must be a real number.");if(S(this,qe,wt).call(this),!i(this,v).video)throw new Error("No video track declared.");if(typeof i(this,v).fastStart=="object"&&i(this,G).samples.length===i(this,v).fastStart.expectedVideoChunks)throw new Error(`Cannot add more video chunks than specified in 'fastStart' (${i(this,v).fastStart.expectedVideoChunks}).`);let l=S(this,He,xt).call(this,i(this,G),e,t,n,r,a,o);if(i(this,v).fastStart==="fragmented"&&i(this,D)){for(;i(this,le).length>0&&i(this,le)[0].decodeTimestamp<=l.decodeTimestamp;){let s=i(this,le).shift();S(this,X,Q).call(this,i(this,D),s)}l.decodeTimestamp<=i(this,D).lastDecodeTimestamp?S(this,X,Q).call(this,i(this,G),l):i(this,ie).push(l)}else S(this,X,Q).call(this,i(this,G),l)}addAudioChunk(e,t,n){if(!(e instanceof EncodedAudioChunk))throw new TypeError("addAudioChunk's first argument (sample) must be of type EncodedAudioChunk.");if(t&&typeof t!="object")throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");if(n!==void 0&&(!Number.isFinite(n)||n<0))throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");let r=new Uint8Array(e.byteLength);e.copyTo(r),this.addAudioChunkRaw(r,e.type,n??e.timestamp,e.duration,t)}addAudioChunkRaw(e,t,n,r,a){if(!(e instanceof Uint8Array))throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");if(t!=="key"&&t!=="delta")throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(n)||n<0)throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(r)||r<0)throw new TypeError("addAudioChunkRaw's fourth argument (duration) must be a non-negative real number.");if(a&&typeof a!="object")throw new TypeError("addAudioChunkRaw's fifth argument (meta), when provided, must be an object.");if(S(this,qe,wt).call(this),!i(this,v).audio)throw new Error("No audio track declared.");if(typeof i(this,v).fastStart=="object"&&i(this,D).samples.length===i(this,v).fastStart.expectedAudioChunks)throw new Error(`Cannot add more audio chunks than specified in 'fastStart' (${i(this,v).fastStart.expectedAudioChunks}).`);let o=S(this,He,xt).call(this,i(this,D),e,t,n,r,a);if(i(this,v).fastStart==="fragmented"&&i(this,G)){for(;i(this,ie).length>0&&i(this,ie)[0].decodeTimestamp<=o.decodeTimestamp;){let l=i(this,ie).shift();S(this,X,Q).call(this,i(this,G),l)}o.decodeTimestamp<=i(this,G).lastDecodeTimestamp?S(this,X,Q).call(this,i(this,D),o):i(this,le).push(o)}else S(this,X,Q).call(this,i(this,D),o)}finalize(){if(i(this,De))throw new Error("Cannot finalize a muxer more than once.");if(i(this,v).fastStart==="fragmented"){for(let t of i(this,ie))S(this,X,Q).call(this,i(this,G),t);for(let t of i(this,le))S(this,X,Q).call(this,i(this,D),t);S(this,Qe,Nt).call(this,!1)}else i(this,G)&&S(this,Ee,Ke).call(this,i(this,G)),i(this,D)&&S(this,Ee,Ke).call(this,i(this,D));let e=[i(this,G),i(this,D)].filter(Boolean);if(i(this,v).fastStart==="in-memory"){let t;for(let r=0;r<2;r++){let a=Oe(e,i(this,ye)),o=i(this,y).measureBox(a);t=i(this,y).measureBox(i(this,N));let l=i(this,y).pos+o+t;for(let s of i(this,be)){s.offset=l;for(let{data:c}of s.samples)l+=c.byteLength,t+=c.byteLength}if(l<2**32)break;t>=2**32&&(i(this,N).largeSize=!0)}let n=Oe(e,i(this,ye));i(this,y).writeBox(n),i(this,N).size=t,i(this,y).writeBox(i(this,N));for(let r of i(this,be))for(let a of r.samples)i(this,y).write(a.data),a.data=null}else if(i(this,v).fastStart==="fragmented"){let t=i(this,y).pos,n=ma(e);i(this,y).writeBox(n);let r=i(this,y).pos-t;i(this,y).seek(i(this,y).pos-4),i(this,y).writeU32(r)}else{let t=i(this,y).offsets.get(i(this,N)),n=i(this,y).pos-t;i(this,N).size=n,i(this,N).largeSize=n>=2**32,i(this,y).patchBox(i(this,N));let r=Oe(e,i(this,ye));if(typeof i(this,v).fastStart=="object"){i(this,y).seek(i(this,Ye)),i(this,y).writeBox(r);let a=t-i(this,y).pos;i(this,y).writeBox(Rr(a))}else i(this,y).writeBox(r)}S(this,xe,ze).call(this),i(this,y).finalize(),E(this,De,!0)}};v=new WeakMap;y=new WeakMap;Ye=new WeakMap;N=new WeakMap;G=new WeakMap;D=new WeakMap;ye=new WeakMap;be=new WeakMap;Gt=new WeakMap;ie=new WeakMap;le=new WeakMap;De=new WeakMap;gt=new WeakSet;sn=function(e){if(typeof e!="object")throw new TypeError("The muxer requires an options object to be passed to its constructor.");if(!(e.target instanceof tt))throw new TypeError("The target must be provided and an instance of Target.");if(e.video){if(!ka.includes(e.video.codec))throw new TypeError(`Unsupported video codec: ${e.video.codec}`);if(!Number.isInteger(e.video.width)||e.video.width<=0)throw new TypeError(`Invalid video width: ${e.video.width}. Must be a positive integer.`);if(!Number.isInteger(e.video.height)||e.video.height<=0)throw new TypeError(`Invalid video height: ${e.video.height}. Must be a positive integer.`);const t=e.video.rotation;if(typeof t=="number"&&![0,90,180,270].includes(t))throw new TypeError(`Invalid video rotation: ${t}. Has to be 0, 90, 180 or 270.`);if(Array.isArray(t)&&(t.length!==9||t.some(n=>typeof n!="number")))throw new TypeError(`Invalid video transformation matrix: ${t.join()}`);if(e.video.frameRate!==void 0&&(!Number.isInteger(e.video.frameRate)||e.video.frameRate<=0))throw new TypeError(`Invalid video frame rate: ${e.video.frameRate}. Must be a positive integer.`)}if(e.audio){if(!Ra.includes(e.audio.codec))throw new TypeError(`Unsupported audio codec: ${e.audio.codec}`);if(!Number.isInteger(e.audio.numberOfChannels)||e.audio.numberOfChannels<=0)throw new TypeError(`Invalid number of audio channels: ${e.audio.numberOfChannels}. Must be a positive integer.`);if(!Number.isInteger(e.audio.sampleRate)||e.audio.sampleRate<=0)throw new TypeError(`Invalid audio sample rate: ${e.audio.sampleRate}. Must be a positive integer.`)}if(e.firstTimestampBehavior&&!Ba.includes(e.firstTimestampBehavior))throw new TypeError(`Invalid first timestamp behavior: ${e.firstTimestampBehavior}`);if(typeof e.fastStart=="object"){if(e.video){if(e.fastStart.expectedVideoChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedVideoChunks'.");if(!Number.isInteger(e.fastStart.expectedVideoChunks)||e.fastStart.expectedVideoChunks<0)throw new TypeError("'expectedVideoChunks' must be a non-negative integer.")}if(e.audio){if(e.fastStart.expectedAudioChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedAudioChunks'.");if(!Number.isInteger(e.fastStart.expectedAudioChunks)||e.fastStart.expectedAudioChunks<0)throw new TypeError("'expectedAudioChunks' must be a non-negative integer.")}}else if(![!1,"in-memory","fragmented"].includes(e.fastStart))throw new TypeError("'fastStart' option must be false, 'in-memory', 'fragmented' or an object.");if(e.minFragmentDuration!==void 0&&(!Number.isFinite(e.minFragmentDuration)||e.minFragmentDuration<0))throw new TypeError("'minFragmentDuration' must be a non-negative number.")};bt=new WeakSet;un=function(){var e;if(i(this,y).writeBox(kr({holdsAvc:((e=i(this,v).video)==null?void 0:e.codec)==="avc",fragmented:i(this,v).fastStart==="fragmented"})),E(this,Ye,i(this,y).pos),i(this,v).fastStart==="in-memory")E(this,N,dt(!1));else if(i(this,v).fastStart!=="fragmented"){if(typeof i(this,v).fastStart=="object"){let t=S(this,It,cn).call(this);i(this,y).seek(i(this,y).pos+t)}E(this,N,dt(!0)),i(this,y).writeBox(i(this,N))}S(this,xe,ze).call(this)};It=new WeakSet;cn=function(){if(typeof i(this,v).fastStart!="object")return;let e=0,t=[i(this,v).fastStart.expectedVideoChunks,i(this,v).fastStart.expectedAudioChunks];for(let n of t)n&&(e+=8*Math.ceil(2/3*n),e+=4*n,e+=12*Math.ceil(2/3*n),e+=4*n,e+=8*n);return e+=4096,e};St=new WeakSet;fn=function(){if(i(this,v).video&&E(this,G,{id:1,info:{type:"video",codec:i(this,v).video.codec,width:i(this,v).video.width,height:i(this,v).video.height,rotation:i(this,v).video.rotation??0,decoderConfig:null},timescale:i(this,v).video.frameRate??57600,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,v).audio&&(E(this,D,{id:i(this,v).video?2:1,info:{type:"audio",codec:i(this,v).audio.codec,numberOfChannels:i(this,v).audio.numberOfChannels,sampleRate:i(this,v).audio.sampleRate,decoderConfig:null},timescale:i(this,v).audio.sampleRate,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,v).audio.codec==="aac")){let e=S(this,Ut,mn).call(this,2,i(this,v).audio.sampleRate,i(this,v).audio.numberOfChannels);i(this,D).info.decoderConfig={codec:i(this,v).audio.codec,description:e,numberOfChannels:i(this,v).audio.numberOfChannels,sampleRate:i(this,v).audio.sampleRate}}};Ut=new WeakSet;mn=function(e,t,n){let a=[96e3,88200,64e3,48e3,44100,32e3,24e3,22050,16e3,12e3,11025,8e3,7350].indexOf(t),o=n,l="";l+=e.toString(2).padStart(5,"0"),l+=a.toString(2).padStart(4,"0"),a===15&&(l+=t.toString(2).padStart(24,"0")),l+=o.toString(2).padStart(4,"0");let s=Math.ceil(l.length/8)*8;l=l.padEnd(s,"0");let c=new Uint8Array(l.length/8);for(let f=0;f<l.length;f+=8)c[f/8]=parseInt(l.slice(f,f+8),2);return c};He=new WeakSet;xt=function(e,t,n,r,a,o,l){let s=r/1e6,c=(r-(l??0))/1e6,f=a/1e6,m=S(this,Wt,pn).call(this,s,c,e);return s=m.presentationTimestamp,c=m.decodeTimestamp,o!=null&&o.decoderConfig&&(e.info.decoderConfig===null?e.info.decoderConfig=o.decoderConfig:Object.assign(e.info.decoderConfig,o.decoderConfig)),{presentationTimestamp:s,decodeTimestamp:c,duration:f,data:t,size:t.byteLength,type:n,timescaleUnitsToNextSample:$(f,e.timescale)}};X=new WeakSet;Q=function(e,t){i(this,v).fastStart!=="fragmented"&&e.samples.push(t);const n=$(t.presentationTimestamp-t.decodeTimestamp,e.timescale);if(e.lastTimescaleUnits!==null){let a=$(t.decodeTimestamp,e.timescale,!1),o=Math.round(a-e.lastTimescaleUnits);if(e.lastTimescaleUnits+=o,e.lastSample.timescaleUnitsToNextSample=o,i(this,v).fastStart!=="fragmented"){let l=Xe(e.timeToSampleTable);l.sampleCount===1?(l.sampleDelta=o,l.sampleCount++):l.sampleDelta===o?l.sampleCount++:(l.sampleCount--,e.timeToSampleTable.push({sampleCount:2,sampleDelta:o}));const s=Xe(e.compositionTimeOffsetTable);s.sampleCompositionTimeOffset===n?s.sampleCount++:e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:n})}}else e.lastTimescaleUnits=0,i(this,v).fastStart!=="fragmented"&&(e.timeToSampleTable.push({sampleCount:1,sampleDelta:$(t.duration,e.timescale)}),e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:n}));e.lastSample=t;let r=!1;if(!e.currentChunk)r=!0;else{let a=t.presentationTimestamp-e.currentChunk.startTimestamp;if(i(this,v).fastStart==="fragmented"){let o=i(this,G)??i(this,D);const l=i(this,v).minFragmentDuration??1;e===o&&t.type==="key"&&a>=l&&(r=!0,S(this,Qe,Nt).call(this))}else r=a>=.5}r&&(e.currentChunk&&S(this,Ee,Ke).call(this,e),e.currentChunk={startTimestamp:t.presentationTimestamp,samples:[]}),e.currentChunk.samples.push(t)};Wt=new WeakSet;pn=function(e,t,n){var l,s;const r=i(this,v).firstTimestampBehavior==="strict",a=n.lastDecodeTimestamp===-1;if(r&&a&&t!==0)throw new Error(`The first chunk for your media track must have a timestamp of 0 (received DTS=${t}).Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of thedocument, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
`);if(i(this,v).firstTimestampBehavior==="offset"||i(this,v).firstTimestampBehavior==="cross-track-offset"){n.firstDecodeTimestamp===void 0&&(n.firstDecodeTimestamp=t);let c;i(this,v).firstTimestampBehavior==="offset"?c=n.firstDecodeTimestamp:c=Math.min(((l=i(this,G))==null?void 0:l.firstDecodeTimestamp)??1/0,((s=i(this,D))==null?void 0:s.firstDecodeTimestamp)??1/0),t-=c,e-=c}if(t<n.lastDecodeTimestamp)throw new Error(`Timestamps must be monotonically increasing (DTS went from ${n.lastDecodeTimestamp*1e6} to ${t*1e6}).`);return n.lastDecodeTimestamp=t,{presentationTimestamp:e,decodeTimestamp:t}};Ee=new WeakSet;Ke=function(e){if(i(this,v).fastStart==="fragmented")throw new Error("Can't finalize individual chunks if 'fastStart' is set to 'fragmented'.");if(e.currentChunk){if(e.finalizedChunks.push(e.currentChunk),i(this,be).push(e.currentChunk),(e.compactlyCodedChunkTable.length===0||Xe(e.compactlyCodedChunkTable).samplesPerChunk!==e.currentChunk.samples.length)&&e.compactlyCodedChunkTable.push({firstChunk:e.finalizedChunks.length,samplesPerChunk:e.currentChunk.samples.length}),i(this,v).fastStart==="in-memory"){e.currentChunk.offset=0;return}e.currentChunk.offset=i(this,y).pos;for(let t of e.currentChunk.samples)i(this,y).write(t.data),t.data=null;S(this,xe,ze).call(this)}};Qe=new WeakSet;Nt=function(e=!0){if(i(this,v).fastStart!=="fragmented")throw new Error("Can't finalize a fragment unless 'fastStart' is set to 'fragmented'.");let t=[i(this,G),i(this,D)].filter(s=>s&&s.currentChunk);if(t.length===0)return;let n=wr(this,Gt)._++;if(n===1){let s=Oe(t,i(this,ye),!0);i(this,y).writeBox(s)}let r=i(this,y).pos,a=Zt(n,t);i(this,y).writeBox(a);{let s=dt(!1),c=0;for(let m of t)for(let d of m.currentChunk.samples)c+=d.size;let f=i(this,y).measureBox(s)+c;f>=2**32&&(s.largeSize=!0,f=i(this,y).measureBox(s)+c),s.size=f,i(this,y).writeBox(s)}for(let s of t){s.currentChunk.offset=i(this,y).pos,s.currentChunk.moofOffset=r;for(let c of s.currentChunk.samples)i(this,y).write(c.data),c.data=null}let o=i(this,y).pos;i(this,y).seek(i(this,y).offsets.get(a));let l=Zt(n,t);i(this,y).writeBox(l),i(this,y).seek(o);for(let s of t)s.finalizedChunks.push(s.currentChunk),i(this,be).push(s.currentChunk),s.currentChunk=null;e&&S(this,xe,ze).call(this)};xe=new WeakSet;ze=function(){i(this,y)instanceof zt&&i(this,y).flush()};qe=new WeakSet;wt=function(){if(i(this,De))throw new Error("Cannot add new video or audio chunks after the file has been finalized.")};function Pa(e,t){const n=e.find(r=>r.id===t);if(!n)throw new Error(`Unknown scene: ${t}. Use list_scenes to find scene IDs.`);return n}function Da(e,t){for(const[n,r]of Object.entries(t)){const a=e.params.find(s=>s.id===n);if(!a)throw new Error(`Unknown parameter ${n} for ${e.id}.`);if(!Number.isFinite(r))throw new Error(`${n} must be a finite number.`);if(a.type!=="float"&&!Number.isInteger(r))throw new Error(`${n} must be an integer.`);const o=a.min??(a.type==="seed"?0:-1/0),l=a.max??(a.type==="seed"?999999:1/0);if(r<o||r>l)throw new Error(`${n} must be between ${o} and ${l}.`)}}function Ea(e){const{duration:t,fps:n,width:r,height:a}=e;if(!Number.isFinite(t)||t<1||t>3600)throw new Error("Duration must be between 1 and 3600 seconds.");if(![30,60].includes(n))throw new Error("FPS must be 30 or 60.");if(!Number.isInteger(r)||!Number.isInteger(a)||r<16||a<16||r>3840||a>2160||r%2||a%2)throw new Error("Video dimensions must be even integers from 16×16 through 3840×2160.")}async function Ma(e){const{canvas:t,render:n,sink:r,signal:a,progress:o,duration:l,fps:s,width:c,height:f}=e;let m,d,g=Promise.resolve(),h,x=!1;const k=()=>{x=!0};t.addEventListener("webglcontextlost",k);const M=()=>{if(a==null||a.throwIfAborted(),x)throw new Error("WebGL context was lost during export.");if(h)throw h};try{if(Ea(e),typeof VideoEncoder>"u"||typeof VideoFrame>"u")throw new Error("Video export requires WebCodecs in a supported Chrome or Edge browser.");const W={codec:c*f*s>1920*1080*30?"avc1.640034":"avc1.640028",width:c,height:f,bitrate:16e6,framerate:s};if(!(await VideoEncoder.isConfigSupported(W)).supported)throw new Error("H.264 encoding is unavailable at this resolution and frame rate.");const T=e.audio,L=T?Math.min(2,T.numberOfChannels):0,he=T?{codec:"mp4a.40.2",sampleRate:T.sampleRate,numberOfChannels:L,bitrate:192e3}:void 0;if(he&&(typeof AudioEncoder>"u"||typeof AudioData>"u"||!(await AudioEncoder.isConfigSupported(he)).supported))throw new Error("AAC soundtrack encoding is unavailable. Use a supported Chrome or Edge browser, or uncheck Include soundtrack in video.");M();const Bn=new Dt({chunked:!0,chunkSize:1024*1024,onData(B,ee){const V=B.slice();g=g.then(async()=>{M(),await r.write(V,ee)}).catch(re=>{h??(h=re)})}}),ot=new Fa({target:Bn,video:{codec:"avc",width:c,height:f,frameRate:s},fastStart:!1,audio:T?{codec:"aac",sampleRate:T.sampleRate,numberOfChannels:L}:void 0,firstTimestampBehavior:"offset"});m=new VideoEncoder({output:(B,ee)=>{try{ot.addVideoChunk(B,ee)}catch(V){h??(h=V)}},error:B=>{h??(h=B)}}),m.configure(W),he&&(d=new AudioEncoder({output(B,ee){try{ot.addAudioChunk(B,ee)}catch(V){h??(h=V)}},error(B){h??(h=B)}}),d.configure(he));let Ce=0;const Fn=T?Math.ceil(l*T.sampleRate):0,Ht=Math.ceil(l*s);for(let B=0;B<Ht;B++){M(),n(B/s,c,f);const ee=new VideoFrame(t,{timestamp:Math.round(B*1e6/s),duration:Math.round((B+1)*1e6/s)-Math.round(B*1e6/s)});try{m.encode(ee,{keyFrame:B%(s*2)===0})}finally{ee.close()}if(T&&d){const V=Math.min(Fn,Math.ceil((B+1)/s*T.sampleRate));for(;Ce<V;){const re=Math.min(1024,V-Ce),qt=new Float32Array(re*L);for(let Ge=0;Ge<L;Ge++){const it=T.getChannelData(Ge);for(let Ie=0;Ie<re;Ie++){const jt=Ce+Ie;qt[Ge*re+Ie]=e.loopAudio?it[jt%it.length]:it[jt]??0}}const Vt=new AudioData({format:"f32-planar",sampleRate:T.sampleRate,numberOfFrames:re,numberOfChannels:L,timestamp:Math.round(Ce*1e6/T.sampleRate),data:qt});try{d.encode(Vt)}finally{Vt.close()}Ce+=re,d.encodeQueueSize>=8&&await d.flush()}}(m.encodeQueueSize>=8||B%s===s-1)&&(await m.flush(),await(d==null?void 0:d.flush())),await g,M(),o==null||o(B+1,Ht),await new Promise(V=>setTimeout(V,0))}await m.flush(),await(d==null?void 0:d.flush()),M(),ot.finalize(),await g,M(),await r.close()}catch(H){throw h??(h=H),await g,await r.abort().catch(()=>{}),H}finally{m&&m.state!=="closed"&&m.close(),d&&d.state!=="closed"&&d.close(),t.removeEventListener("webglcontextlost",k)}}function za(e){var s;const t=new URLSearchParams(location.hash.slice(1)).get("mcp");if(!t||location.hostname!=="127.0.0.1")return;const n=document.createElement("div");n.className="panel-section small",n.textContent="MCP: connecting…",(s=document.querySelector(".panel"))==null||s.prepend(n);let r=!1,a,o;const l=()=>{const c=new AbortController,f=new URL("/control",location.href);f.protocol="ws:",f.searchParams.set("token",t),a=new WebSocket(f);const m=g=>{a.readyState===WebSocket.OPEN&&a.send(JSON.stringify(g))},d={signal:c.signal,progress:(g,h,x)=>m({event:"progress",jobId:g,completed:h,total:x}),sink:g=>({async write(h,x){const k=await fetch(`/exports/${encodeURIComponent(g)}?offset=${x}`,{method:"PUT",headers:{Authorization:`Bearer ${t}`,"Content-Type":"application/octet-stream"},body:h.slice().buffer,signal:c.signal});if(!k.ok)throw new Error(`Video upload failed: ${await k.text()}`)},close:async()=>{},abort:async()=>{}})};a.addEventListener("open",()=>{n.textContent="MCP: connected"}),a.addEventListener("message",async g=>{let h;try{h=JSON.parse(String(g.data))}catch{return}if(!(typeof h.id!="string"||typeof h.command!="string"))try{m({id:h.id,result:await e(h,d)})}catch(x){m({id:h.id,error:x instanceof Error?x.message:String(x)})}}),a.addEventListener("close",g=>{c.abort(new Error("MCP browser connection closed.")),n.textContent=g.code===1008?"MCP: another tab is already connected":"MCP: disconnected",!r&&g.code!==1008&&(o=window.setTimeout(l,2e3))}),a.addEventListener("error",()=>{n.textContent="MCP: connection unavailable"})};l(),window.addEventListener("pagehide",()=>{r=!0,clearTimeout(o),a.close()},{once:!0})}const dn=document.querySelector("#app");if(!dn)throw new Error("Missing #app root element");dn.innerHTML=`
  <div class="app-shell">
    <div class="layout">
      <aside class="panel">
        <div class="panel-section">
          <div class="section-title">Scenes</div>
          <select class="scene-select" id="scene-list"></select>
        </div>
        <div class="panel-section">
          <div class="section-title">Controls</div>
          <div class="panel-actions" id="panel-actions"></div>
          <div class="control-list" id="control-list"></div>
        </div>
        <div class="panel-section">
          <div class="section-title">Record</div>
          <div class="rec-row">
            <button class="ghost small" id="rec-btn">Record</button>
            <span class="rec-badge hidden" id="rec-badge"></span>
          </div>
        </div>
        <div class="panel-section">
          <div class="section-title">Export video</div>
          <div class="offline-controls">
            <div class="offline-row">
              <label>Duration</label>
              <input type="number" id="offline-duration" value="10" min="1" max="3600" step="1" />
              <span class="offline-unit">sec</span>
            </div>
            <div class="offline-row">
              <label>FPS</label>
              <select id="offline-fps">
                <option value="30">30</option>
                <option value="60" selected>60</option>
              </select>
            </div>
            <div class="offline-row">
              <label>Resolution</label>
              <select id="offline-res">
                <option value="1280x720">720p</option>
                <option value="1920x1080" selected>1080p</option>
                <option value="2560x1440">1440p</option>
                <option value="3840x2160">4K</option>
              </select>
            </div>
            <button class="ghost small" id="offline-btn">Export MP4</button>
            <div class="offline-progress hidden" id="offline-progress">
              <div class="offline-progress-bar" id="offline-bar"></div>
            </div>
            <div class="offline-status hidden" id="offline-status"></div>
          </div>
        </div>
        <div class="panel-section small">
          <div class="section-title">Keys</div>
          <div class="key-help" id="key-help"></div>
        </div>
      </aside>
      <main class="stage">
        <canvas id="gl-canvas"></canvas>
        <button class="sidebar-toggle" data-action="toggle-sidebar"></button>
        <div class="hud">
          <div class="hud-title" id="hud-title"></div>
          <div class="hud-desc" id="hud-desc"></div>
        </div>
      </main>
    </div>
  </div>
`;const z=document.querySelector("#gl-canvas"),ge=document.querySelector("#scene-list"),Ct=document.querySelector("#control-list"),Ve=document.querySelector("#panel-actions"),Ae=document.querySelector("#key-help"),Je=document.querySelector("#hud-title"),hn=document.querySelector("#hud-desc"),nt=document.querySelector("[data-action='toggle-sidebar']"),Ot=document.querySelector(".stage");if(!z||!ge||!Ct||!Ve||!Ae||!Je||!hn||!nt||!Ot)throw new Error("Missing required UI elements");const u=z.getContext("webgl2",{antialias:!0});if(!u)throw Ot.innerHTML=`
    <div class="fallback">
      <h2>WebGL2 unavailable</h2>
      <p>Your browser or GPU does not expose WebGL2. Try a different browser.</p>
    </div>
  `,new Error("WebGL2 unavailable");const Ga=u.getExtension("EXT_color_buffer_float"),rt=!!Ga;u.disable(u.DEPTH_TEST);u.disable(u.BLEND);const Tt=Gn(u),I=new En,te=new Un(u,Tt),ce=Wn(I,te,()=>{te.reset(),ne(U.id),at=performance.now()}),vn=new Map,yn={},me={},gn={},kt=new Map;for(const e of pe){const t=ft(u,pt,e.fragment),n=new Set;n.add(e.resolutionUniform),n.add(e.timeUniform),e.loopUniform&&n.add(e.loopUniform),e.stateful&&(n.add(e.passUniform??"uPass"),n.add(e.stateUniform??"uState"),n.add(e.gridUniform??"uGridSize"));for(const l of e.params)n.add(l.uniform);const r=mt(u,t,Array.from(n));vn.set(e.id,{program:t,uniforms:r});const a={},o={};for(const l of e.params)a[l.id]=l,o[l.id]=l.type==="seed"?Math.floor(Math.random()*1e6):l.value;yn[e.id]=a,me[e.id]={...o},gn[e.id]={...o}}let U=pe[0],Rt={},at=performance.now(),et=0,st=null,ut=null,se=null,Ue=[],bn=0,je=null,de=!1;function Ia(){const e=["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm","video/mp4"];for(const t of e)if(MediaRecorder.isTypeSupported(t))return t;return""}function Ua(e){return e.startsWith("video/mp4")?"mp4":"webm"}function Wa(e){const t=Math.floor(e/1e3),n=String(Math.floor(t/60)).padStart(2,"0"),r=String(t%60).padStart(2,"0");return`${n}:${r}`}function Na(){const e=document.getElementById("rec-badge");!e||!de||(e.textContent=`⏺ ${Wa(performance.now()-bn)}`)}async function Oa(){var n;if(ce.loading){alert("Wait for soundtrack analysis to finish.");return}if(ce.includeAudio&&I.buffer&&I.player.paused)try{await I.play()}catch(r){alert(`Cannot play soundtrack: ${r instanceof Error?r.message:r}`);return}const e=Ia();if(!e){alert("Recording is not supported in this browser.");return}const t=z.captureStream(60);if(ce.includeAudio&&I.buffer)for(const r of((n=I.stream)==null?void 0:n.getAudioTracks())??[])t.addTrack(r.clone());Ue=[],se=new MediaRecorder(t,{mimeType:e,videoBitsPerSecond:16e6}),se.ondataavailable=r=>{r.data.size>0&&Ue.push(r.data)},se.onstop=()=>{const r=Ua(e),a=new Blob(Ue,{type:e}),o=URL.createObjectURL(a),l=document.createElement("a");l.href=o,l.download=`${U.id}-${Date.now()}.${r}`,l.click(),URL.revokeObjectURL(o),Ue=[],t.getTracks().forEach(s=>s.stop())},se.start(500),de=!0,bn=performance.now(),Sn(),je=window.setInterval(Na,250)}function La(){se&&se.state!=="inactive"&&se.stop(),de=!1,je!==null&&(clearInterval(je),je=null),Sn()}function _a(){de?La():Oa()}function Sn(){const e=document.getElementById("rec-btn"),t=document.getElementById("rec-badge");!e||!t||(de?(e.textContent="Stop",e.classList.add("recording"),t.classList.remove("hidden")):(e.textContent="Record",e.classList.remove("recording"),t.classList.add("hidden"),t.textContent=""))}let Y=!1;function At(e){nt.classList.toggle("hidden",e)}function Ha(){At(!1),ut!==null&&window.clearTimeout(ut),ut=window.setTimeout(()=>{At(!0)},2500)}function xn(){const e=document.body.classList.contains("sidebar-collapsed");nt.textContent=e?">>":"<<"}function qa(){var e;(e=Je.parentElement)==null||e.classList.remove("hidden"),st!==null&&window.clearTimeout(st),st=window.setTimeout(()=>{var t;(t=Je.parentElement)==null||t.classList.add("hidden")},1e4)}function $t(e){const t=u.createTexture();if(!t)throw new Error("Failed to create state texture");return u.bindTexture(u.TEXTURE_2D,t),u.texParameteri(u.TEXTURE_2D,u.TEXTURE_MIN_FILTER,u.NEAREST),u.texParameteri(u.TEXTURE_2D,u.TEXTURE_MAG_FILTER,u.NEAREST),u.texParameteri(u.TEXTURE_2D,u.TEXTURE_WRAP_S,u.REPEAT),u.texParameteri(u.TEXTURE_2D,u.TEXTURE_WRAP_T,u.REPEAT),u.texImage2D(u.TEXTURE_2D,0,u.RGBA16F,e,e,0,u.RGBA,u.HALF_FLOAT,null),u.bindTexture(u.TEXTURE_2D,null),t}function Va(e){const t=$t(e),n=$t(e),r=u.createFramebuffer(),a=u.createFramebuffer();if(!r||!a)throw new Error("Failed to create framebuffer");return u.bindFramebuffer(u.FRAMEBUFFER,r),u.framebufferTexture2D(u.FRAMEBUFFER,u.COLOR_ATTACHMENT0,u.TEXTURE_2D,t,0),u.bindFramebuffer(u.FRAMEBUFFER,a),u.framebufferTexture2D(u.FRAMEBUFFER,u.COLOR_ATTACHMENT0,u.TEXTURE_2D,n,0),u.bindFramebuffer(u.FRAMEBUFFER,null),{size:e,textures:[t,n],fbos:[r,a],index:0,needsInit:!0}}function wn(e){if(!e.stateful)return null;let t=kt.get(e.id);const n=e.bufferSize??192;return(!t||t.size!==n)&&(t=Va(n),kt.set(e.id,t)),t}function ne(e){const t=kt.get(e);t&&(t.needsInit=!0,t.lastTime=void 0)}function ja(e,t){let n=t;return e.min!==void 0&&(n=Math.max(e.min,n)),e.max!==void 0&&(n=Math.min(e.max,n)),e.type==="int"&&(n=Math.round(n)),n}function Cn(e,t){if(e.type==="int")return String(Math.round(t));const n=e.step??.01,r=n<1?Math.min(4,Math.max(2,Math.ceil(-Math.log10(n)))):0;return t.toFixed(r)}function Xa(e){return e.length===1?e.toLowerCase():e}function Za(e){return e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement||e instanceof HTMLSelectElement}function we(e,t,n,r=!0){const a=yn[e][t];if(!a)return;const o=ja(a,n);if(me[e][t]=o,e===U.id&&r){const l=Rt[t];l!=null&&l.range&&(l.range.value=String(o)),l!=null&&l.number&&(l.number.value=Cn(a,o))}}function Tn(e){var t;for(const n of((t=pe.find(r=>r.id===e))==null?void 0:t.params)??[])n.type==="seed"&&we(e,n.id,Math.floor(Math.random()*1e6));ne(e)}function kn(e){const t=gn[e];for(const[n,r]of Object.entries(t))we(e,n,r,!0);ne(e)}function $a(){ge.innerHTML="";for(const e of pe){const t=document.createElement("option");t.value=e.id,t.textContent=e.name,ge.appendChild(t)}ge.addEventListener("change",()=>{Lt(ge.value)})}function Ya(e){Ve.innerHTML="";const t=document.createElement("button");if(t.className="ghost small",t.textContent="Reset",t.addEventListener("click",()=>kn(e.id)),Ve.appendChild(t),e.params.some(r=>r.type==="seed")){const r=document.createElement("button");r.className="ghost small",r.textContent="Reseed",r.addEventListener("click",()=>Tn(e.id)),Ve.appendChild(r)}}function Ka(e){Ct.innerHTML="",Rt={};for(const t of e.params){if(t.type==="seed")continue;const n=document.createElement("div");n.className="control";const r=document.createElement("div");r.className="control-header";const a=document.createElement("label");if(a.textContent=t.label,r.appendChild(a),t.key){const c=document.createElement("span");c.className="key-cap",c.textContent=`${t.key.inc.toUpperCase()}/${t.key.dec.toUpperCase()}`,r.appendChild(c)}const o=document.createElement("div");o.className="control-inputs";const l=document.createElement("input");l.type="range",l.min=String(t.min??0),l.max=String(t.max??1),l.step=String(t.step??(t.type==="int"?1:.01)),l.value=String(me[e.id][t.id]),l.addEventListener("input",c=>{const f=Number(c.target.value);Number.isNaN(f)||we(e.id,t.id,f)});const s=document.createElement("input");s.type="number",s.min=l.min,s.max=l.max,s.step=l.step,s.value=Cn(t,me[e.id][t.id]),s.addEventListener("input",c=>{const f=Number(c.target.value);Number.isNaN(f)||we(e.id,t.id,f)}),o.appendChild(l),o.appendChild(s),n.appendChild(r),n.appendChild(o),Ct.appendChild(n),Rt[t.id]={range:l,number:s}}}function Qa(e){Ae.innerHTML="";for(const t of e.params){if(!t.key||t.type==="seed")continue;const n=document.createElement("div");n.className="key-row",n.textContent=`${t.key.inc.toUpperCase()}/${t.key.dec.toUpperCase()}  ${t.label}`,Ae.appendChild(n)}Ae.childElementCount||(Ae.textContent="No mapped keys for this scene.")}function Lt(e){const t=pe.find(n=>n.id===e);t&&(U=t,te.reset(),t.stateful&&(wn(t),ne(t.id)),Je.textContent=t.name,hn.textContent=t.description,qa(),Ya(t),Ka(t),Qa(t),ge.value=t.id)}function Ja(e){if(Y||Za(e.target))return;const t=Xa(e.key),n=U.params;for(const r of n){if(!r.key||r.type==="seed")continue;const a=t===r.key.inc,o=t===r.key.dec;if(!a&&!o)continue;const l=e.shiftKey&&r.key.shiftStep?r.key.shiftStep:r.key.step,c=me[U.id][r.id]+l*(a?1:-1);we(U.id,r.id,c),e.preventDefault();break}}function We(e,t,n,r,a){const o=me[e.id],l=t.uniforms,s=l[e.resolutionUniform];s&&u.uniform2f(s,r,a);const c=l[e.timeUniform];if(c)if(e.timeMode==="phase"){const m=e.loopDuration??8,d=n%m/m;u.uniform1f(c,d)}else if(e.timeMode==="looped"){const m=e.loopDuration??8,d=n%m;if(u.uniform1f(c,d),e.loopUniform){const g=l[e.loopUniform];g&&u.uniform1f(g,m)}}else u.uniform1f(c,n);const f={};for(const m of e.params){const d=l[m.uniform],g=o[m.id];if(m.component!==void 0){const h=f[m.uniform]??[0,0,0];h[m.component]=g,f[m.uniform]=h;continue}d&&(m.type==="int"?u.uniform1i(d,Math.round(g)):u.uniform1f(d,g))}for(const[m,d]of Object.entries(f)){const g=l[m];g&&u.uniform3f(g,d[0],d[1],d[2])}}function ct(e,t,n,r){const a=t.uniforms[e.passUniform??"uPass"];a&&u.uniform1i(a,r);const o=t.uniforms[e.gridUniform??"uGridSize"];o&&u.uniform2f(o,n.size,n.size);const l=t.uniforms[e.stateUniform??"uState"];l&&u.uniform1i(l,0)}function eo(e,t,n,r,a){u.bindFramebuffer(u.FRAMEBUFFER,a);const o=vn.get(e.id);if(o){if(e.stateful){if(!rt)return;const l=wn(e);if(!l)return;const s=()=>l.textures[l.index],c=()=>l.fbos[(l.index+1)%2];u.useProgram(o.program),u.bindVertexArray(Tt),l.needsInit&&(u.bindFramebuffer(u.FRAMEBUFFER,c()),u.viewport(0,0,l.size,l.size),u.activeTexture(u.TEXTURE0),u.bindTexture(u.TEXTURE_2D,s()),We(e,o,t,n,r),ct(e,o,l,2),u.drawArrays(u.TRIANGLES,0,3),l.index=(l.index+1)%2,l.needsInit=!1),l.lastTime!==t&&(u.bindFramebuffer(u.FRAMEBUFFER,c()),u.viewport(0,0,l.size,l.size),u.activeTexture(u.TEXTURE0),u.bindTexture(u.TEXTURE_2D,s()),We(e,o,t,n,r),ct(e,o,l,0),u.drawArrays(u.TRIANGLES,0,3),l.index=(l.index+1)%2,l.lastTime=t),u.bindFramebuffer(u.FRAMEBUFFER,a),u.viewport(0,0,n,r),u.clearColor(0,0,0,1),u.clear(u.COLOR_BUFFER_BIT),u.activeTexture(u.TEXTURE0),u.bindTexture(u.TEXTURE_2D,s()),We(e,o,t,n,r),ct(e,o,l,1),u.drawArrays(u.TRIANGLES,0,3);return}u.viewport(0,0,n,r),u.clearColor(0,0,0,1),u.clear(u.COLOR_BUFFER_BIT),u.useProgram(o.program),u.bindVertexArray(Tt),We(e,o,t,n,r),u.drawArrays(u.TRIANGLES,0,3)}}let Yt=0;function Rn(e,t,n,r){!Y&&I.buffer&&t<Yt&&(ne(e.id),te.reset()),Y||(Yt=t);const a=te.begin(n,r);eo(e,t,n,r,a),a&&te.finish(t,I.sample(t),!!I.buffer)}function _t(e){if(Y)return;const t=I.buffer?I.player.currentTime:(e-at)/1e3;ce.update();const{width:n,height:r}=zn(z);Rn(U,t,n,r),et=requestAnimationFrame(_t)}async function An(e,t,n,r){if(Y||de||ce.loading)throw await t.abort(),new Error(ce.loading?"Wait for soundtrack analysis to finish.":"Stop the current recording or export before exporting video.");if(u.isContextLost())throw await t.abort(),new Error("WebGL context is lost. Reload the browser tab.");const a=U;if(a.stateful&&!rt)throw await t.abort(),new Error("This scene requires floating-point WebGL render targets.");const o=!!I.buffer&&!I.player.paused;I.player.pause(),te.reset(),Y=!0,cancelAnimationFrame(et);const l=document.querySelector(".panel");l.inert=!0;const s={width:z.width,height:z.height,styleW:z.style.width,styleH:z.style.height},c=document.getElementById("offline-progress"),f=document.getElementById("offline-bar"),m=document.getElementById("offline-status");c==null||c.classList.remove("hidden"),m==null||m.classList.remove("hidden");try{z.width=e.width,z.height=e.height,z.style.width=e.width+"px",z.style.height=e.height+"px",ne(a.id),await Ma({...e,canvas:z,sink:t,signal:n,audio:ce.includeAudio?I.buffer:void 0,loopAudio:I.player.loop,render(d,g,h){Rn(a,d,g,h),u.finish()},progress(d,g){f&&(f.style.width=d/g*100+"%"),m&&(m.textContent="Rendering frame "+d+"/"+g),(d%e.fps===0||d===g)&&(r==null||r(d,g))}}),m&&(m.textContent="Video saved.")}catch(d){throw m&&(m.textContent=n!=null&&n.aborted?"Export cancelled.":"Export failed: "+(d instanceof Error?d.message:String(d))),d}finally{z.width=s.width,z.height=s.height,z.style.width=s.styleW,z.style.height=s.styleH,ne(a.id),l.inert=!1,Y=!1,at=performance.now(),te.reset(),o&&await I.play().catch(console.error),et=requestAnimationFrame(_t)}}async function to(e,t,n,r,a){let o;try{o=await window.showSaveFilePicker({suggestedName:e.id+"-"+r+"x"+a+"-"+n+"fps-"+t+"s.mp4",types:[{description:"MP4 Video",accept:{"video/mp4":[".mp4"]}}]})}catch(l){l instanceof DOMException&&l.name==="AbortError"||alert(l instanceof Error?l.message:String(l));return}try{const l=await o.createWritable();await An({duration:t,fps:n,width:r,height:a},{write:(s,c)=>l.write({type:"write",position:c,data:s.slice().buffer}),close:()=>l.close(),abort:()=>l.abort()})}catch(l){console.error("Video export failed:",l)}}function Ne(){return{sceneId:U.id,name:U.name,values:{...me[U.id]},recording:de,exporting:Y,capabilities:{floatTargets:rt,webCodecs:typeof VideoEncoder<"u",contextLost:u.isContextLost()}}}let Te;za(async({command:e,args:t},n)=>{if(e==="get_state")return Ne();if(e==="cancel_export")return Te&&Te.jobId===t.jobId&&Te.controller.abort(new Error("Export cancelled.")),Ne();if(Y)throw new Error("Wait for the video export to finish or cancel it before editing the scene.");if(u.isContextLost())throw new Error("WebGL context is lost. Reload the browser tab.");if(e==="export_video"){const a=String(t.jobId),o=new AbortController,l=()=>o.abort(new Error("MCP disconnected during export."));n.signal.addEventListener("abort",l,{once:!0}),n.signal.aborted&&l(),Te={jobId:a,controller:o};try{return await An({duration:Number(t.duration),fps:Number(t.fps),width:Number(t.width),height:Number(t.height)},n.sink(a),o.signal,(s,c)=>n.progress(a,s,c)),Ne()}finally{n.signal.removeEventListener("abort",l),Te=void 0}}const r=Pa(pe,String(t.sceneId));if(e==="select_scene"){if(r.stateful&&!rt)throw new Error("This scene requires floating-point WebGL render targets.");Lt(r.id)}else{if(r.id!==U.id)throw new Error("The active scene changed. Read get_state before editing it.");if(e==="set_parameters"){const a=t.values;Da(r,a);for(const[o,l]of Object.entries(a))we(r.id,o,l);r.params.some(o=>o.type==="seed"&&Object.prototype.hasOwnProperty.call(a,o.id))&&ne(r.id)}else if(e==="reset_scene")kn(r.id);else if(e==="reseed_scene")Tn(r.id);else throw new Error("Unknown browser command: "+e)}return Ne()});nt.addEventListener("click",()=>{document.body.classList.toggle("sidebar-collapsed"),xn()});Ot.addEventListener("mousemove",()=>{Ha()});document.addEventListener("keydown",Ja);document.addEventListener("visibilitychange",()=>{document.hidden||(at=performance.now())});var Kt;(Kt=document.getElementById("rec-btn"))==null||Kt.addEventListener("click",_a);var Qt;(Qt=document.getElementById("offline-btn"))==null||Qt.addEventListener("click",()=>{if(Y)return;const e=document.getElementById("offline-duration"),t=document.getElementById("offline-fps"),n=document.getElementById("offline-res"),r=Math.max(1,Math.min(3600,Number((e==null?void 0:e.value)??10))),a=Number((t==null?void 0:t.value)??60),[o,l]=((n==null?void 0:n.value)??"1920x1080").split("x").map(Number);to(U,r,a,o,l)});$a();Lt(U.id);xn();At(!0);et=requestAnimationFrame(_t);
