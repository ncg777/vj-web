(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))o(a);new MutationObserver(a=>{for(const i of a)if(i.type==="childList")for(const l of i.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&o(l)}).observe(document,{childList:!0,subtree:!0});function t(a){const i={};return a.integrity&&(i.integrity=a.integrity),a.referrerPolicy&&(i.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?i.credentials="include":a.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function o(a){if(a.ep)return;a.ep=!0;const i=t(a);fetch(a.href,i)}})();const Z=`#version 300 es\r
precision highp float;\r
\r
const vec2 verts[3] = vec2[](\r
  vec2(-1.0, -1.0),\r
  vec2(3.0, -1.0),\r
  vec2(-1.0, 3.0)\r
);\r
\r
void main() {\r
  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);\r
}\r
`,K=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uPhase;\r
uniform int uComponents;\r
uniform int uIsoBands;\r
uniform float uLineThickness;\r
uniform float uNoiseAmount;\r
uniform float uSeed;\r
\r
float hash11(float n) {\r
  return fract(sin(n) * 43758.5453123);\r
}\r
\r
float randI(int i, float s, float ch) {\r
  return hash11(float(i) * 17.0 + s * 251.0 + ch * 0.61803);\r
}\r
\r
int randint(int i, float s, float ch, int lo, int hiInclusive) {\r
  float r = randI(i, s, ch);\r
  return lo + int(floor(r * float(hiInclusive - lo + 1)));\r
}\r
\r
float randUniform(int i, float s, float ch, float lo, float hi) {\r
  float r = randI(i, s, ch);\r
  return mix(lo, hi, r);\r
}\r
\r
vec3 hsv2rgb(vec3 c) {\r
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);\r
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);\r
  return c.z * mix(vec3(1.0), rgb, c.y);\r
}\r
\r
void main() {\r
  float minDim = min(uResolution.x, uResolution.y);\r
  vec2 p = (2.0 * gl_FragCoord.xy - uResolution) / minDim;\r
\r
  float rn = length(p) / sqrt(2.0);\r
  float th = atan(p.y, p.x);\r
\r
  int fMin = 1;\r
  int fMax = 4;\r
  float aMin = 0.20;\r
  float aMax = 0.85;\r
  int tCyclesMin = 0;\r
  int tCyclesMax = 2;\r
  int nThetaMin = 1;\r
  int nThetaMax = 4;\r
  int nTimeMin = 1;\r
  int nTimeMax = 2;\r
  int nRadMin = 0;\r
  int nRadMax = 2;\r
\r
  float s = 0.0;\r
  float ampSum = 0.0;\r
  float twoPI = 6.283185307179586;\r
\r
  for (int i = 0; i < 64; ++i) {\r
    if (i >= uComponents) {\r
      break;\r
    }\r
\r
    int f = randint(i, uSeed, 11.0, fMin, fMax);\r
    float amp = randUniform(i, uSeed, 13.0, aMin, aMax);\r
    float phi0 = randUniform(i, uSeed, 17.0, 0.0, twoPI);\r
    int tCyc = randint(i, uSeed, 19.0, tCyclesMin, tCyclesMax);\r
    int nTh = randint(i, uSeed, 23.0, nThetaMin, nThetaMax);\r
    int nTi = randint(i, uSeed, 29.0, nTimeMin, nTimeMax);\r
    int nRa = randint(i, uSeed, 31.0, nRadMin, nRadMax);\r
    float nPhi = randUniform(i, uSeed, 37.0, 0.0, twoPI);\r
\r
    amp *= 1.0 / max(1.0, sqrt(float(uComponents)));\r
    ampSum += abs(amp);\r
\r
    float tTerm = twoPI * (float(tCyc) * uPhase);\r
    float angNoise = sin(float(nTh) * th + twoPI * float(nTi) * uPhase + nPhi);\r
    float radNoise = sin(twoPI * (float(nRa) * rn + float(nTi + 1) * uPhase) + 0.37 * nPhi);\r
    float noise = uNoiseAmount * (rn * angNoise + 0.4 * radNoise);\r
\r
    s += amp * sin(twoPI * (float(f) * rn) + phi0 + tTerm + noise);\r
  }\r
\r
  float ampNorm = (ampSum > 1e-9) ? ampSum : 1.0;\r
  float v = s / ampNorm;\r
\r
  float line = abs(sin(3.141592653589793 * float(uIsoBands) * v));\r
  float lt = clamp(uLineThickness, 0.01, 0.75);\r
\r
  float core = pow(max(0.0, 1.0 - (line / lt)), 1.5);\r
  float glow = pow(max(0.0, 1.0 - (line / (lt * 2.8))), 2.2);\r
  float intensity = min(1.0, core + 0.45 * glow);\r
\r
  if (rn > 0.985) {\r
    float t = clamp((rn - 0.985) / (1.0 - 0.985), 0.0, 1.0);\r
    intensity *= (1.0 - t);\r
  }\r
\r
  float t1 = sin(twoPI * (1.0 * uPhase));\r
  float t2 = cos(twoPI * (2.0 * uPhase));\r
  float hue = 0.24 * rn\r
    + 0.18 * v\r
    + 0.12 * sin(th)\r
    + 0.08 * cos(2.0 * th)\r
    + 0.06 * sin(3.0 * th)\r
    + 0.07 * t1\r
    + 0.05 * t2\r
    + 0.06 * sin(twoPI * (0.25 * rn + 1.0 * uPhase));\r
  hue = hue - floor(hue);\r
\r
  float sat = min(1.0, 0.9 + 0.1 * intensity);\r
  float bri = min(1.0, 0.95 * intensity + 0.35 * glow);\r
\r
  vec3 rgb = hsv2rgb(vec3(hue, sat, bri));\r
  outColor = vec4(rgb, 1.0);\r
}\r
`,J=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uPhase;\r
uniform float uScale;\r
uniform int uOctaves;\r
uniform float uLacunarity;\r
uniform float uGain;\r
uniform int uIsoBands;\r
uniform float uLineThickness;\r
uniform float uSeed;\r
uniform float uBubbleAmp;\r
uniform float uBubbleFreq;\r
uniform float uBubbleDetail;\r
\r
const float PI = 3.14159265358979323846;\r
const float TAU = 6.28318530717958647692;\r
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
float h11(float n) {\r
  return fract(sin(n) * 43758.5453123);\r
}\r
\r
vec2 h21(float n) {\r
  return vec2(h11(n * 19.0 + 0.73), h11(n * 23.0 + 1.91));\r
}\r
\r
float fbm(vec2 p, int octaves, float lac, float gain) {\r
  float sum = 0.0;\r
  float amp = 0.5;\r
  float norm = 0.0;\r
  vec2 pp = p;\r
  for (int i = 0; i < 12; ++i) {\r
    if (i >= octaves) {\r
      break;\r
    }\r
    sum += amp * noise(pp);\r
    norm += amp;\r
    pp *= lac;\r
    amp *= gain;\r
  }\r
  return (norm > 1e-6) ? sum / norm : 0.0;\r
}\r
\r
vec3 hsv2rgb(vec3 c) {\r
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);\r
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);\r
  return c.z * mix(vec3(1.0), rgb, c.y);\r
}\r
\r
void main() {\r
  float minDim = min(uResolution.x, uResolution.y);\r
  vec2 p = (gl_FragCoord.xy - 0.5 * uResolution) / minDim;\r
\r
  vec2 seedShift = (h21(uSeed * 0.137) - 0.5) * 1024.0;\r
  vec2 timeShift = vec2(cos(TAU * uPhase), sin(TAU * uPhase)) * (0.75 * uScale);\r
  vec2 world = p * uScale + seedShift + timeShift;\r
\r
  vec2 warpOff = vec2(cos(TAU * (uPhase + 0.27)), sin(TAU * (uPhase + 0.27))) * (0.33 * uScale);\r
  float base0 = fbm(world + warpOff, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));\r
  float signed0 = base0 * 2.0 - 1.0;\r
  float tanh0 = tanh(1.35 * signed0);\r
\r
  vec2 swirl = vec2(-p.y, p.x);\r
  vec2 warp = (0.18 * uScale) * (swirl * tanh0)\r
    + (0.12 * uScale) * vec2(sin(world.y * 0.8), cos(world.x * 0.8)) * tanh0;\r
  vec2 world2 = world + warp;\r
\r
  float base1 = fbm(world2 + warpOff * 0.6, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));\r
  float signed1 = base1 * 2.0 - 1.0;\r
  float h = 0.5 + 0.5 * tanh(1.25 * signed1);\r
  float hCurve = h * h * (3.0 - 2.0 * h);\r
  float hFinal = mix(h, hCurve, 0.6);\r
\r
  float bubbleDet = max(0.25, uBubbleDetail);\r
  vec2 bubbleTimeShift = vec2(cos(TAU * (uPhase + 0.43)), sin(TAU * (uPhase + 0.43))) * (0.55 * bubbleDet);\r
  float bubbleNoise = fbm(world2 * bubbleDet + bubbleTimeShift, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));\r
  float bubbleWave = sin(TAU * (uBubbleFreq * uPhase) + bubbleNoise * PI + 1.5 * tanh0);\r
  float hBubbled = hFinal + uBubbleAmp * bubbleWave * (0.35 + 0.65 * bubbleNoise);\r
  hBubbled = clamp(hBubbled, 0.0, 1.0);\r
\r
  float e = 1.25 / minDim;\r
  float hx = fbm(world2 + vec2(e, 0.0), uOctaves, uLacunarity, uGain)\r
    - fbm(world2 - vec2(e, 0.0), uOctaves, uLacunarity, uGain);\r
  float hy = fbm(world2 + vec2(0.0, e), uOctaves, uLacunarity, uGain)\r
    - fbm(world2 - vec2(0.0, e), uOctaves, uLacunarity, uGain);\r
  float slope = length(vec2(hx, hy));\r
\r
  int bands = max(1, uIsoBands);\r
  float line = abs(sin(PI * float(bands) * hBubbled));\r
  float lt = clamp(uLineThickness, 0.02, 0.75);\r
\r
  float core = pow(max(0.0, 1.0 - (line / lt)), 1.35);\r
  float glow = pow(max(0.0, 1.0 - (line / (lt * 3.0))), 2.2);\r
  float intensity = clamp(core + 0.5 * glow, 0.0, 1.0);\r
\r
  float r = length(p) / 0.9;\r
  float vignette = smoothstep(1.0, 0.6, r);\r
  intensity *= vignette;\r
\r
  float hue = fract(0.62 * hBubbled + 0.18 * slope + 0.1 * sin(TAU * uPhase));\r
  float sat = mix(0.65, 1.0, intensity);\r
  float bri = mix(0.12, 1.0, intensity);\r
  hue = fract(hue + 0.05 * tanh0 + 0.04 * sin(TAU * (uPhase + hBubbled)));\r
\r
  vec3 rgb = hsv2rgb(vec3(hue, sat, bri));\r
  outColor = vec4(rgb, 1.0);\r
}\r
`,$=`#version 300 es\r
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
float loopTime(float t, float duration) {\r
  float phase = mod(t, duration) / duration;\r
  return phase * TAU;\r
}\r
\r
void main() {\r
  vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;\r
\r
  float t = loopTime(iTime * uSpeed, uLoopDuration);\r
  float phase = t / TAU;\r
\r
  float r = length(uv);\r
  float a = atan(uv.y, uv.x);\r
  a += uTwist * r;\r
\r
  vec2 dir = vec2(cos(a), sin(a));\r
  vec2 tOff = vec2(cos(t), sin(t)) * (0.6 * uNoiseScale);\r
  vec2 np = dir * (0.75 * uNoiseScale) + tOff + r * (2.0 * uNoiseScale);\r
  float n = fbm(np);\r
  r += uNoiseAmp * n;\r
  a += uNoiseAmp * 0.5 * n;\r
\r
  float stripePhase = fract(phase + r * 0.5);\r
  float stripe = smoothstep(0.3, 0.5, sin(a * 8.0 + stripePhase * TAU));\r
\r
  vec3 baseHue = uBaseColor;\r
  vec3 dynamicHue = 0.5 + 0.5 * cos(vec3(0.0, 0.6, 1.2) + a * 2.0 + n * 2.0 + uColorCycle * t);\r
  vec3 col = mix(baseHue, dynamicHue, 0.7);\r
\r
  float stripeMask = stripe;\r
  col = mix(col, vec3(1.0), 0.6 * stripeMask);\r
\r
  float fogBase = exp(-r * uFogDensity);\r
  float glowBase = pow(fogBase, 2.0);\r
  float e = 0.003 * max(0.5, uNoiseScale);\r
  vec2 grad;\r
  grad.x = fbm(np + vec2(e, 0.0)) - fbm(np - vec2(e, 0.0));\r
  grad.y = fbm(np + vec2(0.0, e)) - fbm(np - vec2(0.0, e));\r
  vec2 normal2D = normalize(grad + vec2(1e-6));\r
  float refractStrength = 0.03;\r
  vec2 uvR = uv + normal2D * refractStrength * (0.3 + 0.7 * glowBase);\r
\r
  float rR = length(uvR);\r
  float aR = atan(uvR.y, uvR.x);\r
  aR += uTwist * rR;\r
  vec2 dirR = vec2(cos(aR), sin(aR));\r
  vec2 npR = dirR * (0.75 * uNoiseScale) + tOff + rR * (2.0 * uNoiseScale);\r
  float nR = fbm(npR);\r
  float stripePhaseR = fract(phase + rR * 0.5);\r
  float stripeR = smoothstep(0.3, 0.5, sin(aR * 8.0 + stripePhaseR * TAU));\r
  vec3 dynamicHueR = 0.5 + 0.5 * cos(vec3(0.0, 0.6, 1.2) + aR * 2.0 + nR * 2.0 + uColorCycle * t);\r
  vec3 colR = mix(baseHue, dynamicHueR, 0.7);\r
  colR = mix(colR, vec3(1.0), 0.6 * stripeR);\r
\r
  col = mix(col, colR, 0.6);\r
\r
  float fog = fogBase;\r
  float glow = glowBase;\r
  col *= mix(0.6, 1.6, fog);\r
  col += glow * 0.35 * (0.6 * baseHue + 0.4 * dynamicHue);\r
\r
  col = clamp(col, 0.0, 1.0);\r
  outColor = vec4(col, 1.0);\r
}\r
`,Q=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform int uIterations;\r
uniform float uScale;\r
uniform float uRotation;\r
uniform float uGlowIntensity;\r
uniform vec3 uColorPrimary;\r
uniform vec3 uColorSecondary;\r
\r
float sdSegment(vec2 p, vec2 a, vec2 b) {\r
  vec2 pa = p - a;\r
  vec2 ba = b - a;\r
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);\r
  return length(pa - ba * h);\r
}\r
\r
vec2 rotate(vec2 p, float angle) {\r
  float c = cos(angle);\r
  float s = sin(angle);\r
  return vec2(c * p.x - s * p.y, s * p.x + c * p.y);\r
}\r
\r
float kochSegmentDistanceIter(vec2 p, vec2 a, vec2 b, int iterations) {\r
  vec2 ab = b - a;\r
  float len = max(length(ab), 1e-6);\r
  vec2 dir = ab / len;\r
  vec2 nrm = vec2(-dir.y, dir.x);\r
  vec2 pl = vec2(dot(p - a, dir) / len, dot(p - a, nrm) / len);\r
\r
  const float c60 = 0.5;\r
  const float s60 = 0.8660254037844386;\r
  mat2 invRotPlus = mat2(c60, s60, -s60, c60);\r
  mat2 invRotMinus = mat2(c60, -s60, s60, c60);\r
\r
  const int MAX_ITERS = 8;\r
  int it = min(iterations, MAX_ITERS);\r
  float scaleAccum = 1.0;\r
\r
  for (int i = 0; i < MAX_ITERS; ++i) {\r
    if (i >= it) {\r
      break;\r
    }\r
    pl *= 3.0;\r
    float region = floor(pl.x);\r
\r
    if (region == 1.0) {\r
      vec2 c = vec2(1.5, 0.0);\r
      vec2 pr = pl - c;\r
      vec2 pr1 = invRotPlus * pr;\r
      vec2 pr2 = invRotMinus * pr;\r
      vec2 p1 = pr1 + c;\r
      vec2 p2 = pr2 + c;\r
      pl = (abs(p1.y) < abs(p2.y)) ? p1 : p2;\r
    }\r
\r
    pl.x -= region;\r
    scaleAccum *= (1.0 / 3.0);\r
  }\r
\r
  float dLocal = sdSegment(pl, vec2(0.0, 0.0), vec2(1.0, 0.0));\r
  return dLocal * len * scaleAccum;\r
}\r
\r
float kochSnowflakeDistance(vec2 p, float size, int iterations) {\r
  float h = size * sqrt(3.0) / 2.0;\r
  vec2 v1 = vec2(0.0, h * 2.0 / 3.0);\r
  vec2 v2 = vec2(-size / 2.0, -h / 3.0);\r
  vec2 v3 = vec2(size / 2.0, -h / 3.0);\r
\r
  float d1 = kochSegmentDistanceIter(p, v1, v2, iterations);\r
  float d2 = kochSegmentDistanceIter(p, v2, v3, iterations);\r
  float d3 = kochSegmentDistanceIter(p, v3, v1, iterations);\r
  return min(min(d1, d2), d3);\r
}\r
\r
float trianglePerimeterDistance(vec2 p, float size) {\r
  float h = size * sqrt(3.0) / 2.0;\r
  vec2 v1 = vec2(0.0, h * 2.0 / 3.0);\r
  vec2 v2 = vec2(-size / 2.0, -h / 3.0);\r
  vec2 v3 = vec2(size / 2.0, -h / 3.0);\r
  float d1 = sdSegment(p, v1, v2);\r
  float d2 = sdSegment(p, v2, v3);\r
  float d3 = sdSegment(p, v3, v1);\r
  return min(min(d1, d2), d3);\r
}\r
\r
void main() {\r
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / min(uResolution.x, uResolution.y);\r
  float angle = uTime * uRotation;\r
  uv = rotate(uv, angle);\r
\r
  float distKoch = kochSnowflakeDistance(uv, uScale, uIterations);\r
  float distTri = trianglePerimeterDistance(uv, uScale);\r
  float dist = min(distKoch, distTri * 0.75);\r
\r
  const float lineWidth = 0.004;\r
  const float lineOuterMult = 1.5;\r
  const float lineInnerMult = 0.5;\r
  const float distanceScale = 15.0;\r
  const float timeScale = 2.0;\r
  const float glowMix = 0.4;\r
  const float edgeGlowMult = 0.3;\r
\r
  float line = smoothstep(lineWidth * lineOuterMult, lineWidth * lineInnerMult, dist);\r
  float glow = exp(-dist * distanceScale * uGlowIntensity);\r
  float colorMix = sin(dist * distanceScale - uTime * timeScale) * 0.5 + 0.5;\r
  vec3 color = mix(uColorPrimary, uColorSecondary, colorMix);\r
\r
  vec3 finalColor = color * (line + glow * glowMix);\r
  vec3 edgeGlowColor = vec3(0.2, 0.3, 0.5);\r
  finalColor += edgeGlowColor * glow * uGlowIntensity * edgeGlowMult;\r
\r
  outColor = vec4(finalColor, 1.0);\r
}\r
`,ee=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform int uIterations;\r
uniform float uScale;\r
uniform float uRotation;\r
uniform float uGlowIntensity;\r
uniform vec3 uColorPrimary;\r
uniform vec3 uColorSecondary;\r
\r
mat2 rot2(float a) {\r
  float c = cos(a);\r
  float s = sin(a);\r
  return mat2(c, -s, s, c);\r
}\r
\r
float sdSegment(vec2 p, vec2 a, vec2 b) {\r
  vec2 pa = p - a;\r
  vec2 ba = b - a;\r
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);\r
  return length(pa - ba * h);\r
}\r
\r
float quasi(vec2 p, int waves) {\r
  float A = 0.0;\r
  for (int i = 0; i < 16; ++i) {\r
    if (i >= waves) {\r
      break;\r
    }\r
    float ang = 6.2831853 * float(i) * 0.5 * (sqrt(5.0) - 1.0);\r
    vec2 k = vec2(cos(ang), sin(ang));\r
    A += cos(dot(k, p) * 3.0);\r
  }\r
  return A / float(max(1, waves));\r
}\r
\r
float kochSegmentIter(vec2 p, vec2 a, vec2 b, int it) {\r
  vec2 ex = normalize(b - a);\r
  vec2 ey = vec2(-ex.y, ex.x);\r
  float L = length(b - a);\r
\r
  vec2 v = vec2(dot(p - a, ex), dot(p - a, ey));\r
  vec2 w = v / L;\r
\r
  float s = 1.0;\r
  for (int k = 0; k < 8; ++k) {\r
    if (k >= it) {\r
      break;\r
    }\r
    w *= 3.0;\r
    s /= 3.0;\r
    if (w.x > 1.0 && w.x < 2.0) {\r
      w = rot2(-3.14159265 / 3.0) * (w - vec2(1.0, 0.0));\r
    } else if (w.x >= 2.0) {\r
      w.x -= 2.0;\r
    }\r
  }\r
\r
  float d = sdSegment(w, vec2(0.0), vec2(1.0, 0.0));\r
  return d * L * s;\r
}\r
\r
float kochSnowflakeDist(vec2 p, float size, int it) {\r
  float r = size;\r
  vec2 v0 = r * vec2(cos(0.0), sin(0.0));\r
  vec2 v1 = r * vec2(cos(2.094395102), sin(2.094395102));\r
  vec2 v2 = r * vec2(cos(4.188790205), sin(4.188790205));\r
\r
  float d0 = kochSegmentIter(p, v0, v1, it);\r
  float d1 = kochSegmentIter(p, v1, v2, it);\r
  float d2 = kochSegmentIter(p, v2, v0, it);\r
  return min(d0, min(d1, d2));\r
}\r
\r
void main() {\r
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y;\r
\r
  float r = length(uv);\r
  float vig = smoothstep(1.2, 0.2, r);\r
  vec3 bg = mix(uColorSecondary * 0.06, uColorSecondary * 0.22, vig);\r
\r
  vec2 p = uv * uScale;\r
  p *= rot2(uRotation);\r
  float q1 = quasi(p * 2.8 + 0.3 * vec2(cos(uTime * 0.17), sin(uTime * 0.21)), 9);\r
  float q2 = quasi(p.yx * 3.1 + 0.2 * vec2(sin(uTime * 0.13), cos(uTime * 0.19)), 7);\r
  float warpAmp = 0.06 + 0.045 * (0.5 + 0.5 * sin(uTime * 0.57));\r
  vec2 pWarp = p + warpAmp * vec2(q1, q2);\r
\r
  float maxIt = float(clamp(uIterations, 1, 8));\r
  float minIt = max(1.0, maxIt - 3.0);\r
  float iAnim = mix(minIt, maxIt, 0.5 + 0.5 * sin(uTime * 0.27));\r
  int i0 = int(floor(iAnim));\r
  int i1 = min(i0 + 1, 8);\r
  float itMix = fract(iAnim);\r
\r
  float radius = 0.70 + 0.12 * sin(uTime * 0.41);\r
  float d0 = kochSnowflakeDist(pWarp, radius, i0);\r
  float d1 = kochSnowflakeDist(pWarp, radius, i1);\r
  float d = mix(d0, d1, itMix);\r
\r
  float lineWidth = 0.0035 + 0.0015 * (0.5 + 0.5 * sin(uTime * 0.77));\r
  float edge = smoothstep(lineWidth, 0.0, d);\r
  float glow = exp(-14.0 * d) * uGlowIntensity;\r
\r
  vec3 snow = mix(uColorSecondary, uColorPrimary, edge) + glow * uColorPrimary;\r
  vec3 col = bg + snow;\r
\r
  outColor = vec4(col, 1.0);\r
}\r
`,ne=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform vec2 uGridSize;\r
uniform sampler2D uState;\r
uniform int uPass;\r
uniform float uTime;\r
uniform float uSelfWeight;\r
uniform float uNeighborWeight;\r
uniform float uDecay;\r
uniform float uRotate;\r
uniform float uInjectAmp;\r
uniform float uInjectRadius;\r
uniform float uValueGain;\r
uniform float uSeed;\r
\r
float hash11(float n) {\r
  return fract(sin(n) * 43758.5453123);\r
}\r
\r
vec3 hsv2rgb(vec3 c) {\r
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);\r
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);\r
  return c.z * mix(vec3(1.0), rgb, c.y);\r
}\r
\r
vec2 sampleState(vec2 uv) {\r
  vec2 gridUV = uv * uGridSize - 0.5;\r
  vec2 base = floor(gridUV);\r
  vec2 f = fract(gridUV);\r
  vec2 invGrid = 1.0 / uGridSize;\r
  vec2 p00 = (base + vec2(0.5, 0.5)) * invGrid;\r
  vec2 p10 = (base + vec2(1.5, 0.5)) * invGrid;\r
  vec2 p01 = (base + vec2(0.5, 1.5)) * invGrid;\r
  vec2 p11 = (base + vec2(1.5, 1.5)) * invGrid;\r
  vec2 v00 = texture(uState, p00).rg;\r
  vec2 v10 = texture(uState, p10).rg;\r
  vec2 v01 = texture(uState, p01).rg;\r
  vec2 v11 = texture(uState, p11).rg;\r
  vec2 v0 = mix(v00, v10, f.x);\r
  vec2 v1 = mix(v01, v11, f.x);\r
  return mix(v0, v1, f.y);\r
}\r
\r
vec2 diffuseVec2(vec2 uv, vec2 texel) {\r
  vec2 c = texture(uState, uv).rg;\r
  vec2 sum = c * uSelfWeight;\r
  sum += texture(uState, uv + vec2(texel.x, 0.0)).rg * uNeighborWeight;\r
  sum += texture(uState, uv - vec2(texel.x, 0.0)).rg * uNeighborWeight;\r
  sum += texture(uState, uv + vec2(0.0, texel.y)).rg * uNeighborWeight;\r
  sum += texture(uState, uv - vec2(0.0, texel.y)).rg * uNeighborWeight;\r
  float norm = max(1e-4, uSelfWeight + 4.0 * uNeighborWeight);\r
  return sum / norm;\r
}\r
\r
void main() {\r
  if (uPass == 2) {\r
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;\r
    vec2 ab = vec2(0.0);\r
    float seed = uSeed * 0.001;\r
    float radius = 0.08;\r
    for (int i = 0; i < 3; ++i) {\r
      float fi = float(i);\r
      vec2 pos = vec2(hash11(seed + fi * 3.1 + 1.0), hash11(seed + fi * 4.7 + 2.0));\r
      float ang = hash11(seed + fi * 5.3 + 3.0) * 6.2831853;\r
      float d = distance(uv, pos);\r
      float g = exp(-d * d / (radius * radius));\r
      ab += g * vec2(cos(ang), sin(ang));\r
    }\r
    outColor = vec4(ab, 0.0, 1.0);\r
    return;\r
  }\r
\r
  if (uPass == 0) {\r
    vec2 texel = 1.0 / uGridSize;\r
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;\r
    vec2 ab = diffuseVec2(uv, texel);\r
\r
    float ang = uRotate;\r
    float ca = cos(ang);\r
    float sa = sin(ang);\r
    ab = mat2(ca, -sa, sa, ca) * ab;\r
    ab *= uDecay;\r
\r
    float seed = uSeed * 0.001;\r
    float t = uTime * 0.6 + seed * 3.0;\r
    vec2 pos = 0.5 + 0.32 * vec2(sin(t * 1.1 + seed), cos(t * 1.4 + seed * 1.7));\r
    float injectAng = t * 1.7 + seed * 5.0;\r
    float dist = distance(uv, pos);\r
    float sigma = max(1e-4, uInjectRadius);\r
    float g = exp(-dist * dist / (sigma * sigma));\r
    ab += uInjectAmp * g * vec2(cos(injectAng), sin(injectAng));\r
\r
    outColor = vec4(ab, 0.0, 1.0);\r
    return;\r
  }\r
\r
  vec2 uv = gl_FragCoord.xy / uResolution;\r
  vec2 ab = sampleState(uv);\r
  float angle = atan(ab.y, ab.x);\r
  float mag = length(ab);\r
  float hue = (angle + 3.14159265) / 6.2831853;\r
  float value = clamp(mag * uValueGain, 0.0, 1.0);\r
  vec3 rgb = hsv2rgb(vec3(hue, 1.0, value));\r
  outColor = vec4(rgb, 1.0);\r
}\r
`,re=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform vec2 uGridSize;\r
uniform sampler2D uState;\r
uniform int uPass;\r
uniform float uTime;\r
uniform float uSelfWeight;\r
uniform float uNeighborWeight;\r
uniform float uDecay;\r
uniform float uBlobAmp;\r
uniform float uBlobRadius;\r
uniform float uSpeed;\r
uniform float uFlowGain;\r
uniform float uFlowThreshold;\r
uniform float uSeed;\r
\r
float hash11(float n) {\r
  return fract(sin(n) * 43758.5453123);\r
}\r
\r
vec3 hsv2rgb(vec3 c) {\r
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);\r
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);\r
  return c.z * mix(vec3(1.0), rgb, c.y);\r
}\r
\r
float sampleState(vec2 uv) {\r
  vec2 gridUV = uv * uGridSize - 0.5;\r
  vec2 base = floor(gridUV);\r
  vec2 f = fract(gridUV);\r
  vec2 invGrid = 1.0 / uGridSize;\r
  vec2 p00 = (base + vec2(0.5, 0.5)) * invGrid;\r
  vec2 p10 = (base + vec2(1.5, 0.5)) * invGrid;\r
  vec2 p01 = (base + vec2(0.5, 1.5)) * invGrid;\r
  vec2 p11 = (base + vec2(1.5, 1.5)) * invGrid;\r
  float v00 = texture(uState, p00).r;\r
  float v10 = texture(uState, p10).r;\r
  float v01 = texture(uState, p01).r;\r
  float v11 = texture(uState, p11).r;\r
  float v0 = mix(v00, v10, f.x);\r
  float v1 = mix(v01, v11, f.x);\r
  return mix(v0, v1, f.y);\r
}\r
\r
float diffuseScalar(vec2 uv, vec2 texel) {\r
  float c = texture(uState, uv).r;\r
  float sum = c * uSelfWeight;\r
  sum += texture(uState, uv + vec2(texel.x, 0.0)).r * uNeighborWeight;\r
  sum += texture(uState, uv - vec2(texel.x, 0.0)).r * uNeighborWeight;\r
  sum += texture(uState, uv + vec2(0.0, texel.y)).r * uNeighborWeight;\r
  sum += texture(uState, uv - vec2(0.0, texel.y)).r * uNeighborWeight;\r
  float norm = max(1e-4, uSelfWeight + 4.0 * uNeighborWeight);\r
  return sum / norm;\r
}\r
\r
void main() {\r
  if (uPass == 2) {\r
    outColor = vec4(0.0, 0.0, 0.0, 1.0);\r
    return;\r
  }\r
\r
  if (uPass == 0) {\r
    vec2 texel = 1.0 / uGridSize;\r
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;\r
    float x = diffuseScalar(uv, texel);\r
    x *= uDecay;\r
\r
    float seed = uSeed * 0.001;\r
    float t = uTime * uSpeed + seed * 2.0;\r
    vec2 c1 = 0.5 + 0.34 * vec2(sin(t * 1.2 + seed), cos(t * 1.6 + seed * 1.3));\r
    vec2 c2 = 0.5 + 0.30 * vec2(sin(t * 0.8 + seed * 2.1), cos(t * 1.1 + seed * 0.7));\r
    float sigma = max(1e-4, uBlobRadius);\r
    float g1 = exp(-distance(uv, c1) * distance(uv, c1) / (sigma * sigma));\r
    float g2 = exp(-distance(uv, c2) * distance(uv, c2) / (sigma * sigma));\r
    x += uBlobAmp * (g1 + 0.8 * g2);\r
\r
    x = clamp(x, 0.0, 1.0);\r
    outColor = vec4(x, 0.0, 0.0, 1.0);\r
    return;\r
  }\r
\r
  vec2 uv = gl_FragCoord.xy / uResolution;\r
  vec2 texel = 1.0 / uGridSize;\r
  float xL = sampleState(uv - vec2(texel.x, 0.0));\r
  float xR = sampleState(uv + vec2(texel.x, 0.0));\r
  float xD = sampleState(uv - vec2(0.0, texel.y));\r
  float xU = sampleState(uv + vec2(0.0, texel.y));\r
\r
  vec2 grad = vec2(xR - xL, xU - xD);\r
  float mag = length(grad) * uFlowGain;\r
  float threshold = max(0.0, uFlowThreshold);\r
  float edge = smoothstep(threshold, threshold + 0.05, mag);\r
\r
  float hue = (atan(grad.y, grad.x) + 3.14159265) / 6.2831853;\r
  float value = clamp(mag, 0.0, 1.0) * edge;\r
  vec3 rgb = hsv2rgb(vec3(hue, 0.9, value));\r
  outColor = vec4(rgb, 1.0);\r
}\r
`,te=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform vec2 uGridSize;\r
uniform sampler2D uState;\r
uniform int uPass;\r
uniform float uTime;\r
uniform float uSelfWeight;\r
uniform float uNeighborWeight;\r
uniform float uDecay;\r
uniform float uThreshold;\r
uniform float uSharpness;\r
uniform float uNoiseAmp;\r
uniform float uTurbulence;\r
uniform float uInjectAmp;\r
uniform float uInjectRadius;\r
uniform float uSpeed;\r
uniform float uSeed;\r
\r
float hash11(float n) {\r
  return fract(sin(n) * 43758.5453123);\r
}\r
\r
float hash21(vec2 p) {\r
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);\r
}\r
\r
float diffuseScalar(vec2 uv, vec2 texel) {\r
  float c = texture(uState, uv).r;\r
  float sum = c * uSelfWeight;\r
  sum += texture(uState, uv + vec2(texel.x, 0.0)).r * uNeighborWeight;\r
  sum += texture(uState, uv - vec2(texel.x, 0.0)).r * uNeighborWeight;\r
  sum += texture(uState, uv + vec2(0.0, texel.y)).r * uNeighborWeight;\r
  sum += texture(uState, uv - vec2(0.0, texel.y)).r * uNeighborWeight;\r
  float norm = max(1e-4, uSelfWeight + 4.0 * uNeighborWeight);\r
  return sum / norm;\r
}\r
\r
float sigmoid(float z) {\r
  return 1.0 / (1.0 + exp(-z));\r
}\r
\r
vec3 hsv2rgb(vec3 c) {\r
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);\r
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);\r
  return c.z * mix(vec3(1.0), rgb, c.y);\r
}\r
\r
float sampleState(vec2 uv) {\r
  vec2 gridUV = uv * uGridSize - 0.5;\r
  vec2 base = floor(gridUV);\r
  vec2 f = fract(gridUV);\r
  vec2 invGrid = 1.0 / uGridSize;\r
  vec2 p00 = (base + vec2(0.5, 0.5)) * invGrid;\r
  vec2 p10 = (base + vec2(1.5, 0.5)) * invGrid;\r
  vec2 p01 = (base + vec2(0.5, 1.5)) * invGrid;\r
  vec2 p11 = (base + vec2(1.5, 1.5)) * invGrid;\r
  float x00 = texture(uState, p00).r;\r
  float x10 = texture(uState, p10).r;\r
  float x01 = texture(uState, p01).r;\r
  float x11 = texture(uState, p11).r;\r
  float x0 = mix(x00, x10, f.x);\r
  float x1 = mix(x01, x11, f.x);\r
  return mix(x0, x1, f.y);\r
}\r
\r
vec2 flowField(vec2 uv, float t) {\r
  float s1 = sin(uv.y * 6.0 + t);\r
  float s2 = cos(uv.x * 6.0 - t * 1.1);\r
  float n1 = hash21(uv * uGridSize + t * 0.7);\r
  float n2 = hash21(uv * uGridSize + t * 0.7 + vec2(12.3, 45.6));\r
  vec2 flow = vec2(s1 + (n1 - 0.5) * 1.2, s2 + (n2 - 0.5) * 1.2);\r
  return normalize(flow + vec2(1e-3));\r
}\r
\r
void main() {\r
  if (uPass == 2) {\r
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;\r
    float seed = uSeed * 0.001;\r
    float x = hash21(uv * uGridSize + seed) * 0.25;\r
    outColor = vec4(x, 0.0, 0.0, 1.0);\r
    return;\r
  }\r
\r
  if (uPass == 0) {\r
    vec2 texel = 1.0 / uGridSize;\r
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;\r
    float t = uTime * uSpeed + uSeed * 0.001;\r
    float advect = 0.02 + 0.04 * clamp(uNoiseAmp, 0.0, 1.0);\r
    vec2 flow = flowField(uv, t);\r
    vec2 uvAdv = uv + flow * advect * max(0.0, uTurbulence);\r
    float x = diffuseScalar(uvAdv, texel);\r
\r
    x = sigmoid(uSharpness * (x - uThreshold));\r
    x *= uDecay;\r
\r
    float seed = uSeed * 0.001;\r
    float injectT = uTime * uSpeed + seed * 4.0;\r
    vec2 pos = 0.5 + 0.33 * vec2(sin(injectT * 1.1 + seed), cos(injectT * 1.4 + seed * 1.9));\r
    float sigma = max(1e-4, uInjectRadius);\r
    float g = exp(-distance(uv, pos) * distance(uv, pos) / (sigma * sigma));\r
    x += uInjectAmp * g;\r
\r
    float noise = (hash21(uv * uGridSize + uTime * 2.0 + seed) - 0.5) * uNoiseAmp;\r
    x = clamp(x + noise, 0.0, 1.0);\r
    outColor = vec4(x, 0.0, 0.0, 1.0);\r
    return;\r
  }\r
\r
  vec2 uv = gl_FragCoord.xy / uResolution;\r
  float x = sampleState(uv);\r
  float hue = fract(0.6 + 0.1 * sin(uTime * 0.25) + x * 1.2);\r
  float sat = clamp(0.5 + x * 0.8, 0.0, 1.0);\r
  float val = clamp(0.15 + x * 1.1, 0.0, 1.0);\r
  vec3 rgb = hsv2rgb(vec3(hue, sat, val));\r
  outColor = vec4(rgb, 1.0);\r
}\r
`,oe=`#version 300 es\r
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
`,ae=`#version 300 es\r
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
uniform vec3 uWaterTint;\r
\r
const int NUM_STEPS = 32;\r
const int ITER_GEOMETRY = 3;\r
const int ITER_FRAGMENT = 5;\r
const float PI = 3.141592;\r
const float EPSILON = 1e-3;\r
#define EPSILON_NRM (0.1 / uResolution.x)\r
const mat2 octave_m = mat2(1.6, 1.2, -1.2, 1.6);\r
\r
mat3 fromEuler(vec3 ang) {\r
  vec2 a1 = vec2(sin(ang.x), cos(ang.x));\r
  vec2 a2 = vec2(sin(ang.y), cos(ang.y));\r
  vec2 a3 = vec2(sin(ang.z), cos(ang.z));\r
  mat3 m;\r
  m[0] = vec3(a1.y * a3.y + a1.x * a2.x * a3.x, a1.y * a2.x * a3.x + a3.y * a1.x, -a2.y * a3.x);\r
  m[1] = vec3(-a2.y * a1.x, a1.y * a2.y, a2.x);\r
  m[2] = vec3(a3.y * a1.x * a2.x + a1.y * a3.x, a1.x * a3.x - a1.y * a3.y * a2.x, a2.y * a3.y);\r
  return m;\r
}\r
\r
float hash(vec2 p) {\r
  float h = dot(p, vec2(127.1, 311.7));\r
  return fract(sin(h) * 43758.5453123);\r
}\r
\r
float noise(vec2 p) {\r
  vec2 i = floor(p);\r
  vec2 f = fract(p);\r
  vec2 u = f * f * (3.0 - 2.0 * f);\r
  return -1.0 + 2.0 * mix(\r
    mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),\r
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),\r
    u.y\r
  );\r
}\r
\r
float diffuse(vec3 n, vec3 l, float p) {\r
  return pow(dot(n, l) * 0.4 + 0.6, p);\r
}\r
\r
float specular(vec3 n, vec3 l, vec3 e, float s) {\r
  float nrm = (s + 8.0) / (PI * 8.0);\r
  return pow(max(dot(reflect(e, n), l), 0.0), s) * nrm;\r
}\r
\r
vec3 getSkyColor(vec3 e) {\r
  e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8;\r
  return vec3(pow(1.0 - e.y, 2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4) * uSkyBoost;\r
}\r
\r
float sea_octave(vec2 uv, float choppy) {\r
  uv += noise(uv);\r
  vec2 wv = 1.0 - abs(sin(uv));\r
  vec2 swv = abs(cos(uv));\r
  wv = mix(wv, swv, wv);\r
  return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);\r
}\r
\r
float map(vec3 p, float seaTime) {\r
  float freq = uSeaFreq;\r
  float amp = uSeaHeight;\r
  float choppy = uSeaChoppy;\r
  vec2 uv = p.xz;\r
  uv.x *= 0.75;\r
\r
  float d;\r
  float h = 0.0;\r
  for (int i = 0; i < ITER_GEOMETRY; i++) {\r
    d = sea_octave((uv + seaTime) * freq, choppy);\r
    d += sea_octave((uv - seaTime) * freq, choppy);\r
    h += d * amp;\r
    uv *= octave_m;\r
    freq *= 1.9;\r
    amp *= 0.22;\r
    choppy = mix(choppy, 1.0, 0.2);\r
  }\r
  return p.y - h;\r
}\r
\r
float map_detailed(vec3 p, float seaTime) {\r
  float freq = uSeaFreq;\r
  float amp = uSeaHeight;\r
  float choppy = uSeaChoppy;\r
  vec2 uv = p.xz;\r
  uv.x *= 0.75;\r
\r
  float d;\r
  float h = 0.0;\r
  for (int i = 0; i < ITER_FRAGMENT; i++) {\r
    d = sea_octave((uv + seaTime) * freq, choppy);\r
    d += sea_octave((uv - seaTime) * freq, choppy);\r
    h += d * amp;\r
    uv *= octave_m;\r
    freq *= 1.9;\r
    amp *= 0.22;\r
    choppy = mix(choppy, 1.0, 0.2);\r
  }\r
  return p.y - h;\r
}\r
\r
vec3 getSeaColor(vec3 p, vec3 n, vec3 l, vec3 eye, vec3 dist) {\r
  float fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0);\r
  fresnel = min(fresnel * fresnel * fresnel, 0.5);\r
\r
  vec3 seaBase = uWaterTint * 0.2;\r
  vec3 seaWater = mix(vec3(0.8, 0.9, 0.6), uWaterTint, 0.5) * uWaterBrightness;\r
\r
  vec3 reflected = getSkyColor(reflect(eye, n));\r
  vec3 refracted = seaBase + diffuse(n, l, 80.0) * seaWater * 0.12;\r
\r
  vec3 color = mix(refracted, reflected, fresnel);\r
\r
  float atten = max(1.0 - dot(dist, dist) * 0.001, 0.0);\r
  color += seaWater * (p.y - uSeaHeight) * 0.18 * atten;\r
\r
  color += specular(n, l, eye, 600.0 * inversesqrt(dot(dist, dist)));\r
\r
  return color;\r
}\r
\r
vec3 getNormal(vec3 p, float eps, float seaTime) {\r
  vec3 n;\r
  n.y = map_detailed(p, seaTime);\r
  n.x = map_detailed(vec3(p.x + eps, p.y, p.z), seaTime) - n.y;\r
  n.z = map_detailed(vec3(p.x, p.y, p.z + eps), seaTime) - n.y;\r
  n.y = eps;\r
  return normalize(n);\r
}\r
\r
float heightMapTracing(vec3 ori, vec3 dir, out vec3 p, float seaTime) {\r
  float tm = 0.0;\r
  float tx = 1000.0;\r
  float hx = map(ori + dir * tx, seaTime);\r
  if (hx > 0.0) {\r
    p = ori + dir * tx;\r
    return tx;\r
  }\r
  float hm = map(ori, seaTime);\r
  for (int i = 0; i < NUM_STEPS; i++) {\r
    float tmid = mix(tm, tx, hm / (hm - hx));\r
    p = ori + dir * tmid;\r
    float hmid = map(p, seaTime);\r
    if (hmid < 0.0) {\r
      tx = tmid;\r
      hx = hmid;\r
    } else {\r
      tm = tmid;\r
      hm = hmid;\r
    }\r
    if (abs(hmid) < EPSILON) break;\r
  }\r
  return mix(tm, tx, hm / (hm - hx));\r
}\r
\r
vec3 getPixel(vec2 coord, float time, float seaTime) {\r
  vec2 uv = coord / uResolution.xy;\r
  uv = uv * 2.0 - 1.0;\r
  uv.x *= uResolution.x / uResolution.y;\r
\r
  vec3 ang = vec3(sin(time * 3.0) * 0.1 + uCamPitch, sin(time) * 0.2 + 0.3, time + uCamYaw);\r
  vec3 ori = vec3(0.0, uCamHeight, time * uCamDistance);\r
  vec3 dir = normalize(vec3(uv.xy, -2.0));\r
  dir.z += length(uv) * 0.14;\r
  dir = normalize(dir) * fromEuler(ang);\r
\r
  vec3 p;\r
  heightMapTracing(ori, dir, p, seaTime);\r
  vec3 dist = p - ori;\r
  vec3 n = getNormal(p, dot(dist, dist) * EPSILON_NRM, seaTime);\r
  vec3 light = normalize(vec3(0.0, 1.0, 0.8));\r
\r
  return mix(\r
    getSkyColor(dir),\r
    getSeaColor(p, n, light, dir, dist),\r
    pow(smoothstep(0.0, -0.02, dir.y), 0.2)\r
  );\r
}\r
\r
void main() {\r
  vec2 fragCoord = gl_FragCoord.xy;\r
  float time = uTime * uTimeScale;\r
  float seaTime = 1.0 + time * uSeaSpeed;\r
\r
  vec3 color = getPixel(fragCoord, time, seaTime);\r
  outColor = vec4(pow(color, vec3(0.65)), 1.0);\r
}\r
`,ie=`#version 300 es\r
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
\r
const float TAU = 6.28318530718;\r
const int MAX_ITER = 5;\r
\r
void main() {\r
  float time = uTime * uTimeScale + 23.0;\r
  vec2 uv = gl_FragCoord.xy / uResolution.xy;\r
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
  outColor = vec4(color, 1.0);\r
}\r
`,le=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
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
void main() {\r
  vec2 I = gl_FragCoord.xy;\r
  float t = uTime * uTimeScale;\r
  float i = 0.0;\r
  float z = 0.0;\r
  float d = 0.0;\r
  float s = 0.0;\r
  vec4 O = vec4(0.0);\r
\r
  for (O *= i; i++ < 100.0;) {\r
    vec3 p = z * normalize(vec3(I + I, 0.0) - uResolution.xyy);\r
\r
    for (d = 5.0; d < 200.0; d += d) {\r
      p += uTurbulence * 0.6 * sin(p.yzx * d - 0.2 * t) / d;\r
    }\r
\r
    float height = max(0.05, uCloudHeight);\r
    s = height - abs(p.y);\r
    z += d = uStepBase + max(s, -s * 0.2) / uStepScale;\r
\r
    vec4 phase = vec4(3.0, 4.0, 5.0, 0.0) + uHueShift + uHueSpeed * t;\r
    O += (cos(s / 0.07 + p.x + 0.5 * t - phase) + 1.5) * exp(s / 0.1) / d;\r
  }\r
\r
  O = tanh(O * O / 4e8);\r
  outColor = vec4(O.rgb * uIntensity, 1.0);\r
}\r
`,ue=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uZoom;\r
uniform float uTimeScale;\r
uniform float uTwist;\r
uniform float uWarp;\r
uniform float uPulse;\r
uniform float uIterLimit;\r
uniform float uGlow;\r
uniform float uOffsetX;\r
uniform float uOffsetY;\r
uniform float uColorShift;\r
\r
void main() {\r
  vec2 frag = gl_FragCoord.xy;\r
  vec2 res = uResolution.xy;\r
  vec2 uv = frag;\r
  vec2 v = res;\r
  vec2 offset = vec2(uOffsetX, uOffsetY) * res;\r
\r
  uv = uZoom * (uv + uv - v + offset) / v.y;\r
\r
  vec4 z = vec4(1.0, 2.0, 3.0, 0.0);\r
  vec4 o = z;\r
  float a = 0.5;\r
  float t = uTime * uTimeScale;\r
\r
  for (int i = 0; i < 19; ++i) {\r
    float fi = float(i) + 1.0;\r
    float mask = step(fi, uIterLimit);\r
    float denom = length(\r
      (1.0 + fi * dot(v, v))\r
        * sin(1.5 * uv / (0.5 - dot(uv, uv)) - uTwist * 9.0 * uv.yx + t)\r
    );\r
    o += mask * (1.0 + cos(z + t + uColorShift)) / max(1e-3, denom);\r
\r
    a += 0.03;\r
    float ap = pow(a, fi);\r
    t += 1.0;\r
    v = cos(t - uPulse * 7.0 * uv * ap) - 5.0 * uv;\r
\r
    uv *= mat2(cos(fi + 0.02 * t - vec4(0.0, 11.0, 33.0, 0.0)));\r
    vec2 warp = tanh(uWarp * 40.0 * dot(uv, uv) * cos(100.0 * uv.yx + t)) / 200.0;\r
    uv += warp\r
      + 0.2 * a * uv\r
      + cos(4.0 / exp(dot(o, o) / 100.0) + t) / 300.0;\r
  }\r
\r
  vec4 mapped = (25.6 * uGlow) / (min(o, 13.0) + 164.0 / o);\r
  mapped -= dot(uv, uv) / 250.0;\r
  outColor = vec4(mapped.rgb, 1.0);\r
}\r
`,se=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform float uSpin;\r
uniform float uTwist;\r
uniform float uWarp;\r
uniform float uPulse;\r
uniform float uBoltDensity;\r
uniform float uBoltSharpness;\r
uniform float uBoltIntensity;\r
uniform float uArcSteps;\r
uniform float uCoreSize;\r
uniform float uCoreGlow;\r
uniform float uNoiseAmp;\r
uniform float uPaletteShift;\r
uniform float uSeed;\r
uniform vec3 uColorPrimary;\r
uniform vec3 uColorSecondary;\r
uniform vec3 uColorAccent;\r
\r
const float TAU = 6.28318530718;\r
\r
float hash(vec2 p) {\r
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);\r
}\r
\r
vec3 palette(float t) {\r
  return uColorPrimary + uColorSecondary * cos(TAU * (uColorAccent * t + uPaletteShift));\r
}\r
\r
void main() {\r
  vec2 uv = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;\r
  uv.x *= uResolution.x / uResolution.y;\r
  uv *= uZoom;\r
\r
  float t = uTime * uTimeScale;\r
  float spin = t * uSpin;\r
  mat2 rot = mat2(cos(spin), -sin(spin), sin(spin), cos(spin));\r
  uv = rot * uv;\r
\r
  vec2 twist = vec2(\r
    sin(uv.y * (2.0 + uTwist) + t),\r
    cos(uv.x * (2.5 + uTwist) - t)\r
  );\r
  uv += twist * 0.15 * uWarp;\r
\r
  float radius = length(uv) + 1e-4;\r
  float angle = atan(uv.y, uv.x);\r
\r
  float bolts = 0.0;\r
  for (int i = 0; i < 80; i++) {\r
    float fi = float(i) + 1.0;\r
    float mask = step(fi, uArcSteps);\r
    float phase = t * (0.8 + uPulse) + fi * 0.37 + uSeed * 0.001;\r
    float wave = sin(angle * (uBoltDensity + fi * 0.12) + phase)\r
      + cos(radius * (uBoltDensity * 1.7) - phase);\r
    float d = abs(wave) + 0.12 + radius * uBoltSharpness * 0.35;\r
    float contribution = uBoltIntensity / (d * d);\r
    bolts += mask * contribution;\r
  }\r
\r
  vec2 q = uv;\r
  float spark = 0.0;\r
  for (int i = 0; i < 7; i++) {\r
    float fi = float(i) + 1.0;\r
    float denom = max(0.25, dot(q, q));\r
    q = abs(q) / denom - vec2(0.55, 0.35) * uWarp;\r
    spark += exp(-fi) * (0.5 + 0.5 * sin(fi * 2.3 + t + q.x * 4.0 + q.y * 5.0));\r
  }\r
\r
  float grain = hash(uv * (12.0 + uSeed * 0.001) + t);\r
  spark += uNoiseAmp * (grain - 0.5);\r
\r
  float core = uCoreGlow / (abs(radius - uCoreSize) + 0.02);\r
\r
  float energy = bolts * 0.03 + spark * 0.9 + core * 0.6;\r
  energy *= smoothstep(1.8, 0.2, radius);\r
\r
  vec3 col = palette(radius + spark * 0.35 + t * 0.1);\r
  col *= energy;\r
  col += vec3(bolts * 0.02);\r
  col += vec3(core * 0.15);\r
\r
  col = pow(max(col, 0.0), vec3(0.75));\r
  outColor = vec4(col, 1.0);\r
}\r
`,ce=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uZoom;\r
uniform float uCoreRadius;\r
uniform float uCoreNoiseScale;\r
uniform float uCoreNoiseAmp;\r
uniform float uCoreIntensity;\r
uniform float uBoltLengthMin;\r
uniform float uBoltLengthMax;\r
uniform float uBoltWidth;\r
uniform float uBoltWiggle;\r
uniform float uBoltNoiseScale;\r
uniform float uBoltNoiseSpeed;\r
uniform float uBoltSecondaryScale;\r
uniform float uBoltIntensity;\r
uniform float uFlickerSpeed;\r
uniform float uAngleJitter;\r
uniform float uTwist;\r
uniform float uSeed;\r
uniform int uBoltCount;\r
uniform int uNoiseOctaves;\r
uniform vec3 uColorPrimary;\r
uniform vec3 uColorSecondary;\r
uniform vec3 uColorAccent;\r
\r
const float TAU = 6.28318530718;\r
\r
mat2 Rotate(float angle) {\r
  return mat2(cos(angle), sin(angle), -sin(angle), cos(angle));\r
}\r
\r
float CircleSDF(vec2 p, float r) {\r
  return length(p) - r;\r
}\r
\r
float LineSDF(vec2 p, vec2 a, vec2 b, float s) {\r
  vec2 pa = a - p;\r
  vec2 ba = a - b;\r
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);\r
  return length(pa - ba * h) - s;\r
}\r
\r
float RandomFloat(vec2 seed) {\r
  seed = sin(seed * vec2(123.45, 546.23)) * 345.21 + 12.57;\r
  return fract(seed.x * seed.y);\r
}\r
\r
float SimpleNoise(vec2 uv, int octaves) {\r
  float sn = 0.0;\r
  float amplitude = 1.0;\r
  float deno = 0.0;\r
  int count = clamp(octaves, 1, 6);\r
  for (int i = 1; i <= 6; i++) {\r
    if (i > count) {\r
      break;\r
    }\r
    vec2 grid = smoothstep(0.0, 1.0, fract(uv));\r
    vec2 id = floor(uv);\r
    vec2 offs = vec2(0.0, 1.0);\r
    float bl = RandomFloat(id);\r
    float br = RandomFloat(id + offs.yx);\r
    float tl = RandomFloat(id + offs);\r
    float tr = RandomFloat(id + offs.yy);\r
    sn += mix(mix(bl, br, grid.x), mix(tl, tr, grid.x), grid.y) * amplitude;\r
    deno += amplitude;\r
    uv *= 3.5;\r
    amplitude *= 0.5;\r
  }\r
  return sn / max(1e-4, deno);\r
}\r
\r
vec3 Bolt(vec2 uv, float len, float ind, float time) {\r
  vec2 travel = vec2(0.0, mod(time, 200.0) * uBoltNoiseSpeed);\r
\r
  float sn = SimpleNoise(\r
    uv * uBoltNoiseScale - travel + vec2(ind * 1.5 + uSeed * 0.01, 0.0),\r
    uNoiseOctaves\r
  ) * 2.0 - 1.0;\r
  uv.x += sn * uBoltWiggle * smoothstep(0.0, 0.2, abs(uv.y));\r
\r
  vec3 l = vec3(LineSDF(uv, vec2(0.0), vec2(0.0, len), uBoltWidth));\r
  l = uBoltIntensity / max(vec3(0.0), l) * uColorSecondary;\r
  l = clamp(1.0 - exp(l * -0.02), 0.0, 1.0) * smoothstep(len - 0.01, 0.0, abs(uv.y));\r
  vec3 bolt = l;\r
\r
  uv = Rotate(TAU * uTwist) * uv;\r
  sn = SimpleNoise(\r
    uv * (uBoltNoiseScale * 1.25) - travel * 1.2 + vec2(ind * 2.3 + uSeed * 0.03, 0.0),\r
    uNoiseOctaves\r
  ) * 2.0 - 1.0;\r
  uv.x += sn * uv.y * uBoltSecondaryScale * smoothstep(0.1, 0.25, len);\r
  len *= 0.5;\r
\r
  l = vec3(LineSDF(uv, vec2(0.0), vec2(0.0, len), uBoltWidth * 0.8));\r
  l = uBoltIntensity * 0.7 / max(vec3(0.0), l) * uColorAccent;\r
  l = clamp(1.0 - exp(l * -0.03), 0.0, 1.0) * smoothstep(len * 0.7, 0.0, abs(uv.y));\r
  bolt += l;\r
\r
  float hz = uFlickerSpeed * time * TAU;\r
  float r = RandomFloat(vec2(ind + uSeed * 0.1)) * 0.5 * TAU;\r
  float flicker = sin(hz + r) * 0.5 + 0.5;\r
  return bolt * smoothstep(0.5, 0.0, flicker);\r
}\r
\r
void main() {\r
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;\r
  uv *= uZoom;\r
\r
  float time = uTime * uTimeScale;\r
  vec3 col = vec3(0.0);\r
\r
  float coreNoise = SimpleNoise(\r
    uv * uCoreNoiseScale - vec2(0.0, mod(time, 200.0) * uCoreNoiseScale * 0.2),\r
    uNoiseOctaves\r
  );\r
  float r = uCoreRadius + uCoreNoiseAmp * (coreNoise * 2.0 - 1.0);\r
  vec3 core = uCoreIntensity / max(0.0, CircleSDF(uv, r)) * uColorPrimary;\r
  core = 1.0 - exp(core * -0.05);\r
  col = core;\r
\r
  int count = max(uBoltCount, 1);\r
  for (int i = 0; i < 12; i++) {\r
    if (i >= count) {\r
      break;\r
    }\r
    float fi = float(i);\r
    float angle = fi * TAU / float(count);\r
    angle += (RandomFloat(vec2(float(count) + floor(time * 5.0 + fi) + uSeed)) - 0.5)\r
      * uAngleJitter;\r
    float len = mix(\r
      uBoltLengthMin,\r
      uBoltLengthMax,\r
      RandomFloat(vec2(angle + uSeed, fi * 1.7))\r
    );\r
    col += Bolt(Rotate(angle) * uv, len, fi, time);\r
  }\r
\r
  outColor = vec4(col, 1.0);\r
}\r
`,fe=`#version 300 es\r
precision highp float;\r
precision highp int;\r
\r
out vec4 outColor;\r
\r
uniform vec2 uResolution;\r
uniform float uTime;\r
uniform float uTimeScale;\r
uniform float uPower;\r
uniform float uBulbSpin;\r
uniform float uMaxRayLength;\r
uniform float uTolerance;\r
uniform float uNormOffset;\r
uniform float uInitStep;\r
uniform float uRotSpeedX;\r
uniform float uRotSpeedY;\r
uniform float uCamDistance;\r
uniform float uCamHeight;\r
uniform float uFov;\r
uniform float uSkyBoost;\r
uniform float uGlowBoost;\r
uniform float uGlowFalloff;\r
uniform float uDiffuseBoost;\r
uniform float uMatTransmit;\r
uniform float uMatReflect;\r
uniform float uRefractIndex;\r
uniform float uHueShift;\r
uniform float uGlowHueOffset;\r
uniform float uNebulaMix;\r
uniform float uNebulaHueShift;\r
uniform float uNebulaSat;\r
uniform float uNebulaVal;\r
uniform float uNebulaGlowHue;\r
uniform float uNebulaGlowBoost;\r
uniform float uSkySat;\r
uniform float uSkyVal;\r
uniform float uGlowSat;\r
uniform float uGlowVal;\r
uniform float uDiffuseSat;\r
uniform float uDiffuseVal;\r
uniform vec3 uBeerColor;\r
uniform vec3 uLightPos;\r
uniform int uLoops;\r
uniform int uRayMarches;\r
uniform int uBounces;\r
\r
const float PI = 3.141592654;\r
const float TAU = 6.28318530718;\r
\r
mat3 g_rot = mat3(1.0);\r
\r
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);\r
vec3 hsv2rgb(vec3 c) {\r
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);\r
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);\r
}\r
\r
vec3 sRGB(vec3 t) {\r
  return mix(1.055 * pow(t, vec3(1.0 / 2.4)) - 0.055, 12.92 * t, step(t, vec3(0.0031308)));\r
}\r
\r
vec3 aces_approx(vec3 v) {\r
  v = max(v, 0.0);\r
  v *= 0.6;\r
  float a = 2.51;\r
  float b = 0.03;\r
  float c = 2.43;\r
  float d = 0.59;\r
  float e = 0.14;\r
  return clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0);\r
}\r
\r
float boxSDF(vec2 p, vec2 b) {\r
  vec2 d = abs(p) - b;\r
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);\r
}\r
\r
float rayPlane(vec3 ro, vec3 rd, vec4 p) {\r
  return -(dot(ro, p.xyz) + p.w) / dot(rd, p.xyz);\r
}\r
\r
float mandelBulb(vec3 p, float time) {\r
  vec3 z = p;\r
  float r = 0.0;\r
  float dr = 1.0;\r
\r
  for (int i = 0; i < 6; ++i) {\r
    if (i >= uLoops) {\r
      break;\r
    }\r
    r = length(z);\r
    if (r > 2.0) {\r
      break;\r
    }\r
    r = max(r, 1e-6);\r
    float theta = atan(z.y, z.x);\r
    float phi = asin(clamp(z.z / r, -1.0, 1.0)) + time * uBulbSpin;\r
\r
    dr = pow(r, uPower - 1.0) * dr * uPower + 1.0;\r
    r = pow(r, uPower);\r
    theta *= uPower;\r
    phi *= uPower;\r
    z = r * vec3(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)) + p;\r
  }\r
\r
  return 0.5 * log(max(r, 1e-6)) * r / dr;\r
}\r
\r
mat3 rot_z(float a) {\r
  float c = cos(a);\r
  float s = sin(a);\r
  return mat3(\r
      c, s, 0.0,\r
     -s, c, 0.0,\r
      0.0, 0.0, 1.0\r
    );\r
}\r
\r
mat3 rot_y(float a) {\r
  float c = cos(a);\r
  float s = sin(a);\r
  return mat3(\r
      c, 0.0, s,\r
      0.0, 1.0, 0.0,\r
     -s, 0.0, c\r
    );\r
}\r
\r
mat3 rot_x(float a) {\r
  float c = cos(a);\r
  float s = sin(a);\r
  return mat3(\r
      1.0, 0.0, 0.0,\r
      0.0, c, s,\r
      0.0, -s, c\r
    );\r
}\r
\r
vec3 skyColor(vec3 ro, vec3 rd, vec3 skyCol) {\r
  vec3 col = clamp(vec3(0.0025 / abs(rd.y)) * skyCol, 0.0, 1.0);\r
\r
  float tp0 = rayPlane(ro, rd, vec4(vec3(0.0, 1.0, 0.0), 4.0));\r
  float tp1 = rayPlane(ro, rd, vec4(vec3(0.0, -1.0, 0.0), 6.0));\r
  float tp = max(tp0, tp1);\r
  if (tp > 0.0) {\r
    vec3 pos = ro + tp * rd;\r
    vec2 pp = pos.xz;\r
    float db = boxSDF(pp, vec2(6.0, 9.0)) - 1.0;\r
    col += vec3(4.0) * skyCol * rd.y * rd.y * smoothstep(0.25, 0.0, db);\r
    col += vec3(0.8) * skyCol * exp(-0.5 * max(db, 0.0));\r
  }\r
\r
  if (tp0 > 0.0) {\r
    vec3 pos = ro + tp0 * rd;\r
    vec2 pp = pos.xz;\r
    float ds = length(pp) - 0.5;\r
    col += vec3(0.25) * skyCol * exp(-0.5 * max(ds, 0.0));\r
  }\r
\r
  return clamp(col, 0.0, 10.0);\r
}\r
\r
float df(vec3 p, float time) {\r
  p *= g_rot;\r
  const float z1 = 2.0;\r
  return mandelBulb(p / z1, time) * z1;\r
}\r
\r
vec3 normal(vec3 pos, float time) {\r
  vec2 eps = vec2(uNormOffset, 0.0);\r
  vec3 nor;\r
  nor.x = df(pos + eps.xyy, time) - df(pos - eps.xyy, time);\r
  nor.y = df(pos + eps.yxy, time) - df(pos - eps.yxy, time);\r
  nor.z = df(pos + eps.yyx, time) - df(pos - eps.yyx, time);\r
  return normalize(nor);\r
}\r
\r
float rayMarch(vec3 ro, vec3 rd, float dfactor, float time, out int ii) {\r
  float t = 0.0;\r
  float tol = dfactor * uTolerance;\r
  ii = uRayMarches;\r
  for (int i = 0; i < 96; ++i) {\r
    if (i >= uRayMarches) {\r
      break;\r
    }\r
    if (t > uMaxRayLength) {\r
      t = uMaxRayLength;\r
      break;\r
    }\r
    float d = dfactor * df(ro + rd * t, time);\r
    if (d < tol) {\r
      ii = i;\r
      break;\r
    }\r
    t += d;\r
  }\r
  return t;\r
}\r
\r
vec3 render(vec3 ro, vec3 rd, float time) {\r
  vec3 agg = vec3(0.0);\r
  vec3 ragg = vec3(1.0);\r
\r
  bool isInside = df(ro, time) < 0.0;\r
\r
  vec3 baseSky = hsv2rgb(vec3(uHueShift + 0.6, uSkySat, uSkyVal)) * uSkyBoost;\r
  vec3 baseGlow = hsv2rgb(vec3(uHueShift + uGlowHueOffset, uGlowSat, uGlowVal)) * uGlowBoost;\r
  vec3 baseDiffuse = hsv2rgb(vec3(uHueShift + 0.6, uDiffuseSat, uDiffuseVal)) * uDiffuseBoost;\r
\r
  vec3 nebulaSky = hsv2rgb(vec3(uNebulaHueShift + 0.18, uNebulaSat, uNebulaVal)) * (uSkyBoost * 0.9);\r
  vec3 nebulaGlow = hsv2rgb(vec3(uNebulaGlowHue, uNebulaSat, uNebulaVal * 1.5)) * uNebulaGlowBoost;\r
  vec3 nebulaDiffuse = hsv2rgb(vec3(uNebulaHueShift + 0.55, uNebulaSat * 0.8, uNebulaVal)) * uDiffuseBoost;\r
\r
  float nebulaMix = clamp(uNebulaMix, 0.0, 1.0);\r
  vec3 skyCol = mix(baseSky, nebulaSky, nebulaMix);\r
  vec3 glowCol = mix(baseGlow, nebulaGlow, nebulaMix);\r
  vec3 diffuseCol = mix(baseDiffuse, nebulaDiffuse, nebulaMix);\r
\r
  for (int bounce = 0; bounce < 5; ++bounce) {\r
    if (bounce >= uBounces) {\r
      break;\r
    }\r
    float dfactor = isInside ? -1.0 : 1.0;\r
    float mragg = max(max(ragg.x, ragg.y), ragg.z);\r
    if (mragg < 0.025) {\r
      break;\r
    }\r
    int iter;\r
    float st = rayMarch(ro, rd, dfactor, time, iter);\r
    if (st >= uMaxRayLength) {\r
      agg += ragg * skyColor(ro, rd, skyCol);\r
      break;\r
    }\r
\r
    vec3 sp = ro + rd * st;\r
    vec3 sn = dfactor * normal(sp, time);\r
\r
    float fre = 1.0 + dot(rd, sn);\r
    fre *= fre;\r
    fre = mix(0.1, 1.0, fre);\r
\r
    vec3 ld = normalize(uLightPos - sp);\r
    float dif = max(dot(ld, sn), 0.0);\r
    vec3 ref = reflect(rd, sn);\r
    float re = uRefractIndex;\r
    float ire = 1.0 / re;\r
    vec3 refr = refract(rd, sn, !isInside ? re : ire);\r
    vec3 rsky = skyColor(sp, ref, skyCol);\r
\r
    vec3 col = vec3(0.0);\r
    col += diffuseCol * dif * dif * (1.0 - uMatTransmit);\r
    float edge = smoothstep(1.0, 0.9, fre);\r
    col += rsky * uMatReflect * edge;\r
    col += glowCol * exp(-float(iter) * uGlowFalloff);\r
\r
    if (isInside) {\r
      ragg *= exp(-(st + uInitStep) * uBeerColor);\r
    }\r
    agg += ragg * col;\r
\r
    if (refr == vec3(0.0)) {\r
      rd = ref;\r
    } else {\r
      ragg *= uMatTransmit;\r
      isInside = !isInside;\r
      rd = refr;\r
    }\r
    ro = sp + uInitStep * rd;\r
  }\r
\r
  return agg;\r
}\r
\r
vec3 effect(vec2 p, float time) {\r
  g_rot = rot_x(uRotSpeedX * time) * rot_y(uRotSpeedY * time);\r
  vec3 ro = vec3(0.0, uCamHeight, uCamDistance);\r
  const vec3 la = vec3(0.0);\r
  const vec3 up = vec3(0.0, 1.0, 0.0);\r
\r
  vec3 ww = normalize(la - ro);\r
  vec3 uu = normalize(cross(up, ww));\r
  vec3 vv = cross(ww, uu);\r
  float fov = tan(uFov);\r
  vec3 rd = normalize(-p.x * uu + p.y * vv + fov * ww);\r
\r
  return render(ro, rd, time);\r
}\r
\r
void main() {\r
  vec2 q = gl_FragCoord.xy / uResolution.xy;\r
  vec2 p = -1.0 + 2.0 * q;\r
  p.x *= uResolution.x / uResolution.y;\r
  float time = uTime * uTimeScale;\r
  vec3 col = effect(p, time);\r
  col = aces_approx(col);\r
  col = sRGB(col);\r
  outColor = vec4(col, 1.0);\r
}\r
`,me=`#version 300 es\r
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
\r
const float TAU = 6.28318530718;\r
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
  outColor = vec4(col, 1.0);\r
}\r
`,pe=`#version 300 es\r
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
`,b=[{id:"neon",name:"Neon Isoclines",description:"Electric contour bands driven by seeded radial harmonics.",fragment:K,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"components",label:"Components",uniform:"uComponents",type:"int",value:64,min:1,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:10}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:8,min:1,max:64,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:10}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.25,min:.01,max:.75,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.05}},{id:"noiseAmount",label:"Noise Amount",uniform:"uNoiseAmount",type:"float",value:2.5,min:0,max:5,step:.05,key:{inc:"r",dec:"f",step:.1,shiftStep:.25}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tanh-terrain",name:"Tanh Terrain Isoclines",description:"Tanh warped contours with bubbling noise and topo glow.",fragment:J,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:2.1,min:.1,max:6,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"octaves",label:"Octaves",uniform:"uOctaves",type:"int",value:4,min:1,max:12,step:1,key:{inc:"3",dec:"4",step:1}},{id:"lacunarity",label:"Lacunarity",uniform:"uLacunarity",type:"float",value:1.4,min:1.01,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"gain",label:"Gain",uniform:"uGain",type:"float",value:.5,min:.01,max:.99,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:16,min:1,max:96,step:1,key:{inc:"q",dec:"a",step:4,shiftStep:12}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.2,min:.02,max:.75,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"bubbleAmp",label:"Bubble Amp",uniform:"uBubbleAmp",type:"float",value:.26,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.08}},{id:"bubbleFreq",label:"Bubble Freq",uniform:"uBubbleFreq",type:"float",value:2,min:0,max:6,step:.05,key:{inc:"r",dec:"f",step:.25,shiftStep:.75}},{id:"bubbleDetail",label:"Bubble Detail",uniform:"uBubbleDetail",type:"float",value:1.2,min:.1,max:3,step:.05,key:{inc:"t",dec:"g",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tunnel",name:"Brownian Loop Tunnel",description:"Looped tunnel with Brownian noise, fog, and hue spin.",fragment:$,resolutionUniform:"iResolution",timeUniform:"iTime",timeMode:"looped",loopDuration:8,loopUniform:"uLoopDuration",params:[{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:4,min:0,max:10,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"noiseScale",label:"Noise Scale",uniform:"uNoiseScale",type:"float",value:1.9,min:.1,max:4,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.5,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"fogDensity",label:"Fog Density",uniform:"uFogDensity",type:"float",value:2,min:.1,max:6,step:.05,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"baseRed",label:"Base Red",uniform:"uBaseColor",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:0},{id:"baseGreen",label:"Base Green",uniform:"uBaseColor",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:1},{id:"baseBlue",label:"Base Blue",uniform:"uBaseColor",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:2}]},{id:"prismatic-fold",name:"Prismatic Fold Raymarch",description:"Rotating folded planes with prismatic glow and controllable depth.",fragment:oe,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:24,step:1,key:{inc:"1",dec:"2",step:1,shiftStep:4}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.2,min:-1.5,max:1.5,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.5,min:.1,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:5,min:1.5,max:10,step:.1,key:{inc:"7",dec:"8",step:.2,shiftStep:.6}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"cameraDistance",label:"Camera Distance",uniform:"uCameraDistance",type:"float",value:50,min:10,max:120,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:5}},{id:"cameraSpin",label:"Camera Spin",uniform:"uCameraSpin",type:"float",value:1,min:-3,max:3,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.4}},{id:"colorMix",label:"Color Mix",uniform:"uColorMix",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"alphaGain",label:"Alpha Gain",uniform:"uAlphaGain",type:"float",value:1,min:.3,max:2,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"[",dec:"]",step:.05},component:2}]},{id:"koch",name:"Koch Snowflake",description:"Iterative snowflake edges with neon glow mixing.",fragment:Q,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.8,min:.1,max:2,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:.2,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:2,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.3,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:2}]},{id:"quasi",name:"Quasi Snowflake",description:"Quasicrystal warp with a drifting snowflake outline.",fragment:ee,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:6,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:1.1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.8,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.02,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.03,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.02},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.05,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.02},component:2}]},{id:"tileable-water-plus",name:"Tileable Water Plus",description:"Tileable water ripples with tunable speed, scale, and tint.",fragment:ie,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"tileScale",label:"Tile Scale",uniform:"uTileScale",type:"float",value:1,min:.5,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.2,max:2.5,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.2,min:.3,max:2.5,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"waveShift",label:"Wave Shift",uniform:"uWaveShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"tintRed",label:"Tint Red",uniform:"uTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02},component:0},{id:"tintGreen",label:"Tint Green",uniform:"uTint",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02},component:1},{id:"tintBlue",label:"Tint Blue",uniform:"uTint",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:2}]},{id:"seascape",name:"Seascape Plus",description:"Raymarched ocean with tunable swell and camera drift.",fragment:ae,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Freq",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:1,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:0,min:-1,max:1,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:0,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1.1,min:.6,max:1.6,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"waterBrightness",label:"Water Bright",uniform:"uWaterBrightness",type:"float",value:.6,min:.2,max:1.2,step:.02,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"waterRed",label:"Water Red",uniform:"uWaterTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02},component:0},{id:"waterGreen",label:"Water Green",uniform:"uWaterTint",type:"float",value:.09,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02},component:1},{id:"waterBlue",label:"Water Blue",uniform:"uWaterTint",type:"float",value:.18,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.02},component:2}]},{id:"sunset-plus",name:"Sunset Plus",description:"Volumetric sunset clouds with tunable turbulence and hue drift.",fragment:le,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}}]},{id:"diff-chromatic",name:"Chromatic Flow",description:"Two-channel diffusion with hue-as-angle and drifting color pulses.",fragment:ne,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.998,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"rotate",label:"Rotate",uniform:"uRotate",type:"float",value:.02,min:-.2,max:.2,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.03}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.35,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"w",dec:"s",step:.01,shiftStep:.03}},{id:"valueGain",label:"Value Gain",uniform:"uValueGain",type:"float",value:2.2,min:.2,max:6,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"zippy-zaps",name:"Zippy Zaps Plus",description:"Tanh-warped chromatic flow with twistable energy and glow.",fragment:ue,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.2,min:.05,max:.5,step:.005,key:{inc:"1",dec:"2",step:.01,shiftStep:.03}},{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1,min:.2,max:2,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:1,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"iterLimit",label:"Iter Limit",uniform:"uIterLimit",type:"float",value:19,min:4,max:19,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:3}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.4,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"offsetX",label:"Offset X",uniform:"uOffsetX",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"offsetY",label:"Offset Y",uniform:"uOffsetY",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}}]},{id:"space-lightning-plus",name:"Space Lightning Plus",description:"Funky ion bolts with twistable arcs, palette waves, and core glow.",fragment:se,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.35,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.9,min:.3,max:1.8,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"spin",label:"Spin",uniform:"uSpin",type:"float",value:.4,min:-2,max:2,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1.2,min:0,max:3,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:.9,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1.1,min:0,max:3,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"boltDensity",label:"Bolt Density",uniform:"uBoltDensity",type:"float",value:6.5,min:1,max:20,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"boltSharpness",label:"Bolt Sharpness",uniform:"uBoltSharpness",type:"float",value:.9,min:.1,max:2.5,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:1.2,min:.2,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"arcSteps",label:"Arc Steps",uniform:"uArcSteps",type:"float",value:40,min:6,max:80,step:1,key:{inc:"y",dec:"h",step:1,shiftStep:5}},{id:"coreSize",label:"Core Size",uniform:"uCoreSize",type:"float",value:.25,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"coreGlow",label:"Core Glow",uniform:"uCoreGlow",type:"float",value:.8,min:0,max:2,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.25,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02,shiftStep:.08}},{id:"paletteShift",label:"Palette Shift",uniform:"uPaletteShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"primaryRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.08,min:0,max:1,step:.01,component:0},{id:"primaryGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,component:1},{id:"primaryBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"secondaryRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.9,min:0,max:1,step:.01,component:0},{id:"secondaryGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"secondaryBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.8,min:0,max:1,step:.01,component:2},{id:"accentRed",label:"Accent Red",uniform:"uColorAccent",type:"float",value:1,min:0,max:1,step:.01,component:0},{id:"accentGreen",label:"Accent Green",uniform:"uColorAccent",type:"float",value:.8,min:0,max:1,step:.01,component:1},{id:"accentBlue",label:"Accent Blue",uniform:"uColorAccent",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"lightning-blade-plus",name:"Lightning Blade Plus",description:"Flaring core with jittered blades and controllable noise flicker.",fragment:ce,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.4,min:.1,max:1.2,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"coreRadius",label:"Core Radius",uniform:"uCoreRadius",type:"float",value:.02,min:.005,max:.1,step:.001,key:{inc:"5",dec:"6",step:.002,shiftStep:.01}},{id:"coreNoiseScale",label:"Core Noise Scale",uniform:"uCoreNoiseScale",type:"float",value:50,min:5,max:120,step:.5,key:{inc:"7",dec:"8",step:1,shiftStep:5}},{id:"coreNoiseAmp",label:"Core Noise Amp",uniform:"uCoreNoiseAmp",type:"float",value:.02,min:0,max:.08,step:.001,key:{inc:"q",dec:"a",step:.002,shiftStep:.01}},{id:"coreIntensity",label:"Core Intensity",uniform:"uCoreIntensity",type:"float",value:.6,min:.1,max:2,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"boltCount",label:"Bolt Count",uniform:"uBoltCount",type:"int",value:4,min:1,max:10,step:1,key:{inc:"e",dec:"d",step:1,shiftStep:2}},{id:"boltLengthMin",label:"Bolt Length Min",uniform:"uBoltLengthMin",type:"float",value:.12,min:.05,max:.4,step:.01,key:{inc:"r",dec:"f",step:.01,shiftStep:.05}},{id:"boltLengthMax",label:"Bolt Length Max",uniform:"uBoltLengthMax",type:"float",value:.35,min:.1,max:.7,step:.01,key:{inc:"t",dec:"g",step:.01,shiftStep:.05}},{id:"boltWidth",label:"Bolt Width",uniform:"uBoltWidth",type:"float",value:6e-4,min:1e-4,max:.004,step:1e-4,key:{inc:"y",dec:"h",step:2e-4,shiftStep:.001}},{id:"boltWiggle",label:"Bolt Wiggle",uniform:"uBoltWiggle",type:"float",value:.03,min:0,max:.15,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"boltNoiseScale",label:"Bolt Noise Scale",uniform:"uBoltNoiseScale",type:"float",value:20,min:5,max:60,step:.5,key:{inc:"i",dec:"k",step:1,shiftStep:3}},{id:"boltNoiseSpeed",label:"Bolt Noise Speed",uniform:"uBoltNoiseSpeed",type:"float",value:2,min:0,max:8,step:.05,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"boltSecondaryScale",label:"Bolt Secondary",uniform:"uBoltSecondaryScale",type:"float",value:.8,min:0,max:1.5,step:.02,key:{inc:"p",dec:";",step:.02,shiftStep:.1}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:.25,min:.05,max:1.2,step:.02},{id:"flickerSpeed",label:"Flicker Speed",uniform:"uFlickerSpeed",type:"float",value:4,min:0,max:12,step:.1},{id:"angleJitter",label:"Angle Jitter",uniform:"uAngleJitter",type:"float",value:.5,min:0,max:2,step:.02},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:.125,min:0,max:.5,step:.005},{id:"noiseOctaves",label:"Noise Octaves",uniform:"uNoiseOctaves",type:"int",value:3,min:1,max:6,step:1},{id:"primaryRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"primaryGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.1,min:0,max:1,step:.01,component:1},{id:"primaryBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:.5,min:0,max:1,step:.01,component:2},{id:"secondaryRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"secondaryGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"secondaryBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.8,min:0,max:1,step:.01,component:2},{id:"accentRed",label:"Accent Red",uniform:"uColorAccent",type:"float",value:.8,min:0,max:1,step:.01,component:0},{id:"accentGreen",label:"Accent Green",uniform:"uColorAccent",type:"float",value:.3,min:0,max:1,step:.01,component:1},{id:"accentBlue",label:"Accent Blue",uniform:"uColorAccent",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"mandelbulb-inside-plus",name:"Inside the Mandelbulb Plus",description:"Raymarched mandelbulb interior with tunable optics and palette glow.",fragment:fe,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"power",label:"Power",uniform:"uPower",type:"float",value:8,min:2,max:12,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"bulbSpin",label:"Bulb Spin",uniform:"uBulbSpin",type:"float",value:.2,min:0,max:1.5,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"loops",label:"Loops",uniform:"uLoops",type:"int",value:2,min:1,max:6,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:1}},{id:"rayMarches",label:"Ray Marches",uniform:"uRayMarches",type:"int",value:60,min:20,max:96,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:5}},{id:"maxRayLength",label:"Max Ray Length",uniform:"uMaxRayLength",type:"float",value:20,min:5,max:40,step:.5,key:{inc:"w",dec:"s",step:.5,shiftStep:2}},{id:"tolerance",label:"Tolerance",uniform:"uTolerance",type:"float",value:1e-4,min:1e-5,max:.001,step:1e-5,key:{inc:"e",dec:"d",step:2e-5,shiftStep:1e-4}},{id:"normOffset",label:"Normal Offset",uniform:"uNormOffset",type:"float",value:.005,min:.001,max:.02,step:5e-4,key:{inc:"r",dec:"f",step:5e-4,shiftStep:.002}},{id:"bounces",label:"Bounces",uniform:"uBounces",type:"int",value:5,min:1,max:5,step:1,key:{inc:"t",dec:"g",step:1,shiftStep:1}},{id:"initStep",label:"Init Step",uniform:"uInitStep",type:"float",value:.1,min:.01,max:.3,step:.01,key:{inc:"y",dec:"h",step:.01,shiftStep:.05}},{id:"rotSpeedX",label:"Rot Speed X",uniform:"uRotSpeedX",type:"float",value:.2,min:-1,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.1}},{id:"rotSpeedY",label:"Rot Speed Y",uniform:"uRotSpeedY",type:"float",value:.3,min:-1,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:2,max:10,step:.1,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:2,min:.5,max:5,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"fov",label:"FOV",uniform:"uFov",type:"float",value:.523,min:.3,max:1.2,step:.01},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"glowBoost",label:"Glow Boost",uniform:"uGlowBoost",type:"float",value:1.2,min:0,max:4,step:.05},{id:"glowFalloff",label:"Glow Falloff",uniform:"uGlowFalloff",type:"float",value:.06,min:.01,max:.2,step:.005},{id:"diffuseBoost",label:"Diffuse Boost",uniform:"uDiffuseBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"matTransmit",label:"Mat Transmit",uniform:"uMatTransmit",type:"float",value:.8,min:0,max:1,step:.01},{id:"matReflect",label:"Mat Reflect",uniform:"uMatReflect",type:"float",value:.5,min:0,max:1,step:.01},{id:"refractIndex",label:"Refract Index",uniform:"uRefractIndex",type:"float",value:1.05,min:1,max:2,step:.01},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"glowHueOffset",label:"Glow Hue Offset",uniform:"uGlowHueOffset",type:"float",value:.065,min:-.5,max:.5,step:.005},{id:"nebulaMix",label:"Nebula Mix",uniform:"uNebulaMix",type:"float",value:0,min:0,max:1,step:.01},{id:"nebulaHueShift",label:"Nebula Hue",uniform:"uNebulaHueShift",type:"float",value:.12,min:-1,max:1,step:.01},{id:"nebulaSat",label:"Nebula Sat",uniform:"uNebulaSat",type:"float",value:.9,min:0,max:1,step:.01},{id:"nebulaVal",label:"Nebula Val",uniform:"uNebulaVal",type:"float",value:1.6,min:.2,max:3,step:.02},{id:"nebulaGlowHue",label:"Nebula Glow Hue",uniform:"uNebulaGlowHue",type:"float",value:.35,min:-1,max:1,step:.01},{id:"nebulaGlowBoost",label:"Nebula Glow",uniform:"uNebulaGlowBoost",type:"float",value:1.6,min:0,max:4,step:.05},{id:"skySat",label:"Sky Saturation",uniform:"uSkySat",type:"float",value:.86,min:0,max:1,step:.01},{id:"skyVal",label:"Sky Value",uniform:"uSkyVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"glowSat",label:"Glow Saturation",uniform:"uGlowSat",type:"float",value:.8,min:0,max:1,step:.01},{id:"glowVal",label:"Glow Value",uniform:"uGlowVal",type:"float",value:6,min:.5,max:8,step:.1},{id:"diffuseSat",label:"Diffuse Saturation",uniform:"uDiffuseSat",type:"float",value:.85,min:0,max:1,step:.01},{id:"diffuseVal",label:"Diffuse Value",uniform:"uDiffuseVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"beerRed",label:"Beer Red",uniform:"uBeerColor",type:"float",value:.02,min:0,max:.2,step:.005,component:0},{id:"beerGreen",label:"Beer Green",uniform:"uBeerColor",type:"float",value:.08,min:0,max:.2,step:.005,component:1},{id:"beerBlue",label:"Beer Blue",uniform:"uBeerColor",type:"float",value:.12,min:0,max:.2,step:.005,component:2},{id:"lightX",label:"Light X",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:0},{id:"lightY",label:"Light Y",uniform:"uLightPos",type:"float",value:10,min:-5,max:25,step:.5,component:1},{id:"lightZ",label:"Light Z",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:2}]},{id:"auroras-plus",name:"Auroras Plus",description:"Volumetric auroras with tunable trails, palette waves, and sky glare.",fragment:me,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"auroraSpeed",label:"Aurora Speed",uniform:"uAuroraSpeed",type:"float",value:.06,min:0,max:.2,step:.005,key:{inc:"3",dec:"4",step:.005,shiftStep:.02}},{id:"auroraScale",label:"Aurora Scale",uniform:"uAuroraScale",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"auroraWarp",label:"Aurora Warp",uniform:"uAuroraWarp",type:"float",value:.35,min:0,max:1,step:.02,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"auroraSteps",label:"Aurora Steps",uniform:"uAuroraSteps",type:"int",value:50,min:8,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"auroraBase",label:"Aurora Base",uniform:"uAuroraBase",type:"float",value:.8,min:.2,max:1.6,step:.02,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"auroraStride",label:"Aurora Stride",uniform:"uAuroraStride",type:"float",value:.002,min:2e-4,max:.01,step:2e-4,key:{inc:"e",dec:"d",step:2e-4,shiftStep:.001}},{id:"auroraCurve",label:"Aurora Curve",uniform:"uAuroraCurve",type:"float",value:1.4,min:.8,max:2.2,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"auroraIntensity",label:"Aurora Intensity",uniform:"uAuroraIntensity",type:"float",value:1.8,min:.2,max:4,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"trailBlend",label:"Trail Blend",uniform:"uTrailBlend",type:"float",value:.5,min:.1,max:.9,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"trailFalloff",label:"Trail Falloff",uniform:"uTrailFalloff",type:"float",value:.065,min:.01,max:.15,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"trailFade",label:"Trail Fade",uniform:"uTrailFade",type:"float",value:2.5,min:.5,max:5,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.4}},{id:"ditherStrength",label:"Dither Strength",uniform:"uDitherStrength",type:"float",value:.006,min:0,max:.02,step:5e-4,key:{inc:"o",dec:"l",step:5e-4,shiftStep:.002}},{id:"horizonFade",label:"Horizon Fade",uniform:"uHorizonFade",type:"float",value:.01,min:.001,max:.05,step:.001,key:{inc:"p",dec:";",step:.001,shiftStep:.005}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:-.1,min:-1,max:1,step:.01},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:.1,min:-1,max:1,step:.01},{id:"camWobble",label:"Cam Wobble",uniform:"uCamWobble",type:"float",value:.2,min:0,max:.6,step:.01},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:6.7,min:4,max:12,step:.1},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:0,min:-1,max:2,step:.05},{id:"skyStrength",label:"Sky Strength",uniform:"uSkyStrength",type:"float",value:.63,min:.1,max:2,step:.02},{id:"starDensity",label:"Star Density",uniform:"uStarDensity",type:"float",value:5e-4,min:0,max:.005,step:1e-4},{id:"starIntensity",label:"Star Intensity",uniform:"uStarIntensity",type:"float",value:.8,min:0,max:2,step:.05},{id:"reflectionStrength",label:"Reflection Strength",uniform:"uReflectionStrength",type:"float",value:.6,min:0,max:1.5,step:.05},{id:"reflectionTint",label:"Reflection Tint",uniform:"uReflectionTint",type:"float",value:1,min:0,max:2,step:.05},{id:"reflectionFog",label:"Reflection Fog",uniform:"uReflectionFog",type:"float",value:2,min:0,max:6,step:.1},{id:"colorBand",label:"Color Band",uniform:"uColorBand",type:"float",value:.043,min:0,max:.2,step:.002},{id:"colorSpeed",label:"Color Speed",uniform:"uColorSpeed",type:"float",value:0,min:-1,max:1,step:.01},{id:"auroraRedA",label:"Aurora Red A",uniform:"uAuroraColorA",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenA",label:"Aurora Green A",uniform:"uAuroraColorA",type:"float",value:.9,min:0,max:1,step:.01,component:1},{id:"auroraBlueA",label:"Aurora Blue A",uniform:"uAuroraColorA",type:"float",value:.6,min:0,max:1,step:.01,component:2},{id:"auroraRedB",label:"Aurora Red B",uniform:"uAuroraColorB",type:"float",value:.6,min:0,max:1,step:.01,component:0},{id:"auroraGreenB",label:"Aurora Green B",uniform:"uAuroraColorB",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"auroraBlueB",label:"Aurora Blue B",uniform:"uAuroraColorB",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"auroraRedC",label:"Aurora Red C",uniform:"uAuroraColorC",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenC",label:"Aurora Green C",uniform:"uAuroraColorC",type:"float",value:.6,min:0,max:1,step:.01,component:1},{id:"auroraBlueC",label:"Aurora Blue C",uniform:"uAuroraColorC",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedA",label:"BG Red A",uniform:"uBgColorA",type:"float",value:.05,min:0,max:1,step:.01,component:0},{id:"bgGreenA",label:"BG Green A",uniform:"uBgColorA",type:"float",value:.1,min:0,max:1,step:.01,component:1},{id:"bgBlueA",label:"BG Blue A",uniform:"uBgColorA",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedB",label:"BG Red B",uniform:"uBgColorB",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"bgGreenB",label:"BG Green B",uniform:"uBgColorB",type:"float",value:.05,min:0,max:1,step:.01,component:1},{id:"bgBlueB",label:"BG Blue B",uniform:"uBgColorB",type:"float",value:.2,min:0,max:1,step:.01,component:2}]},{id:"diff-edge-flow",name:"Edge Flow Vectors",description:"Diffusive scalar field rendered as glowing edge-flow vectors.",fragment:re,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.996,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"blobAmp",label:"Blob Amp",uniform:"uBlobAmp",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"blobRadius",label:"Blob Radius",uniform:"uBlobRadius",type:"float",value:.07,min:.01,max:.25,step:.005,key:{inc:"q",dec:"a",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.8,min:0,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"flowGain",label:"Flow Gain",uniform:"uFlowGain",type:"float",value:3,min:.2,max:8,step:.1,key:{inc:"e",dec:"d",step:.2,shiftStep:.6}},{id:"flowThreshold",label:"Flow Threshold",uniform:"uFlowThreshold",type:"float",value:.02,min:0,max:.2,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"diff-threshold",name:"Threshold Feedback",description:"Diffusion with nonlinear feedback for digital fungus crackle.",fragment:te,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:192,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.5,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.995,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"threshold",label:"Threshold",uniform:"uThreshold",type:"float",value:.5,min:.1,max:.9,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.06}},{id:"sharpness",label:"Sharpness",uniform:"uSharpness",type:"float",value:18,min:1,max:40,step:.5,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.08,min:0,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.2,min:0,max:1.5,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractal-fold",name:"Fractal Fold Raymarch",description:"Recursive box-folding fractal with prismatic lighting and IQ palette.",fragment:pe,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.5,max:5,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"distort",label:"Distort",uniform:"uDistort",type:"float",value:2.5,min:1.5,max:4,step:.02,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:1,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.15}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.1,min:-.5,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"maxSteps",label:"Max Steps",uniform:"uMaxSteps",type:"float",value:100,min:20,max:200,step:10,key:{inc:"e",dec:"d",step:10,shiftStep:30}}]}];function E(e,n,t){const o=e.createShader(n);if(!o)throw new Error("Failed to create shader");if(e.shaderSource(o,t),e.compileShader(o),!e.getShaderParameter(o,e.COMPILE_STATUS)){const a=e.getShaderInfoLog(o)||"Unknown shader error";throw e.deleteShader(o),new Error(a)}return o}function de(e,n,t){const o=E(e,e.VERTEX_SHADER,n),a=E(e,e.FRAGMENT_SHADER,t),i=e.createProgram();if(!i)throw new Error("Failed to create program");if(e.attachShader(i,o),e.attachShader(i,a),e.linkProgram(i),e.deleteShader(o),e.deleteShader(a),!e.getProgramParameter(i,e.LINK_STATUS)){const l=e.getProgramInfoLog(i)||"Unknown program error";throw e.deleteProgram(i),new Error(l)}return i}function ve(e,n,t){const o={};for(const a of t)o[a]=e.getUniformLocation(n,a);return o}function ye(e,n=2){const t=Math.min(window.devicePixelRatio||1,n),o=Math.max(1,Math.floor(e.clientWidth*t)),a=Math.max(1,Math.floor(e.clientHeight*t));return(e.width!==o||e.height!==a)&&(e.width=o,e.height=a),{width:o,height:a,dpr:t}}function he(e){const n=e.createVertexArray();if(!n)throw new Error("Failed to create VAO");return e.bindVertexArray(n),n}const W=document.querySelector("#app");if(!W)throw new Error("Missing #app root element");W.innerHTML=`
  <div class="app-shell">
    <div class="layout">
      <aside class="panel">
        <div class="panel-section">
          <div class="section-title">Scenes</div>
          <div class="scene-list" id="scene-list"></div>
        </div>
        <div class="panel-section">
          <div class="section-title">Controls</div>
          <div class="panel-actions" id="panel-actions"></div>
          <div class="control-list" id="control-list"></div>
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
`;const I=document.querySelector("#gl-canvas"),C=document.querySelector("#scene-list"),A=document.querySelector("#control-list"),g=document.querySelector("#panel-actions"),y=document.querySelector("#key-help"),T=document.querySelector("#hud-title"),M=document.querySelector("#hud-desc"),w=document.querySelector("[data-action='toggle-sidebar']"),N=document.querySelector(".stage");if(!I||!C||!A||!g||!y||!T||!M||!w||!N)throw new Error("Missing required UI elements");const r=I.getContext("webgl2",{antialias:!0});if(!r)throw N.innerHTML=`
    <div class="fallback">
      <h2>WebGL2 unavailable</h2>
      <p>Your browser or GPU does not expose WebGL2. Try a different browser.</p>
    </div>
  `,new Error("WebGL2 unavailable");const xe=r.getExtension("EXT_color_buffer_float"),be=!!xe;r.disable(r.DEPTH_TEST);r.disable(r.BLEND);const U=he(r),_=new Map,H={},v={},q={},F=new Map;for(const e of b){const n=de(r,Z,e.fragment),t=new Set;t.add(e.resolutionUniform),t.add(e.timeUniform),e.loopUniform&&t.add(e.loopUniform),e.stateful&&(t.add(e.passUniform??"uPass"),t.add(e.stateUniform??"uState"),t.add(e.gridUniform??"uGridSize"));for(const l of e.params)t.add(l.uniform);const o=ve(r,n,Array.from(t));_.set(e.id,{program:n,uniforms:o});const a={},i={};for(const l of e.params)a[l.id]=l,i[l.id]=l.type==="seed"?Math.floor(Math.random()*1e6):l.value;H[e.id]=a,v[e.id]={...i},q[e.id]={...i}}let u=b[0],P={},O=performance.now(),k=null,R=null;function G(e){w.classList.toggle("hidden",e)}function Se(){G(!1),R!==null&&window.clearTimeout(R),R=window.setTimeout(()=>{G(!0)},2500)}function V(){const e=document.body.classList.contains("sidebar-collapsed");w.textContent=e?">>":"<<"}function ge(){var e;(e=T.parentElement)==null||e.classList.remove("hidden"),k!==null&&window.clearTimeout(k),k=window.setTimeout(()=>{var n;(n=T.parentElement)==null||n.classList.add("hidden")},1e4)}function L(e){const n=r.createTexture();if(!n)throw new Error("Failed to create state texture");return r.bindTexture(r.TEXTURE_2D,n),r.texParameteri(r.TEXTURE_2D,r.TEXTURE_MIN_FILTER,r.NEAREST),r.texParameteri(r.TEXTURE_2D,r.TEXTURE_MAG_FILTER,r.NEAREST),r.texParameteri(r.TEXTURE_2D,r.TEXTURE_WRAP_S,r.REPEAT),r.texParameteri(r.TEXTURE_2D,r.TEXTURE_WRAP_T,r.REPEAT),r.texImage2D(r.TEXTURE_2D,0,r.RGBA16F,e,e,0,r.RGBA,r.HALF_FLOAT,null),r.bindTexture(r.TEXTURE_2D,null),n}function Ce(e){const n=L(e),t=L(e),o=r.createFramebuffer(),a=r.createFramebuffer();if(!o||!a)throw new Error("Failed to create framebuffer");return r.bindFramebuffer(r.FRAMEBUFFER,o),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,n,0),r.bindFramebuffer(r.FRAMEBUFFER,a),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,t,0),r.bindFramebuffer(r.FRAMEBUFFER,null),{size:e,textures:[n,t],fbos:[o,a],index:0,needsInit:!0}}function X(e){if(!e.stateful)return null;let n=F.get(e.id);const t=e.bufferSize??192;return(!n||n.size!==t)&&(n=Ce(t),F.set(e.id,n)),n}function z(e){const n=F.get(e);n&&(n.needsInit=!0)}function Te(e,n){let t=n;return e.min!==void 0&&(t=Math.max(e.min,t)),e.max!==void 0&&(t=Math.min(e.max,t)),e.type==="int"&&(t=Math.round(t)),t}function j(e,n){if(e.type==="int")return String(Math.round(n));const t=e.step??.01,o=t<1?Math.min(4,Math.max(2,Math.ceil(-Math.log10(t)))):0;return n.toFixed(o)}function we(e){return e.length===1?e.toLowerCase():e}function ke(e){return e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement||e instanceof HTMLSelectElement}function x(e,n,t,o=!0){const a=H[e][n];if(!a)return;const i=Te(a,t);if(v[e][n]=i,e===u.id&&o){const l=P[n];l!=null&&l.range&&(l.range.value=String(i)),l!=null&&l.number&&(l.number.value=j(a,i))}}function Re(e){var n;for(const t of((n=b.find(o=>o.id===e))==null?void 0:n.params)??[])t.type==="seed"&&x(e,t.id,Math.floor(Math.random()*1e6));z(e)}function Be(e){const n=q[e];for(const[t,o]of Object.entries(n))x(e,t,o,!0);z(e)}function Ae(){C.innerHTML="";for(const e of b){const n=document.createElement("button");n.className="scene-button",n.textContent=e.name,n.dataset.scene=e.id,n.addEventListener("click",()=>{Y(e.id)}),C.appendChild(n)}}function Fe(e){g.innerHTML="";const n=document.createElement("button");if(n.className="ghost small",n.textContent="Reset",n.addEventListener("click",()=>Be(e.id)),g.appendChild(n),e.params.some(o=>o.type==="seed")){const o=document.createElement("button");o.className="ghost small",o.textContent="Reseed",o.addEventListener("click",()=>Re(e.id)),g.appendChild(o)}}function Pe(e){A.innerHTML="",P={};for(const n of e.params){if(n.type==="seed")continue;const t=document.createElement("div");t.className="control";const o=document.createElement("div");o.className="control-header";const a=document.createElement("label");if(a.textContent=n.label,o.appendChild(a),n.key){const f=document.createElement("span");f.className="key-cap",f.textContent=`${n.key.inc.toUpperCase()}/${n.key.dec.toUpperCase()}`,o.appendChild(f)}const i=document.createElement("div");i.className="control-inputs";const l=document.createElement("input");l.type="range",l.min=String(n.min??0),l.max=String(n.max??1),l.step=String(n.step??(n.type==="int"?1:.01)),l.value=String(v[e.id][n.id]),l.addEventListener("input",f=>{const p=Number(f.target.value);Number.isNaN(p)||x(e.id,n.id,p)});const s=document.createElement("input");s.type="number",s.min=l.min,s.max=l.max,s.step=l.step,s.value=j(n,v[e.id][n.id]),s.addEventListener("input",f=>{const p=Number(f.target.value);Number.isNaN(p)||x(e.id,n.id,p)}),i.appendChild(l),i.appendChild(s),t.appendChild(o),t.appendChild(i),A.appendChild(t),P[n.id]={range:l,number:s}}}function Ge(e){y.innerHTML="";for(const n of e.params){if(!n.key||n.type==="seed")continue;const t=document.createElement("div");t.className="key-row",t.textContent=`${n.key.inc.toUpperCase()}/${n.key.dec.toUpperCase()}  ${n.label}`,y.appendChild(t)}y.childElementCount||(y.textContent="No mapped keys for this scene.")}function Y(e){const n=b.find(t=>t.id===e);n&&(u=n,n.stateful&&(X(n),z(n.id)),T.textContent=n.name,M.textContent=n.description,ge(),Fe(n),Pe(n),Ge(n),C.querySelectorAll(".scene-button").forEach(t=>{const o=t;o.classList.toggle("active",o.dataset.scene===n.id)}))}function Ie(e){if(ke(e.target))return;const n=we(e.key),t=u.params;for(const o of t){if(!o.key||o.type==="seed")continue;const a=n===o.key.inc,i=n===o.key.dec;if(!a&&!i)continue;const l=e.shiftKey&&o.key.shiftStep?o.key.shiftStep:o.key.step,f=v[u.id][o.id]+l*(a?1:-1);x(u.id,o.id,f),e.preventDefault();break}}function S(e,n,t,o,a){const i=v[e.id],l=n.uniforms,s=l[e.resolutionUniform];s&&r.uniform2f(s,o,a);const f=l[e.timeUniform];if(f)if(e.timeMode==="phase"){const c=e.loopDuration??8,m=t%c/c;r.uniform1f(f,m)}else if(e.timeMode==="looped"){const c=e.loopDuration??8,m=t%c;if(r.uniform1f(f,m),e.loopUniform){const d=l[e.loopUniform];d&&r.uniform1f(d,c)}}else r.uniform1f(f,t);const p={};for(const c of e.params){const m=l[c.uniform],d=i[c.id];if(c.component!==void 0){const D=p[c.uniform]??[0,0,0];D[c.component]=d,p[c.uniform]=D;continue}m&&(c.type==="int"?r.uniform1i(m,Math.round(d)):r.uniform1f(m,d))}for(const[c,m]of Object.entries(p)){const d=l[c];d&&r.uniform3f(d,m[0],m[1],m[2])}}function B(e,n,t,o){const a=n.uniforms[e.passUniform??"uPass"];a&&r.uniform1i(a,o);const i=n.uniforms[e.gridUniform??"uGridSize"];i&&r.uniform2f(i,t.size,t.size);const l=n.uniforms[e.stateUniform??"uState"];l&&r.uniform1i(l,0)}function h(e){const n=(e-O)/1e3,{width:t,height:o}=ye(I),a=_.get(u.id);if(a){if(u.stateful){if(!be){M.textContent="Stateful scenes require float render targets (EXT_color_buffer_float).",requestAnimationFrame(h);return}const i=X(u);if(!i){requestAnimationFrame(h);return}const l=()=>i.textures[i.index],s=()=>i.fbos[(i.index+1)%2];r.useProgram(a.program),r.bindVertexArray(U),i.needsInit&&(r.bindFramebuffer(r.FRAMEBUFFER,s()),r.viewport(0,0,i.size,i.size),r.activeTexture(r.TEXTURE0),r.bindTexture(r.TEXTURE_2D,l()),S(u,a,n,t,o),B(u,a,i,2),r.drawArrays(r.TRIANGLES,0,3),i.index=(i.index+1)%2,i.needsInit=!1),r.bindFramebuffer(r.FRAMEBUFFER,s()),r.viewport(0,0,i.size,i.size),r.activeTexture(r.TEXTURE0),r.bindTexture(r.TEXTURE_2D,l()),S(u,a,n,t,o),B(u,a,i,0),r.drawArrays(r.TRIANGLES,0,3),i.index=(i.index+1)%2,r.bindFramebuffer(r.FRAMEBUFFER,null),r.viewport(0,0,t,o),r.clearColor(0,0,0,1),r.clear(r.COLOR_BUFFER_BIT),r.activeTexture(r.TEXTURE0),r.bindTexture(r.TEXTURE_2D,l()),S(u,a,n,t,o),B(u,a,i,1),r.drawArrays(r.TRIANGLES,0,3),requestAnimationFrame(h);return}r.viewport(0,0,t,o),r.clearColor(0,0,0,1),r.clear(r.COLOR_BUFFER_BIT),r.useProgram(a.program),r.bindVertexArray(U),S(u,a,n,t,o),r.drawArrays(r.TRIANGLES,0,3),requestAnimationFrame(h)}}w.addEventListener("click",()=>{document.body.classList.toggle("sidebar-collapsed"),V()});N.addEventListener("mousemove",()=>{Se()});document.addEventListener("keydown",Ie);document.addEventListener("visibilitychange",()=>{document.hidden||(O=performance.now())});Ae();Y(u.id);V();G(!0);requestAnimationFrame(h);
