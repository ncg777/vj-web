(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))a(o);new MutationObserver(o=>{for(const i of o)if(i.type==="childList")for(const l of i.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&a(l)}).observe(document,{childList:!0,subtree:!0});function t(o){const i={};return o.integrity&&(i.integrity=o.integrity),o.referrerPolicy&&(i.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?i.credentials="include":o.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function a(o){if(o.ep)return;o.ep=!0;const i=t(o);fetch(o.href,i)}})();const mt=`#version 300 es
precision highp float;

const vec2 verts[3] = vec2[](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
);

void main() {
  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
`,pt=`#version 300 es
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
`,dt=`#version 300 es
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
`,ht=`#version 300 es\r
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
`,vt=`#version 300 es
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
`,yt=`#version 300 es
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
`,bt=`#version 300 es
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
`,gt=`#version 300 es
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
`,St=`#version 300 es
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
`,xt=`#version 300 es\r
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
`,wt=`#version 300 es
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
`,Ct=`#version 300 es\r
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
}`,Tt=`#version 300 es\r
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
`,kt=`#version 300 es
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
`,Rt=`#version 300 es\r
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
`,At=`#version 300 es
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
`,Bt=`#version 300 es\r
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
const float PI  = 3.14159265359;\r
\r
/* ---- helpers ---- */\r
\r
float hash(vec2 p) {\r
  p = fract(p * vec2(123.34, 456.21));\r
  p += dot(p, p + 45.32);\r
  return fract(p.x * p.y);\r
}\r
\r
float noise(vec2 p) {\r
  vec2 i = floor(p);\r
  vec2 f = fract(p);\r
  f = f * f * (3.0 - 2.0 * f);\r
  float a = hash(i);\r
  float b = hash(i + vec2(1.0, 0.0));\r
  float c = hash(i + vec2(0.0, 1.0));\r
  float d = hash(i + vec2(1.0, 1.0));\r
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);\r
}\r
\r
float fbm(vec2 p, int octaves) {\r
  float value = 0.0, amplitude = 0.5, total = 0.0;\r
  for (int i = 0; i < 8; i++) {\r
    if (i >= octaves) break;\r
    value += noise(p) * amplitude;\r
    total += amplitude;\r
    p = p * 2.0 + vec2(1.7, 9.2);\r
    amplitude *= 0.5;\r
  }\r
  return value / max(total, 0.001);\r
}\r
\r
mat2 rot(float a) {\r
  float c = cos(a), s = sin(a);\r
  return mat2(c, s, -s, c);\r
}\r
\r
float segSDF(vec2 p, vec2 a, vec2 b, float w) {\r
  vec2 pa = p - a, ba = b - a;\r
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);\r
  return length(pa - ba * h) - w;\r
}\r
\r
/* ---- lightning bolt with branches ---- */\r
\r
vec3 bolt(vec2 uv, vec2 start, vec2 end, float eventSeed, float time) {\r
  vec3 result = vec3(0.0);\r
\r
  vec2  dir  = end - start;\r
  float len  = length(dir);\r
  vec2  n    = normalize(dir);\r
  vec2  perp = vec2(-n.y, n.x);\r
\r
  const int SEGS = 8;\r
  vec2 prev = start;\r
\r
  for (int s = 0; s < SEGS; s++) {\r
    float t = float(s + 1) / float(SEGS);\r
    vec2 basePos = start + dir * t;\r
\r
    float nv = noise(vec2(t * uBoltNoiseScale + eventSeed * 7.0,\r
                          time * uBoltNoiseSpeed + eventSeed * 3.0)) * 2.0 - 1.0;\r
    float taper = t * (1.0 - t) * 4.0;\r
    basePos += perp * nv * uBoltWiggle * len * taper;\r
\r
    float d    = segSDF(uv, prev, basePos, uBoltWidth);\r
    float glow = uBoltIntensity / max(d, 0.001);\r
    glow = clamp(1.0 - exp(-glow * 0.01), 0.0, 1.0);\r
    result += glow * uLightningColor;\r
\r
    /* branch */\r
    if (uBoltBranching > 0.0 && s > 0 && s < SEGS - 1) {\r
      float bc = hash(vec2(float(s) + eventSeed * 11.0, 43.0));\r
      if (bc < uBoltBranching) {\r
        float ba2 = (hash(vec2(float(s) + eventSeed, 67.0)) - 0.5) * PI * 0.6;\r
        float bl  = len * 0.25 * hash(vec2(float(s) + eventSeed, 89.0));\r
        vec2  bd  = rot(ba2) * n;\r
        vec2  be  = prev + bd * bl;\r
        vec2  bm  = mix(prev, be, 0.5);\r
        float bnv = noise(vec2(float(s) * 3.0 + eventSeed * 5.0,\r
                               time * uBoltNoiseSpeed * 0.7)) * 2.0 - 1.0;\r
        bm += vec2(-bd.y, bd.x) * bnv * uBoltWiggle * bl;\r
\r
        float d1 = segSDF(uv, prev, bm, uBoltWidth * 0.6);\r
        float d2 = segSDF(uv, bm,  be,  uBoltWidth * 0.4);\r
        float bg = uBoltIntensity * 0.5 / max(min(d1, d2), 0.001);\r
        bg = clamp(1.0 - exp(-bg * 0.008), 0.0, 1.0);\r
        result += bg * uLightningColor * 0.7;\r
      }\r
    }\r
    prev = basePos;\r
  }\r
  return result;\r
}\r
\r
/* ---- main ---- */\r
\r
void main() {\r
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;\r
  uv *= uZoom;\r
\r
  float aspect = uResolution.x / uResolution.y;\r
  float time   = uTime * uTimeScale;\r
\r
  /* ---- cloud layer (top-down view) ---- */\r
  vec2  cUV    = uv * uCloudScale + vec2(time * uCloudSpeed * 0.03,\r
                                          time * uCloudSpeed * 0.02);\r
  float clouds = fbm(cUV, uNoiseOctaves);\r
  float detail = fbm(cUV * 2.5 + vec2(time * uCloudSpeed * 0.01, 0.0),\r
                      max(uNoiseOctaves - 1, 1));\r
  clouds = mix(clouds, clouds * detail, uCloudDetail);\r
  clouds = smoothstep(0.5 - uCloudDensity * 0.5, 0.5 + uCloudDensity * 0.3, clouds);\r
\r
  vec3 col = uCloudColor * clouds;\r
\r
  /* ---- lightning bolts ---- */\r
  float totalIllum = 0.0;\r
  vec3  lightCol   = vec3(0.0);\r
\r
  int count = max(uBoltCount, 1);\r
  for (int i = 0; i < 12; i++) {\r
    if (i >= count) break;\r
    float fi = float(i);\r
\r
    // Chaotic timing: scan a small window of candidate event indices\r
    // and treat each candidate time as (index + jitter)/rate. This\r
    // produces irregular, non-periodic strike times while remaining\r
    // deterministic and cheap to evaluate in a shader.\r
    float laneSeed = fi + uSeed * 0.37;\r
    float maxRate = max(0.08, uFlickerSpeed * 1.2);\r
    float baseIdxF = floor(time * maxRate);\r
    bool eventFound = false;\r
    float eventId = 0.0;\r
    float localT = 0.0;\r
    float flash = 0.0;\r
    float flicker = 1.0;\r
\r
    // check a few recent candidate events (most recent first)\r
    for (int k = 0; k < 8; k++) {\r
      float idxF = baseIdxF - float(k);\r
      float jitter = hash(vec2(idxF + laneSeed, 99.9));\r
      float eventTime = (idxF + jitter) / maxRate;\r
      float lt = (time - eventTime) * maxRate; // normalized local time in [0,1)\r
      if (lt >= 0.0 && lt < 1.0) {\r
        float spawnChance = hash(vec2(idxF + fi * 3.7, 17.0 + uSeed));\r
        if (spawnChance < 0.55) {\r
          break; // candidate didn't spawn this time\r
        }\r
        eventId = idxF;\r
        localT = lt;\r
        flash = smoothstep(0.0, 0.02, localT) * (1.0 - smoothstep(0.05, 0.4, localT));\r
        flicker = 1.0 - 0.3 * smoothstep(0.0, 1.0, sin(localT * 40.0 + fi * 10.0));\r
        flash *= flicker;\r
        if (flash < 0.001) {\r
          break;\r
        }\r
        eventFound = true;\r
        break;\r
      }\r
    }\r
\r
    if (!eventFound) continue;\r
\r
    vec2 startPos = vec2(\r
      hash(vec2(eventId + fi, 21.0 + uSeed)) * 2.0 - 1.0,\r
      hash(vec2(eventId + fi, 25.0 + uSeed)) * 2.0 - 1.0\r
    ) * vec2(aspect * 0.45, 0.45);\r
\r
    float boltLen = mix(uBoltLengthMin, uBoltLengthMax,\r
                        hash(vec2(eventId + fi, 37.0 + uSeed)));\r
    float angle   = hash(vec2(eventId + fi, 29.0 + uSeed)) * TAU;\r
    vec2  endPos  = startPos + vec2(cos(angle), sin(angle)) * boltLen;\r
\r
    float eventSeed = hash(vec2(eventId, fi + 41.0 + uSeed));\r
\r
    lightCol += bolt(uv, startPos, endPos, eventSeed, time) * flash;\r
\r
    /* cloud illumination near the strike */\r
    float dMid  = length(uv - mix(startPos, endPos, 0.5));\r
    float illum = flash * uCloudIllumination * exp(-dMid * 4.0);\r
    totalIllum += illum;\r
  }\r
\r
  col += uCloudColor * totalIllum * 1.5;\r
  col += lightCol;\r
\r
  outColor = vec4(col, 1.0);\r
}\r
`,Ft=`#version 300 es
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
`,Pt=`#version 300 es\r
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
`,Dt=`#version 300 es\r
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
`,Mt=`#version 300 es\r
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
`,zt=`#version 300 es\r
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
`,It=`#version 300 es\r
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
`,Et=`#version 300 es\r
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
`,Gt=`#version 300 es\r
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
}`,Ut=`#version 300 es\r
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
}`,Nt=`#version 300 es\r
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
}`,Ot=`#version 300 es\r
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
}`,Wt=`#version 300 es\r
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
}`,Lt=`#version 300 es\r
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
}`,_t=`#version 300 es\r
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
}`,oe=[{id:"neon",name:"Neon Isoclines",description:"Electric contour bands driven by seeded radial harmonics.",fragment:pt,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"components",label:"Components",uniform:"uComponents",type:"int",value:64,min:1,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:10}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:8,min:1,max:64,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:10}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.25,min:.01,max:.75,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.05}},{id:"noiseAmount",label:"Noise Amount",uniform:"uNoiseAmount",type:"float",value:2.5,min:0,max:5,step:.05,key:{inc:"r",dec:"f",step:.1,shiftStep:.25}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tanh-terrain",name:"Tanh Terrain Isoclines",description:"Tanh warped contours with bubbling noise and topo glow.",fragment:dt,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:2.1,min:.1,max:6,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"octaves",label:"Octaves",uniform:"uOctaves",type:"int",value:4,min:1,max:12,step:1,key:{inc:"3",dec:"4",step:1}},{id:"lacunarity",label:"Lacunarity",uniform:"uLacunarity",type:"float",value:1.4,min:1.01,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"gain",label:"Gain",uniform:"uGain",type:"float",value:.5,min:.01,max:.99,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:16,min:1,max:96,step:1,key:{inc:"q",dec:"a",step:4,shiftStep:12}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.2,min:.02,max:.75,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"bubbleAmp",label:"Bubble Amp",uniform:"uBubbleAmp",type:"float",value:.26,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.08}},{id:"bubbleFreq",label:"Bubble Freq",uniform:"uBubbleFreq",type:"float",value:2,min:0,max:6,step:.05,key:{inc:"r",dec:"f",step:.25,shiftStep:.75}},{id:"bubbleDetail",label:"Bubble Detail",uniform:"uBubbleDetail",type:"float",value:1.2,min:.1,max:3,step:.05,key:{inc:"t",dec:"g",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tunnel",name:"Brownian Loop Tunnel",description:"Looped tunnel with Brownian noise, fog, and hue spin.",fragment:ht,resolutionUniform:"iResolution",timeUniform:"iTime",timeMode:"looped",loopDuration:18,loopUniform:"uLoopDuration",params:[{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:4,min:0,max:10,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"noiseScale",label:"Noise Scale",uniform:"uNoiseScale",type:"float",value:1.9,min:.1,max:4,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.5,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"fogDensity",label:"Fog Density",uniform:"uFogDensity",type:"float",value:2,min:.1,max:6,step:.05,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"baseRed",label:"Base Red",uniform:"uBaseColor",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:0},{id:"baseGreen",label:"Base Green",uniform:"uBaseColor",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:1},{id:"baseBlue",label:"Base Blue",uniform:"uBaseColor",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:2}]},{id:"prismatic-fold",name:"Prismatic Fold Raymarch",description:"Rotating folded planes with prismatic glow and controllable depth.",fragment:xt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:24,step:1,key:{inc:"1",dec:"2",step:1,shiftStep:4}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.2,min:-1.5,max:1.5,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.5,min:.1,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:5,min:1.5,max:10,step:.1,key:{inc:"7",dec:"8",step:.2,shiftStep:.6}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"cameraDistance",label:"Camera Distance",uniform:"uCameraDistance",type:"float",value:50,min:10,max:120,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:5}},{id:"cameraSpin",label:"Camera Spin",uniform:"uCameraSpin",type:"float",value:1,min:-3,max:3,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.4}},{id:"colorMix",label:"Color Mix",uniform:"uColorMix",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"alphaGain",label:"Alpha Gain",uniform:"uAlphaGain",type:"float",value:1,min:.3,max:2,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"[",dec:"]",step:.05},component:2}]},{id:"koch",name:"Koch Snowflake",description:"Iterative snowflake edges with neon glow mixing.",fragment:vt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.8,min:.1,max:2,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:.2,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:2,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.3,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:2}]},{id:"quasi",name:"Quasi Snowflake",description:"Quasicrystal warp with a drifting snowflake outline.",fragment:yt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:6,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:1.1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.8,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.02,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.03,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.02},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.05,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.02},component:2}]},{id:"watercolor-drips",name:"Watercolor Drips",description:"Textured paper painted with wobbly closed-loop watercolor strokes built from vibrating harmonic curves.",fragment:Et,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:4,step:.02},{id:"loopCount",label:"Loop Count",uniform:"uLoopCount",type:"int",value:12,min:1,max:24,step:1},{id:"cycleMin",label:"Cycle Min",uniform:"uCycleMin",type:"float",value:14,min:2,max:40,step:.25},{id:"cycleMax",label:"Cycle Max",uniform:"uCycleMax",type:"float",value:26,min:4,max:60,step:.25},{id:"radiusMin",label:"Radius Min",uniform:"uRadiusMin",type:"float",value:.06,min:.01,max:.4,step:.005},{id:"radiusMax",label:"Radius Max",uniform:"uRadiusMax",type:"float",value:.18,min:.02,max:.6,step:.005},{id:"centerDrift",label:"Center Drift",uniform:"uCenterDrift",type:"float",value:.03,min:0,max:.25,step:.005},{id:"centerDriftSpeed",label:"Center Drift Speed",uniform:"uCenterDriftSpeed",type:"float",value:.4,min:0,max:4,step:.05},{id:"harmonicCount",label:"Harmonic Count",uniform:"uHarmonicCount",type:"int",value:6,min:0,max:8,step:1},{id:"wobbleAmount",label:"Wobble Amount",uniform:"uWobbleAmount",type:"float",value:.22,min:0,max:.8,step:.01},{id:"wobbleFalloff",label:"Wobble Falloff",uniform:"uWobbleFalloff",type:"float",value:1.1,min:.1,max:3,step:.05},{id:"morphSpeed",label:"Morph Speed",uniform:"uMorphSpeed",type:"float",value:.5,min:-4,max:4,step:.05},{id:"vibrationAmount",label:"Vibration Amount",uniform:"uVibrationAmount",type:"float",value:.035,min:0,max:.3,step:.005},{id:"vibrationFreq",label:"Vibration Freq",uniform:"uVibrationFreq",type:"int",value:14,min:1,max:48,step:1},{id:"vibrationSpeed",label:"Vibration Speed",uniform:"uVibrationSpeed",type:"float",value:1.6,min:-8,max:8,step:.05},{id:"strokeWidthMin",label:"Stroke Width Min",uniform:"uStrokeWidthMin",type:"float",value:.08,min:.01,max:.8,step:.01},{id:"strokeWidthMax",label:"Stroke Width Max",uniform:"uStrokeWidthMax",type:"float",value:.22,min:.02,max:1.2,step:.01},{id:"widthMod",label:"Width Modulation",uniform:"uWidthMod",type:"float",value:.4,min:0,max:1,step:.02},{id:"inkTexture",label:"Ink Texture",uniform:"uInkTexture",type:"float",value:.55,min:0,max:1,step:.02},{id:"drawSoftness",label:"Draw Softness",uniform:"uDrawSoftness",type:"float",value:.16,min:.02,max:.55,step:.01},{id:"bleedStrength",label:"Bleed Strength",uniform:"uBleedStrength",type:"float",value:1.1,min:0,max:3,step:.02},{id:"bleedSpread",label:"Bleed Spread",uniform:"uBleedSpread",type:"float",value:3.2,min:.5,max:8,step:.1},{id:"soakStrength",label:"Soak Strength",uniform:"uSoakStrength",type:"float",value:.5,min:0,max:2,step:.02},{id:"dryMix",label:"Dry Mix",uniform:"uDryMix",type:"float",value:1.1,min:0,max:3,step:.02},{id:"wetMix",label:"Wet Mix",uniform:"uWetMix",type:"float",value:1.4,min:0,max:3,step:.02},{id:"absorbStrength",label:"Absorb Strength",uniform:"uAbsorbStrength",type:"float",value:1.5,min:0,max:3,step:.02},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"saturationBoost",label:"Saturation Boost",uniform:"uSaturationBoost",type:"float",value:0,min:-1,max:1,step:.01},{id:"valueBoost",label:"Value Boost",uniform:"uValueBoost",type:"float",value:0,min:-1,max:1,step:.01},{id:"pastelMix",label:"Pastel Mix",uniform:"uPastelMix",type:"float",value:.14,min:0,max:1,step:.01},{id:"paperGrainScale",label:"Paper Grain Scale",uniform:"uPaperGrainScale",type:"float",value:1,min:.05,max:4,step:.05},{id:"paperGrainAmount",label:"Paper Grain Amount",uniform:"uPaperGrainAmount",type:"float",value:.08,min:0,max:.5,step:.005},{id:"paperPulpAmount",label:"Paper Pulp Amount",uniform:"uPaperPulpAmount",type:"float",value:.05,min:0,max:.4,step:.005},{id:"paperRingDensity",label:"Paper Ring Density",uniform:"uPaperRingDensity",type:"float",value:22,min:2,max:80,step:.5},{id:"paperRingWobble",label:"Paper Ring Wobble",uniform:"uPaperRingWobble",type:"float",value:.12,min:0,max:.6,step:.01},{id:"paperRingAmount",label:"Paper Ring Amount",uniform:"uPaperRingAmount",type:"float",value:1,min:0,max:3,step:.02},{id:"paperFleckAmount",label:"Paper Fleck Amount",uniform:"uPaperFleckAmount",type:"float",value:.018,min:0,max:.25,step:.002},{id:"paperBlotchAmount",label:"Paper Blotch Amount",uniform:"uPaperBlotchAmount",type:"float",value:.06,min:0,max:.4,step:.005},{id:"vignetteStrength",label:"Vignette Strength",uniform:"uVignetteStrength",type:"float",value:.07,min:0,max:1,step:.01}]},{id:"tileable-water-plus",name:"Tileable Water Plus",description:"Tileable water ripples with tunable speed, scale, and tint.",fragment:Tt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"tileScale",label:"Tile Scale",uniform:"uTileScale",type:"float",value:1,min:.5,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.2,max:2.5,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.2,min:.3,max:2.5,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"waveShift",label:"Wave Shift",uniform:"uWaveShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"tintRed",label:"Tint Red",uniform:"uTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02},component:0},{id:"tintGreen",label:"Tint Green",uniform:"uTint",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02},component:1},{id:"tintBlue",label:"Tint Blue",uniform:"uTint",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:2},{id:"swirlStrength",label:"Swirl Strength",uniform:"uSwirlStrength",type:"float",value:1.2,min:-6,max:6,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"swirlGridScale",label:"Swirl Grid Scale",uniform:"uSwirlGridScale",type:"float",value:3,min:.5,max:12,step:.1,key:{inc:"y",dec:"h",step:.1,shiftStep:.5}},{id:"swirlRadius",label:"Swirl Radius",uniform:"uSwirlRadius",type:"float",value:.55,min:.05,max:1.5,step:.01},{id:"swirlWobble",label:"Swirl Wobble",uniform:"uSwirlWobble",type:"float",value:.3,min:0,max:1.5,step:.02},{id:"swirlWobbleSpeed",label:"Swirl Wobble Speed",uniform:"uSwirlWobbleSpeed",type:"float",value:.6,min:0,max:4,step:.02},{id:"swirlSpin",label:"Swirl Spin",uniform:"uSwirlSpin",type:"float",value:.5,min:-4,max:4,step:.02},{id:"swirlDesync",label:"Swirl Desync",uniform:"uSwirlDesync",type:"float",value:1,min:0,max:3,step:.05},{id:"swirlPulse",label:"Swirl Pulse",uniform:"uSwirlPulse",type:"float",value:.35,min:0,max:1,step:.02},{id:"swirlColorTwist",label:"Swirl Color Twist",uniform:"uSwirlColorTwist",type:"float",value:.4,min:-2,max:2,step:.02},{id:"rainbowStrength",label:"Rainbow Strength",uniform:"uRainbowStrength",type:"float",value:.45,min:0,max:1,step:.02},{id:"rainbowScale",label:"Rainbow Scale",uniform:"uRainbowScale",type:"float",value:1.4,min:.1,max:6,step:.05},{id:"rainbowSpeed",label:"Rainbow Speed",uniform:"uRainbowSpeed",type:"float",value:.5,min:-4,max:4,step:.05},{id:"rainbowContrast",label:"Rainbow Contrast",uniform:"uRainbowContrast",type:"float",value:1.2,min:.1,max:4,step:.05}]},{id:"seascape",name:"Seascape Plus",description:"Raymarched ocean with tunable swell and camera drift.",fragment:wt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Freq",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:1,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:0,min:-1,max:1,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:0,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1.1,min:.6,max:1.6,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"waterBrightness",label:"Water Bright",uniform:"uWaterBrightness",type:"float",value:.6,min:.2,max:1.2,step:.02,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"waterRed",label:"Water Red",uniform:"uWaterTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02},component:0},{id:"waterGreen",label:"Water Green",uniform:"uWaterTint",type:"float",value:.09,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02},component:1},{id:"waterBlue",label:"Water Blue",uniform:"uWaterTint",type:"float",value:.18,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.02},component:2}]},{id:"acidscape",name:"Acidscape",description:"A raymarched ocean skinned with flowing oil diffraction, spectral wave crests, and racing plasma veins.",fragment:Ct,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Frequency",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Camera Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Camera Distance",uniform:"uCamDistance",type:"float",value:5,min:1,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Camera Yaw",uniform:"uCamYaw",type:"float",value:0,min:-1,max:1,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Camera Pitch",uniform:"uCamPitch",type:"float",value:0,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1,min:.4,max:1.8,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"waterBrightness",label:"Film Brightness",uniform:"uWaterBrightness",type:"float",value:.75,min:.1,max:1.8,step:.02,key:{inc:"u",dec:"j",step:.03,shiftStep:.1}},{id:"filmThickness",label:"Film Thickness",uniform:"uFilmThickness",type:"float",value:1,min:.15,max:3.5,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"diffraction",label:"Diffraction Bands",uniform:"uDiffraction",type:"float",value:1.35,min:.2,max:5,step:.03,key:{inc:"o",dec:"l",step:.08,shiftStep:.3}},{id:"oilScale",label:"Oil Scale",uniform:"uOilScale",type:"float",value:.35,min:.05,max:1.5,step:.01,key:{inc:"p",dec:";",step:.03,shiftStep:.1}},{id:"oilWarp",label:"Oil Warp",uniform:"uOilWarp",type:"float",value:1,min:0,max:3,step:.02},{id:"plasmaDensity",label:"Plasma Density",uniform:"uPlasmaDensity",type:"float",value:1,min:.2,max:4,step:.03},{id:"dischargeSpeed",label:"Discharge Speed",uniform:"uDischargeSpeed",type:"float",value:1,min:-4,max:4,step:.03},{id:"spectralContrast",label:"Spectral Contrast",uniform:"uSpectralContrast",type:"float",value:1.1,min:.2,max:3,step:.02},{id:"plasmaGlow",label:"Plasma Glow",uniform:"uPlasmaGlow",type:"float",value:1.2,min:.2,max:3,step:.02},{id:"acidSky",label:"Acid Sky",uniform:"uAcidSky",type:"float",value:.35,min:0,max:1,step:.01},{id:"hueShift",label:"Spectral Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"sunset-plus",name:"Sunset Plus",description:"Volumetric sunset clouds with tunable turbulence and hue drift.",fragment:kt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}}]},{id:"sunset-orbit",name:"Sunset Orbit",description:"Volumetric sunset with horizon wrapped around an animated periodic orbit.",fragment:Rt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"warpAmp",label:"Warp Amplitude",uniform:"uWarpAmp",type:"float",value:.4,min:0,max:2,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.1}},{id:"warpFreq",label:"Warp Frequency",uniform:"uWarpFreq",type:"float",value:1.5,min:.1,max:8,step:.1,key:{inc:"y",dec:"h",step:.1,shiftStep:.5}},{id:"warpSpeed",label:"Warp Speed",uniform:"uWarpSpeed",type:"float",value:.5,min:0,max:3,step:.05,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"warpHarmonics",label:"Warp Harmonics",uniform:"uWarpHarmonics",type:"float",value:3,min:1,max:5,step:1,key:{inc:"i",dec:"k",step:1}},{id:"orbitRadius",label:"Orbit Radius",uniform:"uOrbitRadius",type:"float",value:.3,min:0,max:2,step:.02,key:{inc:"o",dec:"l",step:.02,shiftStep:.1}},{id:"orbitSpeed",label:"Orbit Speed",uniform:"uOrbitSpeed",type:"float",value:.3,min:0,max:3,step:.05,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"orbitEcc",label:"Orbit Eccentricity",uniform:"uOrbitEcc",type:"float",value:.6,min:0,max:2,step:.05},{id:"tiltAngle",label:"Tilt Angle",uniform:"uTiltAngle",type:"float",value:0,min:-1.57,max:1.57,step:.02},{id:"cloudDensity",label:"Cloud Density",uniform:"uCloudDensity",type:"float",value:1,min:.1,max:3,step:.05},{id:"fogFalloff",label:"Fog Falloff",uniform:"uFogFalloff",type:"float",value:.01,min:0,max:.1,step:.002},{id:"colorSep",label:"Color Separation",uniform:"uColorSep",type:"float",value:.15,min:0,max:1,step:.01}]},{id:"diff-chromatic",name:"Chromatic Flow",description:"Two-channel diffusion with hue-as-angle and drifting color pulses.",fragment:bt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.998,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"rotate",label:"Rotate",uniform:"uRotate",type:"float",value:.02,min:-.2,max:.2,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.03}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.35,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"w",dec:"s",step:.01,shiftStep:.03}},{id:"valueGain",label:"Value Gain",uniform:"uValueGain",type:"float",value:2.2,min:.2,max:6,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"zippy-zaps",name:"Zippy Zaps Plus",description:"Tanh-warped chromatic flow with twistable energy and glow.",fragment:At,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.2,min:.05,max:.5,step:.005,key:{inc:"1",dec:"2",step:.01,shiftStep:.03}},{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1,min:.2,max:2,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:1,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"iterLimit",label:"Iter Limit",uniform:"uIterLimit",type:"float",value:19,min:4,max:19,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:3}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.4,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"offsetX",label:"Offset X",uniform:"uOffsetX",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"offsetY",label:"Offset Y",uniform:"uOffsetY",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}}]},{id:"plasma-menagerie",name:"Plasma Menagerie",description:"A charged kaleidoscope where luminous amoeba-glyphs orbit, blink, and trail electric tendrils through living plasma.",fragment:Gt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"plasmaScale",label:"Plasma Scale",uniform:"uPlasmaScale",type:"float",value:5.5,min:1,max:14,step:.1,key:{inc:"5",dec:"6",step:.2,shiftStep:.8}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:.85,min:0,max:2.5,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"fold",label:"Fold Symmetry",uniform:"uFold",type:"int",value:7,min:1,max:16,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:3}},{id:"morph",label:"Creature Morph",uniform:"uMorph",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"shapeCount",label:"Creature Count",uniform:"uShapeCount",type:"int",value:6,min:1,max:8,step:1,key:{inc:"e",dec:"d",step:1,shiftStep:2}},{id:"orbit",label:"Orbit Reach",uniform:"uOrbit",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"filament",label:"Filament Density",uniform:"uFilament",type:"float",value:.65,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.03,shiftStep:.1}},{id:"glow",label:"Discharge Glow",uniform:"uGlow",type:"float",value:1.1,min:.2,max:2.5,step:.02,key:{inc:"y",dec:"h",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:-3,max:3,step:.02,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-coral-choir",name:"Plasma Coral Choir",description:"Branching electric colonies grow antlers, pulse charged buds, and sing through a smoky plasma reef.",fragment:Ut,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.4,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"branches",label:"Branch Count",uniform:"uBranches",type:"int",value:7,min:3,max:16,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:3}},{id:"growth",label:"Colony Growth",uniform:"uGrowth",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"curl",label:"Branch Curl",uniform:"uCurl",type:"float",value:1.15,min:0,max:4,step:.03,key:{inc:"q",dec:"a",step:.08,shiftStep:.3}},{id:"pulse",label:"Choir Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:4,step:.03,key:{inc:"w",dec:"s",step:.08,shiftStep:.3}},{id:"arcDensity",label:"Arc Density",uniform:"uArcDensity",type:"float",value:1,min:.2,max:3,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"glow",label:"Bioelectric Glow",uniform:"uGlow",type:"float",value:1.2,min:.2,max:3,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"hue",label:"Reef Hue",uniform:"uHue",type:"float",value:.05,min:-1,max:1,step:.01,key:{inc:"t",dec:"g",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-crystal-reactor",name:"Plasma Crystal Reactor",description:"A polygonal reactor fractures its surroundings into restless crystal cells and ejects prismatic shock rings.",fragment:Nt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cellScale",label:"Crystal Scale",uniform:"uCellScale",type:"float",value:5,min:1.5,max:12,step:.1,key:{inc:"5",dec:"6",step:.2,shiftStep:.8}},{id:"jitter",label:"Lattice Jitter",uniform:"uJitter",type:"float",value:.75,min:0,max:1.5,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"fracture",label:"Fracture Sharpness",uniform:"uFracture",type:"float",value:.65,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"coreRadius",label:"Core Radius",uniform:"uCoreRadius",type:"float",value:.2,min:.05,max:.55,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"spin",label:"Reactor Spin",uniform:"uSpin",type:"float",value:1,min:-4,max:4,step:.03,key:{inc:"e",dec:"d",step:.08,shiftStep:.3}},{id:"shockwave",label:"Shockwave Density",uniform:"uShockwave",type:"float",value:1,min:.2,max:3,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"glow",label:"Crystal Glow",uniform:"uGlow",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"hue",label:"Prism Hue",uniform:"uHue",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"y",dec:"h",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-ferrofluid-oracle",name:"Plasma Ferrofluid Oracle",description:"Wandering magnetic eyes pull liquid metal into impossible bodies while luminous field lines reveal their predictions.",fragment:Ot,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"poleCount",label:"Magnetic Poles",uniform:"uPoleCount",type:"int",value:5,min:2,max:7,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:2}},{id:"fieldLines",label:"Field Line Density",uniform:"uFieldLines",type:"float",value:2,min:.3,max:6,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.4}},{id:"viscosity",label:"Fluid Viscosity",uniform:"uViscosity",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"blobSize",label:"Oracle Size",uniform:"uBlobSize",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"poleOrbit",label:"Pole Orbit",uniform:"uPoleOrbit",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"eyeStrength",label:"Oracle Eyes",uniform:"uEyeStrength",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"glow",label:"Magnetic Glow",uniform:"uGlow",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"hue",label:"Oil Hue",uniform:"uHue",type:"float",value:.05,min:-1,max:1,step:.01,key:{inc:"y",dec:"h",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-signal-cathedral",name:"Plasma Signal Cathedral",description:"An endless procession of alien plasma runes transmits through bending arches and a radiant central portal.",fragment:Wt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"columns",label:"Rune Columns",uniform:"uColumns",type:"int",value:4,min:2,max:10,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:2}},{id:"glyphComplexity",label:"Glyph Complexity",uniform:"uGlyphComplexity",type:"int",value:7,min:3,max:16,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:3}},{id:"procession",label:"Rune Procession",uniform:"uProcession",type:"float",value:1,min:-3,max:3,step:.03,key:{inc:"q",dec:"a",step:.08,shiftStep:.3}},{id:"portalBend",label:"Portal Bend",uniform:"uPortalBend",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"signalNoise",label:"Signal Corruption",uniform:"uSignalNoise",type:"float",value:.65,min:0,max:2,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"scanRate",label:"Scan Rate",uniform:"uScanRate",type:"float",value:1,min:-4,max:4,step:.03,key:{inc:"r",dec:"f",step:.08,shiftStep:.3}},{id:"glow",label:"Sanctum Glow",uniform:"uGlow",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"hue",label:"Signal Hue",uniform:"uHue",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"y",dec:"h",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasmic-deep-sea-medusas",name:"Plasmic Deep-Sea Medusas",description:"Translucent abyssal medusas breathe, ascend, and trail charged tendrils through caustic haze and marine snow.",fragment:Lt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"medusaCount",label:"Medusa Count",uniform:"uMedusaCount",type:"int",value:6,min:1,max:8,step:1,key:{inc:"5",dec:"6",step:1,shiftStep:2}},{id:"bellRibs",label:"Bell Ribs",uniform:"uBellRibs",type:"int",value:9,min:3,max:18,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:3}},{id:"tentacles",label:"Tentacles",uniform:"uTentacles",type:"int",value:7,min:3,max:10,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:2}},{id:"bellSize",label:"Bell Size",uniform:"uBellSize",type:"float",value:1,min:.45,max:2,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"pulse",label:"Breathing Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:4,step:.03,key:{inc:"e",dec:"d",step:.08,shiftStep:.3}},{id:"riseSpeed",label:"Ascent Speed",uniform:"uRiseSpeed",type:"float",value:1,min:-2,max:4,step:.03,key:{inc:"r",dec:"f",step:.08,shiftStep:.3}},{id:"tentacleLength",label:"Tentacle Length",uniform:"uTentacleLength",type:"float",value:.82,min:.25,max:1.6,step:.02,key:{inc:"t",dec:"g",step:.04,shiftStep:.15}},{id:"tentacleSway",label:"Tentacle Sway",uniform:"uTentacleSway",type:"float",value:1,min:0,max:3,step:.03,key:{inc:"y",dec:"h",step:.08,shiftStep:.3}},{id:"transparency",label:"Bell Transparency",uniform:"uTransparency",type:"float",value:1,min:0,max:2.5,step:.02,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"glow",label:"Bioluminescence",uniform:"uGlow",type:"float",value:1.2,min:.2,max:3,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"marineSnow",label:"Marine Snow",uniform:"uMarineSnow",type:"float",value:.75,min:0,max:2.5,step:.02,key:{inc:"o",dec:"l",step:.05,shiftStep:.2}},{id:"hue",label:"Abyssal Hue",uniform:"uHue",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"p",dec:";",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"plasma-oil-diffraction",name:"Plasma Oil Diffraction",description:"Charged filaments ignite along flowing oil membranes while thin-film diffraction splits every ripple into spectral bands.",fragment:_t,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.35,max:2.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"filmThickness",label:"Film Thickness",uniform:"uFilmThickness",type:"float",value:1,min:.15,max:3.5,step:.02,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"diffraction",label:"Diffraction Bands",uniform:"uDiffraction",type:"float",value:1.35,min:.2,max:5,step:.03,key:{inc:"7",dec:"8",step:.08,shiftStep:.3}},{id:"fluidWarp",label:"Fluid Warp",uniform:"uFluidWarp",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"dropletScale",label:"Droplet Scale",uniform:"uDropletScale",type:"float",value:3.8,min:1,max:10,step:.1,key:{inc:"w",dec:"s",step:.2,shiftStep:.8}},{id:"plasmaDensity",label:"Plasma Density",uniform:"uPlasmaDensity",type:"float",value:1,min:.2,max:4,step:.03,key:{inc:"e",dec:"d",step:.08,shiftStep:.3}},{id:"dischargeSpeed",label:"Discharge Speed",uniform:"uDischargeSpeed",type:"float",value:1,min:-4,max:4,step:.03,key:{inc:"r",dec:"f",step:.08,shiftStep:.3}},{id:"spectralContrast",label:"Spectral Contrast",uniform:"uSpectralContrast",type:"float",value:1.15,min:.2,max:3,step:.02,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"glow",label:"Plasma Glow",uniform:"uGlow",type:"float",value:1.2,min:.2,max:3,step:.02,key:{inc:"y",dec:"h",step:.05,shiftStep:.2}},{id:"hueShift",label:"Spectral Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01,key:{inc:"u",dec:"j",step:.03,shiftStep:.1}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"thunderstorm",name:"Thunderstorm",description:"Top-down storm clouds with stepped-leader lightning, return strokes, restrikes, and branching plasma channels.",fragment:Bt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.2,max:3,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"cloudScale",label:"Cloud Scale",uniform:"uCloudScale",type:"float",value:3,min:.5,max:10,step:.1,key:{inc:"5",dec:"6",step:.2,shiftStep:1}},{id:"cloudSpeed",label:"Cloud Speed",uniform:"uCloudSpeed",type:"float",value:1,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"cloudDensity",label:"Cloud Density",uniform:"uCloudDensity",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.02,shiftStep:.1}},{id:"cloudDetail",label:"Cloud Detail",uniform:"uCloudDetail",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.1}},{id:"boltCount",label:"Bolt Count",uniform:"uBoltCount",type:"int",value:6,min:1,max:12,step:1,key:{inc:"e",dec:"d",step:1,shiftStep:2}},{id:"boltLengthMin",label:"Bolt Length Min",uniform:"uBoltLengthMin",type:"float",value:.1,min:.02,max:.4,step:.01,key:{inc:"r",dec:"f",step:.01,shiftStep:.05}},{id:"boltLengthMax",label:"Bolt Length Max",uniform:"uBoltLengthMax",type:"float",value:.35,min:.1,max:.8,step:.01,key:{inc:"t",dec:"g",step:.01,shiftStep:.05}},{id:"boltWidth",label:"Bolt Width",uniform:"uBoltWidth",type:"float",value:8e-4,min:1e-4,max:.005,step:1e-4,key:{inc:"y",dec:"h",step:2e-4,shiftStep:.001}},{id:"boltWiggle",label:"Bolt Wiggle",uniform:"uBoltWiggle",type:"float",value:.04,min:0,max:.2,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"boltNoiseScale",label:"Bolt Noise Scale",uniform:"uBoltNoiseScale",type:"float",value:15,min:3,max:60,step:.5,key:{inc:"i",dec:"k",step:1,shiftStep:3}},{id:"boltNoiseSpeed",label:"Bolt Noise Speed",uniform:"uBoltNoiseSpeed",type:"float",value:2,min:0,max:8,step:.05,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"boltBranching",label:"Bolt Branching",uniform:"uBoltBranching",type:"float",value:.5,min:0,max:1,step:.02,key:{inc:"p",dec:";",step:.02,shiftStep:.1}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:.3,min:.05,max:1.5,step:.02},{id:"flickerSpeed",label:"Flicker Speed",uniform:"uFlickerSpeed",type:"float",value:2,min:0,max:8,step:.1},{id:"cloudIllumination",label:"Cloud Illumination",uniform:"uCloudIllumination",type:"float",value:1,min:0,max:3,step:.05},{id:"noiseOctaves",label:"Noise Octaves",uniform:"uNoiseOctaves",type:"int",value:5,min:1,max:8,step:1},{id:"cloudRed",label:"Cloud Red",uniform:"uCloudColor",type:"float",value:.12,min:0,max:1,step:.01,component:0},{id:"cloudGreen",label:"Cloud Green",uniform:"uCloudColor",type:"float",value:.12,min:0,max:1,step:.01,component:1},{id:"cloudBlue",label:"Cloud Blue",uniform:"uCloudColor",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"lightningRed",label:"Lightning Red",uniform:"uLightningColor",type:"float",value:.7,min:0,max:1,step:.01,component:0},{id:"lightningGreen",label:"Lightning Green",uniform:"uLightningColor",type:"float",value:.7,min:0,max:1,step:.01,component:1},{id:"lightningBlue",label:"Lightning Blue",uniform:"uLightningColor",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"mandelbulb-inside-plus",name:"Inside the Mandelbulb Plus",description:"Raymarched mandelbulb interior with tunable optics and palette glow.",fragment:Ft,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"power",label:"Power",uniform:"uPower",type:"float",value:8,min:2,max:12,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"bulbSpin",label:"Bulb Spin",uniform:"uBulbSpin",type:"float",value:.2,min:0,max:1.5,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"loops",label:"Loops",uniform:"uLoops",type:"int",value:2,min:1,max:6,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:1}},{id:"rayMarches",label:"Ray Marches",uniform:"uRayMarches",type:"int",value:60,min:20,max:96,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:5}},{id:"maxRayLength",label:"Max Ray Length",uniform:"uMaxRayLength",type:"float",value:20,min:5,max:40,step:.5,key:{inc:"w",dec:"s",step:.5,shiftStep:2}},{id:"tolerance",label:"Tolerance",uniform:"uTolerance",type:"float",value:1e-4,min:1e-5,max:.001,step:1e-5,key:{inc:"e",dec:"d",step:2e-5,shiftStep:1e-4}},{id:"normOffset",label:"Normal Offset",uniform:"uNormOffset",type:"float",value:.005,min:.001,max:.02,step:5e-4,key:{inc:"r",dec:"f",step:5e-4,shiftStep:.002}},{id:"bounces",label:"Bounces",uniform:"uBounces",type:"int",value:5,min:1,max:5,step:1,key:{inc:"t",dec:"g",step:1,shiftStep:1}},{id:"initStep",label:"Init Step",uniform:"uInitStep",type:"float",value:.1,min:.01,max:.3,step:.01,key:{inc:"y",dec:"h",step:.01,shiftStep:.05}},{id:"rotSpeedX",label:"Rot Speed X",uniform:"uRotSpeedX",type:"float",value:.2,min:-1,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.1}},{id:"rotSpeedY",label:"Rot Speed Y",uniform:"uRotSpeedY",type:"float",value:.3,min:-1,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:2,max:10,step:.1,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:2,min:.5,max:5,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"fov",label:"FOV",uniform:"uFov",type:"float",value:.523,min:.3,max:1.2,step:.01},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"glowBoost",label:"Glow Boost",uniform:"uGlowBoost",type:"float",value:1.2,min:0,max:4,step:.05},{id:"glowFalloff",label:"Glow Falloff",uniform:"uGlowFalloff",type:"float",value:.06,min:.01,max:.2,step:.005},{id:"diffuseBoost",label:"Diffuse Boost",uniform:"uDiffuseBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"matTransmit",label:"Mat Transmit",uniform:"uMatTransmit",type:"float",value:.8,min:0,max:1,step:.01},{id:"matReflect",label:"Mat Reflect",uniform:"uMatReflect",type:"float",value:.5,min:0,max:1,step:.01},{id:"refractIndex",label:"Refract Index",uniform:"uRefractIndex",type:"float",value:1.05,min:1,max:2,step:.01},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"glowHueOffset",label:"Glow Hue Offset",uniform:"uGlowHueOffset",type:"float",value:.065,min:-.5,max:.5,step:.005},{id:"nebulaMix",label:"Nebula Mix",uniform:"uNebulaMix",type:"float",value:0,min:0,max:1,step:.01},{id:"nebulaHueShift",label:"Nebula Hue",uniform:"uNebulaHueShift",type:"float",value:.12,min:-1,max:1,step:.01},{id:"nebulaSat",label:"Nebula Sat",uniform:"uNebulaSat",type:"float",value:.9,min:0,max:1,step:.01},{id:"nebulaVal",label:"Nebula Val",uniform:"uNebulaVal",type:"float",value:1.6,min:.2,max:3,step:.02},{id:"nebulaGlowHue",label:"Nebula Glow Hue",uniform:"uNebulaGlowHue",type:"float",value:.35,min:-1,max:1,step:.01},{id:"nebulaGlowBoost",label:"Nebula Glow",uniform:"uNebulaGlowBoost",type:"float",value:1.6,min:0,max:4,step:.05},{id:"skySat",label:"Sky Saturation",uniform:"uSkySat",type:"float",value:.86,min:0,max:1,step:.01},{id:"skyVal",label:"Sky Value",uniform:"uSkyVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"glowSat",label:"Glow Saturation",uniform:"uGlowSat",type:"float",value:.8,min:0,max:1,step:.01},{id:"glowVal",label:"Glow Value",uniform:"uGlowVal",type:"float",value:6,min:.5,max:8,step:.1},{id:"diffuseSat",label:"Diffuse Saturation",uniform:"uDiffuseSat",type:"float",value:.85,min:0,max:1,step:.01},{id:"diffuseVal",label:"Diffuse Value",uniform:"uDiffuseVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"beerRed",label:"Beer Red",uniform:"uBeerColor",type:"float",value:.02,min:0,max:.2,step:.005,component:0},{id:"beerGreen",label:"Beer Green",uniform:"uBeerColor",type:"float",value:.08,min:0,max:.2,step:.005,component:1},{id:"beerBlue",label:"Beer Blue",uniform:"uBeerColor",type:"float",value:.12,min:0,max:.2,step:.005,component:2},{id:"lightX",label:"Light X",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:0},{id:"lightY",label:"Light Y",uniform:"uLightPos",type:"float",value:10,min:-5,max:25,step:.5,component:1},{id:"lightZ",label:"Light Z",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:2}]},{id:"auroras-plus",name:"Auroras Plus",description:"Volumetric auroras with tunable trails, palette waves, and sky glare.",fragment:Pt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"auroraSpeed",label:"Aurora Speed",uniform:"uAuroraSpeed",type:"float",value:.06,min:0,max:.2,step:.005,key:{inc:"3",dec:"4",step:.005,shiftStep:.02}},{id:"auroraScale",label:"Aurora Scale",uniform:"uAuroraScale",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"auroraWarp",label:"Aurora Warp",uniform:"uAuroraWarp",type:"float",value:.35,min:0,max:1,step:.02,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"auroraSteps",label:"Aurora Steps",uniform:"uAuroraSteps",type:"int",value:50,min:8,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"auroraBase",label:"Aurora Base",uniform:"uAuroraBase",type:"float",value:.8,min:.2,max:1.6,step:.02,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"auroraStride",label:"Aurora Stride",uniform:"uAuroraStride",type:"float",value:.002,min:2e-4,max:.01,step:2e-4,key:{inc:"e",dec:"d",step:2e-4,shiftStep:.001}},{id:"auroraCurve",label:"Aurora Curve",uniform:"uAuroraCurve",type:"float",value:1.4,min:.8,max:2.2,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"auroraIntensity",label:"Aurora Intensity",uniform:"uAuroraIntensity",type:"float",value:1.8,min:.2,max:4,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"trailBlend",label:"Trail Blend",uniform:"uTrailBlend",type:"float",value:.5,min:.1,max:.9,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"trailFalloff",label:"Trail Falloff",uniform:"uTrailFalloff",type:"float",value:.065,min:.01,max:.15,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"trailFade",label:"Trail Fade",uniform:"uTrailFade",type:"float",value:2.5,min:.5,max:5,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.4}},{id:"ditherStrength",label:"Dither Strength",uniform:"uDitherStrength",type:"float",value:.006,min:0,max:.02,step:5e-4,key:{inc:"o",dec:"l",step:5e-4,shiftStep:.002}},{id:"horizonFade",label:"Horizon Fade",uniform:"uHorizonFade",type:"float",value:.01,min:.001,max:.05,step:.001,key:{inc:"p",dec:";",step:.001,shiftStep:.005}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:-.1,min:-1,max:1,step:.01},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:.1,min:-1,max:1,step:.01},{id:"camWobble",label:"Cam Wobble",uniform:"uCamWobble",type:"float",value:.2,min:0,max:.6,step:.01},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:6.7,min:4,max:12,step:.1},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:0,min:-1,max:2,step:.05},{id:"skyStrength",label:"Sky Strength",uniform:"uSkyStrength",type:"float",value:.63,min:.1,max:2,step:.02},{id:"starDensity",label:"Star Density",uniform:"uStarDensity",type:"float",value:5e-4,min:0,max:.005,step:1e-4},{id:"starIntensity",label:"Star Intensity",uniform:"uStarIntensity",type:"float",value:.8,min:0,max:2,step:.05},{id:"reflectionStrength",label:"Reflection Strength",uniform:"uReflectionStrength",type:"float",value:.6,min:0,max:1.5,step:.05},{id:"reflectionTint",label:"Reflection Tint",uniform:"uReflectionTint",type:"float",value:1,min:0,max:2,step:.05},{id:"reflectionFog",label:"Reflection Fog",uniform:"uReflectionFog",type:"float",value:2,min:0,max:6,step:.1},{id:"colorBand",label:"Color Band",uniform:"uColorBand",type:"float",value:.043,min:0,max:.2,step:.002},{id:"colorSpeed",label:"Color Speed",uniform:"uColorSpeed",type:"float",value:0,min:-1,max:1,step:.01},{id:"vortexStrength",label:"Vortex Strength",uniform:"uVortexStrength",type:"float",value:.8,min:-4,max:4,step:.02},{id:"vortexGridScale",label:"Vortex Grid Scale",uniform:"uVortexGridScale",type:"float",value:2.5,min:.5,max:10,step:.1},{id:"vortexRadius",label:"Vortex Radius",uniform:"uVortexRadius",type:"float",value:.45,min:.05,max:1.5,step:.01},{id:"vortexWobble",label:"Vortex Wobble",uniform:"uVortexWobble",type:"float",value:.35,min:0,max:1.5,step:.02},{id:"vortexWobbleSpeed",label:"Vortex Wobble Speed",uniform:"uVortexWobbleSpeed",type:"float",value:.5,min:0,max:4,step:.02},{id:"vortexSpin",label:"Vortex Spin",uniform:"uVortexSpin",type:"float",value:.4,min:-4,max:4,step:.02},{id:"vortexDesync",label:"Vortex Desync",uniform:"uVortexDesync",type:"float",value:1,min:0,max:3,step:.05},{id:"vortexDrift",label:"Vortex Drift",uniform:"uVortexDrift",type:"float",value:.2,min:-2,max:2,step:.02},{id:"vortexColorShift",label:"Vortex Color Shift",uniform:"uVortexColorShift",type:"float",value:.5,min:-2,max:2,step:.02},{id:"oilStrength",label:"Oil Film Strength",uniform:"uOilStrength",type:"float",value:.35,min:0,max:1,step:.02},{id:"oilScale",label:"Oil Film Scale",uniform:"uOilScale",type:"float",value:1.6,min:.1,max:8,step:.05},{id:"oilSpeed",label:"Oil Film Speed",uniform:"uOilSpeed",type:"float",value:.6,min:-4,max:4,step:.05},{id:"oilContrast",label:"Oil Film Contrast",uniform:"uOilContrast",type:"float",value:1.4,min:.1,max:4,step:.05},{id:"auroraRedA",label:"Aurora Red A",uniform:"uAuroraColorA",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenA",label:"Aurora Green A",uniform:"uAuroraColorA",type:"float",value:.9,min:0,max:1,step:.01,component:1},{id:"auroraBlueA",label:"Aurora Blue A",uniform:"uAuroraColorA",type:"float",value:.6,min:0,max:1,step:.01,component:2},{id:"auroraRedB",label:"Aurora Red B",uniform:"uAuroraColorB",type:"float",value:.6,min:0,max:1,step:.01,component:0},{id:"auroraGreenB",label:"Aurora Green B",uniform:"uAuroraColorB",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"auroraBlueB",label:"Aurora Blue B",uniform:"uAuroraColorB",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"auroraRedC",label:"Aurora Red C",uniform:"uAuroraColorC",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenC",label:"Aurora Green C",uniform:"uAuroraColorC",type:"float",value:.6,min:0,max:1,step:.01,component:1},{id:"auroraBlueC",label:"Aurora Blue C",uniform:"uAuroraColorC",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedA",label:"BG Red A",uniform:"uBgColorA",type:"float",value:.05,min:0,max:1,step:.01,component:0},{id:"bgGreenA",label:"BG Green A",uniform:"uBgColorA",type:"float",value:.1,min:0,max:1,step:.01,component:1},{id:"bgBlueA",label:"BG Blue A",uniform:"uBgColorA",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedB",label:"BG Red B",uniform:"uBgColorB",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"bgGreenB",label:"BG Green B",uniform:"uBgColorB",type:"float",value:.05,min:0,max:1,step:.01,component:1},{id:"bgBlueB",label:"BG Blue B",uniform:"uBgColorB",type:"float",value:.2,min:0,max:1,step:.01,component:2}]},{id:"diff-edge-flow",name:"Edge Flow Vectors",description:"Diffusive scalar field rendered as glowing edge-flow vectors.",fragment:gt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.996,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"blobAmp",label:"Blob Amp",uniform:"uBlobAmp",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"blobRadius",label:"Blob Radius",uniform:"uBlobRadius",type:"float",value:.07,min:.01,max:.25,step:.005,key:{inc:"q",dec:"a",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.8,min:0,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"flowGain",label:"Flow Gain",uniform:"uFlowGain",type:"float",value:3,min:.2,max:8,step:.1,key:{inc:"e",dec:"d",step:.2,shiftStep:.6}},{id:"flowThreshold",label:"Flow Threshold",uniform:"uFlowThreshold",type:"float",value:.02,min:0,max:.2,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"diff-threshold",name:"Threshold Feedback",description:"Diffusion with nonlinear feedback for digital fungus crackle.",fragment:St,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:192,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.5,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.995,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"threshold",label:"Threshold",uniform:"uThreshold",type:"float",value:.5,min:.1,max:.9,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.06}},{id:"sharpness",label:"Sharpness",uniform:"uSharpness",type:"float",value:18,min:1,max:40,step:.5,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.08,min:0,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.2,min:0,max:1.5,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractal-fold",name:"Fractal Fold Raymarch",description:"Recursive box-folding fractal with prismatic lighting and IQ palette.",fragment:Dt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.5,max:5,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"distort",label:"Distort",uniform:"uDistort",type:"float",value:2.5,min:1.5,max:4,step:.02,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:1,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.15}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.1,min:-.5,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"maxSteps",label:"Max Steps",uniform:"uMaxSteps",type:"float",value:100,min:20,max:200,step:10,key:{inc:"e",dec:"d",step:10,shiftStep:30}}]},{id:"hadamard-disk",name:"Hadamard Disk",description:"Spinning Hadamard matrix mapped onto a disk with animated colour cycling.",fragment:Mt,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"order",label:"Order",uniform:"uOrder",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"rotSpeed",label:"Rotation Speed",uniform:"uRotSpeed",type:"float",value:1,min:-4,max:4,step:.1,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.95,min:.2,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"radialPow",label:"Radial Power",uniform:"uRadialPow",type:"float",value:1,min:.1,max:5,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"spiral",label:"Spiral",uniform:"uSpiral",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"smooth",label:"Smooth",uniform:"uSmooth",type:"float",value:0,min:0,max:1,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"gap",label:"Cell Gap",uniform:"uGap",type:"float",value:0,min:0,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.05}},{id:"fadeStart",label:"Fade Start",uniform:"uFadeStart",type:"float",value:.9,min:.3,max:1,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.05}},{id:"fadeWidth",label:"Fade Width",uniform:"uFadeWidth",type:"float",value:.08,min:.01,max:.5,step:.01,key:{inc:"5",dec:"6",step:.01,shiftStep:.05}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:0,min:0,max:2,step:.05,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:0,min:0,max:2,step:.05,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"bgBright",label:"Background",uniform:"uBgBright",type:"float",value:0,min:0,max:.5,step:.01},{id:"baseR",label:"Base Red",uniform:"uBaseR",type:"float",value:.75,min:0,max:1,step:.01},{id:"baseG",label:"Base Green",uniform:"uBaseG",type:"float",value:.75,min:0,max:1,step:.01},{id:"baseB",label:"Base Blue",uniform:"uBaseB",type:"float",value:.75,min:0,max:1,step:.01},{id:"ampR",label:"Amp Red",uniform:"uAmpR",type:"float",value:-.25,min:-1,max:1,step:.01},{id:"ampG",label:"Amp Green",uniform:"uAmpG",type:"float",value:.25,min:-1,max:1,step:.01},{id:"ampB",label:"Amp Blue",uniform:"uAmpB",type:"float",value:.25,min:-1,max:1,step:.01},{id:"freqR",label:"Freq Red",uniform:"uFreqR",type:"float",value:1,min:0,max:16,step:.5},{id:"freqG",label:"Freq Green",uniform:"uFreqG",type:"float",value:2,min:0,max:16,step:.5},{id:"freqB",label:"Freq Blue",uniform:"uFreqB",type:"float",value:4,min:0,max:16,step:.5},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractalReverie",name:"Fractal Reverie",description:"Continuous fractal color fields with smooth drifting motion and a saturated glowing backdrop.",fragment:zt,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.05,min:0,max:3,step:.01,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"fractalIters",label:"Fractal Iters",uniform:"uFractalIters",type:"int",value:2,min:1,max:12,step:1,key:{inc:"w",dec:"s",step:1}},{id:"foldScale",label:"Fold Scale",uniform:"uFoldScale",type:"float",value:1.18,min:1,max:4,step:.01,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.42,min:0,max:3,step:.01,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"rotSpeed",label:"Rot Speed",uniform:"uRotSpeed",type:"float",value:.03,min:0,max:3,step:.01},{id:"detailLevel",label:"Detail Level",uniform:"uDetailLevel",type:"float",value:1,min:.2,max:5,step:.1},{id:"lightIntensity",label:"Light Intensity",uniform:"uLightIntensity",type:"float",value:1.5,min:0,max:5,step:.05,key:{inc:"u",dec:"j",step:.1,shiftStep:.5}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.05,min:0,max:1,step:.005},{id:"saturation",label:"Saturation",uniform:"uSaturation",type:"float",value:2,min:0,max:3,step:.01,key:{inc:"o",dec:"l",step:.05,shiftStep:.2}},{id:"brightness",label:"Brightness",uniform:"uBrightness",type:"float",value:1.8,min:.1,max:4,step:.05},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.1,min:.5,max:3,step:.01,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"glowIntensity",label:"Glow Intensity",uniform:"uGlowIntensity",type:"float",value:4.5,min:0,max:8,step:.1,key:{inc:"[",dec:"]",step:.2,shiftStep:.5}},{id:"chromaShift",label:"Chroma Shift",uniform:"uChromaShift",type:"float",value:.5,min:0,max:20,step:.1},{id:"smoothBlend",label:"Smooth Blend",uniform:"uSmoothBlend",type:"float",value:.9,min:0,max:1,step:.01},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.68,min:.2,max:3,step:.01},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:.3,min:-3,max:3,step:.05},{id:"camOrbit",label:"Cam Orbit",uniform:"uCamOrbit",type:"float",value:.018,min:0,max:1,step:.005},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"polar-rose",name:"Polar Rose",description:"Animated polar rose curves with C1-continuous Bezier segments and chromatic glow.",fragment:It,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"symmetry",label:"Symmetry",uniform:"uSymmetry",type:"int",value:5,min:1,max:32,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"subdivisions",label:"Subdivisions",uniform:"uSubdivisions",type:"int",value:64,min:16,max:1024,step:16,key:{inc:"w",dec:"s",step:16,shiftStep:64}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.85,min:.05,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.1}},{id:"sinAmp",label:"Sin Amplitude",uniform:"uSinAmp",type:"float",value:.4,min:0,max:2,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.1}},{id:"baseFreq",label:"Base Frequency",uniform:"uBaseFreq",type:"float",value:3,min:.5,max:12,step:.1,key:{inc:"t",dec:"g",step:.1,shiftStep:.5}},{id:"modAmp",label:"Mod Amplitude",uniform:"uModAmp",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"y",dec:"h",step:.05,shiftStep:.25}},{id:"modFreq",label:"Mod Frequency",uniform:"uModFreq",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"u",dec:"j",step:.1,shiftStep:.5}},{id:"modDiv",label:"Mod Divisor",uniform:"uModDiv",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.5}},{id:"thetaScale",label:"Theta Scale",uniform:"uThetaScale",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"o",dec:"l",step:.05,shiftStep:.25}},{id:"lineWidth",label:"Line Width",uniform:"uLineWidth",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"[",dec:"]",step:.1,shiftStep:.5}},{id:"hueCycles",label:"Hue Cycles",uniform:"uHueCycles",type:"float",value:4,min:0,max:12,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]}];oe.sort((e,n)=>e.name.localeCompare(n.name));function An(e,n,t){const a=e.createShader(n);if(!a)throw new Error("Failed to create shader");if(e.shaderSource(a,t),e.compileShader(a),!e.getShaderParameter(a,e.COMPILE_STATUS)){const o=e.getShaderInfoLog(a)||"Unknown shader error";throw e.deleteShader(a),new Error(o)}return a}function Ht(e,n,t){const a=An(e,e.VERTEX_SHADER,n),o=An(e,e.FRAGMENT_SHADER,t),i=e.createProgram();if(!i)throw new Error("Failed to create program");if(e.attachShader(i,a),e.attachShader(i,o),e.linkProgram(i),e.deleteShader(a),e.deleteShader(o),!e.getProgramParameter(i,e.LINK_STATUS)){const l=e.getProgramInfoLog(i)||"Unknown program error";throw e.deleteProgram(i),new Error(l)}return i}function qt(e,n,t){const a={};for(const o of t)a[o]=e.getUniformLocation(n,o);return a}function Vt(e,n=2){const t=Math.min(window.devicePixelRatio||1,n),a=Math.max(1,Math.floor(e.clientWidth*t)),o=Math.max(1,Math.floor(e.clientHeight*t));return(e.width!==a||e.height!==o)&&(e.width=a,e.height=o),{width:a,height:o,dpr:t}}function jt(e){const n=e.createVertexArray();if(!n)throw new Error("Failed to create VAO");return e.bindVertexArray(n),n}var ln=(e,n,t)=>{if(!n.has(e))throw TypeError("Cannot "+t)},r=(e,n,t)=>(ln(e,n,"read from private field"),t?t.call(e):n.get(e)),d=(e,n,t)=>{if(n.has(e))throw TypeError("Cannot add the same private member more than once");n instanceof WeakSet?n.add(e):n.set(e,t)},B=(e,n,t,a)=>(ln(e,n,"write to private field"),n.set(e,t),t),Xt=(e,n,t,a)=>({set _(o){B(e,n,o)},get _(){return r(e,n,a)}}),h=(e,n,t)=>(ln(e,n,"access private method"),t),v=new Uint8Array(8),U=new DataView(v.buffer),R=e=>[(e%256+256)%256],g=e=>(U.setUint16(0,e,!1),[v[0],v[1]]),Zt=e=>(U.setInt16(0,e,!1),[v[0],v[1]]),zn=e=>(U.setUint32(0,e,!1),[v[1],v[2],v[3]]),f=e=>(U.setUint32(0,e,!1),[v[0],v[1],v[2],v[3]]),Yt=e=>(U.setInt32(0,e,!1),[v[0],v[1],v[2],v[3]]),Y=e=>(U.setUint32(0,Math.floor(e/2**32),!1),U.setUint32(4,e,!1),[v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]]),sn=e=>(U.setInt16(0,2**8*e,!1),[v[0],v[1]]),E=e=>(U.setInt32(0,2**16*e,!1),[v[0],v[1],v[2],v[3]]),He=e=>(U.setInt32(0,2**30*e,!1),[v[0],v[1],v[2],v[3]]),M=(e,n=!1)=>{let t=Array(e.length).fill(null).map((a,o)=>e.charCodeAt(o));return n&&t.push(0),t},De=e=>e&&e[e.length-1],un=e=>{let n;for(let t of e)(!n||t.presentationTimestamp>n.presentationTimestamp)&&(n=t);return n},G=(e,n,t=!0)=>{let a=e*n;return t?Math.round(a):a},In=e=>{let n=e*(Math.PI/180),t=Math.cos(n),a=Math.sin(n);return[t,a,0,-a,t,0,0,0,1]},En=In(0),Gn=e=>[E(e[0]),E(e[1]),He(e[2]),E(e[3]),E(e[4]),He(e[5]),E(e[6]),E(e[7]),He(e[8])],ue=e=>!e||typeof e!="object"?e:Array.isArray(e)?e.map(ue):Object.fromEntries(Object.entries(e).map(([n,t])=>[n,ue(t)])),te=e=>e>=0&&e<2**32,T=(e,n,t)=>({type:e,contents:n&&new Uint8Array(n.flat(10)),children:t}),x=(e,n,t,a,o)=>T(e,[R(n),zn(t),a??[]],o),$t=e=>{let n=512;return e.fragmented?T("ftyp",[M("iso5"),f(n),M("iso5"),M("iso6"),M("mp41")]):T("ftyp",[M("isom"),f(n),M("isom"),e.holdsAvc?M("avc1"):[],M("mp41")])},Xe=e=>({type:"mdat",largeSize:e}),Kt=e=>({type:"free",size:e}),Te=(e,n,t=!1)=>T("moov",null,[Qt(n,e),...e.map(a=>Jt(a,n)),t?Pr(e):null]),Qt=(e,n)=>{let t=G(Math.max(0,...n.filter(l=>l.samples.length>0).map(l=>{const u=un(l.samples);return u.presentationTimestamp+u.duration})),$e),a=Math.max(...n.map(l=>l.id))+1,o=!te(e)||!te(t),i=o?Y:f;return x("mvhd",+o,0,[i(e),i(e),f($e),i(t),E(1),sn(1),Array(10).fill(0),Gn(En),Array(24).fill(0),f(a)])},Jt=(e,n)=>T("trak",null,[er(e,n),nr(e,n)]),er=(e,n)=>{let t=un(e.samples),a=G(t?t.presentationTimestamp+t.duration:0,$e),o=!te(n)||!te(a),i=o?Y:f,l;return e.info.type==="video"?l=typeof e.info.rotation=="number"?In(e.info.rotation):e.info.rotation:l=En,x("tkhd",+o,3,[i(n),i(n),f(e.id),f(0),i(a),Array(8).fill(0),g(0),g(0),sn(e.info.type==="audio"?1:0),g(0),Gn(l),E(e.info.type==="video"?e.info.width:0),E(e.info.type==="video"?e.info.height:0)])},nr=(e,n)=>T("mdia",null,[tr(e,n),rr(e.info.type==="video"?"vide":"soun"),ar(e)]),tr=(e,n)=>{let t=un(e.samples),a=G(t?t.presentationTimestamp+t.duration:0,e.timescale),o=!te(n)||!te(a),i=o?Y:f;return x("mdhd",+o,0,[i(n),i(n),f(e.timescale),i(a),g(21956),g(0)])},rr=e=>x("hdlr",0,0,[M("mhlr"),M(e),f(0),f(0),f(0),M("mp4-muxer-hdlr",!0)]),ar=e=>T("minf",null,[e.info.type==="video"?or():ir(),lr(),fr(e)]),or=()=>x("vmhd",0,1,[g(0),g(0),g(0),g(0)]),ir=()=>x("smhd",0,0,[g(0),g(0)]),lr=()=>T("dinf",null,[sr()]),sr=()=>x("dref",0,0,[f(1)],[ur()]),ur=()=>x("url ",0,1),fr=e=>{const n=e.compositionTimeOffsetTable.length>1||e.compositionTimeOffsetTable.some(t=>t.sampleCompositionTimeOffset!==0);return T("stbl",null,[cr(e),Tr(e),kr(e),Rr(e),Ar(e),Br(e),n?Fr(e):null])},cr=e=>x("stsd",0,0,[f(1)],[e.info.type==="video"?mr(Wr[e.info.codec],e):xr(_r[e.info.codec],e)]),mr=(e,n)=>T(e,[Array(6).fill(0),g(1),g(0),g(0),Array(12).fill(0),g(n.info.width),g(n.info.height),f(4718592),f(4718592),f(0),g(1),Array(32).fill(0),g(24),Zt(65535)],[Lr[n.info.codec](n),n.info.decoderConfig.colorSpace?vr(n):null]),pr={bt709:1,bt470bg:5,smpte170m:6},dr={bt709:1,smpte170m:6,"iec61966-2-1":13},hr={rgb:0,bt709:1,bt470bg:5,smpte170m:6},vr=e=>T("colr",[M("nclx"),g(pr[e.info.decoderConfig.colorSpace.primaries]),g(dr[e.info.decoderConfig.colorSpace.transfer]),g(hr[e.info.decoderConfig.colorSpace.matrix]),R((e.info.decoderConfig.colorSpace.fullRange?1:0)<<7)]),yr=e=>e.info.decoderConfig&&T("avcC",[...new Uint8Array(e.info.decoderConfig.description)]),br=e=>e.info.decoderConfig&&T("hvcC",[...new Uint8Array(e.info.decoderConfig.description)]),gr=e=>{if(!e.info.decoderConfig)return null;let n=e.info.decoderConfig;if(!n.colorSpace)throw new Error("'colorSpace' is required in the decoder config for VP9.");let t=n.codec.split("."),a=Number(t[1]),o=Number(t[2]),u=(Number(t[3])<<4)+(0<<1)+Number(n.colorSpace.fullRange);return x("vpcC",1,0,[R(a),R(o),R(u),R(2),R(2),R(2),g(0)])},Sr=()=>{let t=(1<<7)+1;return T("av1C",[t,0,0,0])},xr=(e,n)=>T(e,[Array(6).fill(0),g(1),g(0),g(0),f(0),g(n.info.numberOfChannels),g(16),g(0),g(0),E(n.info.sampleRate)],[Hr[n.info.codec](n)]),wr=e=>{let n=new Uint8Array(e.info.decoderConfig.description);return x("esds",0,0,[f(58753152),R(32+n.byteLength),g(1),R(0),f(75530368),R(18+n.byteLength),R(64),R(21),zn(0),f(130071),f(130071),f(92307584),R(n.byteLength),...n,f(109084800),R(1),R(2)])},Cr=e=>{var o;let n=3840,t=0;const a=(o=e.info.decoderConfig)==null?void 0:o.description;if(a){if(a.byteLength<18)throw new TypeError("Invalid decoder description provided for Opus; must be at least 18 bytes long.");const i=ArrayBuffer.isView(a)?new DataView(a.buffer,a.byteOffset,a.byteLength):new DataView(a);n=i.getUint16(10,!0),t=i.getInt16(14,!0)}return T("dOps",[R(0),R(e.info.numberOfChannels),g(n),f(e.info.sampleRate),sn(t),R(0)])},Tr=e=>x("stts",0,0,[f(e.timeToSampleTable.length),e.timeToSampleTable.map(n=>[f(n.sampleCount),f(n.sampleDelta)])]),kr=e=>{if(e.samples.every(t=>t.type==="key"))return null;let n=[...e.samples.entries()].filter(([,t])=>t.type==="key");return x("stss",0,0,[f(n.length),n.map(([t])=>f(t+1))])},Rr=e=>x("stsc",0,0,[f(e.compactlyCodedChunkTable.length),e.compactlyCodedChunkTable.map(n=>[f(n.firstChunk),f(n.samplesPerChunk),f(1)])]),Ar=e=>x("stsz",0,0,[f(0),f(e.samples.length),e.samples.map(n=>f(n.size))]),Br=e=>e.finalizedChunks.length>0&&De(e.finalizedChunks).offset>=2**32?x("co64",0,0,[f(e.finalizedChunks.length),e.finalizedChunks.map(n=>Y(n.offset))]):x("stco",0,0,[f(e.finalizedChunks.length),e.finalizedChunks.map(n=>f(n.offset))]),Fr=e=>x("ctts",0,0,[f(e.compositionTimeOffsetTable.length),e.compositionTimeOffsetTable.map(n=>[f(n.sampleCount),f(n.sampleCompositionTimeOffset)])]),Pr=e=>T("mvex",null,e.map(Dr)),Dr=e=>x("trex",0,0,[f(e.id),f(1),f(0),f(0),f(0)]),Bn=(e,n)=>T("moof",null,[Mr(e),...n.map(zr)]),Mr=e=>x("mfhd",0,0,[f(e)]),Un=e=>{let n=0,t=0,a=0,o=0,i=e.type==="delta";return t|=+i,i?n|=1:n|=2,n<<24|t<<16|a<<8|o},zr=e=>T("traf",null,[Ir(e),Er(e),Gr(e)]),Ir=e=>{let n=0;n|=8,n|=16,n|=32,n|=131072;let t=e.currentChunk.samples[1]??e.currentChunk.samples[0],a={duration:t.timescaleUnitsToNextSample,size:t.size,flags:Un(t)};return x("tfhd",0,n,[f(e.id),f(a.duration),f(a.size),f(a.flags)])},Er=e=>x("tfdt",1,0,[Y(G(e.currentChunk.startTimestamp,e.timescale))]),Gr=e=>{let n=e.currentChunk.samples.map(z=>z.timescaleUnitsToNextSample),t=e.currentChunk.samples.map(z=>z.size),a=e.currentChunk.samples.map(Un),o=e.currentChunk.samples.map(z=>G(z.presentationTimestamp-z.decodeTimestamp,e.timescale)),i=new Set(n),l=new Set(t),u=new Set(a),p=new Set(o),y=u.size===2&&a[0]!==a[1],b=i.size>1,w=l.size>1,S=!y&&u.size>1,_=p.size>1||[...p].some(z=>z!==0),N=0;return N|=1,N|=4*+y,N|=256*+b,N|=512*+w,N|=1024*+S,N|=2048*+_,x("trun",1,N,[f(e.currentChunk.samples.length),f(e.currentChunk.offset-e.currentChunk.moofOffset||0),y?f(a[0]):[],e.currentChunk.samples.map((z,$)=>[b?f(n[$]):[],w?f(t[$]):[],S?f(a[$]):[],_?Yt(o[$]):[]])])},Ur=e=>T("mfra",null,[...e.map(Nr),Or()]),Nr=(e,n)=>x("tfra",1,0,[f(e.id),f(63),f(e.finalizedChunks.length),e.finalizedChunks.map(a=>[Y(G(a.startTimestamp,e.timescale)),Y(a.moofOffset),f(n+1),f(1),f(1)])]),Or=()=>x("mfro",0,0,[f(0)]),Wr={avc:"avc1",hevc:"hvc1",vp9:"vp09",av1:"av01"},Lr={avc:yr,hevc:br,vp9:gr,av1:Sr},_r={aac:"mp4a",opus:"Opus"},Hr={aac:wr,opus:Cr},Oe=class{},qr=class extends Oe{constructor(){super(...arguments),this.buffer=null}},Nn=class extends Oe{constructor(e){if(super(),this.options=e,typeof e!="object")throw new TypeError("StreamTarget requires an options object to be passed to its constructor.");if(e.onData){if(typeof e.onData!="function")throw new TypeError("options.onData, when provided, must be a function.");if(e.onData.length<2)throw new TypeError("options.onData, when provided, must be a function that takes in at least two arguments (data and position). Ignoring the position argument, which specifies the byte offset at which the data is to be written, can lead to broken outputs.")}if(e.chunked!==void 0&&typeof e.chunked!="boolean")throw new TypeError("options.chunked, when provided, must be a boolean.");if(e.chunkSize!==void 0&&(!Number.isInteger(e.chunkSize)||e.chunkSize<1024))throw new TypeError("options.chunkSize, when provided, must be an integer and not smaller than 1024.")}},On=class extends Oe{constructor(e,n){if(super(),this.stream=e,this.options=n,!(e instanceof FileSystemWritableFileStream))throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");if(n!==void 0&&typeof n!="object")throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");if(n&&n.chunkSize!==void 0&&(!Number.isInteger(n.chunkSize)||n.chunkSize<=0))throw new TypeError("options.chunkSize, when provided, must be a positive integer")}},H,Q,Wn=class{constructor(){this.pos=0,d(this,H,new Uint8Array(8)),d(this,Q,new DataView(r(this,H).buffer)),this.offsets=new WeakMap}seek(e){this.pos=e}writeU32(e){r(this,Q).setUint32(0,e,!1),this.write(r(this,H).subarray(0,4))}writeU64(e){r(this,Q).setUint32(0,Math.floor(e/2**32),!1),r(this,Q).setUint32(4,e,!1),this.write(r(this,H).subarray(0,8))}writeAscii(e){for(let n=0;n<e.length;n++)r(this,Q).setUint8(n%8,e.charCodeAt(n)),n%8===7&&this.write(r(this,H));e.length%8!==0&&this.write(r(this,H).subarray(0,e.length%8))}writeBox(e){if(this.offsets.set(e,this.pos),e.contents&&!e.children)this.writeBoxHeader(e,e.size??e.contents.byteLength+8),this.write(e.contents);else{let n=this.pos;if(this.writeBoxHeader(e,0),e.contents&&this.write(e.contents),e.children)for(let o of e.children)o&&this.writeBox(o);let t=this.pos,a=e.size??t-n;this.seek(n),this.writeBoxHeader(e,a),this.seek(t)}}writeBoxHeader(e,n){this.writeU32(e.largeSize?1:n),this.writeAscii(e.type),e.largeSize&&this.writeU64(n)}measureBoxHeader(e){return 8+(e.largeSize?8:0)}patchBox(e){let n=this.pos;this.seek(this.offsets.get(e)),this.writeBox(e),this.seek(n)}measureBox(e){if(e.contents&&!e.children)return this.measureBoxHeader(e)+e.contents.byteLength;{let n=this.measureBoxHeader(e);if(e.contents&&(n+=e.contents.byteLength),e.children)for(let t of e.children)t&&(n+=this.measureBox(t));return n}}};H=new WeakMap;Q=new WeakMap;var ke,Z,de,ie,Re,Ze,Vr=class extends Wn{constructor(e){super(),d(this,Re),d(this,ke,void 0),d(this,Z,new ArrayBuffer(2**16)),d(this,de,new Uint8Array(r(this,Z))),d(this,ie,0),B(this,ke,e)}write(e){h(this,Re,Ze).call(this,this.pos+e.byteLength),r(this,de).set(e,this.pos),this.pos+=e.byteLength,B(this,ie,Math.max(r(this,ie),this.pos))}finalize(){h(this,Re,Ze).call(this,this.pos),r(this,ke).buffer=r(this,Z).slice(0,Math.max(r(this,ie),this.pos))}};ke=new WeakMap;Z=new WeakMap;de=new WeakMap;ie=new WeakMap;Re=new WeakSet;Ze=function(e){let n=r(this,Z).byteLength;for(;n<e;)n*=2;if(n===r(this,Z).byteLength)return;let t=new ArrayBuffer(n),a=new Uint8Array(t);a.set(r(this,de),0),B(this,Z,t),B(this,de,a)};var jr=2**24,Xr=2,fe,q,le,W,D,Me,Ye,fn,Ln,cn,_n,ce,ze,mn=class extends Wn{constructor(e){var n,t;super(),d(this,Me),d(this,fn),d(this,cn),d(this,ce),d(this,fe,void 0),d(this,q,[]),d(this,le,void 0),d(this,W,void 0),d(this,D,[]),B(this,fe,e),B(this,le,((n=e.options)==null?void 0:n.chunked)??!1),B(this,W,((t=e.options)==null?void 0:t.chunkSize)??jr)}write(e){r(this,q).push({data:e.slice(),start:this.pos}),this.pos+=e.byteLength}flush(){var t,a;if(r(this,q).length===0)return;let e=[],n=[...r(this,q)].sort((o,i)=>o.start-i.start);e.push({start:n[0].start,size:n[0].data.byteLength});for(let o=1;o<n.length;o++){let i=e[e.length-1],l=n[o];l.start<=i.start+i.size?i.size=Math.max(i.size,l.start+l.data.byteLength-i.start):e.push({start:l.start,size:l.data.byteLength})}for(let o of e){o.data=new Uint8Array(o.size);for(let i of r(this,q))o.start<=i.start&&i.start<o.start+o.size&&o.data.set(i.data,i.start-o.start);r(this,le)?(h(this,Me,Ye).call(this,o.data,o.start),h(this,ce,ze).call(this)):(a=(t=r(this,fe).options).onData)==null||a.call(t,o.data,o.start)}r(this,q).length=0}finalize(){r(this,le)&&h(this,ce,ze).call(this,!0)}};fe=new WeakMap;q=new WeakMap;le=new WeakMap;W=new WeakMap;D=new WeakMap;Me=new WeakSet;Ye=function(e,n){let t=r(this,D).findIndex(u=>u.start<=n&&n<u.start+r(this,W));t===-1&&(t=h(this,cn,_n).call(this,n));let a=r(this,D)[t],o=n-a.start,i=e.subarray(0,Math.min(r(this,W)-o,e.byteLength));a.data.set(i,o);let l={start:o,end:o+i.byteLength};if(h(this,fn,Ln).call(this,a,l),a.written[0].start===0&&a.written[0].end===r(this,W)&&(a.shouldFlush=!0),r(this,D).length>Xr){for(let u=0;u<r(this,D).length-1;u++)r(this,D)[u].shouldFlush=!0;h(this,ce,ze).call(this)}i.byteLength<e.byteLength&&h(this,Me,Ye).call(this,e.subarray(i.byteLength),n+i.byteLength)};fn=new WeakSet;Ln=function(e,n){let t=0,a=e.written.length-1,o=-1;for(;t<=a;){let i=Math.floor(t+(a-t+1)/2);e.written[i].start<=n.start?(t=i+1,o=i):a=i-1}for(e.written.splice(o+1,0,n),(o===-1||e.written[o].end<n.start)&&o++;o<e.written.length-1&&e.written[o].end>=e.written[o+1].start;)e.written[o].end=Math.max(e.written[o].end,e.written[o+1].end),e.written.splice(o+1,1)};cn=new WeakSet;_n=function(e){let t={start:Math.floor(e/r(this,W))*r(this,W),data:new Uint8Array(r(this,W)),written:[],shouldFlush:!1};return r(this,D).push(t),r(this,D).sort((a,o)=>a.start-o.start),r(this,D).indexOf(t)};ce=new WeakSet;ze=function(e=!1){var n,t;for(let a=0;a<r(this,D).length;a++){let o=r(this,D)[a];if(!(!o.shouldFlush&&!e)){for(let i of o.written)(t=(n=r(this,fe).options).onData)==null||t.call(n,o.data.subarray(i.start,i.end),o.start+i.start);r(this,D).splice(a--,1)}}};var Zr=class extends mn{constructor(e){var n;super(new Nn({onData:(t,a)=>e.stream.write({type:"write",data:t,position:a}),chunked:!0,chunkSize:(n=e.options)==null?void 0:n.chunkSize}))}},$e=1e3,Yr=["avc","hevc","vp9","av1"],$r=["aac","opus"],Kr=2082844800,Qr=["strict","offset","cross-track-offset"],c,m,Ie,P,F,A,J,ne,pn,V,j,me,Ke,Hn,Qe,qn,dn,Vn,Je,jn,hn,Xn,Ae,en,I,O,vn,Zn,pe,Ee,Ge,yn,re,ye,Be,nn,Jr=class{constructor(e){if(d(this,Ke),d(this,Qe),d(this,dn),d(this,Je),d(this,hn),d(this,Ae),d(this,I),d(this,vn),d(this,pe),d(this,Ge),d(this,re),d(this,Be),d(this,c,void 0),d(this,m,void 0),d(this,Ie,void 0),d(this,P,void 0),d(this,F,null),d(this,A,null),d(this,J,Math.floor(Date.now()/1e3)+Kr),d(this,ne,[]),d(this,pn,1),d(this,V,[]),d(this,j,[]),d(this,me,!1),h(this,Ke,Hn).call(this,e),e.video=ue(e.video),e.audio=ue(e.audio),e.fastStart=ue(e.fastStart),this.target=e.target,B(this,c,{firstTimestampBehavior:"strict",...e}),e.target instanceof qr)B(this,m,new Vr(e.target));else if(e.target instanceof Nn)B(this,m,new mn(e.target));else if(e.target instanceof On)B(this,m,new Zr(e.target));else throw new Error(`Invalid target: ${e.target}`);h(this,Je,jn).call(this),h(this,Qe,qn).call(this)}addVideoChunk(e,n,t,a){if(!(e instanceof EncodedVideoChunk))throw new TypeError("addVideoChunk's first argument (sample) must be of type EncodedVideoChunk.");if(n&&typeof n!="object")throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");if(t!==void 0&&(!Number.isFinite(t)||t<0))throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");if(a!==void 0&&!Number.isFinite(a))throw new TypeError("addVideoChunk's fourth argument (compositionTimeOffset), when provided, must be a real number.");let o=new Uint8Array(e.byteLength);e.copyTo(o),this.addVideoChunkRaw(o,e.type,t??e.timestamp,e.duration,n,a)}addVideoChunkRaw(e,n,t,a,o,i){if(!(e instanceof Uint8Array))throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");if(n!=="key"&&n!=="delta")throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(t)||t<0)throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(a)||a<0)throw new TypeError("addVideoChunkRaw's fourth argument (duration) must be a non-negative real number.");if(o&&typeof o!="object")throw new TypeError("addVideoChunkRaw's fifth argument (meta), when provided, must be an object.");if(i!==void 0&&!Number.isFinite(i))throw new TypeError("addVideoChunkRaw's sixth argument (compositionTimeOffset), when provided, must be a real number.");if(h(this,Be,nn).call(this),!r(this,c).video)throw new Error("No video track declared.");if(typeof r(this,c).fastStart=="object"&&r(this,F).samples.length===r(this,c).fastStart.expectedVideoChunks)throw new Error(`Cannot add more video chunks than specified in 'fastStart' (${r(this,c).fastStart.expectedVideoChunks}).`);let l=h(this,Ae,en).call(this,r(this,F),e,n,t,a,o,i);if(r(this,c).fastStart==="fragmented"&&r(this,A)){for(;r(this,j).length>0&&r(this,j)[0].decodeTimestamp<=l.decodeTimestamp;){let u=r(this,j).shift();h(this,I,O).call(this,r(this,A),u)}l.decodeTimestamp<=r(this,A).lastDecodeTimestamp?h(this,I,O).call(this,r(this,F),l):r(this,V).push(l)}else h(this,I,O).call(this,r(this,F),l)}addAudioChunk(e,n,t){if(!(e instanceof EncodedAudioChunk))throw new TypeError("addAudioChunk's first argument (sample) must be of type EncodedAudioChunk.");if(n&&typeof n!="object")throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");if(t!==void 0&&(!Number.isFinite(t)||t<0))throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");let a=new Uint8Array(e.byteLength);e.copyTo(a),this.addAudioChunkRaw(a,e.type,t??e.timestamp,e.duration,n)}addAudioChunkRaw(e,n,t,a,o){if(!(e instanceof Uint8Array))throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");if(n!=="key"&&n!=="delta")throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(t)||t<0)throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(a)||a<0)throw new TypeError("addAudioChunkRaw's fourth argument (duration) must be a non-negative real number.");if(o&&typeof o!="object")throw new TypeError("addAudioChunkRaw's fifth argument (meta), when provided, must be an object.");if(h(this,Be,nn).call(this),!r(this,c).audio)throw new Error("No audio track declared.");if(typeof r(this,c).fastStart=="object"&&r(this,A).samples.length===r(this,c).fastStart.expectedAudioChunks)throw new Error(`Cannot add more audio chunks than specified in 'fastStart' (${r(this,c).fastStart.expectedAudioChunks}).`);let i=h(this,Ae,en).call(this,r(this,A),e,n,t,a,o);if(r(this,c).fastStart==="fragmented"&&r(this,F)){for(;r(this,V).length>0&&r(this,V)[0].decodeTimestamp<=i.decodeTimestamp;){let l=r(this,V).shift();h(this,I,O).call(this,r(this,F),l)}i.decodeTimestamp<=r(this,F).lastDecodeTimestamp?h(this,I,O).call(this,r(this,A),i):r(this,j).push(i)}else h(this,I,O).call(this,r(this,A),i)}finalize(){if(r(this,me))throw new Error("Cannot finalize a muxer more than once.");if(r(this,c).fastStart==="fragmented"){for(let n of r(this,V))h(this,I,O).call(this,r(this,F),n);for(let n of r(this,j))h(this,I,O).call(this,r(this,A),n);h(this,Ge,yn).call(this,!1)}else r(this,F)&&h(this,pe,Ee).call(this,r(this,F)),r(this,A)&&h(this,pe,Ee).call(this,r(this,A));let e=[r(this,F),r(this,A)].filter(Boolean);if(r(this,c).fastStart==="in-memory"){let n;for(let a=0;a<2;a++){let o=Te(e,r(this,J)),i=r(this,m).measureBox(o);n=r(this,m).measureBox(r(this,P));let l=r(this,m).pos+i+n;for(let u of r(this,ne)){u.offset=l;for(let{data:p}of u.samples)l+=p.byteLength,n+=p.byteLength}if(l<2**32)break;n>=2**32&&(r(this,P).largeSize=!0)}let t=Te(e,r(this,J));r(this,m).writeBox(t),r(this,P).size=n,r(this,m).writeBox(r(this,P));for(let a of r(this,ne))for(let o of a.samples)r(this,m).write(o.data),o.data=null}else if(r(this,c).fastStart==="fragmented"){let n=r(this,m).pos,t=Ur(e);r(this,m).writeBox(t);let a=r(this,m).pos-n;r(this,m).seek(r(this,m).pos-4),r(this,m).writeU32(a)}else{let n=r(this,m).offsets.get(r(this,P)),t=r(this,m).pos-n;r(this,P).size=t,r(this,P).largeSize=t>=2**32,r(this,m).patchBox(r(this,P));let a=Te(e,r(this,J));if(typeof r(this,c).fastStart=="object"){r(this,m).seek(r(this,Ie)),r(this,m).writeBox(a);let o=n-r(this,m).pos;r(this,m).writeBox(Kt(o))}else r(this,m).writeBox(a)}h(this,re,ye).call(this),r(this,m).finalize(),B(this,me,!0)}};c=new WeakMap;m=new WeakMap;Ie=new WeakMap;P=new WeakMap;F=new WeakMap;A=new WeakMap;J=new WeakMap;ne=new WeakMap;pn=new WeakMap;V=new WeakMap;j=new WeakMap;me=new WeakMap;Ke=new WeakSet;Hn=function(e){if(typeof e!="object")throw new TypeError("The muxer requires an options object to be passed to its constructor.");if(!(e.target instanceof Oe))throw new TypeError("The target must be provided and an instance of Target.");if(e.video){if(!Yr.includes(e.video.codec))throw new TypeError(`Unsupported video codec: ${e.video.codec}`);if(!Number.isInteger(e.video.width)||e.video.width<=0)throw new TypeError(`Invalid video width: ${e.video.width}. Must be a positive integer.`);if(!Number.isInteger(e.video.height)||e.video.height<=0)throw new TypeError(`Invalid video height: ${e.video.height}. Must be a positive integer.`);const n=e.video.rotation;if(typeof n=="number"&&![0,90,180,270].includes(n))throw new TypeError(`Invalid video rotation: ${n}. Has to be 0, 90, 180 or 270.`);if(Array.isArray(n)&&(n.length!==9||n.some(t=>typeof t!="number")))throw new TypeError(`Invalid video transformation matrix: ${n.join()}`);if(e.video.frameRate!==void 0&&(!Number.isInteger(e.video.frameRate)||e.video.frameRate<=0))throw new TypeError(`Invalid video frame rate: ${e.video.frameRate}. Must be a positive integer.`)}if(e.audio){if(!$r.includes(e.audio.codec))throw new TypeError(`Unsupported audio codec: ${e.audio.codec}`);if(!Number.isInteger(e.audio.numberOfChannels)||e.audio.numberOfChannels<=0)throw new TypeError(`Invalid number of audio channels: ${e.audio.numberOfChannels}. Must be a positive integer.`);if(!Number.isInteger(e.audio.sampleRate)||e.audio.sampleRate<=0)throw new TypeError(`Invalid audio sample rate: ${e.audio.sampleRate}. Must be a positive integer.`)}if(e.firstTimestampBehavior&&!Qr.includes(e.firstTimestampBehavior))throw new TypeError(`Invalid first timestamp behavior: ${e.firstTimestampBehavior}`);if(typeof e.fastStart=="object"){if(e.video){if(e.fastStart.expectedVideoChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedVideoChunks'.");if(!Number.isInteger(e.fastStart.expectedVideoChunks)||e.fastStart.expectedVideoChunks<0)throw new TypeError("'expectedVideoChunks' must be a non-negative integer.")}if(e.audio){if(e.fastStart.expectedAudioChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedAudioChunks'.");if(!Number.isInteger(e.fastStart.expectedAudioChunks)||e.fastStart.expectedAudioChunks<0)throw new TypeError("'expectedAudioChunks' must be a non-negative integer.")}}else if(![!1,"in-memory","fragmented"].includes(e.fastStart))throw new TypeError("'fastStart' option must be false, 'in-memory', 'fragmented' or an object.");if(e.minFragmentDuration!==void 0&&(!Number.isFinite(e.minFragmentDuration)||e.minFragmentDuration<0))throw new TypeError("'minFragmentDuration' must be a non-negative number.")};Qe=new WeakSet;qn=function(){var e;if(r(this,m).writeBox($t({holdsAvc:((e=r(this,c).video)==null?void 0:e.codec)==="avc",fragmented:r(this,c).fastStart==="fragmented"})),B(this,Ie,r(this,m).pos),r(this,c).fastStart==="in-memory")B(this,P,Xe(!1));else if(r(this,c).fastStart!=="fragmented"){if(typeof r(this,c).fastStart=="object"){let n=h(this,dn,Vn).call(this);r(this,m).seek(r(this,m).pos+n)}B(this,P,Xe(!0)),r(this,m).writeBox(r(this,P))}h(this,re,ye).call(this)};dn=new WeakSet;Vn=function(){if(typeof r(this,c).fastStart!="object")return;let e=0,n=[r(this,c).fastStart.expectedVideoChunks,r(this,c).fastStart.expectedAudioChunks];for(let t of n)t&&(e+=8*Math.ceil(2/3*t),e+=4*t,e+=12*Math.ceil(2/3*t),e+=4*t,e+=8*t);return e+=4096,e};Je=new WeakSet;jn=function(){if(r(this,c).video&&B(this,F,{id:1,info:{type:"video",codec:r(this,c).video.codec,width:r(this,c).video.width,height:r(this,c).video.height,rotation:r(this,c).video.rotation??0,decoderConfig:null},timescale:r(this,c).video.frameRate??57600,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),r(this,c).audio&&(B(this,A,{id:r(this,c).video?2:1,info:{type:"audio",codec:r(this,c).audio.codec,numberOfChannels:r(this,c).audio.numberOfChannels,sampleRate:r(this,c).audio.sampleRate,decoderConfig:null},timescale:r(this,c).audio.sampleRate,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),r(this,c).audio.codec==="aac")){let e=h(this,hn,Xn).call(this,2,r(this,c).audio.sampleRate,r(this,c).audio.numberOfChannels);r(this,A).info.decoderConfig={codec:r(this,c).audio.codec,description:e,numberOfChannels:r(this,c).audio.numberOfChannels,sampleRate:r(this,c).audio.sampleRate}}};hn=new WeakSet;Xn=function(e,n,t){let o=[96e3,88200,64e3,48e3,44100,32e3,24e3,22050,16e3,12e3,11025,8e3,7350].indexOf(n),i=t,l="";l+=e.toString(2).padStart(5,"0"),l+=o.toString(2).padStart(4,"0"),o===15&&(l+=n.toString(2).padStart(24,"0")),l+=i.toString(2).padStart(4,"0");let u=Math.ceil(l.length/8)*8;l=l.padEnd(u,"0");let p=new Uint8Array(l.length/8);for(let y=0;y<l.length;y+=8)p[y/8]=parseInt(l.slice(y,y+8),2);return p};Ae=new WeakSet;en=function(e,n,t,a,o,i,l){let u=a/1e6,p=(a-(l??0))/1e6,y=o/1e6,b=h(this,vn,Zn).call(this,u,p,e);return u=b.presentationTimestamp,p=b.decodeTimestamp,i!=null&&i.decoderConfig&&(e.info.decoderConfig===null?e.info.decoderConfig=i.decoderConfig:Object.assign(e.info.decoderConfig,i.decoderConfig)),{presentationTimestamp:u,decodeTimestamp:p,duration:y,data:n,size:n.byteLength,type:t,timescaleUnitsToNextSample:G(y,e.timescale)}};I=new WeakSet;O=function(e,n){r(this,c).fastStart!=="fragmented"&&e.samples.push(n);const t=G(n.presentationTimestamp-n.decodeTimestamp,e.timescale);if(e.lastTimescaleUnits!==null){let o=G(n.decodeTimestamp,e.timescale,!1),i=Math.round(o-e.lastTimescaleUnits);if(e.lastTimescaleUnits+=i,e.lastSample.timescaleUnitsToNextSample=i,r(this,c).fastStart!=="fragmented"){let l=De(e.timeToSampleTable);l.sampleCount===1?(l.sampleDelta=i,l.sampleCount++):l.sampleDelta===i?l.sampleCount++:(l.sampleCount--,e.timeToSampleTable.push({sampleCount:2,sampleDelta:i}));const u=De(e.compositionTimeOffsetTable);u.sampleCompositionTimeOffset===t?u.sampleCount++:e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:t})}}else e.lastTimescaleUnits=0,r(this,c).fastStart!=="fragmented"&&(e.timeToSampleTable.push({sampleCount:1,sampleDelta:G(n.duration,e.timescale)}),e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:t}));e.lastSample=n;let a=!1;if(!e.currentChunk)a=!0;else{let o=n.presentationTimestamp-e.currentChunk.startTimestamp;if(r(this,c).fastStart==="fragmented"){let i=r(this,F)??r(this,A);const l=r(this,c).minFragmentDuration??1;e===i&&n.type==="key"&&o>=l&&(a=!0,h(this,Ge,yn).call(this))}else a=o>=.5}a&&(e.currentChunk&&h(this,pe,Ee).call(this,e),e.currentChunk={startTimestamp:n.presentationTimestamp,samples:[]}),e.currentChunk.samples.push(n)};vn=new WeakSet;Zn=function(e,n,t){var l,u;const a=r(this,c).firstTimestampBehavior==="strict",o=t.lastDecodeTimestamp===-1;if(a&&o&&n!==0)throw new Error(`The first chunk for your media track must have a timestamp of 0 (received DTS=${n}).Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of thedocument, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
`);if(r(this,c).firstTimestampBehavior==="offset"||r(this,c).firstTimestampBehavior==="cross-track-offset"){t.firstDecodeTimestamp===void 0&&(t.firstDecodeTimestamp=n);let p;r(this,c).firstTimestampBehavior==="offset"?p=t.firstDecodeTimestamp:p=Math.min(((l=r(this,F))==null?void 0:l.firstDecodeTimestamp)??1/0,((u=r(this,A))==null?void 0:u.firstDecodeTimestamp)??1/0),n-=p,e-=p}if(n<t.lastDecodeTimestamp)throw new Error(`Timestamps must be monotonically increasing (DTS went from ${t.lastDecodeTimestamp*1e6} to ${n*1e6}).`);return t.lastDecodeTimestamp=n,{presentationTimestamp:e,decodeTimestamp:n}};pe=new WeakSet;Ee=function(e){if(r(this,c).fastStart==="fragmented")throw new Error("Can't finalize individual chunks if 'fastStart' is set to 'fragmented'.");if(e.currentChunk){if(e.finalizedChunks.push(e.currentChunk),r(this,ne).push(e.currentChunk),(e.compactlyCodedChunkTable.length===0||De(e.compactlyCodedChunkTable).samplesPerChunk!==e.currentChunk.samples.length)&&e.compactlyCodedChunkTable.push({firstChunk:e.finalizedChunks.length,samplesPerChunk:e.currentChunk.samples.length}),r(this,c).fastStart==="in-memory"){e.currentChunk.offset=0;return}e.currentChunk.offset=r(this,m).pos;for(let n of e.currentChunk.samples)r(this,m).write(n.data),n.data=null;h(this,re,ye).call(this)}};Ge=new WeakSet;yn=function(e=!0){if(r(this,c).fastStart!=="fragmented")throw new Error("Can't finalize a fragment unless 'fastStart' is set to 'fragmented'.");let n=[r(this,F),r(this,A)].filter(u=>u&&u.currentChunk);if(n.length===0)return;let t=Xt(this,pn)._++;if(t===1){let u=Te(n,r(this,J),!0);r(this,m).writeBox(u)}let a=r(this,m).pos,o=Bn(t,n);r(this,m).writeBox(o);{let u=Xe(!1),p=0;for(let b of n)for(let w of b.currentChunk.samples)p+=w.size;let y=r(this,m).measureBox(u)+p;y>=2**32&&(u.largeSize=!0,y=r(this,m).measureBox(u)+p),u.size=y,r(this,m).writeBox(u)}for(let u of n){u.currentChunk.offset=r(this,m).pos,u.currentChunk.moofOffset=a;for(let p of u.currentChunk.samples)r(this,m).write(p.data),p.data=null}let i=r(this,m).pos;r(this,m).seek(r(this,m).offsets.get(o));let l=Bn(t,n);r(this,m).writeBox(l),r(this,m).seek(i);for(let u of n)u.finalizedChunks.push(u.currentChunk),r(this,ne).push(u.currentChunk),u.currentChunk=null;e&&h(this,re,ye).call(this)};re=new WeakSet;ye=function(){r(this,m)instanceof mn&&r(this,m).flush()};Be=new WeakSet;nn=function(){if(r(this,me))throw new Error("Cannot add new video or audio chunks after the file has been finalized.")};const Yn=document.querySelector("#app");if(!Yn)throw new Error("Missing #app root element");Yn.innerHTML=`
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
          <div class="section-title">Offline Render</div>
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
            <button class="ghost small" id="offline-btn">Generate</button>
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
`;const k=document.querySelector("#gl-canvas"),ee=document.querySelector("#scene-list"),tn=document.querySelector("#control-list"),Fe=document.querySelector("#panel-actions"),se=document.querySelector("#key-help"),Ue=document.querySelector("#hud-title"),$n=document.querySelector("#hud-desc"),We=document.querySelector("[data-action='toggle-sidebar']"),bn=document.querySelector(".stage");if(!k||!ee||!tn||!Fe||!se||!Ue||!$n||!We||!bn)throw new Error("Missing required UI elements");const s=k.getContext("webgl2",{antialias:!0});if(!s)throw bn.innerHTML=`
    <div class="fallback">
      <h2>WebGL2 unavailable</h2>
      <p>Your browser or GPU does not expose WebGL2. Try a different browser.</p>
    </div>
  `,new Error("WebGL2 unavailable");const ea=s.getExtension("EXT_color_buffer_float"),na=!!ea;s.disable(s.DEPTH_TEST);s.disable(s.BLEND);const Fn=jt(s),Kn=new Map,Qn={},ae={},Jn={},rn=new Map;for(const e of oe){const n=Ht(s,mt,e.fragment),t=new Set;t.add(e.resolutionUniform),t.add(e.timeUniform),e.loopUniform&&t.add(e.loopUniform),e.stateful&&(t.add(e.passUniform??"uPass"),t.add(e.stateUniform??"uState"),t.add(e.gridUniform??"uGridSize"));for(const l of e.params)t.add(l.uniform);const a=qt(s,n,Array.from(t));Kn.set(e.id,{program:n,uniforms:a});const o={},i={};for(const l of e.params)o[l.id]=l,i[l.id]=l.type==="seed"?Math.floor(Math.random()*1e6):l.value;Qn[e.id]=o,ae[e.id]={...i},Jn[e.id]={...i}}let L=oe[0],an={},gn=performance.now(),qe=null,Ve=null,X=null,we=[],et=0,Pe=null,be=!1;function ta(){const e=["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm","video/mp4"];for(const n of e)if(MediaRecorder.isTypeSupported(n))return n;return""}function ra(e){return e.startsWith("video/mp4")?"mp4":"webm"}function aa(e){const n=Math.floor(e/1e3),t=String(Math.floor(n/60)).padStart(2,"0"),a=String(n%60).padStart(2,"0");return`${t}:${a}`}function oa(){const e=document.getElementById("rec-badge");!e||!be||(e.textContent=`⏺ ${aa(performance.now()-et)}`)}function ia(){const e=ta();if(!e){alert("Recording is not supported in this browser.");return}const n=k.captureStream(60);we=[],X=new MediaRecorder(n,{mimeType:e,videoBitsPerSecond:16e6}),X.ondataavailable=t=>{t.data.size>0&&we.push(t.data)},X.onstop=()=>{const t=ra(e),a=new Blob(we,{type:e}),o=URL.createObjectURL(a),i=document.createElement("a");i.href=o,i.download=`${L.id}-${Date.now()}.${t}`,i.click(),URL.revokeObjectURL(o),we=[]},X.start(500),be=!0,et=performance.now(),nt(),Pe=window.setInterval(oa,250)}function la(){X&&X.state!=="inactive"&&X.stop(),be=!1,Pe!==null&&(clearInterval(Pe),Pe=null),nt()}function sa(){be?la():ia()}function nt(){const e=document.getElementById("rec-btn"),n=document.getElementById("rec-badge");!e||!n||(be?(e.textContent="Stop",e.classList.add("recording"),n.classList.remove("hidden")):(e.textContent="Record",e.classList.remove("recording"),n.classList.add("hidden"),n.textContent=""))}let Ne=!1;function on(e){We.classList.toggle("hidden",e)}function ua(){on(!1),Ve!==null&&window.clearTimeout(Ve),Ve=window.setTimeout(()=>{on(!0)},2500)}function tt(){const e=document.body.classList.contains("sidebar-collapsed");We.textContent=e?">>":"<<"}function fa(){var e;(e=Ue.parentElement)==null||e.classList.remove("hidden"),qe!==null&&window.clearTimeout(qe),qe=window.setTimeout(()=>{var n;(n=Ue.parentElement)==null||n.classList.add("hidden")},1e4)}function Pn(e){const n=s.createTexture();if(!n)throw new Error("Failed to create state texture");return s.bindTexture(s.TEXTURE_2D,n),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_MIN_FILTER,s.NEAREST),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_MAG_FILTER,s.NEAREST),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_S,s.REPEAT),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_T,s.REPEAT),s.texImage2D(s.TEXTURE_2D,0,s.RGBA16F,e,e,0,s.RGBA,s.HALF_FLOAT,null),s.bindTexture(s.TEXTURE_2D,null),n}function ca(e){const n=Pn(e),t=Pn(e),a=s.createFramebuffer(),o=s.createFramebuffer();if(!a||!o)throw new Error("Failed to create framebuffer");return s.bindFramebuffer(s.FRAMEBUFFER,a),s.framebufferTexture2D(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,n,0),s.bindFramebuffer(s.FRAMEBUFFER,o),s.framebufferTexture2D(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,t,0),s.bindFramebuffer(s.FRAMEBUFFER,null),{size:e,textures:[n,t],fbos:[a,o],index:0,needsInit:!0}}function Sn(e){if(!e.stateful)return null;let n=rn.get(e.id);const t=e.bufferSize??192;return(!n||n.size!==t)&&(n=ca(t),rn.set(e.id,n)),n}function he(e){const n=rn.get(e);n&&(n.needsInit=!0)}function ma(e,n){let t=n;return e.min!==void 0&&(t=Math.max(e.min,t)),e.max!==void 0&&(t=Math.min(e.max,t)),e.type==="int"&&(t=Math.round(t)),t}function rt(e,n){if(e.type==="int")return String(Math.round(n));const t=e.step??.01,a=t<1?Math.min(4,Math.max(2,Math.ceil(-Math.log10(t)))):0;return n.toFixed(a)}function pa(e){return e.length===1?e.toLowerCase():e}function da(e){return e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement||e instanceof HTMLSelectElement}function ve(e,n,t,a=!0){const o=Qn[e][n];if(!o)return;const i=ma(o,t);if(ae[e][n]=i,e===L.id&&a){const l=an[n];l!=null&&l.range&&(l.range.value=String(i)),l!=null&&l.number&&(l.number.value=rt(o,i))}}function ha(e){var n;for(const t of((n=oe.find(a=>a.id===e))==null?void 0:n.params)??[])t.type==="seed"&&ve(e,t.id,Math.floor(Math.random()*1e6));he(e)}function va(e){const n=Jn[e];for(const[t,a]of Object.entries(n))ve(e,t,a,!0);he(e)}function ya(){ee.innerHTML="";for(const e of oe){const n=document.createElement("option");n.value=e.id,n.textContent=e.name,ee.appendChild(n)}ee.addEventListener("change",()=>{at(ee.value)})}function ba(e){Fe.innerHTML="";const n=document.createElement("button");if(n.className="ghost small",n.textContent="Reset",n.addEventListener("click",()=>va(e.id)),Fe.appendChild(n),e.params.some(a=>a.type==="seed")){const a=document.createElement("button");a.className="ghost small",a.textContent="Reseed",a.addEventListener("click",()=>ha(e.id)),Fe.appendChild(a)}}function ga(e){tn.innerHTML="",an={};for(const n of e.params){if(n.type==="seed")continue;const t=document.createElement("div");t.className="control";const a=document.createElement("div");a.className="control-header";const o=document.createElement("label");if(o.textContent=n.label,a.appendChild(o),n.key){const p=document.createElement("span");p.className="key-cap",p.textContent=`${n.key.inc.toUpperCase()}/${n.key.dec.toUpperCase()}`,a.appendChild(p)}const i=document.createElement("div");i.className="control-inputs";const l=document.createElement("input");l.type="range",l.min=String(n.min??0),l.max=String(n.max??1),l.step=String(n.step??(n.type==="int"?1:.01)),l.value=String(ae[e.id][n.id]),l.addEventListener("input",p=>{const y=Number(p.target.value);Number.isNaN(y)||ve(e.id,n.id,y)});const u=document.createElement("input");u.type="number",u.min=l.min,u.max=l.max,u.step=l.step,u.value=rt(n,ae[e.id][n.id]),u.addEventListener("input",p=>{const y=Number(p.target.value);Number.isNaN(y)||ve(e.id,n.id,y)}),i.appendChild(l),i.appendChild(u),t.appendChild(a),t.appendChild(i),tn.appendChild(t),an[n.id]={range:l,number:u}}}function Sa(e){se.innerHTML="";for(const n of e.params){if(!n.key||n.type==="seed")continue;const t=document.createElement("div");t.className="key-row",t.textContent=`${n.key.inc.toUpperCase()}/${n.key.dec.toUpperCase()}  ${n.label}`,se.appendChild(t)}se.childElementCount||(se.textContent="No mapped keys for this scene.")}function at(e){const n=oe.find(t=>t.id===e);n&&(L=n,n.stateful&&(Sn(n),he(n.id)),Ue.textContent=n.name,$n.textContent=n.description,fa(),ba(n),ga(n),Sa(n),ee.value=n.id)}function xa(e){if(da(e.target))return;const n=pa(e.key),t=L.params;for(const a of t){if(!a.key||a.type==="seed")continue;const o=n===a.key.inc,i=n===a.key.dec;if(!o&&!i)continue;const l=e.shiftKey&&a.key.shiftStep?a.key.shiftStep:a.key.step,p=ae[L.id][a.id]+l*(o?1:-1);ve(L.id,a.id,p),e.preventDefault();break}}function Ce(e,n,t,a,o){const i=ae[e.id],l=n.uniforms,u=l[e.resolutionUniform];u&&s.uniform2f(u,a,o);const p=l[e.timeUniform];if(p)if(e.timeMode==="phase"){const b=e.loopDuration??8,w=t%b/b;s.uniform1f(p,w)}else if(e.timeMode==="looped"){const b=e.loopDuration??8,w=t%b;if(s.uniform1f(p,w),e.loopUniform){const S=l[e.loopUniform];S&&s.uniform1f(S,b)}}else s.uniform1f(p,t);const y={};for(const b of e.params){const w=l[b.uniform],S=i[b.id];if(b.component!==void 0){const _=y[b.uniform]??[0,0,0];_[b.component]=S,y[b.uniform]=_;continue}w&&(b.type==="int"?s.uniform1i(w,Math.round(S)):s.uniform1f(w,S))}for(const[b,w]of Object.entries(y)){const S=l[b];S&&s.uniform3f(S,w[0],w[1],w[2])}}function je(e,n,t,a){const o=n.uniforms[e.passUniform??"uPass"];o&&s.uniform1i(o,a);const i=n.uniforms[e.gridUniform??"uGridSize"];i&&s.uniform2f(i,t.size,t.size);const l=n.uniforms[e.stateUniform??"uState"];l&&s.uniform1i(l,0)}function ot(e,n,t,a){const o=Kn.get(e.id);if(o){if(e.stateful){if(!na)return;const i=Sn(e);if(!i)return;const l=()=>i.textures[i.index],u=()=>i.fbos[(i.index+1)%2];s.useProgram(o.program),s.bindVertexArray(Fn),i.needsInit&&(s.bindFramebuffer(s.FRAMEBUFFER,u()),s.viewport(0,0,i.size,i.size),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,l()),Ce(e,o,n,t,a),je(e,o,i,2),s.drawArrays(s.TRIANGLES,0,3),i.index=(i.index+1)%2,i.needsInit=!1),s.bindFramebuffer(s.FRAMEBUFFER,u()),s.viewport(0,0,i.size,i.size),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,l()),Ce(e,o,n,t,a),je(e,o,i,0),s.drawArrays(s.TRIANGLES,0,3),i.index=(i.index+1)%2,s.bindFramebuffer(s.FRAMEBUFFER,null),s.viewport(0,0,t,a),s.clearColor(0,0,0,1),s.clear(s.COLOR_BUFFER_BIT),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,l()),Ce(e,o,n,t,a),je(e,o,i,1),s.drawArrays(s.TRIANGLES,0,3);return}s.viewport(0,0,t,a),s.clearColor(0,0,0,1),s.clear(s.COLOR_BUFFER_BIT),s.useProgram(o.program),s.bindVertexArray(Fn),Ce(e,o,n,t,a),s.drawArrays(s.TRIANGLES,0,3)}}function xn(e){if(Ne)return;const n=(e-gn)/1e3,{width:t,height:a}=Vt(k);ot(L,n,t,a),requestAnimationFrame(xn)}function wa(){return new Promise(e=>requestAnimationFrame(()=>e()))}async function Ca(e,n,t,a,o){if(typeof VideoEncoder>"u"||typeof VideoFrame>"u"){alert("Offline rendering requires the WebCodecs API (Chrome/Edge 94+, Safari 16.4+).");return}const i={codec:"avc1.640028",width:a,height:o,bitrate:16e6,framerate:t};if(!(await VideoEncoder.isConfigSupported(i)).supported){alert("H.264 video encoding is not supported on this device.");return}let u;try{u=await window.showSaveFilePicker({suggestedName:`${e.id}-${a}x${o}-${t}fps-${n}s.mp4`,types:[{description:"MP4 Video",accept:{"video/mp4":[".mp4"]}}]})}catch{return}const p=await u.createWritable(),y=document.getElementById("offline-btn"),b=document.getElementById("offline-progress"),w=document.getElementById("offline-bar"),S=document.getElementById("offline-status");y&&(y.disabled=!0),b==null||b.classList.remove("hidden"),S==null||S.classList.remove("hidden"),Ne=!0,e.stateful&&(Sn(e),he(e.id));const _=k.width,N=k.height,z=k.style.width,$=k.style.height;k.width=a,k.height=o,k.style.width=`${a}px`,k.style.height=`${o}px`;const it=new On(p),wn=new Jr({target:it,video:{codec:"avc",width:a,height:o},fastStart:!1});let ge=null;const K=new VideoEncoder({output:(C,_e)=>wn.addVideoChunk(C,_e??void 0),error:C=>{ge=C}});K.configure(i);const Se=Math.ceil(n*t),Cn=Math.round(1e6/t),lt=t*2,st=t*30,ut=performance.now();let xe=!1;const Tn=C=>{C.preventDefault(),xe=!0};k.addEventListener("webglcontextlost",Tn);for(let C=0;C<Se&&!(ge||xe);C++){const _e=C/t;ot(e,_e,a,o),s.finish();const kn=new VideoFrame(k,{timestamp:C*Cn,duration:Cn});K.encode(kn,{keyFrame:C%lt===0}),kn.close(),C>0&&C%st===0&&await K.flush();const ft=(C+1)/Se*100;if(w&&(w.style.width=`${ft}%`),C%30===0){const ct=(performance.now()-ut)/1e3/(C+1)*(Se-C-1);S&&(S.textContent=`Frame ${C+1}/${Se}  —  ~${Math.ceil(ct)}s left`)}K.encodeQueueSize>10&&await new Promise(Rn=>setTimeout(Rn,1)),await wa()}k.removeEventListener("webglcontextlost",Tn);try{await K.flush(),K.close(),wn.finalize(),await p.close()}catch(C){console.error("Finalization failed:",C);try{await p.close()}catch{}}k.style.width=z,k.style.height=$,k.width=_,k.height=N,e.stateful&&he(e.id),Ne=!1,gn=performance.now(),requestAnimationFrame(xn),y&&(y.disabled=!1);const Le=ge?ge.message:null;S&&(xe?S.textContent="Context lost — partial video saved.":Le?S.textContent=`Error: ${Le}`:S.textContent="Done!"),setTimeout(()=>{b==null||b.classList.add("hidden"),S==null||S.classList.add("hidden"),w&&(w.style.width="0%")},xe||!!Le?8e3:3e3)}We.addEventListener("click",()=>{document.body.classList.toggle("sidebar-collapsed"),tt()});bn.addEventListener("mousemove",()=>{ua()});document.addEventListener("keydown",xa);document.addEventListener("visibilitychange",()=>{document.hidden||(gn=performance.now())});var Dn;(Dn=document.getElementById("rec-btn"))==null||Dn.addEventListener("click",sa);var Mn;(Mn=document.getElementById("offline-btn"))==null||Mn.addEventListener("click",()=>{if(Ne)return;const e=document.getElementById("offline-duration"),n=document.getElementById("offline-fps"),t=document.getElementById("offline-res"),a=Math.max(1,Math.min(3600,Number((e==null?void 0:e.value)??10))),o=Number((n==null?void 0:n.value)??60),[i,l]=((t==null?void 0:t.value)??"1920x1080").split("x").map(Number);Ca(L,a,o,i,l)});ya();at(L.id);tt();on(!0);requestAnimationFrame(xn);
