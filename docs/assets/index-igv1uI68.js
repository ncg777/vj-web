(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))o(a);new MutationObserver(a=>{for(const l of a)if(l.type==="childList")for(const r of l.addedNodes)r.tagName==="LINK"&&r.rel==="modulepreload"&&o(r)}).observe(document,{childList:!0,subtree:!0});function n(a){const l={};return a.integrity&&(l.integrity=a.integrity),a.referrerPolicy&&(l.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?l.credentials="include":a.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function o(a){if(a.ep)return;a.ep=!0;const l=n(a);fetch(a.href,l)}})();const mn=`#version 300 es
precision highp float;

const vec2 verts[3] = vec2[](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
);

void main() {
  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
`,pn=`#version 300 es
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
`,dn=`#version 300 es
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
`,hn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 iResolution;
uniform float iTime;

uniform float uLoopDuration;
uniform float uSpeed;
uniform float uTwist;
uniform float uNoiseScale;
uniform float uNoiseAmp;
uniform float uColorCycle;
uniform float uFogDensity;
uniform vec3 uBaseColor;

const float TAU = 6.28318530718;

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

float fbm(vec2 p) {
  float sum = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    sum += amp * noise(p);
    p *= 2.0;
    amp *= 0.5;
  }
  return sum;
}

vec3 tunnelPalette(float t) {
  return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.17, 0.36)));
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

  float phase = mod(iTime, uLoopDuration) / max(uLoopDuration, 0.001);
  float theta = phase * TAU;
  vec2 loopA = vec2(cos(theta), sin(theta));
  vec2 loopB = vec2(cos(2.0 * theta + 0.7), sin(3.0 * theta - 0.5));

  float baseR = length(uv);
  float a = atan(uv.y, uv.x);
  float twist = uTwist * (0.65 + 0.35 * loopA.x);
  a += twist * baseR + 0.25 * loopB.y;

  vec2 dir = vec2(cos(a), sin(a));
  vec2 flow = loopA * (0.7 + 0.3 * uSpeed) + loopB * (0.25 + 0.2 * uSpeed);
  vec2 np = dir * (0.9 * uNoiseScale) + uv * (2.2 * uNoiseScale) + flow * (0.75 * uNoiseScale);
  float n = fbm(np);
  float n2 = fbm(np * 1.9 + loopB * 3.0 + loopA.yx * 1.7);

  float r = baseR + uNoiseAmp * ((n - 0.5) * 1.4 + (n2 - 0.5) * 0.8);
  float lane = 0.5 + 0.5 * sin(a * 9.0 + n * 4.0 + 2.8 * loopB.x);
  float rings = 0.5 + 0.5 * cos(r * 22.0 - n2 * 5.5 + 2.2 * loopA.y);
  float tunnel = smoothstep(0.28, 0.92, lane * 0.75 + rings * 0.95);

  float huePhase = a / TAU + 0.25 * n + 0.12 * n2 + 0.12 * uColorCycle * loopA.y;
  vec3 dynamicHue = tunnelPalette(huePhase);
  vec3 oilHue = tunnelPalette(huePhase + 0.12 * loopB.x + 0.08 * loopA.y);
  vec3 col = mix(uBaseColor, dynamicHue, 0.45);
  col = mix(col, oilHue, 0.45 + 0.25 * lane);
  col = mix(col, vec3(1.0), 0.55 * tunnel);

  float fogBase = exp(-baseR * uFogDensity);
  float glowBase = pow(fogBase, 1.8);
  float e = 0.003 * max(0.5, uNoiseScale);
  vec2 grad;
  grad.x = fbm(np + vec2(e, 0.0)) - fbm(np - vec2(e, 0.0));
  grad.y = fbm(np + vec2(0.0, e)) - fbm(np - vec2(0.0, e));
  vec2 normal2D = normalize(grad + vec2(1e-6));
  vec2 uvR = uv + normal2D * (0.028 + 0.02 * glowBase);

  float rR = length(uvR);
  float aR = atan(uvR.y, uvR.x) + twist * rR + 0.25 * loopB.x;
  vec2 dirR = vec2(cos(aR), sin(aR));
  vec2 npR = dirR * (0.9 * uNoiseScale) + uvR * (2.2 * uNoiseScale) + flow * (0.75 * uNoiseScale);
  float nR = fbm(npR);
  float nR2 = fbm(npR * 1.9 + loopB * 3.0 + loopA.yx * 1.7);
  float laneR = 0.5 + 0.5 * sin(aR * 9.0 + nR * 4.0 + 2.8 * loopA.x);
  float ringsR = 0.5 + 0.5 * cos(rR * 22.0 - nR2 * 5.5 + 2.2 * loopB.y);
  vec3 colR = mix(uBaseColor, tunnelPalette(aR / TAU + 0.25 * nR + 0.12 * nR2 + 0.12 * uColorCycle * loopB.x), 0.45);
  colR = mix(colR, tunnelPalette(aR / TAU + 0.14 * loopA.x + 0.08 * nR), 0.45 + 0.25 * laneR);
  colR = mix(colR, vec3(1.0), 0.45 * smoothstep(0.28, 0.92, laneR * 0.75 + ringsR * 0.95));

  col = mix(col, colR, 0.58);
  col *= mix(0.55, 1.7, fogBase);
  col += glowBase * 0.32 * (0.5 * dynamicHue + 0.5 * oilHue);

  col = clamp(col, 0.0, 1.0);
  outColor = vec4(col, 1.0);
}
`,vn=`#version 300 es
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
`,yn=`#version 300 es
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
`,bn=`#version 300 es
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
`,gn=`#version 300 es
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
`,Sn=`#version 300 es
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
`,xn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform float uTime;
uniform vec2 uResolution;
uniform int uIterations;
uniform float uRotateSpeed;
uniform float uFoldOffset;
uniform float uStepScale;
uniform float uGlow;
uniform float uCameraDistance;
uniform float uCameraSpin;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;
uniform float uColorMix;
uniform float uAlphaGain;

vec3 palette(float d) {
    vec3 base = mix(uColorPrimary, uColorSecondary, clamp(d, 0.0, 1.0));
    return mix(base, base * base, uColorMix);
}

vec2 rotate2D(vec2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, s, -s, c) * p;
}

float mapFunc(vec3 p) {
    float t = uTime * uRotateSpeed;
    for (int i = 0; i < 64; ++i) {
        if (i >= uIterations) break;
        p.xz = rotate2D(p.xz, t);
        p.xy = rotate2D(p.xy, t * 1.89);
        p.xz = abs(p.xz);
        p.xz -= vec2(uFoldOffset);
    }
    return dot(sign(p), p) / uStepScale;
}

vec4 rm(vec3 ro, vec3 rd) {
    float t = 0.0;
    vec3 col = vec3(0.0);
    float d = 1.0;

    for (int i = 0; i < 72; ++i) {
        vec3 p = ro + rd * t;
        d = mapFunc(p) * 0.5;

        if (d < 0.02) break;
        if (d > 120.0) break;

        float shade = length(p) * 0.08;
        col += palette(shade) * uGlow / (400.0 * d);
        t += d;
    }

    float alpha = 1.0 / (max(d, 0.01) * 100.0);
    return vec4(col, clamp(alpha * uAlphaGain, 0.0, 1.0));
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = (fragCoord - (uResolution * 0.5)) / uResolution.x;

    vec3 ro = vec3(0.0, 0.0, -uCameraDistance);
    ro.xz = rotate2D(ro.xz, uTime * uCameraSpin);

    vec3 cf = normalize(-ro);
    vec3 cs = normalize(cross(cf, vec3(0.0, 1.0, 0.0)));
    vec3 cu = normalize(cross(cf, cs));

    vec3 uuv = ro + cf * 3.0 + uv.x * cs + uv.y * cu;
    vec3 rd = normalize(uuv - ro);

    outColor = rm(ro, rd);
}
`,wn=`#version 300 es
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
}
`,Cn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uTileScale;
uniform float uIntensity;
uniform float uContrast;
uniform float uWaveShift;
uniform vec3 uTint;

const float TAU = 6.28318530718;
const int MAX_ITER = 5;

void main() {
  float time = uTime * uTimeScale + 23.0;
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  vec2 p = mod(uv * TAU * uTileScale, TAU) - 250.0;
  vec2 i = p;
  float c = 1.0;
  float inten = 0.005;

  for (int n = 0; n < MAX_ITER; n++) {
    float t = time * (1.0 - (3.5 / float(n + 1))) + uWaveShift;
    i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
    vec2 denom = vec2(p.x / (sin(i.x + t) / inten), p.y / (cos(i.y + t) / inten));
    c += 1.0 / length(denom);
  }

  c /= float(MAX_ITER);
  c = 1.17 - pow(c, 1.4);
  float cAdj = pow(clamp(c, 0.0, 1.0), max(0.1, uContrast));
  vec3 color = vec3(pow(abs(cAdj), 8.0)) * uIntensity;
  color = clamp(color + uTint, 0.0, 1.0);

  outColor = vec4(color, 1.0);
}
`,Tn=`#version 300 es
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
`,kn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uTurbulence;
uniform float uCloudHeight;
uniform float uStepBase;
uniform float uStepScale;
uniform float uHueShift;
uniform float uHueSpeed;
uniform float uIntensity;

// --- new degrees of freedom ---
uniform float uWarpAmp;       // amplitude of horizon sine warp
uniform float uWarpFreq;      // spatial frequency of the warp
uniform float uWarpSpeed;     // temporal speed of the warp animation
uniform float uWarpHarmonics; // add harmonic overtones for richer shape
uniform float uOrbitRadius;   // radius of circular horizon orbit
uniform float uOrbitSpeed;    // angular speed of the orbit
uniform float uOrbitEcc;      // eccentricity (ellipse stretch)
uniform float uTiltAngle;     // tilt the horizon plane
uniform float uCloudDensity;  // per-step density multiplier
uniform float uFogFalloff;    // controls exponential fog rolloff
uniform float uColorSep;      // chromatic separation between RGB channels

/*
   Smooth periodic horizon function.
   The horizon is no longer a single flat plane at y=0.
   Instead it follows a warped surface whose height varies with (x,z)
   and is additionally orbited in the (y, z) plane over time.
*/
float horizonHeight(vec3 p, float t) {
    // Base warp: sum of harmonics of a spatial sine wave
    float h = 0.0;
    float amp = uWarpAmp;
    float freq = uWarpFreq;
    for (float k = 1.0; k <= 5.0; k += 1.0) {
        if (k > uWarpHarmonics) break;
        // phase varies with time, direction alternates each harmonic
        float phase = uWarpSpeed * t * (0.7 + 0.3 * k) + k * 1.37;
        h += amp * sin(freq * p.x + phase)
           * cos(freq * 0.6 * p.z + phase * 0.8);
        amp  *= 0.55;   // each harmonic is weaker
        freq *= 1.8;    // and higher frequency
    }

    // Orbit: translate the center of the horizon along an elliptical path
    float orbitAngle = uOrbitSpeed * t;
    float oy = uOrbitRadius * sin(orbitAngle);
    float oz = uOrbitRadius * uOrbitEcc * cos(orbitAngle);

    // Tilt: rotate horizon normal by uTiltAngle around the x-axis
    float ct = cos(uTiltAngle);
    float st = sin(uTiltAngle);
    float tiltedY = ct * (p.y - oy) - st * (p.z - oz);

    return h - tiltedY;   // positive = inside clouds
}

void main() {
    vec2 I = gl_FragCoord.xy;
    float t = uTime * uTimeScale;
    float z = 0.0;
    float d = 0.0;
    float s = 0.0;
    vec4 O = vec4(0.0);

    vec3 rd = normalize(vec3(I + I, 0.0) - uResolution.xyy);

    for (float i = 0.0; i < 100.0; i++) {
        vec3 p = z * rd;

        // Volumetric turbulence (same family as sunset_plus)
        for (d = 5.0; d < 200.0; d += d) {
            p += uTurbulence * 0.6 * sin(p.yzx * d - 0.2 * t) / d;
        }

        float height = max(0.05, uCloudHeight);

        // Sample the periodic horizon surface
        s = horizonHeight(p, t);
        s = height - abs(s);            // cloud shell thickness

        // Adaptive ray step
        z += d = uStepBase + max(s, -s * 0.2) / uStepScale;

        // Colour with per-channel separation for richer sunsets
        vec4 phaseR = vec4(3.0, 4.0, 5.0, 0.0) + uHueShift + uHueSpeed * t;
        vec4 phaseG = phaseR + uColorSep;
        vec4 phaseB = phaseR - uColorSep;

        float envelope = exp(s / 0.1) / d;
        float density  = uCloudDensity * envelope;

        float fogAtt = exp(-z * uFogFalloff);

        vec4 cR = (cos(s / 0.07 + p.x + 0.5 * t - phaseR) + 1.5) * density;
        vec4 cG = (cos(s / 0.07 + p.x + 0.5 * t - phaseG) + 1.5) * density;
        vec4 cB = (cos(s / 0.07 + p.x + 0.5 * t - phaseB) + 1.5) * density;

        O.r += cR.r * fogAtt;
        O.g += cG.g * fogAtt;
        O.b += cB.b * fogAtt;
    }

    O = tanh(O * O / 4e8);
    outColor = vec4(O.rgb * uIntensity, 1.0);
}
`,Rn=`#version 300 es
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
`,Bn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uSpin;
uniform float uTwist;
uniform float uWarp;
uniform float uPulse;
uniform float uBoltDensity;
uniform float uBoltSharpness;
uniform float uBoltIntensity;
uniform float uArcSteps;
uniform float uCoreSize;
uniform float uCoreGlow;
uniform float uNoiseAmp;
uniform float uPaletteShift;
uniform float uSeed;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;
uniform vec3 uColorAccent;

const float TAU = 6.28318530718;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 palette(float t) {
  return uColorPrimary + uColorSecondary * cos(TAU * (uColorAccent * t + uPaletteShift));
}

void main() {
  vec2 uv = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;
  uv *= uZoom;

  float t = uTime * uTimeScale;
  float spin = t * uSpin;
  mat2 rot = mat2(cos(spin), -sin(spin), sin(spin), cos(spin));
  uv = rot * uv;

  vec2 twist = vec2(
    sin(uv.y * (2.0 + uTwist) + t),
    cos(uv.x * (2.5 + uTwist) - t)
  );
  uv += twist * 0.15 * uWarp;

  float radius = length(uv) + 1e-4;
  float angle = atan(uv.y, uv.x);

  float bolts = 0.0;
  for (int i = 0; i < 80; i++) {
    float fi = float(i) + 1.0;
    float mask = step(fi, uArcSteps);
    float phase = t * (0.8 + uPulse) + fi * 0.37 + uSeed * 0.001;
    float wave = sin(angle * (uBoltDensity + fi * 0.12) + phase)
      + cos(radius * (uBoltDensity * 1.7) - phase);
    float d = abs(wave) + 0.12 + radius * uBoltSharpness * 0.35;
    float contribution = uBoltIntensity / (d * d);
    bolts += mask * contribution;
  }

  vec2 q = uv;
  float spark = 0.0;
  for (int i = 0; i < 7; i++) {
    float fi = float(i) + 1.0;
    float denom = max(0.25, dot(q, q));
    q = abs(q) / denom - vec2(0.55, 0.35) * uWarp;
    spark += exp(-fi) * (0.5 + 0.5 * sin(fi * 2.3 + t + q.x * 4.0 + q.y * 5.0));
  }

  float grain = hash(uv * (12.0 + uSeed * 0.001) + t);
  spark += uNoiseAmp * (grain - 0.5);

  float core = uCoreGlow / (abs(radius - uCoreSize) + 0.02);

  float energy = bolts * 0.03 + spark * 0.9 + core * 0.6;
  energy *= smoothstep(1.8, 0.2, radius);

  vec3 col = palette(radius + spark * 0.35 + t * 0.1);
  col *= energy;
  col += vec3(bolts * 0.02);
  col += vec3(core * 0.15);

  col = pow(max(col, 0.0), vec3(0.75));
  outColor = vec4(col, 1.0);
}
`,An=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uCloudScale;
uniform float uCloudSpeed;
uniform float uCloudDensity;
uniform float uCloudDetail;
uniform float uBoltLengthMin;
uniform float uBoltLengthMax;
uniform float uBoltWidth;
uniform float uBoltWiggle;
uniform float uBoltNoiseScale;
uniform float uBoltNoiseSpeed;
uniform float uBoltBranching;
uniform float uBoltIntensity;
uniform float uFlickerSpeed;
uniform float uCloudIllumination;
uniform float uSeed;
uniform int   uBoltCount;
uniform int   uNoiseOctaves;
uniform vec3  uCloudColor;
uniform vec3  uLightningColor;

const float TAU = 6.28318530718;
const float PI  = 3.14159265359;

/* ---- helpers ---- */

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p, int octaves) {
  float value = 0.0, amplitude = 0.5, total = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= octaves) break;
    value += noise(p) * amplitude;
    total += amplitude;
    p = p * 2.0 + vec2(1.7, 9.2);
    amplitude *= 0.5;
  }
  return value / max(total, 0.001);
}

mat2 rot(float a) {
  float c = cos(a), s = sin(a);
  return mat2(c, s, -s, c);
}

float segSDF(vec2 p, vec2 a, vec2 b, float w) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - w;
}

/* ---- lightning bolt with branches ---- */

vec3 bolt(vec2 uv, vec2 start, vec2 end, float eventSeed, float time) {
  vec3 result = vec3(0.0);

  vec2  dir  = end - start;
  float len  = length(dir);
  vec2  n    = normalize(dir);
  vec2  perp = vec2(-n.y, n.x);

  const int SEGS = 8;
  vec2 prev = start;

  for (int s = 0; s < SEGS; s++) {
    float t = float(s + 1) / float(SEGS);
    vec2 basePos = start + dir * t;

    float nv = noise(vec2(t * uBoltNoiseScale + eventSeed * 7.0,
                          time * uBoltNoiseSpeed + eventSeed * 3.0)) * 2.0 - 1.0;
    float taper = t * (1.0 - t) * 4.0;
    basePos += perp * nv * uBoltWiggle * len * taper;

    float d    = segSDF(uv, prev, basePos, uBoltWidth);
    float glow = uBoltIntensity / max(d, 0.001);
    glow = clamp(1.0 - exp(-glow * 0.01), 0.0, 1.0);
    result += glow * uLightningColor;

    /* branch */
    if (uBoltBranching > 0.0 && s > 0 && s < SEGS - 1) {
      float bc = hash(vec2(float(s) + eventSeed * 11.0, 43.0));
      if (bc < uBoltBranching) {
        float ba2 = (hash(vec2(float(s) + eventSeed, 67.0)) - 0.5) * PI * 0.6;
        float bl  = len * 0.25 * hash(vec2(float(s) + eventSeed, 89.0));
        vec2  bd  = rot(ba2) * n;
        vec2  be  = prev + bd * bl;
        vec2  bm  = mix(prev, be, 0.5);
        float bnv = noise(vec2(float(s) * 3.0 + eventSeed * 5.0,
                               time * uBoltNoiseSpeed * 0.7)) * 2.0 - 1.0;
        bm += vec2(-bd.y, bd.x) * bnv * uBoltWiggle * bl;

        float d1 = segSDF(uv, prev, bm, uBoltWidth * 0.6);
        float d2 = segSDF(uv, bm,  be,  uBoltWidth * 0.4);
        float bg = uBoltIntensity * 0.5 / max(min(d1, d2), 0.001);
        bg = clamp(1.0 - exp(-bg * 0.008), 0.0, 1.0);
        result += bg * uLightningColor * 0.7;
      }
    }
    prev = basePos;
  }
  return result;
}

/* ---- main ---- */

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
  uv *= uZoom;

  float aspect = uResolution.x / uResolution.y;
  float time   = uTime * uTimeScale;

  /* ---- cloud layer (top-down view) ---- */
  vec2  cUV    = uv * uCloudScale + vec2(time * uCloudSpeed * 0.03,
                                          time * uCloudSpeed * 0.02);
  float clouds = fbm(cUV, uNoiseOctaves);
  float detail = fbm(cUV * 2.5 + vec2(time * uCloudSpeed * 0.01, 0.0),
                      max(uNoiseOctaves - 1, 1));
  clouds = mix(clouds, clouds * detail, uCloudDetail);
  clouds = smoothstep(0.5 - uCloudDensity * 0.5, 0.5 + uCloudDensity * 0.3, clouds);

  vec3 col = uCloudColor * clouds;

  /* ---- lightning bolts ---- */
  float totalIllum = 0.0;
  vec3  lightCol   = vec3(0.0);

  int count = max(uBoltCount, 1);
  for (int i = 0; i < 12; i++) {
    if (i >= count) break;
    float fi = float(i);

    float laneRate  = uFlickerSpeed * (0.5 + 0.8 * hash(vec2(fi + uSeed, 7.3)));
    float lanePhase = hash(vec2(fi + uSeed * 0.5, 13.1)) * 30.0;
    float laneTime  = time * laneRate + lanePhase;
    float eventId   = floor(laneTime);
    float localT    = fract(laneTime);

    float spawnChance = hash(vec2(eventId + fi * 3.7, 17.0 + uSeed));
    if (spawnChance < 0.55) continue;

    float flash   = smoothstep(0.0, 0.02, localT)
                  * (1.0 - smoothstep(0.05, 0.4, localT));
    float flicker = 1.0 - 0.3 * smoothstep(0.0, 1.0,
                        sin(localT * 40.0 + fi * 10.0));
    flash *= flicker;
    if (flash < 0.001) continue;

    vec2 startPos = vec2(
      hash(vec2(eventId + fi, 21.0 + uSeed)) * 2.0 - 1.0,
      hash(vec2(eventId + fi, 25.0 + uSeed)) * 2.0 - 1.0
    ) * vec2(aspect * 0.45, 0.45);

    float boltLen = mix(uBoltLengthMin, uBoltLengthMax,
                        hash(vec2(eventId + fi, 37.0 + uSeed)));
    float angle   = hash(vec2(eventId + fi, 29.0 + uSeed)) * TAU;
    vec2  endPos  = startPos + vec2(cos(angle), sin(angle)) * boltLen;

    float eventSeed = hash(vec2(eventId, fi + 41.0 + uSeed));

    lightCol += bolt(uv, startPos, endPos, eventSeed, time) * flash;

    /* cloud illumination near the strike */
    float dMid  = length(uv - mix(startPos, endPos, 0.5));
    float illum = flash * uCloudIllumination * exp(-dMid * 4.0);
    totalIllum += illum;
  }

  col += uCloudColor * totalIllum * 1.5;
  col += lightCol;

  outColor = vec4(col, 1.0);
}
`,Fn=`#version 300 es
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
`,zn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uAuroraSpeed;
uniform float uAuroraScale;
uniform float uAuroraWarp;
uniform float uAuroraBase;
uniform float uAuroraStride;
uniform float uAuroraCurve;
uniform float uAuroraIntensity;
uniform float uTrailBlend;
uniform float uTrailFalloff;
uniform float uTrailFade;
uniform float uDitherStrength;
uniform float uHorizonFade;
uniform float uCamYaw;
uniform float uCamPitch;
uniform float uCamWobble;
uniform float uCamDistance;
uniform float uCamHeight;
uniform float uSkyStrength;
uniform float uStarDensity;
uniform float uStarIntensity;
uniform float uReflectionStrength;
uniform float uReflectionTint;
uniform float uReflectionFog;
uniform float uColorBand;
uniform float uColorSpeed;
uniform vec3 uAuroraColorA;
uniform vec3 uAuroraColorB;
uniform vec3 uAuroraColorC;
uniform vec3 uBgColorA;
uniform vec3 uBgColorB;
uniform int uAuroraSteps;

const float TAU = 6.28318530718;

mat2 mm2(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat2(c, s, -s, c);
}

mat2 m2 = mat2(0.95534, 0.29552, -0.29552, 0.95534);

float tri(float x) {
  return clamp(abs(fract(x) - 0.5), 0.01, 0.49);
}

vec2 tri2(vec2 p) {
  return vec2(tri(p.x) + tri(p.y), tri(p.y + tri(p.x)));
}

float triNoise2d(vec2 p, float spd, float time) {
  float z = 1.8;
  float z2 = 2.5;
  float rz = 0.0;
  p *= mm2(p.x * 0.06);
  vec2 bp = p;
  for (float i = 0.0; i < 5.0; i++) {
    vec2 dg = tri2(bp * 1.85) * 0.75;
    dg *= mm2(time * spd);
    p -= dg / z2;

    bp *= 1.3;
    z2 *= 0.45;
    z *= 0.42;
    p *= 1.21 + (rz - 1.0) * 0.02;

    rz += tri(p.x + tri(p.y)) * z;
    p *= -m2;
  }
  return clamp(1.0 / pow(rz * 29.0, 1.3), 0.0, 0.55);
}

float hash21(vec2 n) {
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec4 aurora(vec3 ro, vec3 rd, float time) {
  vec4 col = vec4(0.0);
  vec4 avgCol = vec4(0.0);
  int steps = max(uAuroraSteps, 1);

  for (int i = 0; i < 64; i++) {
    if (i >= steps) {
      break;
    }
    float fi = float(i);
    float of = uDitherStrength * hash21(gl_FragCoord.xy) * smoothstep(0.0, 15.0, fi);
    float pt = ((uAuroraBase + pow(fi, uAuroraCurve) * uAuroraStride) - ro.y) / (rd.y * 2.0 + 0.4);
    pt -= of;
    vec3 bpos = ro + pt * rd;
    vec2 p = bpos.zx;
    float rzt = triNoise2d(p * uAuroraScale, uAuroraSpeed, time);
    rzt = mix(rzt, pow(rzt, 1.0 + uAuroraWarp), uAuroraWarp);

    vec3 wave = sin(vec3(0.0, 2.1, 4.2) + fi * uColorBand + time * uColorSpeed);
    vec3 palette = mix(uAuroraColorA, uAuroraColorB, 0.5 + 0.5 * wave);
    palette = mix(palette, uAuroraColorC, rzt);

    vec4 col2 = vec4(palette * rzt * uAuroraIntensity, rzt);
    avgCol = mix(avgCol, col2, uTrailBlend);
    col += avgCol * exp2(-fi * uTrailFalloff - uTrailFade) * smoothstep(0.0, 5.0, fi);
  }

  col *= clamp(rd.y * 15.0 + 0.4, 0.0, 1.0);
  return col;
}

vec3 nmzHash33(vec3 q) {
  uvec3 p = uvec3(ivec3(q));
  p = p * uvec3(374761393U, 1103515245U, 668265263U) + p.zxy + p.yzx;
  p = p.yzx * (p.zxy ^ (p >> 3U));
  return vec3(p ^ (p >> 16U)) * (1.0 / vec3(0xffffffffU));
}

vec3 stars(vec3 p) {
  vec3 c = vec3(0.0);
  float res = uResolution.x * 1.0;

  for (float i = 0.0; i < 4.0; i++) {
    vec3 q = fract(p * (0.15 * res)) - 0.5;
    vec3 id = floor(p * (0.15 * res));
    vec2 rn = nmzHash33(id).xy;
    float c2 = 1.0 - smoothstep(0.0, 0.6, length(q));
    c2 *= step(rn.x, uStarDensity + i * i * 0.001);
    c += c2 * (mix(vec3(1.0, 0.49, 0.1), vec3(0.75, 0.9, 1.0), rn.y) * 0.1 + 0.9);
    p *= 1.3;
  }
  return c * c * uStarIntensity;
}

vec3 bg(vec3 rd) {
  float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd) * 0.5 + 0.5;
  sd = pow(sd, 5.0);
  vec3 col = mix(uBgColorA, uBgColorB, sd);
  return col * uSkyStrength;
}

void main() {
  vec2 q = gl_FragCoord.xy / uResolution.xy;
  vec2 p = q - 0.5;
  p.x *= uResolution.x / uResolution.y;

  float time = uTime * uTimeScale;

  vec3 ro = vec3(0.0, uCamHeight, -uCamDistance);
  vec3 rd = normalize(vec3(p, 1.3));
  rd.yz *= mm2(uCamPitch + sin(time * 0.05) * uCamWobble);
  rd.xz *= mm2(uCamYaw + sin(time * 0.05) * uCamWobble);

  vec3 col = vec3(0.0);
  float fade = smoothstep(0.0, uHorizonFade, abs(rd.y)) * 0.1 + 0.9;
  col = bg(rd) * fade;

  if (rd.y > 0.0) {
    vec4 aur = smoothstep(0.0, 1.5, aurora(ro, rd, time)) * fade;
    col += stars(rd);
    col = col * (1.0 - aur.a) + aur.rgb;
  } else {
    rd.y = abs(rd.y);
    col = bg(rd) * fade * uReflectionStrength;
    vec4 aur = smoothstep(0.0, 2.5, aurora(ro, rd, time));
    col += stars(rd) * 0.1;
    col = col * (1.0 - aur.a) + aur.rgb;
    vec3 pos = ro + ((0.5 - ro.y) / rd.y) * rd;
    float nz2 = triNoise2d(pos.xz * vec2(0.5, 0.7), 0.0, time);
    vec3 waterTint = mix(vec3(0.2, 0.25, 0.5) * 0.08, vec3(0.3, 0.3, 0.5) * 0.7, nz2 * 0.4);
    col += waterTint * uReflectionTint;
    col *= mix(1.0, exp(-abs(rd.y) * uReflectionFog), uReflectionStrength);
  }

  outColor = vec4(col, 1.0);
}
`,In=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

// --- UNIFORMS ---
uniform float uTime;
uniform vec2 uResolution;
uniform float uZoom;
uniform float uColorShift;
uniform int uIterations;
uniform float uDistort;
uniform float uRotateSpeed;
uniform float uMaxSteps;

// --- MATH HELPERS ---

// Rotation Matrix
mat2 rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

// Palette function for coloring (IQ style)
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

// The SDF (Signed Distance Function) - The Core Math
// This function returns the distance from point 'p' to the fractal surface
float map(vec3 p, float time, float distort) {
    float scale = 1.0;
    float offset = 1.0;
    
    // Recursive Folding Loop - Menger-like fractal
    for (int i = 0; i < 8; i++) {
        if (i >= uIterations) break;
        
        // Rotate space
        p.xy *= rot(time);
        p.yz *= rot(time * 0.7);
        
        // Folding - creates symmetry
        p = abs(p);
        
        // Menger fold
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;
        
        // Scale and translate
        p = p * distort - offset * (distort - 1.0);
        scale *= distort;
    }
    
    // Return distance to a box, scaled back
    float d = (length(p) - 1.5) / scale;
    return d;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    
    // 1. Setup Camera
    vec2 uv = (fragCoord - uResolution * 0.5) / min(uResolution.x, uResolution.y);
    uv *= uZoom;

    vec3 ro = vec3(0.0, 0.0, -3.0); // Ray Origin
    vec3 rd = normalize(vec3(uv, 1.0)); // Ray Direction

    float time = uTime * uRotateSpeed;

    // 2. Raymarching Loop
    float t = 0.0; // Total distance traveled
    float d = 0.0; // Distance to surface
    int maxSteps = int(uMaxSteps);

    vec3 col = vec3(0.0);
    vec3 p = ro;
    float glow = 0.0;

    for (int i = 0; i < 200; i++) {
        if (i >= maxSteps) break;

        p = ro + rd * t;
        d = map(p, time, uDistort); // Get distance to fractal

        // Accumulate glow based on proximity
        glow += 0.02 / (0.1 + abs(d));

        // If we hit the surface
        if (abs(d) < 0.001) {
            // Calculate Normal
            vec2 e = vec2(0.001, 0.0);
            vec3 n = normalize(vec3(
                map(p + e.xyy, time, uDistort) - map(p - e.xyy, time, uDistort),
                map(p + e.yxy, time, uDistort) - map(p - e.yxy, time, uDistort),
                map(p + e.yyx, time, uDistort) - map(p - e.yyx, time, uDistort)
            ));

            // Lighting
            vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
            float diff = max(dot(n, lightDir), 0.0);
            float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 16.0);

            // Coloring based on position and normal
            float fresnel = pow(1.0 + dot(rd, n), 3.0);
            
            // Dynamic Palette
            vec3 paletteColor = palette(
                length(p) * 0.4 + uTime * 0.1 + uColorShift, 
                vec3(0.5), 
                vec3(0.5), 
                vec3(1.0), 
                vec3(0.263, 0.416, 0.557)
            );

            col = paletteColor * (diff * 0.8 + 0.2) + vec3(1.0) * spec * 0.5;
            col = mix(col, vec3(1.0), fresnel * 0.3);
            break;
        }

        // Move ray forward
        t += d * 0.5; // Use smaller steps for safety
        
        // Stop if too far
        if (t > 20.0) break;
    }

    // Add glow effect for missed rays
    col += glow * 0.02 * palette(
        uTime * 0.05 + uColorShift,
        vec3(0.5), 
        vec3(0.5), 
        vec3(1.0), 
        vec3(0.263, 0.416, 0.557)
    );

    // 3. Post-Processing (Vignette)
    vec2 vUv = fragCoord / uResolution;
    col *= 1.0 - length(vUv - 0.5) * 0.5;

    outColor = vec4(col, 1.0);
}
`,Mn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2  uResolution;
uniform float uPhase;

// Structure
uniform int   uOrder;       // Hadamard order: matrix size = 2^order
uniform float uRotSpeed;    // rotations per loop cycle
uniform float uZoom;        // disk scale
uniform float uRadialPow;   // power curve on radial mapping
uniform float uSpiral;      // spiral twist amount

// Cell appearance
uniform float uSmooth;      // smoothing between cells (0 = crisp)
uniform float uGap;         // visible gap between cells
uniform float uFadeStart;   // edge fade begin (fraction of radius)
uniform float uFadeWidth;   // edge fade extent
uniform float uPulse;       // radial breathing amount

// Color
uniform float uBaseR;
uniform float uBaseG;
uniform float uBaseB;
uniform float uAmpR;        // red modulation amplitude (cos)
uniform float uAmpG;        // green modulation amplitude (sin)
uniform float uAmpB;        // blue modulation amplitude (sin)
uniform float uFreqR;       // red oscillation cycles per loop
uniform float uFreqG;       // green oscillation cycles per loop
uniform float uFreqB;       // blue oscillation cycles per loop

// Extra
uniform float uGlow;        // glow halo intensity
uniform float uBgBright;    // background brightness
uniform float uSeed;        // random angular offset

#define PI  3.14159265359
#define TAU 6.28318530718

/* ── Hadamard value (Sylvester / natural order) ────────────────
   H[row][col] = (-1)^popcount(row & col)
   Returns 1.0 for +1 entries, 0.0 for -1 entries. */
int popcount(int x) {
    int c = 0;
    int v = x;
    for (int i = 0; i < 16; i++) {
        c += v & 1;
        v >>= 1;
        if (v == 0) break;
    }
    return c;
}

float hadamard(int row, int col) {
    return float(1 - (popcount(row & col) & 1));
}

/* Simple hash for seed-based offset */
float hash(float n) {
    return fract(sin(n * 127.1) * 43758.5453);
}

void main() {
    float minDim = min(uResolution.x, uResolution.y);
    vec2 uv = (2.0 * gl_FragCoord.xy - uResolution) / minDim;

    // Zoom
    uv /= max(uZoom, 0.01);

    float r     = length(uv);
    float theta = atan(uv.y, uv.x);
    float t     = uPhase;                       // 0 → 1 per loop

    // Seed-based static angular offset
    theta += hash(uSeed * 13.37) * TAU;

    // Rotation
    theta += t * uRotSpeed * TAU;

    // Spiral warp: angle shifts proportionally to radius
    theta += uSpiral * r * TAU;

    // Radial pulse (breathing)
    float rAdj = r + uPulse * 0.08 * sin(t * TAU);

    // Matrix dimensions
    int   size  = 1 << clamp(uOrder, 1, 10);
    float fSize = float(size);

    // ── Map polar coords → matrix indices ─────────────────────
    // Radius  → row  (center = row 0, edge = last row)
    float rN   = clamp(rAdj, 0.0, 0.9999);
    rN         = pow(rN, uRadialPow);
    float rowF = rN * fSize;
    int   row  = clamp(int(floor(rowF)), 0, size - 1);

    // Angle → column (wraps)
    float aN   = fract(theta / TAU);
    float colF = aN * fSize;
    int   col  = clamp(int(floor(colF)), 0, size - 1);

    // ── Compute cell value ────────────────────────────────────
    float v;
    if (uSmooth > 0.001) {
        // Bilinear interpolation between neighbouring cells
        int row2 = min(row + 1, size - 1);
        int col2 = (col + 1) % size;          // wrap angularly

        float fr = fract(rowF);
        float fc = fract(colF);

        // Smoothstep the fractional parts for a tuneable blend
        float sf = uSmooth;
        fr = smoothstep(0.5 - 0.5 * sf, 0.5 + 0.5 * sf, fr);
        fc = smoothstep(0.5 - 0.5 * sf, 0.5 + 0.5 * sf, fc);

        v = mix(
            mix(hadamard(row, col),  hadamard(row2, col),  fr),
            mix(hadamard(row, col2), hadamard(row2, col2), fr),
            fc
        );
    } else {
        v = hadamard(row, col);
    }

    // ── Cell gap ──────────────────────────────────────────────
    if (uGap > 0.001) {
        float edgeR = min(fract(rowF), 1.0 - fract(rowF));
        float edgeC = min(fract(colF), 1.0 - fract(colF));
        float edge  = min(edgeR, edgeC);
        v *= smoothstep(0.0, uGap * 0.5 + 0.005, edge);
    }

    // ── Time-varying colour (matches original Java formula) ───
    float cr = (uBaseR + uAmpR * cos(uFreqR * TAU * t)) * v;
    float cg = (uBaseG + uAmpG * sin(uFreqG * TAU * t)) * v;
    float cb = (uBaseB + uAmpB * sin(uFreqB * TAU * t)) * v;

    // ── Edge fade ─────────────────────────────────────────────
    float fadeEnd = uFadeStart + max(uFadeWidth, 0.001);
    float alpha   = smoothstep(fadeEnd, uFadeStart, r);

    // ── Glow halo around disk edge ────────────────────────────
    vec3 glowCol = vec3(
        uBaseR + uAmpR * cos(uFreqR * TAU * t),
        uBaseG + uAmpG * sin(uFreqG * TAU * t),
        uBaseB + uAmpB * sin(uFreqB * TAU * t)
    );
    float glowFactor = uGlow * 0.35 * exp(-10.0 * max(r - uFadeStart, 0.0));

    // ── Compose ───────────────────────────────────────────────
    vec3 fg    = clamp(vec3(cr, cg, cb), 0.0, 1.0);
    vec3 bg    = vec3(uBgBright);
    vec3 color = mix(bg, fg, alpha) + glowFactor * glowCol;

    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
`,En=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeed;

uniform int uFractalIters;
uniform float uFoldScale;
uniform float uFoldOffset;
uniform float uRotSpeed;

uniform float uDetailLevel;

uniform float uLightIntensity;

uniform float uHueShift;
uniform float uHueSpeed;
uniform float uSaturation;
uniform float uBrightness;
uniform float uContrast;
uniform float uGlowIntensity;
uniform float uChromaShift;
uniform float uSmoothBlend;

uniform float uZoom;
uniform float uCamHeight;
uniform float uCamOrbit;

const float TAU = 6.28318530718;
const vec2 HASH_SCALE = vec2(234.34, 435.45);
const float HASH_BIAS = 34.23;

mat2 rot2(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

float hash21(vec2 p) {
    p = fract(p * HASH_SCALE);
    p += dot(p, p + HASH_BIAS + fract(uSeed * 0.000001));
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.55;
    float norm = 0.0;
    int octaves = clamp(uFractalIters + 3, 3, 8);
    float persistence = mix(0.46, 0.60, uSmoothBlend);

    for (int i = 0; i < 8; i++) {
        if (i >= octaves) {
            break;
        }
        sum += amp * noise(p);
        norm += amp;
        p = rot2(0.45 + 0.03 * float(i)) * p * 2.02 + vec2(0.31, -0.27);
        amp *= persistence;
    }

    return sum / max(norm, 0.0001);
}

vec3 palNeon(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.0, 0.8, 0.6) + vec3(0.00, 0.33, 0.67)));
}

vec3 palLava(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.7, 0.9, 1.3) + vec3(0.00, 0.18, 0.55)));
}

vec3 palFire(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.3, 0.5, 0.8) + vec3(0.25, 0.00, 0.55)));
}

vec3 palIce(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.6, 0.9, 1.2) + vec3(0.55, 0.70, 0.85)));
}

vec2 warp(vec2 p, float time) {
    float drift = time * (0.05 + uRotSpeed * 0.12);
    float warpScale = 0.75 + uFoldScale * 0.35;
    float warpGain = 0.22 + uFoldOffset * 0.25;

    vec2 q = vec2(
        fbm(p * warpScale + vec2(0.0, drift)),
        fbm(p * warpScale + vec2(4.8, -drift * 0.8))
    );

    vec2 r = vec2(
        fbm(p * (warpScale + 0.6) + (q - 0.5) * 2.0 + vec2(1.7, -2.6) + drift * 0.4),
        fbm(p * (warpScale + 0.4) + (q - 0.5) * 2.0 + vec2(-3.1, 0.9) - drift * 0.3)
    );

    return p + (q + r - 1.0) * warpGain;
}

vec3 background(vec2 uv, vec3 rd, float time) {
    float colorT = uHueShift + time * uHueSpeed * 0.25 + fract(uSeed * 0.000001);
    vec2 p = uv * (1.8 + uZoom * 0.55);
    p += vec2(time * uCamOrbit * 0.6, uCamHeight * 0.22);

    vec2 q = warp(p, time);
    vec2 r = warp(q * 1.35 + vec2(2.4, -1.8), time * 0.7);

    float field = fbm(q);
    float filaments = fbm(r * 1.7 + vec2(time * 0.03, -time * 0.025));
    float mist = fbm(p * 0.65 - vec2(time * 0.02, time * 0.015));
    float ridges = 1.0 - abs(2.0 * fbm(q * (1.4 + 0.2 * uDetailLevel)) - 1.0);
    float glowMask = smoothstep(0.30, 0.88, mix(field, ridges, 0.55));
    float veil = smoothstep(0.18, 0.82, mix(mist, filaments, 0.4));

    float horizon = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    float radial = length(uv);

    vec3 cool = mix(
        palIce(colorT + field * 0.45 + mist * 0.2),
        palNeon(colorT + filaments * 0.35 + horizon * 0.15),
        0.45 + 0.25 * horizon
    );

    vec3 warm = mix(
        palLava(colorT + ridges * 0.50 + 0.08 * radial),
        palFire(colorT + glowMask * 0.65 + filaments * 0.15),
        glowMask
    );

    vec3 col = mix(cool * (0.22 + 0.18 * horizon), warm, veil);
    col += palFire(colorT + filaments + radial * 0.1) * glowMask * glowMask
        * (0.10 + 0.05 * uGlowIntensity);
    col += palNeon(colorT + mist * 0.6 + 0.12 * radial)
        * pow(max(1.0 - radial * 0.75, 0.0), 2.0)
        * (0.06 + 0.03 * uLightIntensity);

    float haze = smoothstep(0.25, 1.35, radial + (1.0 - horizon) * 0.35);
    col = mix(col, cool * 0.35, haze * 0.35);

    return col * (0.75 + 0.25 * uBrightness);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - uResolution * 0.5) / min(uResolution.x, uResolution.y);
    float time = uTime * uTimeScale;
    vec3 rd = normalize(vec3(uv, 1.9 + 0.25 * uZoom));

    vec3 col = background(uv, rd, time);

    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(luma), col, uSaturation);

    float ca = uChromaShift * 0.003;
    float r2 = dot(uv, uv);
    col.r *= 1.0 + ca * r2 * 1.7;
    col.b *= 1.0 - ca * r2 * 1.7;

    col = mix(vec3(0.5), col, uContrast);
    col = max(col, 0.0);
    col = col / (col + 0.35);

    vec2 vUv = gl_FragCoord.xy / uResolution;
    col *= 1.0 - length(vUv - 0.5) * 0.5;
    col = pow(max(col, 0.0), vec3(0.85));

    outColor = vec4(col, 1.0);
}
`,Pn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2  uResolution;     // pixels
uniform float uPhase;          // [0,1] loop time
uniform int   uSymmetry;       // rotational symmetry copies
uniform int   uSubdivisions;   // curve sample points per copy
uniform float uScale;          // overall radius scale (0.0 - 1.0)

uniform float uSinAmp;         // amplitude of the outer sine
uniform float uBaseFreq;       // base frequency term
uniform float uModAmp;         // amplitude modulator
uniform float uModFreq;        // inner modulation frequency
uniform float uModDiv;         // inner modulation divisor
uniform float uThetaScale;     // scale of theta

uniform float uLineWidth;      // antialiased line width in pixels
uniform float uHueCycles;      // hue cycles per loop

uniform float uSeed;           // random seed for variation

const float PI  = 3.14159265358979323846;
const float TAU = 6.28318530717958647692;

float dot2(vec2 v) { return dot(v, v); }

// Exact SDF to quadratic Bezier (Inigo Quilez), returns vec2(distance, closest_t)
// where closest_t in [0,1] is the parameter on the Bezier nearest to pos.
vec2 sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C) {
  vec2 a = B - A;
  vec2 b = A - 2.0*B + C;
  if (dot(b, b) < 1e-10) {
    // Degenerate: line fallback — compute t along segment
    vec2 pa = pos - A, ba = C - A;
    float ht = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return vec2(length(pa - ba * ht), ht);
  }
  vec2 c = a * 2.0;
  vec2 d = A - pos;
  float kk = 1.0 / dot(b, b);
  float kx = kk * dot(a, b);
  float ky = kk * (2.0*dot(a,a) + dot(d,b)) / 3.0;
  float kz = kk * dot(d, a);
  float p  = ky - kx*kx;
  float p3 = p*p*p;
  float q  = kx*(2.0*kx*kx - 3.0*ky) + kz;
  float h  = q*q + 4.0*p3;
  float res;
  float bestT;
  if (h >= 0.0) {
    h = sqrt(h);
    vec2 x = (vec2(h, -h) - q) / 2.0;
    vec2 uv = sign(x) * pow(abs(x), vec2(1.0/3.0));
    bestT = clamp(uv.x + uv.y - kx, 0.0, 1.0);
    res = dot2(d + (c + b*bestT)*bestT);
  } else {
    float z = sqrt(-p);
    float v = acos(clamp(q/(p*z*2.0), -1.0, 1.0)) / 3.0;
    float m = cos(v);
    float n = sin(v) * 1.732050808;
    vec3  t = clamp(vec3(m+m, -n-m, n-m)*z - kx, 0.0, 1.0);
    float d0 = dot2(d + (c + b*t.x)*t.x);
    float d1 = dot2(d + (c + b*t.y)*t.y);
    if (d0 < d1) { res = d0; bestT = t.x; }
    else         { res = d1; bestT = t.y; }
  }
  return vec2(sqrt(res), bestT);
}

// The transformation function
float transformation(float t) {
  float inner = sin(t * uModFreq * PI);
  float outer = uBaseFreq + uModAmp * sin(TAU * inner / uModDiv);
  return (1.0 + uSinAmp * sin(outer * t * uThetaScale * PI)) * PI;
}

// Evaluate a point on the polar rose curve with symmetry rotation
vec2 curvePoint(float t, float cosR, float sinR) {
  float theta = transformation(t);
  float r = sin(t * TAU + uPhase * TAU) * uScale * 0.5;
  vec2 c = vec2(r * cos(theta), r * sin(theta));
  return vec2(cosR * c.x - sinR * c.y, sinR * c.x + cosR * c.y);
}

void main() {
  float minDim = min(uResolution.x, uResolution.y);
  vec2 px = (gl_FragCoord.xy - 0.5 * uResolution) / minDim;

  float dMin = 1e9;
  float closestT = 0.0;

  int N   = max(3, uSubdivisions);
  int sym = max(1, uSymmetry);

  // Bounding-box margin covers the glow radius
  float margin = uLineWidth * 3.0 / minDim;

  for (int s = 0; s < 128; ++s) {
    if (s >= sym) break;
    float rotAngle = float(s) * TAU / float(sym);
    float cosR = cos(rotAngle);
    float sinR = sin(rotAngle);

    // Closed-loop Catmull-Rom Bezier: N sample points, indices wrap mod N
    // so the curve forms a seamless loop with no dangling endpoints.
    vec2 Pprev = curvePoint(float(N - 1) / float(N), cosR, sinR);
    vec2 Pcurr = curvePoint(0.0, cosR, sinR);

    for (int i = 0; i < 8192; ++i) {
      if (i >= N) break;

      vec2 Pnext = curvePoint(float((i + 1) % N) / float(N), cosR, sinR);

      // Catmull-Rom midpoint Bezier: mid(prev,curr) → curr → mid(curr,next)
      // Gives C1-continuous joins — no sharp corners anywhere.
      vec2 A = 0.5 * (Pprev + Pcurr);
      vec2 B = Pcurr;
      vec2 C = 0.5 * (Pcurr + Pnext);

      // Bounding-box culling: skip the expensive sdBezier when far away.
      vec2 lo = min(A, min(B, C)) - margin;
      vec2 hi = max(A, max(B, C)) + margin;

      if (px.x >= lo.x && px.x <= hi.x && px.y >= lo.y && px.y <= hi.y) {
        vec2 db = sdBezier(px, A, B, C);  // .x = distance, .y = bezier param [0,1]
        if (db.x < dMin) {
          dMin = db.x;
          // Continuous curve parameter: segment i, interpolated by Bezier parameter
          closestT = (float(i) + db.y) / float(N);
        }
      }

      Pprev = Pcurr;
      Pcurr = Pnext;
    }
  }

  // --- Sharp core + soft bloom glow ---
  float lineHalf = uLineWidth * 0.5 / minDim;

  // Sharp antialiased core
  float core = 1.0 - smoothstep(lineHalf * 0.35, lineHalf, dMin);

  // Soft bloom glow extending beyond the core
  float glowR = lineHalf * 5.0;
  float glow  = exp(-dMin * dMin / (glowR * glowR * 0.18)) * 0.3;

  float alpha = max(core, glow);

  // Hue from curve parameter
  float hue = 0.5 + 0.5 * sin(closestT * TAU * uHueCycles);
  hue = fract(hue + 0.1 * sin(uPhase * TAU));

  // Glow is slightly desaturated; core is vivid
  float sat = mix(0.55, 0.9, core);
  float val = 0.99;

  // HSV → RGB
  vec3 rgb;
  {
    vec3 cv = vec3(hue, sat, val);
    vec3 q  = abs(fract(cv.xxx + vec3(0.0, 2.0/6.0, 4.0/6.0)) * 6.0 - 3.0);
    rgb = cv.z * mix(vec3(1.0), clamp(q - 1.0, 0.0, 1.0), cv.y);
  }

  outColor = vec4(rgb * alpha, alpha);
}
`,ae=[{id:"neon",name:"Neon Isoclines",description:"Electric contour bands driven by seeded radial harmonics.",fragment:pn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"components",label:"Components",uniform:"uComponents",type:"int",value:64,min:1,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:10}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:8,min:1,max:64,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:10}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.25,min:.01,max:.75,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.05}},{id:"noiseAmount",label:"Noise Amount",uniform:"uNoiseAmount",type:"float",value:2.5,min:0,max:5,step:.05,key:{inc:"r",dec:"f",step:.1,shiftStep:.25}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tanh-terrain",name:"Tanh Terrain Isoclines",description:"Tanh warped contours with bubbling noise and topo glow.",fragment:dn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:2.1,min:.1,max:6,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"octaves",label:"Octaves",uniform:"uOctaves",type:"int",value:4,min:1,max:12,step:1,key:{inc:"3",dec:"4",step:1}},{id:"lacunarity",label:"Lacunarity",uniform:"uLacunarity",type:"float",value:1.4,min:1.01,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"gain",label:"Gain",uniform:"uGain",type:"float",value:.5,min:.01,max:.99,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:16,min:1,max:96,step:1,key:{inc:"q",dec:"a",step:4,shiftStep:12}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.2,min:.02,max:.75,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"bubbleAmp",label:"Bubble Amp",uniform:"uBubbleAmp",type:"float",value:.26,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.08}},{id:"bubbleFreq",label:"Bubble Freq",uniform:"uBubbleFreq",type:"float",value:2,min:0,max:6,step:.05,key:{inc:"r",dec:"f",step:.25,shiftStep:.75}},{id:"bubbleDetail",label:"Bubble Detail",uniform:"uBubbleDetail",type:"float",value:1.2,min:.1,max:3,step:.05,key:{inc:"t",dec:"g",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tunnel",name:"Brownian Loop Tunnel",description:"Looped tunnel with Brownian noise, fog, and hue spin.",fragment:hn,resolutionUniform:"iResolution",timeUniform:"iTime",timeMode:"looped",loopDuration:18,loopUniform:"uLoopDuration",params:[{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:4,min:0,max:10,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"noiseScale",label:"Noise Scale",uniform:"uNoiseScale",type:"float",value:1.9,min:.1,max:4,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.5,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"fogDensity",label:"Fog Density",uniform:"uFogDensity",type:"float",value:2,min:.1,max:6,step:.05,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"baseRed",label:"Base Red",uniform:"uBaseColor",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:0},{id:"baseGreen",label:"Base Green",uniform:"uBaseColor",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:1},{id:"baseBlue",label:"Base Blue",uniform:"uBaseColor",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:2}]},{id:"prismatic-fold",name:"Prismatic Fold Raymarch",description:"Rotating folded planes with prismatic glow and controllable depth.",fragment:xn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:24,step:1,key:{inc:"1",dec:"2",step:1,shiftStep:4}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.2,min:-1.5,max:1.5,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.5,min:.1,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:5,min:1.5,max:10,step:.1,key:{inc:"7",dec:"8",step:.2,shiftStep:.6}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"cameraDistance",label:"Camera Distance",uniform:"uCameraDistance",type:"float",value:50,min:10,max:120,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:5}},{id:"cameraSpin",label:"Camera Spin",uniform:"uCameraSpin",type:"float",value:1,min:-3,max:3,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.4}},{id:"colorMix",label:"Color Mix",uniform:"uColorMix",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"alphaGain",label:"Alpha Gain",uniform:"uAlphaGain",type:"float",value:1,min:.3,max:2,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"[",dec:"]",step:.05},component:2}]},{id:"koch",name:"Koch Snowflake",description:"Iterative snowflake edges with neon glow mixing.",fragment:vn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.8,min:.1,max:2,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:.2,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:2,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.3,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:2}]},{id:"quasi",name:"Quasi Snowflake",description:"Quasicrystal warp with a drifting snowflake outline.",fragment:yn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:6,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:1.1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.8,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.02,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.03,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.02},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.05,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.02},component:2}]},{id:"tileable-water-plus",name:"Tileable Water Plus",description:"Tileable water ripples with tunable speed, scale, and tint.",fragment:Cn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"tileScale",label:"Tile Scale",uniform:"uTileScale",type:"float",value:1,min:.5,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.2,max:2.5,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.2,min:.3,max:2.5,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"waveShift",label:"Wave Shift",uniform:"uWaveShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"tintRed",label:"Tint Red",uniform:"uTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02},component:0},{id:"tintGreen",label:"Tint Green",uniform:"uTint",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02},component:1},{id:"tintBlue",label:"Tint Blue",uniform:"uTint",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:2}]},{id:"seascape",name:"Seascape Plus",description:"Raymarched ocean with tunable swell and camera drift.",fragment:wn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Freq",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:1,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:0,min:-1,max:1,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:0,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1.1,min:.6,max:1.6,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"waterBrightness",label:"Water Bright",uniform:"uWaterBrightness",type:"float",value:.6,min:.2,max:1.2,step:.02,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"waterRed",label:"Water Red",uniform:"uWaterTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02},component:0},{id:"waterGreen",label:"Water Green",uniform:"uWaterTint",type:"float",value:.09,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02},component:1},{id:"waterBlue",label:"Water Blue",uniform:"uWaterTint",type:"float",value:.18,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.02},component:2}]},{id:"sunset-plus",name:"Sunset Plus",description:"Volumetric sunset clouds with tunable turbulence and hue drift.",fragment:Tn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}}]},{id:"sunset-orbit",name:"Sunset Orbit",description:"Volumetric sunset with horizon wrapped around an animated periodic orbit.",fragment:kn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"warpAmp",label:"Warp Amplitude",uniform:"uWarpAmp",type:"float",value:.4,min:0,max:2,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.1}},{id:"warpFreq",label:"Warp Frequency",uniform:"uWarpFreq",type:"float",value:1.5,min:.1,max:8,step:.1,key:{inc:"y",dec:"h",step:.1,shiftStep:.5}},{id:"warpSpeed",label:"Warp Speed",uniform:"uWarpSpeed",type:"float",value:.5,min:0,max:3,step:.05,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"warpHarmonics",label:"Warp Harmonics",uniform:"uWarpHarmonics",type:"float",value:3,min:1,max:5,step:1,key:{inc:"i",dec:"k",step:1}},{id:"orbitRadius",label:"Orbit Radius",uniform:"uOrbitRadius",type:"float",value:.3,min:0,max:2,step:.02,key:{inc:"o",dec:"l",step:.02,shiftStep:.1}},{id:"orbitSpeed",label:"Orbit Speed",uniform:"uOrbitSpeed",type:"float",value:.3,min:0,max:3,step:.05,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"orbitEcc",label:"Orbit Eccentricity",uniform:"uOrbitEcc",type:"float",value:.6,min:0,max:2,step:.05},{id:"tiltAngle",label:"Tilt Angle",uniform:"uTiltAngle",type:"float",value:0,min:-1.57,max:1.57,step:.02},{id:"cloudDensity",label:"Cloud Density",uniform:"uCloudDensity",type:"float",value:1,min:.1,max:3,step:.05},{id:"fogFalloff",label:"Fog Falloff",uniform:"uFogFalloff",type:"float",value:.01,min:0,max:.1,step:.002},{id:"colorSep",label:"Color Separation",uniform:"uColorSep",type:"float",value:.15,min:0,max:1,step:.01}]},{id:"diff-chromatic",name:"Chromatic Flow",description:"Two-channel diffusion with hue-as-angle and drifting color pulses.",fragment:bn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.998,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"rotate",label:"Rotate",uniform:"uRotate",type:"float",value:.02,min:-.2,max:.2,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.03}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.35,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"w",dec:"s",step:.01,shiftStep:.03}},{id:"valueGain",label:"Value Gain",uniform:"uValueGain",type:"float",value:2.2,min:.2,max:6,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"zippy-zaps",name:"Zippy Zaps Plus",description:"Tanh-warped chromatic flow with twistable energy and glow.",fragment:Rn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.2,min:.05,max:.5,step:.005,key:{inc:"1",dec:"2",step:.01,shiftStep:.03}},{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1,min:.2,max:2,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:1,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"iterLimit",label:"Iter Limit",uniform:"uIterLimit",type:"float",value:19,min:4,max:19,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:3}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.4,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"offsetX",label:"Offset X",uniform:"uOffsetX",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"offsetY",label:"Offset Y",uniform:"uOffsetY",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}}]},{id:"space-lightning-plus",name:"Space Lightning Plus",description:"Funky ion bolts with twistable arcs, palette waves, and core glow.",fragment:Bn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.35,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.9,min:.3,max:1.8,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"spin",label:"Spin",uniform:"uSpin",type:"float",value:.4,min:-2,max:2,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1.2,min:0,max:3,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:.9,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1.1,min:0,max:3,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"boltDensity",label:"Bolt Density",uniform:"uBoltDensity",type:"float",value:6.5,min:1,max:20,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"boltSharpness",label:"Bolt Sharpness",uniform:"uBoltSharpness",type:"float",value:.9,min:.1,max:2.5,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:1.2,min:.2,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"arcSteps",label:"Arc Steps",uniform:"uArcSteps",type:"float",value:40,min:6,max:80,step:1,key:{inc:"y",dec:"h",step:1,shiftStep:5}},{id:"coreSize",label:"Core Size",uniform:"uCoreSize",type:"float",value:.25,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"coreGlow",label:"Core Glow",uniform:"uCoreGlow",type:"float",value:.8,min:0,max:2,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.25,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02,shiftStep:.08}},{id:"paletteShift",label:"Palette Shift",uniform:"uPaletteShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"primaryRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.08,min:0,max:1,step:.01,component:0},{id:"primaryGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,component:1},{id:"primaryBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"secondaryRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.9,min:0,max:1,step:.01,component:0},{id:"secondaryGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"secondaryBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.8,min:0,max:1,step:.01,component:2},{id:"accentRed",label:"Accent Red",uniform:"uColorAccent",type:"float",value:1,min:0,max:1,step:.01,component:0},{id:"accentGreen",label:"Accent Green",uniform:"uColorAccent",type:"float",value:.8,min:0,max:1,step:.01,component:1},{id:"accentBlue",label:"Accent Blue",uniform:"uColorAccent",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"thunderstorm",name:"Thunderstorm",description:"Top-down view of a thunderstorm over dense clouds with sporadic branching lightning.",fragment:An,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.2,max:3,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"cloudScale",label:"Cloud Scale",uniform:"uCloudScale",type:"float",value:3,min:.5,max:10,step:.1,key:{inc:"5",dec:"6",step:.2,shiftStep:1}},{id:"cloudSpeed",label:"Cloud Speed",uniform:"uCloudSpeed",type:"float",value:1,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"cloudDensity",label:"Cloud Density",uniform:"uCloudDensity",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.02,shiftStep:.1}},{id:"cloudDetail",label:"Cloud Detail",uniform:"uCloudDetail",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.1}},{id:"boltCount",label:"Bolt Count",uniform:"uBoltCount",type:"int",value:6,min:1,max:12,step:1,key:{inc:"e",dec:"d",step:1,shiftStep:2}},{id:"boltLengthMin",label:"Bolt Length Min",uniform:"uBoltLengthMin",type:"float",value:.1,min:.02,max:.4,step:.01,key:{inc:"r",dec:"f",step:.01,shiftStep:.05}},{id:"boltLengthMax",label:"Bolt Length Max",uniform:"uBoltLengthMax",type:"float",value:.35,min:.1,max:.8,step:.01,key:{inc:"t",dec:"g",step:.01,shiftStep:.05}},{id:"boltWidth",label:"Bolt Width",uniform:"uBoltWidth",type:"float",value:8e-4,min:1e-4,max:.005,step:1e-4,key:{inc:"y",dec:"h",step:2e-4,shiftStep:.001}},{id:"boltWiggle",label:"Bolt Wiggle",uniform:"uBoltWiggle",type:"float",value:.04,min:0,max:.2,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"boltNoiseScale",label:"Bolt Noise Scale",uniform:"uBoltNoiseScale",type:"float",value:15,min:3,max:60,step:.5,key:{inc:"i",dec:"k",step:1,shiftStep:3}},{id:"boltNoiseSpeed",label:"Bolt Noise Speed",uniform:"uBoltNoiseSpeed",type:"float",value:2,min:0,max:8,step:.05,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"boltBranching",label:"Bolt Branching",uniform:"uBoltBranching",type:"float",value:.5,min:0,max:1,step:.02,key:{inc:"p",dec:";",step:.02,shiftStep:.1}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:.3,min:.05,max:1.5,step:.02},{id:"flickerSpeed",label:"Flicker Speed",uniform:"uFlickerSpeed",type:"float",value:2,min:0,max:8,step:.1},{id:"cloudIllumination",label:"Cloud Illumination",uniform:"uCloudIllumination",type:"float",value:1,min:0,max:3,step:.05},{id:"noiseOctaves",label:"Noise Octaves",uniform:"uNoiseOctaves",type:"int",value:5,min:1,max:8,step:1},{id:"cloudRed",label:"Cloud Red",uniform:"uCloudColor",type:"float",value:.12,min:0,max:1,step:.01,component:0},{id:"cloudGreen",label:"Cloud Green",uniform:"uCloudColor",type:"float",value:.12,min:0,max:1,step:.01,component:1},{id:"cloudBlue",label:"Cloud Blue",uniform:"uCloudColor",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"lightningRed",label:"Lightning Red",uniform:"uLightningColor",type:"float",value:.7,min:0,max:1,step:.01,component:0},{id:"lightningGreen",label:"Lightning Green",uniform:"uLightningColor",type:"float",value:.7,min:0,max:1,step:.01,component:1},{id:"lightningBlue",label:"Lightning Blue",uniform:"uLightningColor",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"mandelbulb-inside-plus",name:"Inside the Mandelbulb Plus",description:"Raymarched mandelbulb interior with tunable optics and palette glow.",fragment:Fn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"power",label:"Power",uniform:"uPower",type:"float",value:8,min:2,max:12,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"bulbSpin",label:"Bulb Spin",uniform:"uBulbSpin",type:"float",value:.2,min:0,max:1.5,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"loops",label:"Loops",uniform:"uLoops",type:"int",value:2,min:1,max:6,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:1}},{id:"rayMarches",label:"Ray Marches",uniform:"uRayMarches",type:"int",value:60,min:20,max:96,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:5}},{id:"maxRayLength",label:"Max Ray Length",uniform:"uMaxRayLength",type:"float",value:20,min:5,max:40,step:.5,key:{inc:"w",dec:"s",step:.5,shiftStep:2}},{id:"tolerance",label:"Tolerance",uniform:"uTolerance",type:"float",value:1e-4,min:1e-5,max:.001,step:1e-5,key:{inc:"e",dec:"d",step:2e-5,shiftStep:1e-4}},{id:"normOffset",label:"Normal Offset",uniform:"uNormOffset",type:"float",value:.005,min:.001,max:.02,step:5e-4,key:{inc:"r",dec:"f",step:5e-4,shiftStep:.002}},{id:"bounces",label:"Bounces",uniform:"uBounces",type:"int",value:5,min:1,max:5,step:1,key:{inc:"t",dec:"g",step:1,shiftStep:1}},{id:"initStep",label:"Init Step",uniform:"uInitStep",type:"float",value:.1,min:.01,max:.3,step:.01,key:{inc:"y",dec:"h",step:.01,shiftStep:.05}},{id:"rotSpeedX",label:"Rot Speed X",uniform:"uRotSpeedX",type:"float",value:.2,min:-1,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.1}},{id:"rotSpeedY",label:"Rot Speed Y",uniform:"uRotSpeedY",type:"float",value:.3,min:-1,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:2,max:10,step:.1,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:2,min:.5,max:5,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"fov",label:"FOV",uniform:"uFov",type:"float",value:.523,min:.3,max:1.2,step:.01},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"glowBoost",label:"Glow Boost",uniform:"uGlowBoost",type:"float",value:1.2,min:0,max:4,step:.05},{id:"glowFalloff",label:"Glow Falloff",uniform:"uGlowFalloff",type:"float",value:.06,min:.01,max:.2,step:.005},{id:"diffuseBoost",label:"Diffuse Boost",uniform:"uDiffuseBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"matTransmit",label:"Mat Transmit",uniform:"uMatTransmit",type:"float",value:.8,min:0,max:1,step:.01},{id:"matReflect",label:"Mat Reflect",uniform:"uMatReflect",type:"float",value:.5,min:0,max:1,step:.01},{id:"refractIndex",label:"Refract Index",uniform:"uRefractIndex",type:"float",value:1.05,min:1,max:2,step:.01},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"glowHueOffset",label:"Glow Hue Offset",uniform:"uGlowHueOffset",type:"float",value:.065,min:-.5,max:.5,step:.005},{id:"nebulaMix",label:"Nebula Mix",uniform:"uNebulaMix",type:"float",value:0,min:0,max:1,step:.01},{id:"nebulaHueShift",label:"Nebula Hue",uniform:"uNebulaHueShift",type:"float",value:.12,min:-1,max:1,step:.01},{id:"nebulaSat",label:"Nebula Sat",uniform:"uNebulaSat",type:"float",value:.9,min:0,max:1,step:.01},{id:"nebulaVal",label:"Nebula Val",uniform:"uNebulaVal",type:"float",value:1.6,min:.2,max:3,step:.02},{id:"nebulaGlowHue",label:"Nebula Glow Hue",uniform:"uNebulaGlowHue",type:"float",value:.35,min:-1,max:1,step:.01},{id:"nebulaGlowBoost",label:"Nebula Glow",uniform:"uNebulaGlowBoost",type:"float",value:1.6,min:0,max:4,step:.05},{id:"skySat",label:"Sky Saturation",uniform:"uSkySat",type:"float",value:.86,min:0,max:1,step:.01},{id:"skyVal",label:"Sky Value",uniform:"uSkyVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"glowSat",label:"Glow Saturation",uniform:"uGlowSat",type:"float",value:.8,min:0,max:1,step:.01},{id:"glowVal",label:"Glow Value",uniform:"uGlowVal",type:"float",value:6,min:.5,max:8,step:.1},{id:"diffuseSat",label:"Diffuse Saturation",uniform:"uDiffuseSat",type:"float",value:.85,min:0,max:1,step:.01},{id:"diffuseVal",label:"Diffuse Value",uniform:"uDiffuseVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"beerRed",label:"Beer Red",uniform:"uBeerColor",type:"float",value:.02,min:0,max:.2,step:.005,component:0},{id:"beerGreen",label:"Beer Green",uniform:"uBeerColor",type:"float",value:.08,min:0,max:.2,step:.005,component:1},{id:"beerBlue",label:"Beer Blue",uniform:"uBeerColor",type:"float",value:.12,min:0,max:.2,step:.005,component:2},{id:"lightX",label:"Light X",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:0},{id:"lightY",label:"Light Y",uniform:"uLightPos",type:"float",value:10,min:-5,max:25,step:.5,component:1},{id:"lightZ",label:"Light Z",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:2}]},{id:"auroras-plus",name:"Auroras Plus",description:"Volumetric auroras with tunable trails, palette waves, and sky glare.",fragment:zn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"auroraSpeed",label:"Aurora Speed",uniform:"uAuroraSpeed",type:"float",value:.06,min:0,max:.2,step:.005,key:{inc:"3",dec:"4",step:.005,shiftStep:.02}},{id:"auroraScale",label:"Aurora Scale",uniform:"uAuroraScale",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"auroraWarp",label:"Aurora Warp",uniform:"uAuroraWarp",type:"float",value:.35,min:0,max:1,step:.02,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"auroraSteps",label:"Aurora Steps",uniform:"uAuroraSteps",type:"int",value:50,min:8,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"auroraBase",label:"Aurora Base",uniform:"uAuroraBase",type:"float",value:.8,min:.2,max:1.6,step:.02,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"auroraStride",label:"Aurora Stride",uniform:"uAuroraStride",type:"float",value:.002,min:2e-4,max:.01,step:2e-4,key:{inc:"e",dec:"d",step:2e-4,shiftStep:.001}},{id:"auroraCurve",label:"Aurora Curve",uniform:"uAuroraCurve",type:"float",value:1.4,min:.8,max:2.2,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"auroraIntensity",label:"Aurora Intensity",uniform:"uAuroraIntensity",type:"float",value:1.8,min:.2,max:4,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"trailBlend",label:"Trail Blend",uniform:"uTrailBlend",type:"float",value:.5,min:.1,max:.9,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"trailFalloff",label:"Trail Falloff",uniform:"uTrailFalloff",type:"float",value:.065,min:.01,max:.15,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"trailFade",label:"Trail Fade",uniform:"uTrailFade",type:"float",value:2.5,min:.5,max:5,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.4}},{id:"ditherStrength",label:"Dither Strength",uniform:"uDitherStrength",type:"float",value:.006,min:0,max:.02,step:5e-4,key:{inc:"o",dec:"l",step:5e-4,shiftStep:.002}},{id:"horizonFade",label:"Horizon Fade",uniform:"uHorizonFade",type:"float",value:.01,min:.001,max:.05,step:.001,key:{inc:"p",dec:";",step:.001,shiftStep:.005}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:-.1,min:-1,max:1,step:.01},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:.1,min:-1,max:1,step:.01},{id:"camWobble",label:"Cam Wobble",uniform:"uCamWobble",type:"float",value:.2,min:0,max:.6,step:.01},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:6.7,min:4,max:12,step:.1},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:0,min:-1,max:2,step:.05},{id:"skyStrength",label:"Sky Strength",uniform:"uSkyStrength",type:"float",value:.63,min:.1,max:2,step:.02},{id:"starDensity",label:"Star Density",uniform:"uStarDensity",type:"float",value:5e-4,min:0,max:.005,step:1e-4},{id:"starIntensity",label:"Star Intensity",uniform:"uStarIntensity",type:"float",value:.8,min:0,max:2,step:.05},{id:"reflectionStrength",label:"Reflection Strength",uniform:"uReflectionStrength",type:"float",value:.6,min:0,max:1.5,step:.05},{id:"reflectionTint",label:"Reflection Tint",uniform:"uReflectionTint",type:"float",value:1,min:0,max:2,step:.05},{id:"reflectionFog",label:"Reflection Fog",uniform:"uReflectionFog",type:"float",value:2,min:0,max:6,step:.1},{id:"colorBand",label:"Color Band",uniform:"uColorBand",type:"float",value:.043,min:0,max:.2,step:.002},{id:"colorSpeed",label:"Color Speed",uniform:"uColorSpeed",type:"float",value:0,min:-1,max:1,step:.01},{id:"auroraRedA",label:"Aurora Red A",uniform:"uAuroraColorA",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenA",label:"Aurora Green A",uniform:"uAuroraColorA",type:"float",value:.9,min:0,max:1,step:.01,component:1},{id:"auroraBlueA",label:"Aurora Blue A",uniform:"uAuroraColorA",type:"float",value:.6,min:0,max:1,step:.01,component:2},{id:"auroraRedB",label:"Aurora Red B",uniform:"uAuroraColorB",type:"float",value:.6,min:0,max:1,step:.01,component:0},{id:"auroraGreenB",label:"Aurora Green B",uniform:"uAuroraColorB",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"auroraBlueB",label:"Aurora Blue B",uniform:"uAuroraColorB",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"auroraRedC",label:"Aurora Red C",uniform:"uAuroraColorC",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenC",label:"Aurora Green C",uniform:"uAuroraColorC",type:"float",value:.6,min:0,max:1,step:.01,component:1},{id:"auroraBlueC",label:"Aurora Blue C",uniform:"uAuroraColorC",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedA",label:"BG Red A",uniform:"uBgColorA",type:"float",value:.05,min:0,max:1,step:.01,component:0},{id:"bgGreenA",label:"BG Green A",uniform:"uBgColorA",type:"float",value:.1,min:0,max:1,step:.01,component:1},{id:"bgBlueA",label:"BG Blue A",uniform:"uBgColorA",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedB",label:"BG Red B",uniform:"uBgColorB",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"bgGreenB",label:"BG Green B",uniform:"uBgColorB",type:"float",value:.05,min:0,max:1,step:.01,component:1},{id:"bgBlueB",label:"BG Blue B",uniform:"uBgColorB",type:"float",value:.2,min:0,max:1,step:.01,component:2}]},{id:"diff-edge-flow",name:"Edge Flow Vectors",description:"Diffusive scalar field rendered as glowing edge-flow vectors.",fragment:gn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.996,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"blobAmp",label:"Blob Amp",uniform:"uBlobAmp",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"blobRadius",label:"Blob Radius",uniform:"uBlobRadius",type:"float",value:.07,min:.01,max:.25,step:.005,key:{inc:"q",dec:"a",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.8,min:0,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"flowGain",label:"Flow Gain",uniform:"uFlowGain",type:"float",value:3,min:.2,max:8,step:.1,key:{inc:"e",dec:"d",step:.2,shiftStep:.6}},{id:"flowThreshold",label:"Flow Threshold",uniform:"uFlowThreshold",type:"float",value:.02,min:0,max:.2,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"diff-threshold",name:"Threshold Feedback",description:"Diffusion with nonlinear feedback for digital fungus crackle.",fragment:Sn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:192,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.5,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.995,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"threshold",label:"Threshold",uniform:"uThreshold",type:"float",value:.5,min:.1,max:.9,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.06}},{id:"sharpness",label:"Sharpness",uniform:"uSharpness",type:"float",value:18,min:1,max:40,step:.5,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.08,min:0,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.2,min:0,max:1.5,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractal-fold",name:"Fractal Fold Raymarch",description:"Recursive box-folding fractal with prismatic lighting and IQ palette.",fragment:In,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.5,max:5,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"distort",label:"Distort",uniform:"uDistort",type:"float",value:2.5,min:1.5,max:4,step:.02,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:1,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.15}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.1,min:-.5,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"maxSteps",label:"Max Steps",uniform:"uMaxSteps",type:"float",value:100,min:20,max:200,step:10,key:{inc:"e",dec:"d",step:10,shiftStep:30}}]},{id:"hadamard-disk",name:"Hadamard Disk",description:"Spinning Hadamard matrix mapped onto a disk with animated colour cycling.",fragment:Mn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"order",label:"Order",uniform:"uOrder",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"rotSpeed",label:"Rotation Speed",uniform:"uRotSpeed",type:"float",value:1,min:-4,max:4,step:.1,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.95,min:.2,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"radialPow",label:"Radial Power",uniform:"uRadialPow",type:"float",value:1,min:.1,max:5,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"spiral",label:"Spiral",uniform:"uSpiral",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"smooth",label:"Smooth",uniform:"uSmooth",type:"float",value:0,min:0,max:1,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"gap",label:"Cell Gap",uniform:"uGap",type:"float",value:0,min:0,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.05}},{id:"fadeStart",label:"Fade Start",uniform:"uFadeStart",type:"float",value:.9,min:.3,max:1,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.05}},{id:"fadeWidth",label:"Fade Width",uniform:"uFadeWidth",type:"float",value:.08,min:.01,max:.5,step:.01,key:{inc:"5",dec:"6",step:.01,shiftStep:.05}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:0,min:0,max:2,step:.05,key:{inc:"u",dec:"j",step:.05,shiftStep:.2}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:0,min:0,max:2,step:.05,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"bgBright",label:"Background",uniform:"uBgBright",type:"float",value:0,min:0,max:.5,step:.01},{id:"baseR",label:"Base Red",uniform:"uBaseR",type:"float",value:.75,min:0,max:1,step:.01},{id:"baseG",label:"Base Green",uniform:"uBaseG",type:"float",value:.75,min:0,max:1,step:.01},{id:"baseB",label:"Base Blue",uniform:"uBaseB",type:"float",value:.75,min:0,max:1,step:.01},{id:"ampR",label:"Amp Red",uniform:"uAmpR",type:"float",value:-.25,min:-1,max:1,step:.01},{id:"ampG",label:"Amp Green",uniform:"uAmpG",type:"float",value:.25,min:-1,max:1,step:.01},{id:"ampB",label:"Amp Blue",uniform:"uAmpB",type:"float",value:.25,min:-1,max:1,step:.01},{id:"freqR",label:"Freq Red",uniform:"uFreqR",type:"float",value:1,min:0,max:16,step:.5},{id:"freqG",label:"Freq Green",uniform:"uFreqG",type:"float",value:2,min:0,max:16,step:.5},{id:"freqB",label:"Freq Blue",uniform:"uFreqB",type:"float",value:4,min:0,max:16,step:.5},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractalReverie",name:"Fractal Reverie",description:"Continuous fractal color fields with smooth drifting motion and a saturated glowing backdrop.",fragment:En,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.05,min:0,max:3,step:.01,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"fractalIters",label:"Fractal Iters",uniform:"uFractalIters",type:"int",value:2,min:1,max:12,step:1,key:{inc:"w",dec:"s",step:1}},{id:"foldScale",label:"Fold Scale",uniform:"uFoldScale",type:"float",value:1.18,min:1,max:4,step:.01,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.42,min:0,max:3,step:.01,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"rotSpeed",label:"Rot Speed",uniform:"uRotSpeed",type:"float",value:.03,min:0,max:3,step:.01},{id:"detailLevel",label:"Detail Level",uniform:"uDetailLevel",type:"float",value:1,min:.2,max:5,step:.1},{id:"lightIntensity",label:"Light Intensity",uniform:"uLightIntensity",type:"float",value:1.5,min:0,max:5,step:.05,key:{inc:"u",dec:"j",step:.1,shiftStep:.5}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.05,min:0,max:1,step:.005},{id:"saturation",label:"Saturation",uniform:"uSaturation",type:"float",value:2,min:0,max:3,step:.01,key:{inc:"o",dec:"l",step:.05,shiftStep:.2}},{id:"brightness",label:"Brightness",uniform:"uBrightness",type:"float",value:1.8,min:.1,max:4,step:.05},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.1,min:.5,max:3,step:.01,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"glowIntensity",label:"Glow Intensity",uniform:"uGlowIntensity",type:"float",value:4.5,min:0,max:8,step:.1,key:{inc:"[",dec:"]",step:.2,shiftStep:.5}},{id:"chromaShift",label:"Chroma Shift",uniform:"uChromaShift",type:"float",value:.5,min:0,max:20,step:.1},{id:"smoothBlend",label:"Smooth Blend",uniform:"uSmoothBlend",type:"float",value:.9,min:0,max:1,step:.01},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.68,min:.2,max:3,step:.01},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:.3,min:-3,max:3,step:.05},{id:"camOrbit",label:"Cam Orbit",uniform:"uCamOrbit",type:"float",value:.018,min:0,max:1,step:.005},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"polar-rose",name:"Polar Rose",description:"Animated polar rose curves with C1-continuous Bezier segments and chromatic glow.",fragment:Pn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"symmetry",label:"Symmetry",uniform:"uSymmetry",type:"int",value:5,min:1,max:32,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"subdivisions",label:"Subdivisions",uniform:"uSubdivisions",type:"int",value:64,min:16,max:1024,step:16,key:{inc:"w",dec:"s",step:16,shiftStep:64}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.85,min:.05,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.1}},{id:"sinAmp",label:"Sin Amplitude",uniform:"uSinAmp",type:"float",value:.4,min:0,max:2,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.1}},{id:"baseFreq",label:"Base Frequency",uniform:"uBaseFreq",type:"float",value:3,min:.5,max:12,step:.1,key:{inc:"t",dec:"g",step:.1,shiftStep:.5}},{id:"modAmp",label:"Mod Amplitude",uniform:"uModAmp",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"y",dec:"h",step:.05,shiftStep:.25}},{id:"modFreq",label:"Mod Frequency",uniform:"uModFreq",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"u",dec:"j",step:.1,shiftStep:.5}},{id:"modDiv",label:"Mod Divisor",uniform:"uModDiv",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.5}},{id:"thetaScale",label:"Theta Scale",uniform:"uThetaScale",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"o",dec:"l",step:.05,shiftStep:.25}},{id:"lineWidth",label:"Line Width",uniform:"uLineWidth",type:"float",value:2,min:.5,max:8,step:.1,key:{inc:"[",dec:"]",step:.1,shiftStep:.5}},{id:"hueCycles",label:"Hue Cycles",uniform:"uHueCycles",type:"float",value:4,min:0,max:12,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]}];ae.sort((e,t)=>e.name.localeCompare(t.name));function Rt(e,t,n){const o=e.createShader(t);if(!o)throw new Error("Failed to create shader");if(e.shaderSource(o,n),e.compileShader(o),!e.getShaderParameter(o,e.COMPILE_STATUS)){const a=e.getShaderInfoLog(o)||"Unknown shader error";throw e.deleteShader(o),new Error(a)}return o}function Dn(e,t,n){const o=Rt(e,e.VERTEX_SHADER,t),a=Rt(e,e.FRAGMENT_SHADER,n),l=e.createProgram();if(!l)throw new Error("Failed to create program");if(e.attachShader(l,o),e.attachShader(l,a),e.linkProgram(l),e.deleteShader(o),e.deleteShader(a),!e.getProgramParameter(l,e.LINK_STATUS)){const r=e.getProgramInfoLog(l)||"Unknown program error";throw e.deleteProgram(l),new Error(r)}return l}function Un(e,t,n){const o={};for(const a of n)o[a]=e.getUniformLocation(t,a);return o}function Gn(e,t=2){const n=Math.min(window.devicePixelRatio||1,t),o=Math.max(1,Math.floor(e.clientWidth*n)),a=Math.max(1,Math.floor(e.clientHeight*n));return(e.width!==o||e.height!==a)&&(e.width=o,e.height=a),{width:o,height:a,dpr:n}}function Nn(e){const t=e.createVertexArray();if(!t)throw new Error("Failed to create VAO");return e.bindVertexArray(t),t}var lt=(e,t,n)=>{if(!t.has(e))throw TypeError("Cannot "+n)},i=(e,t,n)=>(lt(e,t,"read from private field"),n?n.call(e):t.get(e)),d=(e,t,n)=>{if(t.has(e))throw TypeError("Cannot add the same private member more than once");t instanceof WeakSet?t.add(e):t.set(e,n)},A=(e,t,n,o)=>(lt(e,t,"write to private field"),t.set(e,n),n),_n=(e,t,n,o)=>({set _(a){A(e,t,a)},get _(){return i(e,t,o)}}),h=(e,t,n)=>(lt(e,t,"access private method"),n),v=new Uint8Array(8),G=new DataView(v.buffer),R=e=>[(e%256+256)%256],g=e=>(G.setUint16(0,e,!1),[v[0],v[1]]),Ln=e=>(G.setInt16(0,e,!1),[v[0],v[1]]),Mt=e=>(G.setUint32(0,e,!1),[v[1],v[2],v[3]]),f=e=>(G.setUint32(0,e,!1),[v[0],v[1],v[2],v[3]]),On=e=>(G.setInt32(0,e,!1),[v[0],v[1],v[2],v[3]]),Z=e=>(G.setUint32(0,Math.floor(e/2**32),!1),G.setUint32(4,e,!1),[v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]]),rt=e=>(G.setInt16(0,2**8*e,!1),[v[0],v[1]]),D=e=>(G.setInt32(0,2**16*e,!1),[v[0],v[1],v[2],v[3]]),He=e=>(G.setInt32(0,2**30*e,!1),[v[0],v[1],v[2],v[3]]),M=(e,t=!1)=>{let n=Array(e.length).fill(null).map((o,a)=>e.charCodeAt(a));return t&&n.push(0),n},Ie=e=>e&&e[e.length-1],st=e=>{let t;for(let n of e)(!t||n.presentationTimestamp>t.presentationTimestamp)&&(t=n);return t},U=(e,t,n=!0)=>{let o=e*t;return n?Math.round(o):o},Et=e=>{let t=e*(Math.PI/180),n=Math.cos(t),o=Math.sin(t);return[n,o,0,-o,n,0,0,0,1]},Pt=Et(0),Dt=e=>[D(e[0]),D(e[1]),He(e[2]),D(e[3]),D(e[4]),He(e[5]),D(e[6]),D(e[7]),He(e[8])],ue=e=>!e||typeof e!="object"?e:Array.isArray(e)?e.map(ue):Object.fromEntries(Object.entries(e).map(([t,n])=>[t,ue(n)])),ne=e=>e>=0&&e<2**32,T=(e,t,n)=>({type:e,contents:t&&new Uint8Array(t.flat(10)),children:n}),x=(e,t,n,o,a)=>T(e,[R(t),Mt(n),o??[]],a),Wn=e=>{let t=512;return e.fragmented?T("ftyp",[M("iso5"),f(t),M("iso5"),M("iso6"),M("mp41")]):T("ftyp",[M("isom"),f(t),M("isom"),e.holdsAvc?M("avc1"):[],M("mp41")])},Xe=e=>({type:"mdat",largeSize:e}),Hn=e=>({type:"free",size:e}),Te=(e,t,n=!1)=>T("moov",null,[qn(t,e),...e.map(o=>Vn(o,t)),n?wi(e):null]),qn=(e,t)=>{let n=U(Math.max(0,...t.filter(r=>r.samples.length>0).map(r=>{const u=st(r.samples);return u.presentationTimestamp+u.duration})),Ye),o=Math.max(...t.map(r=>r.id))+1,a=!ne(e)||!ne(n),l=a?Z:f;return x("mvhd",+a,0,[l(e),l(e),f(Ye),l(n),D(1),rt(1),Array(10).fill(0),Dt(Pt),Array(24).fill(0),f(o)])},Vn=(e,t)=>T("trak",null,[jn(e,t),Xn(e,t)]),jn=(e,t)=>{let n=st(e.samples),o=U(n?n.presentationTimestamp+n.duration:0,Ye),a=!ne(t)||!ne(o),l=a?Z:f,r;return e.info.type==="video"?r=typeof e.info.rotation=="number"?Et(e.info.rotation):e.info.rotation:r=Pt,x("tkhd",+a,3,[l(t),l(t),f(e.id),f(0),l(o),Array(8).fill(0),g(0),g(0),rt(e.info.type==="audio"?1:0),g(0),Dt(r),D(e.info.type==="video"?e.info.width:0),D(e.info.type==="video"?e.info.height:0)])},Xn=(e,t)=>T("mdia",null,[$n(e,t),Zn(e.info.type==="video"?"vide":"soun"),Yn(e)]),$n=(e,t)=>{let n=st(e.samples),o=U(n?n.presentationTimestamp+n.duration:0,e.timescale),a=!ne(t)||!ne(o),l=a?Z:f;return x("mdhd",+a,0,[l(t),l(t),f(e.timescale),l(o),g(21956),g(0)])},Zn=e=>x("hdlr",0,0,[M("mhlr"),M(e),f(0),f(0),f(0),M("mp4-muxer-hdlr",!0)]),Yn=e=>T("minf",null,[e.info.type==="video"?Kn():Qn(),Jn(),ni(e)]),Kn=()=>x("vmhd",0,1,[g(0),g(0),g(0),g(0)]),Qn=()=>x("smhd",0,0,[g(0),g(0)]),Jn=()=>T("dinf",null,[ei()]),ei=()=>x("dref",0,0,[f(1)],[ti()]),ti=()=>x("url ",0,1),ni=e=>{const t=e.compositionTimeOffsetTable.length>1||e.compositionTimeOffsetTable.some(n=>n.sampleCompositionTimeOffset!==0);return T("stbl",null,[ii(e),vi(e),yi(e),bi(e),gi(e),Si(e),t?xi(e):null])},ii=e=>x("stsd",0,0,[f(1)],[e.info.type==="video"?oi(Mi[e.info.codec],e):pi(Pi[e.info.codec],e)]),oi=(e,t)=>T(e,[Array(6).fill(0),g(1),g(0),g(0),Array(12).fill(0),g(t.info.width),g(t.info.height),f(4718592),f(4718592),f(0),g(1),Array(32).fill(0),g(24),Ln(65535)],[Ei[t.info.codec](t),t.info.decoderConfig.colorSpace?si(t):null]),ai={bt709:1,bt470bg:5,smpte170m:6},li={bt709:1,smpte170m:6,"iec61966-2-1":13},ri={rgb:0,bt709:1,bt470bg:5,smpte170m:6},si=e=>T("colr",[M("nclx"),g(ai[e.info.decoderConfig.colorSpace.primaries]),g(li[e.info.decoderConfig.colorSpace.transfer]),g(ri[e.info.decoderConfig.colorSpace.matrix]),R((e.info.decoderConfig.colorSpace.fullRange?1:0)<<7)]),ui=e=>e.info.decoderConfig&&T("avcC",[...new Uint8Array(e.info.decoderConfig.description)]),fi=e=>e.info.decoderConfig&&T("hvcC",[...new Uint8Array(e.info.decoderConfig.description)]),ci=e=>{if(!e.info.decoderConfig)return null;let t=e.info.decoderConfig;if(!t.colorSpace)throw new Error("'colorSpace' is required in the decoder config for VP9.");let n=t.codec.split("."),o=Number(n[1]),a=Number(n[2]),u=(Number(n[3])<<4)+(0<<1)+Number(t.colorSpace.fullRange);return x("vpcC",1,0,[R(o),R(a),R(u),R(2),R(2),R(2),g(0)])},mi=()=>{let n=(1<<7)+1;return T("av1C",[n,0,0,0])},pi=(e,t)=>T(e,[Array(6).fill(0),g(1),g(0),g(0),f(0),g(t.info.numberOfChannels),g(16),g(0),g(0),D(t.info.sampleRate)],[Di[t.info.codec](t)]),di=e=>{let t=new Uint8Array(e.info.decoderConfig.description);return x("esds",0,0,[f(58753152),R(32+t.byteLength),g(1),R(0),f(75530368),R(18+t.byteLength),R(64),R(21),Mt(0),f(130071),f(130071),f(92307584),R(t.byteLength),...t,f(109084800),R(1),R(2)])},hi=e=>{var a;let t=3840,n=0;const o=(a=e.info.decoderConfig)==null?void 0:a.description;if(o){if(o.byteLength<18)throw new TypeError("Invalid decoder description provided for Opus; must be at least 18 bytes long.");const l=ArrayBuffer.isView(o)?new DataView(o.buffer,o.byteOffset,o.byteLength):new DataView(o);t=l.getUint16(10,!0),n=l.getInt16(14,!0)}return T("dOps",[R(0),R(e.info.numberOfChannels),g(t),f(e.info.sampleRate),rt(n),R(0)])},vi=e=>x("stts",0,0,[f(e.timeToSampleTable.length),e.timeToSampleTable.map(t=>[f(t.sampleCount),f(t.sampleDelta)])]),yi=e=>{if(e.samples.every(n=>n.type==="key"))return null;let t=[...e.samples.entries()].filter(([,n])=>n.type==="key");return x("stss",0,0,[f(t.length),t.map(([n])=>f(n+1))])},bi=e=>x("stsc",0,0,[f(e.compactlyCodedChunkTable.length),e.compactlyCodedChunkTable.map(t=>[f(t.firstChunk),f(t.samplesPerChunk),f(1)])]),gi=e=>x("stsz",0,0,[f(0),f(e.samples.length),e.samples.map(t=>f(t.size))]),Si=e=>e.finalizedChunks.length>0&&Ie(e.finalizedChunks).offset>=2**32?x("co64",0,0,[f(e.finalizedChunks.length),e.finalizedChunks.map(t=>Z(t.offset))]):x("stco",0,0,[f(e.finalizedChunks.length),e.finalizedChunks.map(t=>f(t.offset))]),xi=e=>x("ctts",0,0,[f(e.compositionTimeOffsetTable.length),e.compositionTimeOffsetTable.map(t=>[f(t.sampleCount),f(t.sampleCompositionTimeOffset)])]),wi=e=>T("mvex",null,e.map(Ci)),Ci=e=>x("trex",0,0,[f(e.id),f(1),f(0),f(0),f(0)]),Bt=(e,t)=>T("moof",null,[Ti(e),...t.map(ki)]),Ti=e=>x("mfhd",0,0,[f(e)]),Ut=e=>{let t=0,n=0,o=0,a=0,l=e.type==="delta";return n|=+l,l?t|=1:t|=2,t<<24|n<<16|o<<8|a},ki=e=>T("traf",null,[Ri(e),Bi(e),Ai(e)]),Ri=e=>{let t=0;t|=8,t|=16,t|=32,t|=131072;let n=e.currentChunk.samples[1]??e.currentChunk.samples[0],o={duration:n.timescaleUnitsToNextSample,size:n.size,flags:Ut(n)};return x("tfhd",0,t,[f(e.id),f(o.duration),f(o.size),f(o.flags)])},Bi=e=>x("tfdt",1,0,[Z(U(e.currentChunk.startTimestamp,e.timescale))]),Ai=e=>{let t=e.currentChunk.samples.map(E=>E.timescaleUnitsToNextSample),n=e.currentChunk.samples.map(E=>E.size),o=e.currentChunk.samples.map(Ut),a=e.currentChunk.samples.map(E=>U(E.presentationTimestamp-E.decodeTimestamp,e.timescale)),l=new Set(t),r=new Set(n),u=new Set(o),p=new Set(a),y=u.size===2&&o[0]!==o[1],b=l.size>1,w=r.size>1,S=!y&&u.size>1,W=p.size>1||[...p].some(E=>E!==0),N=0;return N|=1,N|=4*+y,N|=256*+b,N|=512*+w,N|=1024*+S,N|=2048*+W,x("trun",1,N,[f(e.currentChunk.samples.length),f(e.currentChunk.offset-e.currentChunk.moofOffset||0),y?f(o[0]):[],e.currentChunk.samples.map((E,Y)=>[b?f(t[Y]):[],w?f(n[Y]):[],S?f(o[Y]):[],W?On(a[Y]):[]])])},Fi=e=>T("mfra",null,[...e.map(zi),Ii()]),zi=(e,t)=>x("tfra",1,0,[f(e.id),f(63),f(e.finalizedChunks.length),e.finalizedChunks.map(o=>[Z(U(o.startTimestamp,e.timescale)),Z(o.moofOffset),f(t+1),f(1),f(1)])]),Ii=()=>x("mfro",0,0,[f(0)]),Mi={avc:"avc1",hevc:"hvc1",vp9:"vp09",av1:"av01"},Ei={avc:ui,hevc:fi,vp9:ci,av1:mi},Pi={aac:"mp4a",opus:"Opus"},Di={aac:di,opus:hi},_e=class{},Ui=class extends _e{constructor(){super(...arguments),this.buffer=null}},Gt=class extends _e{constructor(e){if(super(),this.options=e,typeof e!="object")throw new TypeError("StreamTarget requires an options object to be passed to its constructor.");if(e.onData){if(typeof e.onData!="function")throw new TypeError("options.onData, when provided, must be a function.");if(e.onData.length<2)throw new TypeError("options.onData, when provided, must be a function that takes in at least two arguments (data and position). Ignoring the position argument, which specifies the byte offset at which the data is to be written, can lead to broken outputs.")}if(e.chunked!==void 0&&typeof e.chunked!="boolean")throw new TypeError("options.chunked, when provided, must be a boolean.");if(e.chunkSize!==void 0&&(!Number.isInteger(e.chunkSize)||e.chunkSize<1024))throw new TypeError("options.chunkSize, when provided, must be an integer and not smaller than 1024.")}},Nt=class extends _e{constructor(e,t){if(super(),this.stream=e,this.options=t,!(e instanceof FileSystemWritableFileStream))throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");if(t!==void 0&&typeof t!="object")throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");if(t&&t.chunkSize!==void 0&&(!Number.isInteger(t.chunkSize)||t.chunkSize<=0))throw new TypeError("options.chunkSize, when provided, must be a positive integer")}},H,Q,_t=class{constructor(){this.pos=0,d(this,H,new Uint8Array(8)),d(this,Q,new DataView(i(this,H).buffer)),this.offsets=new WeakMap}seek(e){this.pos=e}writeU32(e){i(this,Q).setUint32(0,e,!1),this.write(i(this,H).subarray(0,4))}writeU64(e){i(this,Q).setUint32(0,Math.floor(e/2**32),!1),i(this,Q).setUint32(4,e,!1),this.write(i(this,H).subarray(0,8))}writeAscii(e){for(let t=0;t<e.length;t++)i(this,Q).setUint8(t%8,e.charCodeAt(t)),t%8===7&&this.write(i(this,H));e.length%8!==0&&this.write(i(this,H).subarray(0,e.length%8))}writeBox(e){if(this.offsets.set(e,this.pos),e.contents&&!e.children)this.writeBoxHeader(e,e.size??e.contents.byteLength+8),this.write(e.contents);else{let t=this.pos;if(this.writeBoxHeader(e,0),e.contents&&this.write(e.contents),e.children)for(let a of e.children)a&&this.writeBox(a);let n=this.pos,o=e.size??n-t;this.seek(t),this.writeBoxHeader(e,o),this.seek(n)}}writeBoxHeader(e,t){this.writeU32(e.largeSize?1:t),this.writeAscii(e.type),e.largeSize&&this.writeU64(t)}measureBoxHeader(e){return 8+(e.largeSize?8:0)}patchBox(e){let t=this.pos;this.seek(this.offsets.get(e)),this.writeBox(e),this.seek(t)}measureBox(e){if(e.contents&&!e.children)return this.measureBoxHeader(e)+e.contents.byteLength;{let t=this.measureBoxHeader(e);if(e.contents&&(t+=e.contents.byteLength),e.children)for(let n of e.children)n&&(t+=this.measureBox(n));return t}}};H=new WeakMap;Q=new WeakMap;var ke,$,de,le,Re,$e,Gi=class extends _t{constructor(e){super(),d(this,Re),d(this,ke,void 0),d(this,$,new ArrayBuffer(2**16)),d(this,de,new Uint8Array(i(this,$))),d(this,le,0),A(this,ke,e)}write(e){h(this,Re,$e).call(this,this.pos+e.byteLength),i(this,de).set(e,this.pos),this.pos+=e.byteLength,A(this,le,Math.max(i(this,le),this.pos))}finalize(){h(this,Re,$e).call(this,this.pos),i(this,ke).buffer=i(this,$).slice(0,Math.max(i(this,le),this.pos))}};ke=new WeakMap;$=new WeakMap;de=new WeakMap;le=new WeakMap;Re=new WeakSet;$e=function(e){let t=i(this,$).byteLength;for(;t<e;)t*=2;if(t===i(this,$).byteLength)return;let n=new ArrayBuffer(t),o=new Uint8Array(n);o.set(i(this,de),0),A(this,$,n),A(this,de,o)};var Ni=2**24,_i=2,fe,q,re,L,I,Me,Ze,ut,Lt,ft,Ot,ce,Ee,ct=class extends _t{constructor(e){var t,n;super(),d(this,Me),d(this,ut),d(this,ft),d(this,ce),d(this,fe,void 0),d(this,q,[]),d(this,re,void 0),d(this,L,void 0),d(this,I,[]),A(this,fe,e),A(this,re,((t=e.options)==null?void 0:t.chunked)??!1),A(this,L,((n=e.options)==null?void 0:n.chunkSize)??Ni)}write(e){i(this,q).push({data:e.slice(),start:this.pos}),this.pos+=e.byteLength}flush(){var n,o;if(i(this,q).length===0)return;let e=[],t=[...i(this,q)].sort((a,l)=>a.start-l.start);e.push({start:t[0].start,size:t[0].data.byteLength});for(let a=1;a<t.length;a++){let l=e[e.length-1],r=t[a];r.start<=l.start+l.size?l.size=Math.max(l.size,r.start+r.data.byteLength-l.start):e.push({start:r.start,size:r.data.byteLength})}for(let a of e){a.data=new Uint8Array(a.size);for(let l of i(this,q))a.start<=l.start&&l.start<a.start+a.size&&a.data.set(l.data,l.start-a.start);i(this,re)?(h(this,Me,Ze).call(this,a.data,a.start),h(this,ce,Ee).call(this)):(o=(n=i(this,fe).options).onData)==null||o.call(n,a.data,a.start)}i(this,q).length=0}finalize(){i(this,re)&&h(this,ce,Ee).call(this,!0)}};fe=new WeakMap;q=new WeakMap;re=new WeakMap;L=new WeakMap;I=new WeakMap;Me=new WeakSet;Ze=function(e,t){let n=i(this,I).findIndex(u=>u.start<=t&&t<u.start+i(this,L));n===-1&&(n=h(this,ft,Ot).call(this,t));let o=i(this,I)[n],a=t-o.start,l=e.subarray(0,Math.min(i(this,L)-a,e.byteLength));o.data.set(l,a);let r={start:a,end:a+l.byteLength};if(h(this,ut,Lt).call(this,o,r),o.written[0].start===0&&o.written[0].end===i(this,L)&&(o.shouldFlush=!0),i(this,I).length>_i){for(let u=0;u<i(this,I).length-1;u++)i(this,I)[u].shouldFlush=!0;h(this,ce,Ee).call(this)}l.byteLength<e.byteLength&&h(this,Me,Ze).call(this,e.subarray(l.byteLength),t+l.byteLength)};ut=new WeakSet;Lt=function(e,t){let n=0,o=e.written.length-1,a=-1;for(;n<=o;){let l=Math.floor(n+(o-n+1)/2);e.written[l].start<=t.start?(n=l+1,a=l):o=l-1}for(e.written.splice(a+1,0,t),(a===-1||e.written[a].end<t.start)&&a++;a<e.written.length-1&&e.written[a].end>=e.written[a+1].start;)e.written[a].end=Math.max(e.written[a].end,e.written[a+1].end),e.written.splice(a+1,1)};ft=new WeakSet;Ot=function(e){let n={start:Math.floor(e/i(this,L))*i(this,L),data:new Uint8Array(i(this,L)),written:[],shouldFlush:!1};return i(this,I).push(n),i(this,I).sort((o,a)=>o.start-a.start),i(this,I).indexOf(n)};ce=new WeakSet;Ee=function(e=!1){var t,n;for(let o=0;o<i(this,I).length;o++){let a=i(this,I)[o];if(!(!a.shouldFlush&&!e)){for(let l of a.written)(n=(t=i(this,fe).options).onData)==null||n.call(t,a.data.subarray(l.start,l.end),a.start+l.start);i(this,I).splice(o--,1)}}};var Li=class extends ct{constructor(e){var t;super(new Gt({onData:(n,o)=>e.stream.write({type:"write",data:n,position:o}),chunked:!0,chunkSize:(t=e.options)==null?void 0:t.chunkSize}))}},Ye=1e3,Oi=["avc","hevc","vp9","av1"],Wi=["aac","opus"],Hi=2082844800,qi=["strict","offset","cross-track-offset"],c,m,Pe,z,F,B,J,te,mt,V,j,me,Ke,Wt,Qe,Ht,pt,qt,Je,Vt,dt,jt,Be,et,P,_,ht,Xt,pe,De,Ue,vt,ie,ye,Ae,tt,Vi=class{constructor(e){if(d(this,Ke),d(this,Qe),d(this,pt),d(this,Je),d(this,dt),d(this,Be),d(this,P),d(this,ht),d(this,pe),d(this,Ue),d(this,ie),d(this,Ae),d(this,c,void 0),d(this,m,void 0),d(this,Pe,void 0),d(this,z,void 0),d(this,F,null),d(this,B,null),d(this,J,Math.floor(Date.now()/1e3)+Hi),d(this,te,[]),d(this,mt,1),d(this,V,[]),d(this,j,[]),d(this,me,!1),h(this,Ke,Wt).call(this,e),e.video=ue(e.video),e.audio=ue(e.audio),e.fastStart=ue(e.fastStart),this.target=e.target,A(this,c,{firstTimestampBehavior:"strict",...e}),e.target instanceof Ui)A(this,m,new Gi(e.target));else if(e.target instanceof Gt)A(this,m,new ct(e.target));else if(e.target instanceof Nt)A(this,m,new Li(e.target));else throw new Error(`Invalid target: ${e.target}`);h(this,Je,Vt).call(this),h(this,Qe,Ht).call(this)}addVideoChunk(e,t,n,o){if(!(e instanceof EncodedVideoChunk))throw new TypeError("addVideoChunk's first argument (sample) must be of type EncodedVideoChunk.");if(t&&typeof t!="object")throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");if(n!==void 0&&(!Number.isFinite(n)||n<0))throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");if(o!==void 0&&!Number.isFinite(o))throw new TypeError("addVideoChunk's fourth argument (compositionTimeOffset), when provided, must be a real number.");let a=new Uint8Array(e.byteLength);e.copyTo(a),this.addVideoChunkRaw(a,e.type,n??e.timestamp,e.duration,t,o)}addVideoChunkRaw(e,t,n,o,a,l){if(!(e instanceof Uint8Array))throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");if(t!=="key"&&t!=="delta")throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(n)||n<0)throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(o)||o<0)throw new TypeError("addVideoChunkRaw's fourth argument (duration) must be a non-negative real number.");if(a&&typeof a!="object")throw new TypeError("addVideoChunkRaw's fifth argument (meta), when provided, must be an object.");if(l!==void 0&&!Number.isFinite(l))throw new TypeError("addVideoChunkRaw's sixth argument (compositionTimeOffset), when provided, must be a real number.");if(h(this,Ae,tt).call(this),!i(this,c).video)throw new Error("No video track declared.");if(typeof i(this,c).fastStart=="object"&&i(this,F).samples.length===i(this,c).fastStart.expectedVideoChunks)throw new Error(`Cannot add more video chunks than specified in 'fastStart' (${i(this,c).fastStart.expectedVideoChunks}).`);let r=h(this,Be,et).call(this,i(this,F),e,t,n,o,a,l);if(i(this,c).fastStart==="fragmented"&&i(this,B)){for(;i(this,j).length>0&&i(this,j)[0].decodeTimestamp<=r.decodeTimestamp;){let u=i(this,j).shift();h(this,P,_).call(this,i(this,B),u)}r.decodeTimestamp<=i(this,B).lastDecodeTimestamp?h(this,P,_).call(this,i(this,F),r):i(this,V).push(r)}else h(this,P,_).call(this,i(this,F),r)}addAudioChunk(e,t,n){if(!(e instanceof EncodedAudioChunk))throw new TypeError("addAudioChunk's first argument (sample) must be of type EncodedAudioChunk.");if(t&&typeof t!="object")throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");if(n!==void 0&&(!Number.isFinite(n)||n<0))throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");let o=new Uint8Array(e.byteLength);e.copyTo(o),this.addAudioChunkRaw(o,e.type,n??e.timestamp,e.duration,t)}addAudioChunkRaw(e,t,n,o,a){if(!(e instanceof Uint8Array))throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");if(t!=="key"&&t!=="delta")throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(n)||n<0)throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(o)||o<0)throw new TypeError("addAudioChunkRaw's fourth argument (duration) must be a non-negative real number.");if(a&&typeof a!="object")throw new TypeError("addAudioChunkRaw's fifth argument (meta), when provided, must be an object.");if(h(this,Ae,tt).call(this),!i(this,c).audio)throw new Error("No audio track declared.");if(typeof i(this,c).fastStart=="object"&&i(this,B).samples.length===i(this,c).fastStart.expectedAudioChunks)throw new Error(`Cannot add more audio chunks than specified in 'fastStart' (${i(this,c).fastStart.expectedAudioChunks}).`);let l=h(this,Be,et).call(this,i(this,B),e,t,n,o,a);if(i(this,c).fastStart==="fragmented"&&i(this,F)){for(;i(this,V).length>0&&i(this,V)[0].decodeTimestamp<=l.decodeTimestamp;){let r=i(this,V).shift();h(this,P,_).call(this,i(this,F),r)}l.decodeTimestamp<=i(this,F).lastDecodeTimestamp?h(this,P,_).call(this,i(this,B),l):i(this,j).push(l)}else h(this,P,_).call(this,i(this,B),l)}finalize(){if(i(this,me))throw new Error("Cannot finalize a muxer more than once.");if(i(this,c).fastStart==="fragmented"){for(let t of i(this,V))h(this,P,_).call(this,i(this,F),t);for(let t of i(this,j))h(this,P,_).call(this,i(this,B),t);h(this,Ue,vt).call(this,!1)}else i(this,F)&&h(this,pe,De).call(this,i(this,F)),i(this,B)&&h(this,pe,De).call(this,i(this,B));let e=[i(this,F),i(this,B)].filter(Boolean);if(i(this,c).fastStart==="in-memory"){let t;for(let o=0;o<2;o++){let a=Te(e,i(this,J)),l=i(this,m).measureBox(a);t=i(this,m).measureBox(i(this,z));let r=i(this,m).pos+l+t;for(let u of i(this,te)){u.offset=r;for(let{data:p}of u.samples)r+=p.byteLength,t+=p.byteLength}if(r<2**32)break;t>=2**32&&(i(this,z).largeSize=!0)}let n=Te(e,i(this,J));i(this,m).writeBox(n),i(this,z).size=t,i(this,m).writeBox(i(this,z));for(let o of i(this,te))for(let a of o.samples)i(this,m).write(a.data),a.data=null}else if(i(this,c).fastStart==="fragmented"){let t=i(this,m).pos,n=Fi(e);i(this,m).writeBox(n);let o=i(this,m).pos-t;i(this,m).seek(i(this,m).pos-4),i(this,m).writeU32(o)}else{let t=i(this,m).offsets.get(i(this,z)),n=i(this,m).pos-t;i(this,z).size=n,i(this,z).largeSize=n>=2**32,i(this,m).patchBox(i(this,z));let o=Te(e,i(this,J));if(typeof i(this,c).fastStart=="object"){i(this,m).seek(i(this,Pe)),i(this,m).writeBox(o);let a=t-i(this,m).pos;i(this,m).writeBox(Hn(a))}else i(this,m).writeBox(o)}h(this,ie,ye).call(this),i(this,m).finalize(),A(this,me,!0)}};c=new WeakMap;m=new WeakMap;Pe=new WeakMap;z=new WeakMap;F=new WeakMap;B=new WeakMap;J=new WeakMap;te=new WeakMap;mt=new WeakMap;V=new WeakMap;j=new WeakMap;me=new WeakMap;Ke=new WeakSet;Wt=function(e){if(typeof e!="object")throw new TypeError("The muxer requires an options object to be passed to its constructor.");if(!(e.target instanceof _e))throw new TypeError("The target must be provided and an instance of Target.");if(e.video){if(!Oi.includes(e.video.codec))throw new TypeError(`Unsupported video codec: ${e.video.codec}`);if(!Number.isInteger(e.video.width)||e.video.width<=0)throw new TypeError(`Invalid video width: ${e.video.width}. Must be a positive integer.`);if(!Number.isInteger(e.video.height)||e.video.height<=0)throw new TypeError(`Invalid video height: ${e.video.height}. Must be a positive integer.`);const t=e.video.rotation;if(typeof t=="number"&&![0,90,180,270].includes(t))throw new TypeError(`Invalid video rotation: ${t}. Has to be 0, 90, 180 or 270.`);if(Array.isArray(t)&&(t.length!==9||t.some(n=>typeof n!="number")))throw new TypeError(`Invalid video transformation matrix: ${t.join()}`);if(e.video.frameRate!==void 0&&(!Number.isInteger(e.video.frameRate)||e.video.frameRate<=0))throw new TypeError(`Invalid video frame rate: ${e.video.frameRate}. Must be a positive integer.`)}if(e.audio){if(!Wi.includes(e.audio.codec))throw new TypeError(`Unsupported audio codec: ${e.audio.codec}`);if(!Number.isInteger(e.audio.numberOfChannels)||e.audio.numberOfChannels<=0)throw new TypeError(`Invalid number of audio channels: ${e.audio.numberOfChannels}. Must be a positive integer.`);if(!Number.isInteger(e.audio.sampleRate)||e.audio.sampleRate<=0)throw new TypeError(`Invalid audio sample rate: ${e.audio.sampleRate}. Must be a positive integer.`)}if(e.firstTimestampBehavior&&!qi.includes(e.firstTimestampBehavior))throw new TypeError(`Invalid first timestamp behavior: ${e.firstTimestampBehavior}`);if(typeof e.fastStart=="object"){if(e.video){if(e.fastStart.expectedVideoChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedVideoChunks'.");if(!Number.isInteger(e.fastStart.expectedVideoChunks)||e.fastStart.expectedVideoChunks<0)throw new TypeError("'expectedVideoChunks' must be a non-negative integer.")}if(e.audio){if(e.fastStart.expectedAudioChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedAudioChunks'.");if(!Number.isInteger(e.fastStart.expectedAudioChunks)||e.fastStart.expectedAudioChunks<0)throw new TypeError("'expectedAudioChunks' must be a non-negative integer.")}}else if(![!1,"in-memory","fragmented"].includes(e.fastStart))throw new TypeError("'fastStart' option must be false, 'in-memory', 'fragmented' or an object.");if(e.minFragmentDuration!==void 0&&(!Number.isFinite(e.minFragmentDuration)||e.minFragmentDuration<0))throw new TypeError("'minFragmentDuration' must be a non-negative number.")};Qe=new WeakSet;Ht=function(){var e;if(i(this,m).writeBox(Wn({holdsAvc:((e=i(this,c).video)==null?void 0:e.codec)==="avc",fragmented:i(this,c).fastStart==="fragmented"})),A(this,Pe,i(this,m).pos),i(this,c).fastStart==="in-memory")A(this,z,Xe(!1));else if(i(this,c).fastStart!=="fragmented"){if(typeof i(this,c).fastStart=="object"){let t=h(this,pt,qt).call(this);i(this,m).seek(i(this,m).pos+t)}A(this,z,Xe(!0)),i(this,m).writeBox(i(this,z))}h(this,ie,ye).call(this)};pt=new WeakSet;qt=function(){if(typeof i(this,c).fastStart!="object")return;let e=0,t=[i(this,c).fastStart.expectedVideoChunks,i(this,c).fastStart.expectedAudioChunks];for(let n of t)n&&(e+=8*Math.ceil(2/3*n),e+=4*n,e+=12*Math.ceil(2/3*n),e+=4*n,e+=8*n);return e+=4096,e};Je=new WeakSet;Vt=function(){if(i(this,c).video&&A(this,F,{id:1,info:{type:"video",codec:i(this,c).video.codec,width:i(this,c).video.width,height:i(this,c).video.height,rotation:i(this,c).video.rotation??0,decoderConfig:null},timescale:i(this,c).video.frameRate??57600,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,c).audio&&(A(this,B,{id:i(this,c).video?2:1,info:{type:"audio",codec:i(this,c).audio.codec,numberOfChannels:i(this,c).audio.numberOfChannels,sampleRate:i(this,c).audio.sampleRate,decoderConfig:null},timescale:i(this,c).audio.sampleRate,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,c).audio.codec==="aac")){let e=h(this,dt,jt).call(this,2,i(this,c).audio.sampleRate,i(this,c).audio.numberOfChannels);i(this,B).info.decoderConfig={codec:i(this,c).audio.codec,description:e,numberOfChannels:i(this,c).audio.numberOfChannels,sampleRate:i(this,c).audio.sampleRate}}};dt=new WeakSet;jt=function(e,t,n){let a=[96e3,88200,64e3,48e3,44100,32e3,24e3,22050,16e3,12e3,11025,8e3,7350].indexOf(t),l=n,r="";r+=e.toString(2).padStart(5,"0"),r+=a.toString(2).padStart(4,"0"),a===15&&(r+=t.toString(2).padStart(24,"0")),r+=l.toString(2).padStart(4,"0");let u=Math.ceil(r.length/8)*8;r=r.padEnd(u,"0");let p=new Uint8Array(r.length/8);for(let y=0;y<r.length;y+=8)p[y/8]=parseInt(r.slice(y,y+8),2);return p};Be=new WeakSet;et=function(e,t,n,o,a,l,r){let u=o/1e6,p=(o-(r??0))/1e6,y=a/1e6,b=h(this,ht,Xt).call(this,u,p,e);return u=b.presentationTimestamp,p=b.decodeTimestamp,l!=null&&l.decoderConfig&&(e.info.decoderConfig===null?e.info.decoderConfig=l.decoderConfig:Object.assign(e.info.decoderConfig,l.decoderConfig)),{presentationTimestamp:u,decodeTimestamp:p,duration:y,data:t,size:t.byteLength,type:n,timescaleUnitsToNextSample:U(y,e.timescale)}};P=new WeakSet;_=function(e,t){i(this,c).fastStart!=="fragmented"&&e.samples.push(t);const n=U(t.presentationTimestamp-t.decodeTimestamp,e.timescale);if(e.lastTimescaleUnits!==null){let a=U(t.decodeTimestamp,e.timescale,!1),l=Math.round(a-e.lastTimescaleUnits);if(e.lastTimescaleUnits+=l,e.lastSample.timescaleUnitsToNextSample=l,i(this,c).fastStart!=="fragmented"){let r=Ie(e.timeToSampleTable);r.sampleCount===1?(r.sampleDelta=l,r.sampleCount++):r.sampleDelta===l?r.sampleCount++:(r.sampleCount--,e.timeToSampleTable.push({sampleCount:2,sampleDelta:l}));const u=Ie(e.compositionTimeOffsetTable);u.sampleCompositionTimeOffset===n?u.sampleCount++:e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:n})}}else e.lastTimescaleUnits=0,i(this,c).fastStart!=="fragmented"&&(e.timeToSampleTable.push({sampleCount:1,sampleDelta:U(t.duration,e.timescale)}),e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:n}));e.lastSample=t;let o=!1;if(!e.currentChunk)o=!0;else{let a=t.presentationTimestamp-e.currentChunk.startTimestamp;if(i(this,c).fastStart==="fragmented"){let l=i(this,F)??i(this,B);const r=i(this,c).minFragmentDuration??1;e===l&&t.type==="key"&&a>=r&&(o=!0,h(this,Ue,vt).call(this))}else o=a>=.5}o&&(e.currentChunk&&h(this,pe,De).call(this,e),e.currentChunk={startTimestamp:t.presentationTimestamp,samples:[]}),e.currentChunk.samples.push(t)};ht=new WeakSet;Xt=function(e,t,n){var r,u;const o=i(this,c).firstTimestampBehavior==="strict",a=n.lastDecodeTimestamp===-1;if(o&&a&&t!==0)throw new Error(`The first chunk for your media track must have a timestamp of 0 (received DTS=${t}).Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of thedocument, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
`);if(i(this,c).firstTimestampBehavior==="offset"||i(this,c).firstTimestampBehavior==="cross-track-offset"){n.firstDecodeTimestamp===void 0&&(n.firstDecodeTimestamp=t);let p;i(this,c).firstTimestampBehavior==="offset"?p=n.firstDecodeTimestamp:p=Math.min(((r=i(this,F))==null?void 0:r.firstDecodeTimestamp)??1/0,((u=i(this,B))==null?void 0:u.firstDecodeTimestamp)??1/0),t-=p,e-=p}if(t<n.lastDecodeTimestamp)throw new Error(`Timestamps must be monotonically increasing (DTS went from ${n.lastDecodeTimestamp*1e6} to ${t*1e6}).`);return n.lastDecodeTimestamp=t,{presentationTimestamp:e,decodeTimestamp:t}};pe=new WeakSet;De=function(e){if(i(this,c).fastStart==="fragmented")throw new Error("Can't finalize individual chunks if 'fastStart' is set to 'fragmented'.");if(e.currentChunk){if(e.finalizedChunks.push(e.currentChunk),i(this,te).push(e.currentChunk),(e.compactlyCodedChunkTable.length===0||Ie(e.compactlyCodedChunkTable).samplesPerChunk!==e.currentChunk.samples.length)&&e.compactlyCodedChunkTable.push({firstChunk:e.finalizedChunks.length,samplesPerChunk:e.currentChunk.samples.length}),i(this,c).fastStart==="in-memory"){e.currentChunk.offset=0;return}e.currentChunk.offset=i(this,m).pos;for(let t of e.currentChunk.samples)i(this,m).write(t.data),t.data=null;h(this,ie,ye).call(this)}};Ue=new WeakSet;vt=function(e=!0){if(i(this,c).fastStart!=="fragmented")throw new Error("Can't finalize a fragment unless 'fastStart' is set to 'fragmented'.");let t=[i(this,F),i(this,B)].filter(u=>u&&u.currentChunk);if(t.length===0)return;let n=_n(this,mt)._++;if(n===1){let u=Te(t,i(this,J),!0);i(this,m).writeBox(u)}let o=i(this,m).pos,a=Bt(n,t);i(this,m).writeBox(a);{let u=Xe(!1),p=0;for(let b of t)for(let w of b.currentChunk.samples)p+=w.size;let y=i(this,m).measureBox(u)+p;y>=2**32&&(u.largeSize=!0,y=i(this,m).measureBox(u)+p),u.size=y,i(this,m).writeBox(u)}for(let u of t){u.currentChunk.offset=i(this,m).pos,u.currentChunk.moofOffset=o;for(let p of u.currentChunk.samples)i(this,m).write(p.data),p.data=null}let l=i(this,m).pos;i(this,m).seek(i(this,m).offsets.get(a));let r=Bt(n,t);i(this,m).writeBox(r),i(this,m).seek(l);for(let u of t)u.finalizedChunks.push(u.currentChunk),i(this,te).push(u.currentChunk),u.currentChunk=null;e&&h(this,ie,ye).call(this)};ie=new WeakSet;ye=function(){i(this,m)instanceof ct&&i(this,m).flush()};Ae=new WeakSet;tt=function(){if(i(this,me))throw new Error("Cannot add new video or audio chunks after the file has been finalized.")};const $t=document.querySelector("#app");if(!$t)throw new Error("Missing #app root element");$t.innerHTML=`
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
`;const k=document.querySelector("#gl-canvas"),ee=document.querySelector("#scene-list"),nt=document.querySelector("#control-list"),Fe=document.querySelector("#panel-actions"),se=document.querySelector("#key-help"),Ge=document.querySelector("#hud-title"),Zt=document.querySelector("#hud-desc"),Le=document.querySelector("[data-action='toggle-sidebar']"),yt=document.querySelector(".stage");if(!k||!ee||!nt||!Fe||!se||!Ge||!Zt||!Le||!yt)throw new Error("Missing required UI elements");const s=k.getContext("webgl2",{antialias:!0});if(!s)throw yt.innerHTML=`
    <div class="fallback">
      <h2>WebGL2 unavailable</h2>
      <p>Your browser or GPU does not expose WebGL2. Try a different browser.</p>
    </div>
  `,new Error("WebGL2 unavailable");const ji=s.getExtension("EXT_color_buffer_float"),Xi=!!ji;s.disable(s.DEPTH_TEST);s.disable(s.BLEND);const At=Nn(s),Yt=new Map,Kt={},oe={},Qt={},it=new Map;for(const e of ae){const t=Dn(s,mn,e.fragment),n=new Set;n.add(e.resolutionUniform),n.add(e.timeUniform),e.loopUniform&&n.add(e.loopUniform),e.stateful&&(n.add(e.passUniform??"uPass"),n.add(e.stateUniform??"uState"),n.add(e.gridUniform??"uGridSize"));for(const r of e.params)n.add(r.uniform);const o=Un(s,t,Array.from(n));Yt.set(e.id,{program:t,uniforms:o});const a={},l={};for(const r of e.params)a[r.id]=r,l[r.id]=r.type==="seed"?Math.floor(Math.random()*1e6):r.value;Kt[e.id]=a,oe[e.id]={...l},Qt[e.id]={...l}}let O=ae[0],ot={},bt=performance.now(),qe=null,Ve=null,X=null,we=[],Jt=0,ze=null,be=!1;function $i(){const e=["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm","video/mp4"];for(const t of e)if(MediaRecorder.isTypeSupported(t))return t;return""}function Zi(e){return e.startsWith("video/mp4")?"mp4":"webm"}function Yi(e){const t=Math.floor(e/1e3),n=String(Math.floor(t/60)).padStart(2,"0"),o=String(t%60).padStart(2,"0");return`${n}:${o}`}function Ki(){const e=document.getElementById("rec-badge");!e||!be||(e.textContent=`⏺ ${Yi(performance.now()-Jt)}`)}function Qi(){const e=$i();if(!e){alert("Recording is not supported in this browser.");return}const t=k.captureStream(60);we=[],X=new MediaRecorder(t,{mimeType:e,videoBitsPerSecond:16e6}),X.ondataavailable=n=>{n.data.size>0&&we.push(n.data)},X.onstop=()=>{const n=Zi(e),o=new Blob(we,{type:e}),a=URL.createObjectURL(o),l=document.createElement("a");l.href=a,l.download=`${O.id}-${Date.now()}.${n}`,l.click(),URL.revokeObjectURL(a),we=[]},X.start(500),be=!0,Jt=performance.now(),en(),ze=window.setInterval(Ki,250)}function Ji(){X&&X.state!=="inactive"&&X.stop(),be=!1,ze!==null&&(clearInterval(ze),ze=null),en()}function eo(){be?Ji():Qi()}function en(){const e=document.getElementById("rec-btn"),t=document.getElementById("rec-badge");!e||!t||(be?(e.textContent="Stop",e.classList.add("recording"),t.classList.remove("hidden")):(e.textContent="Record",e.classList.remove("recording"),t.classList.add("hidden"),t.textContent=""))}let Ne=!1;function at(e){Le.classList.toggle("hidden",e)}function to(){at(!1),Ve!==null&&window.clearTimeout(Ve),Ve=window.setTimeout(()=>{at(!0)},2500)}function tn(){const e=document.body.classList.contains("sidebar-collapsed");Le.textContent=e?">>":"<<"}function no(){var e;(e=Ge.parentElement)==null||e.classList.remove("hidden"),qe!==null&&window.clearTimeout(qe),qe=window.setTimeout(()=>{var t;(t=Ge.parentElement)==null||t.classList.add("hidden")},1e4)}function Ft(e){const t=s.createTexture();if(!t)throw new Error("Failed to create state texture");return s.bindTexture(s.TEXTURE_2D,t),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_MIN_FILTER,s.NEAREST),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_MAG_FILTER,s.NEAREST),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_S,s.REPEAT),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_T,s.REPEAT),s.texImage2D(s.TEXTURE_2D,0,s.RGBA16F,e,e,0,s.RGBA,s.HALF_FLOAT,null),s.bindTexture(s.TEXTURE_2D,null),t}function io(e){const t=Ft(e),n=Ft(e),o=s.createFramebuffer(),a=s.createFramebuffer();if(!o||!a)throw new Error("Failed to create framebuffer");return s.bindFramebuffer(s.FRAMEBUFFER,o),s.framebufferTexture2D(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,t,0),s.bindFramebuffer(s.FRAMEBUFFER,a),s.framebufferTexture2D(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,n,0),s.bindFramebuffer(s.FRAMEBUFFER,null),{size:e,textures:[t,n],fbos:[o,a],index:0,needsInit:!0}}function gt(e){if(!e.stateful)return null;let t=it.get(e.id);const n=e.bufferSize??192;return(!t||t.size!==n)&&(t=io(n),it.set(e.id,t)),t}function he(e){const t=it.get(e);t&&(t.needsInit=!0)}function oo(e,t){let n=t;return e.min!==void 0&&(n=Math.max(e.min,n)),e.max!==void 0&&(n=Math.min(e.max,n)),e.type==="int"&&(n=Math.round(n)),n}function nn(e,t){if(e.type==="int")return String(Math.round(t));const n=e.step??.01,o=n<1?Math.min(4,Math.max(2,Math.ceil(-Math.log10(n)))):0;return t.toFixed(o)}function ao(e){return e.length===1?e.toLowerCase():e}function lo(e){return e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement||e instanceof HTMLSelectElement}function ve(e,t,n,o=!0){const a=Kt[e][t];if(!a)return;const l=oo(a,n);if(oe[e][t]=l,e===O.id&&o){const r=ot[t];r!=null&&r.range&&(r.range.value=String(l)),r!=null&&r.number&&(r.number.value=nn(a,l))}}function ro(e){var t;for(const n of((t=ae.find(o=>o.id===e))==null?void 0:t.params)??[])n.type==="seed"&&ve(e,n.id,Math.floor(Math.random()*1e6));he(e)}function so(e){const t=Qt[e];for(const[n,o]of Object.entries(t))ve(e,n,o,!0);he(e)}function uo(){ee.innerHTML="";for(const e of ae){const t=document.createElement("option");t.value=e.id,t.textContent=e.name,ee.appendChild(t)}ee.addEventListener("change",()=>{on(ee.value)})}function fo(e){Fe.innerHTML="";const t=document.createElement("button");if(t.className="ghost small",t.textContent="Reset",t.addEventListener("click",()=>so(e.id)),Fe.appendChild(t),e.params.some(o=>o.type==="seed")){const o=document.createElement("button");o.className="ghost small",o.textContent="Reseed",o.addEventListener("click",()=>ro(e.id)),Fe.appendChild(o)}}function co(e){nt.innerHTML="",ot={};for(const t of e.params){if(t.type==="seed")continue;const n=document.createElement("div");n.className="control";const o=document.createElement("div");o.className="control-header";const a=document.createElement("label");if(a.textContent=t.label,o.appendChild(a),t.key){const p=document.createElement("span");p.className="key-cap",p.textContent=`${t.key.inc.toUpperCase()}/${t.key.dec.toUpperCase()}`,o.appendChild(p)}const l=document.createElement("div");l.className="control-inputs";const r=document.createElement("input");r.type="range",r.min=String(t.min??0),r.max=String(t.max??1),r.step=String(t.step??(t.type==="int"?1:.01)),r.value=String(oe[e.id][t.id]),r.addEventListener("input",p=>{const y=Number(p.target.value);Number.isNaN(y)||ve(e.id,t.id,y)});const u=document.createElement("input");u.type="number",u.min=r.min,u.max=r.max,u.step=r.step,u.value=nn(t,oe[e.id][t.id]),u.addEventListener("input",p=>{const y=Number(p.target.value);Number.isNaN(y)||ve(e.id,t.id,y)}),l.appendChild(r),l.appendChild(u),n.appendChild(o),n.appendChild(l),nt.appendChild(n),ot[t.id]={range:r,number:u}}}function mo(e){se.innerHTML="";for(const t of e.params){if(!t.key||t.type==="seed")continue;const n=document.createElement("div");n.className="key-row",n.textContent=`${t.key.inc.toUpperCase()}/${t.key.dec.toUpperCase()}  ${t.label}`,se.appendChild(n)}se.childElementCount||(se.textContent="No mapped keys for this scene.")}function on(e){const t=ae.find(n=>n.id===e);t&&(O=t,t.stateful&&(gt(t),he(t.id)),Ge.textContent=t.name,Zt.textContent=t.description,no(),fo(t),co(t),mo(t),ee.value=t.id)}function po(e){if(lo(e.target))return;const t=ao(e.key),n=O.params;for(const o of n){if(!o.key||o.type==="seed")continue;const a=t===o.key.inc,l=t===o.key.dec;if(!a&&!l)continue;const r=e.shiftKey&&o.key.shiftStep?o.key.shiftStep:o.key.step,p=oe[O.id][o.id]+r*(a?1:-1);ve(O.id,o.id,p),e.preventDefault();break}}function Ce(e,t,n,o,a){const l=oe[e.id],r=t.uniforms,u=r[e.resolutionUniform];u&&s.uniform2f(u,o,a);const p=r[e.timeUniform];if(p)if(e.timeMode==="phase"){const b=e.loopDuration??8,w=n%b/b;s.uniform1f(p,w)}else if(e.timeMode==="looped"){const b=e.loopDuration??8,w=n%b;if(s.uniform1f(p,w),e.loopUniform){const S=r[e.loopUniform];S&&s.uniform1f(S,b)}}else s.uniform1f(p,n);const y={};for(const b of e.params){const w=r[b.uniform],S=l[b.id];if(b.component!==void 0){const W=y[b.uniform]??[0,0,0];W[b.component]=S,y[b.uniform]=W;continue}w&&(b.type==="int"?s.uniform1i(w,Math.round(S)):s.uniform1f(w,S))}for(const[b,w]of Object.entries(y)){const S=r[b];S&&s.uniform3f(S,w[0],w[1],w[2])}}function je(e,t,n,o){const a=t.uniforms[e.passUniform??"uPass"];a&&s.uniform1i(a,o);const l=t.uniforms[e.gridUniform??"uGridSize"];l&&s.uniform2f(l,n.size,n.size);const r=t.uniforms[e.stateUniform??"uState"];r&&s.uniform1i(r,0)}function an(e,t,n,o){const a=Yt.get(e.id);if(a){if(e.stateful){if(!Xi)return;const l=gt(e);if(!l)return;const r=()=>l.textures[l.index],u=()=>l.fbos[(l.index+1)%2];s.useProgram(a.program),s.bindVertexArray(At),l.needsInit&&(s.bindFramebuffer(s.FRAMEBUFFER,u()),s.viewport(0,0,l.size,l.size),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,r()),Ce(e,a,t,n,o),je(e,a,l,2),s.drawArrays(s.TRIANGLES,0,3),l.index=(l.index+1)%2,l.needsInit=!1),s.bindFramebuffer(s.FRAMEBUFFER,u()),s.viewport(0,0,l.size,l.size),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,r()),Ce(e,a,t,n,o),je(e,a,l,0),s.drawArrays(s.TRIANGLES,0,3),l.index=(l.index+1)%2,s.bindFramebuffer(s.FRAMEBUFFER,null),s.viewport(0,0,n,o),s.clearColor(0,0,0,1),s.clear(s.COLOR_BUFFER_BIT),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,r()),Ce(e,a,t,n,o),je(e,a,l,1),s.drawArrays(s.TRIANGLES,0,3);return}s.viewport(0,0,n,o),s.clearColor(0,0,0,1),s.clear(s.COLOR_BUFFER_BIT),s.useProgram(a.program),s.bindVertexArray(At),Ce(e,a,t,n,o),s.drawArrays(s.TRIANGLES,0,3)}}function St(e){if(Ne)return;const t=(e-bt)/1e3,{width:n,height:o}=Gn(k);an(O,t,n,o),requestAnimationFrame(St)}function ho(){return new Promise(e=>requestAnimationFrame(()=>e()))}async function vo(e,t,n,o,a){if(typeof VideoEncoder>"u"||typeof VideoFrame>"u"){alert("Offline rendering requires the WebCodecs API (Chrome/Edge 94+, Safari 16.4+).");return}const l={codec:"avc1.640028",width:o,height:a,bitrate:16e6,framerate:n};if(!(await VideoEncoder.isConfigSupported(l)).supported){alert("H.264 video encoding is not supported on this device.");return}let u;try{u=await window.showSaveFilePicker({suggestedName:`${e.id}-${o}x${a}-${n}fps-${t}s.mp4`,types:[{description:"MP4 Video",accept:{"video/mp4":[".mp4"]}}]})}catch{return}const p=await u.createWritable(),y=document.getElementById("offline-btn"),b=document.getElementById("offline-progress"),w=document.getElementById("offline-bar"),S=document.getElementById("offline-status");y&&(y.disabled=!0),b==null||b.classList.remove("hidden"),S==null||S.classList.remove("hidden"),Ne=!0,e.stateful&&(gt(e),he(e.id));const W=k.width,N=k.height,E=k.style.width,Y=k.style.height;k.width=o,k.height=a,k.style.width=`${o}px`,k.style.height=`${a}px`;const ln=new Nt(p),xt=new Vi({target:ln,video:{codec:"avc",width:o,height:a},fastStart:!1});let ge=null;const K=new VideoEncoder({output:(C,We)=>xt.addVideoChunk(C,We??void 0),error:C=>{ge=C}});K.configure(l);const Se=Math.ceil(t*n),wt=Math.round(1e6/n),rn=n*2,sn=n*30,un=performance.now();let xe=!1;const Ct=C=>{C.preventDefault(),xe=!0};k.addEventListener("webglcontextlost",Ct);for(let C=0;C<Se&&!(ge||xe);C++){const We=C/n;an(e,We,o,a),s.finish();const Tt=new VideoFrame(k,{timestamp:C*wt,duration:wt});K.encode(Tt,{keyFrame:C%rn===0}),Tt.close(),C>0&&C%sn===0&&await K.flush();const fn=(C+1)/Se*100;if(w&&(w.style.width=`${fn}%`),C%30===0){const cn=(performance.now()-un)/1e3/(C+1)*(Se-C-1);S&&(S.textContent=`Frame ${C+1}/${Se}  —  ~${Math.ceil(cn)}s left`)}K.encodeQueueSize>10&&await new Promise(kt=>setTimeout(kt,1)),await ho()}k.removeEventListener("webglcontextlost",Ct);try{await K.flush(),K.close(),xt.finalize(),await p.close()}catch(C){console.error("Finalization failed:",C);try{await p.close()}catch{}}k.style.width=E,k.style.height=Y,k.width=W,k.height=N,e.stateful&&he(e.id),Ne=!1,bt=performance.now(),requestAnimationFrame(St),y&&(y.disabled=!1);const Oe=ge?ge.message:null;S&&(xe?S.textContent="Context lost — partial video saved.":Oe?S.textContent=`Error: ${Oe}`:S.textContent="Done!"),setTimeout(()=>{b==null||b.classList.add("hidden"),S==null||S.classList.add("hidden"),w&&(w.style.width="0%")},xe||!!Oe?8e3:3e3)}Le.addEventListener("click",()=>{document.body.classList.toggle("sidebar-collapsed"),tn()});yt.addEventListener("mousemove",()=>{to()});document.addEventListener("keydown",po);document.addEventListener("visibilitychange",()=>{document.hidden||(bt=performance.now())});var zt;(zt=document.getElementById("rec-btn"))==null||zt.addEventListener("click",eo);var It;(It=document.getElementById("offline-btn"))==null||It.addEventListener("click",()=>{if(Ne)return;const e=document.getElementById("offline-duration"),t=document.getElementById("offline-fps"),n=document.getElementById("offline-res"),o=Math.max(1,Math.min(3600,Number((e==null?void 0:e.value)??10))),a=Number((t==null?void 0:t.value)??60),[l,r]=((n==null?void 0:n.value)??"1920x1080").split("x").map(Number);vo(O,o,a,l,r)});uo();on(O.id);tn();at(!0);requestAnimationFrame(St);
