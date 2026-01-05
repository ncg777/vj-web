(function(){const n=document.createElement("link").relList;if(n&&n.supports&&n.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))t(i);new MutationObserver(i=>{for(const l of i)if(l.type==="childList")for(const a of l.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&t(a)}).observe(document,{childList:!0,subtree:!0});function o(i){const l={};return i.integrity&&(l.integrity=i.integrity),i.referrerPolicy&&(l.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?l.credentials="include":i.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function t(i){if(i.ep)return;i.ep=!0;const l=o(i);fetch(i.href,l)}})();const z=`#version 300 es
precision highp float;

const vec2 verts[3] = vec2[](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
);

void main() {
  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
`,_=`#version 300 es
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
`,W=`#version 300 es
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
`,V=`#version 300 es
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

float loopTime(float t, float duration) {
  float phase = mod(t, duration) / duration;
  return phase * TAU;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

  float t = loopTime(iTime * uSpeed, uLoopDuration);
  float phase = t / TAU;

  float r = length(uv);
  float a = atan(uv.y, uv.x);
  a += uTwist * r;

  vec2 dir = vec2(cos(a), sin(a));
  vec2 tOff = vec2(cos(t), sin(t)) * (0.6 * uNoiseScale);
  vec2 np = dir * (0.75 * uNoiseScale) + tOff + r * (2.0 * uNoiseScale);
  float n = fbm(np);
  r += uNoiseAmp * n;
  a += uNoiseAmp * 0.5 * n;

  float stripePhase = fract(phase + r * 0.5);
  float stripe = smoothstep(0.3, 0.5, sin(a * 8.0 + stripePhase * TAU));

  vec3 baseHue = uBaseColor;
  vec3 dynamicHue = 0.5 + 0.5 * cos(vec3(0.0, 0.6, 1.2) + a * 2.0 + n * 2.0 + uColorCycle * t);
  vec3 col = mix(baseHue, dynamicHue, 0.7);

  float stripeMask = stripe;
  col = mix(col, vec3(1.0), 0.6 * stripeMask);

  float fogBase = exp(-r * uFogDensity);
  float glowBase = pow(fogBase, 2.0);
  float e = 0.003 * max(0.5, uNoiseScale);
  vec2 grad;
  grad.x = fbm(np + vec2(e, 0.0)) - fbm(np - vec2(e, 0.0));
  grad.y = fbm(np + vec2(0.0, e)) - fbm(np - vec2(0.0, e));
  vec2 normal2D = normalize(grad + vec2(1e-6));
  float refractStrength = 0.03;
  vec2 uvR = uv + normal2D * refractStrength * (0.3 + 0.7 * glowBase);

  float rR = length(uvR);
  float aR = atan(uvR.y, uvR.x);
  aR += uTwist * rR;
  vec2 dirR = vec2(cos(aR), sin(aR));
  vec2 npR = dirR * (0.75 * uNoiseScale) + tOff + rR * (2.0 * uNoiseScale);
  float nR = fbm(npR);
  float stripePhaseR = fract(phase + rR * 0.5);
  float stripeR = smoothstep(0.3, 0.5, sin(aR * 8.0 + stripePhaseR * TAU));
  vec3 dynamicHueR = 0.5 + 0.5 * cos(vec3(0.0, 0.6, 1.2) + aR * 2.0 + nR * 2.0 + uColorCycle * t);
  vec3 colR = mix(baseHue, dynamicHueR, 0.7);
  colR = mix(colR, vec3(1.0), 0.6 * stripeR);

  col = mix(col, colR, 0.6);

  float fog = fogBase;
  float glow = glowBase;
  col *= mix(0.6, 1.6, fog);
  col += glow * 0.35 * (0.6 * baseHue + 0.4 * dynamicHue);

  col = clamp(col, 0.0, 1.0);
  outColor = vec4(col, 1.0);
}
`,K=`#version 300 es
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
`,$=`#version 300 es
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
`,g=[{id:"neon",name:"Neon Isoclines",description:"Electric contour bands driven by seeded radial harmonics.",fragment:_,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"components",label:"Components",uniform:"uComponents",type:"int",value:64,min:1,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:10}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:8,min:1,max:64,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:10}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.25,min:.01,max:.75,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.05}},{id:"noiseAmount",label:"Noise Amount",uniform:"uNoiseAmount",type:"float",value:2.5,min:0,max:5,step:.05,key:{inc:"r",dec:"f",step:.1,shiftStep:.25}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tanh-terrain",name:"Tanh Terrain Isoclines",description:"Tanh warped contours with bubbling noise and topo glow.",fragment:W,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:2.1,min:.1,max:6,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"octaves",label:"Octaves",uniform:"uOctaves",type:"int",value:4,min:1,max:12,step:1,key:{inc:"3",dec:"4",step:1}},{id:"lacunarity",label:"Lacunarity",uniform:"uLacunarity",type:"float",value:1.4,min:1.01,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"gain",label:"Gain",uniform:"uGain",type:"float",value:.5,min:.01,max:.99,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:16,min:1,max:96,step:1,key:{inc:"q",dec:"a",step:4,shiftStep:12}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.2,min:.02,max:.75,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"bubbleAmp",label:"Bubble Amp",uniform:"uBubbleAmp",type:"float",value:.26,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.08}},{id:"bubbleFreq",label:"Bubble Freq",uniform:"uBubbleFreq",type:"float",value:2,min:0,max:6,step:.05,key:{inc:"r",dec:"f",step:.25,shiftStep:.75}},{id:"bubbleDetail",label:"Bubble Detail",uniform:"uBubbleDetail",type:"float",value:1.2,min:.1,max:3,step:.05,key:{inc:"t",dec:"g",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tunnel",name:"Brownian Loop Tunnel",description:"Looped tunnel with Brownian noise, fog, and hue spin.",fragment:V,resolutionUniform:"iResolution",timeUniform:"iTime",timeMode:"looped",loopDuration:8,loopUniform:"uLoopDuration",params:[{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:4,min:0,max:10,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"noiseScale",label:"Noise Scale",uniform:"uNoiseScale",type:"float",value:1.9,min:.1,max:4,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.5,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"fogDensity",label:"Fog Density",uniform:"uFogDensity",type:"float",value:2,min:.1,max:6,step:.05,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"baseRed",label:"Base Red",uniform:"uBaseColor",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:0},{id:"baseGreen",label:"Base Green",uniform:"uBaseColor",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:1},{id:"baseBlue",label:"Base Blue",uniform:"uBaseColor",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:2}]},{id:"koch",name:"Koch Snowflake",description:"Iterative snowflake edges with neon glow mixing.",fragment:K,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.8,min:.1,max:2,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:.2,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:2,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.3,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:2}]},{id:"quasi",name:"Quasi Snowflake",description:"Quasicrystal warp with a drifting snowflake outline.",fragment:$,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:6,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:1.1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.8,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.02,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.03,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.02},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.05,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.02},component:2}]}];function L(e,n,o){const t=e.createShader(n);if(!t)throw new Error("Failed to create shader");if(e.shaderSource(t,o),e.compileShader(t),!e.getShaderParameter(t,e.COMPILE_STATUS)){const i=e.getShaderInfoLog(t)||"Unknown shader error";throw e.deleteShader(t),new Error(i)}return t}function X(e,n,o){const t=L(e,e.VERTEX_SHADER,n),i=L(e,e.FRAGMENT_SHADER,o),l=e.createProgram();if(!l)throw new Error("Failed to create program");if(e.attachShader(l,t),e.attachShader(l,i),e.linkProgram(l),e.deleteShader(t),e.deleteShader(i),!e.getProgramParameter(l,e.LINK_STATUS)){const a=e.getProgramInfoLog(l)||"Unknown program error";throw e.deleteProgram(l),new Error(a)}return l}function j(e,n,o){const t={};for(const i of o)t[i]=e.getUniformLocation(n,i);return t}function Q(e,n=2){const o=Math.min(window.devicePixelRatio||1,n),t=Math.max(1,Math.floor(e.clientWidth*o)),i=Math.max(1,Math.floor(e.clientHeight*o));return(e.width!==t||e.height!==i)&&(e.width=t,e.height=i),{width:t,height:i,dpr:o}}function J(e){const n=e.createVertexArray();if(!n)throw new Error("Failed to create VAO");return e.bindVertexArray(n),n}const A=document.querySelector("#app");if(!A)throw new Error("Missing #app root element");A.innerHTML=`
  <div class="app-shell">
    <header class="top-bar">
      <div class="brand">
        <div class="badge">VJ LAB</div>
        <div class="brand-title">GL Field Deck</div>
        <div class="brand-sub">Live shader scenes for performance and capture</div>
      </div>
      <div class="top-actions">
        <button class="ghost" data-action="toggle-ui">Hide UI</button>
        <button class="accent" data-action="fullscreen">Fullscreen</button>
      </div>
    </header>
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
        <button class="floating-toggle" data-action="toggle-ui-floating">Show UI</button>
        <div class="hud">
          <div class="hud-title" id="hud-title"></div>
          <div class="hud-desc" id="hud-desc"></div>
        </div>
      </main>
    </div>
  </div>
`;const C=document.querySelector("#gl-canvas"),S=document.querySelector("#scene-list"),w=document.querySelector("#control-list"),x=document.querySelector("#panel-actions"),y=document.querySelector("#key-help"),B=document.querySelector("#hud-title"),U=document.querySelector("#hud-desc"),T=document.querySelector("[data-action='toggle-ui']"),R=document.querySelector("[data-action='toggle-ui-floating']"),M=document.querySelector("[data-action='fullscreen']"),P=document.querySelector(".stage");if(!C||!S||!w||!x||!y||!B||!U||!T||!R||!M||!P)throw new Error("Missing required UI elements");const r=C.getContext("webgl2",{antialias:!0});if(!r)throw P.innerHTML=`
    <div class="fallback">
      <h2>WebGL2 unavailable</h2>
      <p>Your browser or GPU does not expose WebGL2. Try a different browser.</p>
    </div>
  `,new Error("WebGL2 unavailable");r.disable(r.DEPTH_TEST);r.disable(r.BLEND);const Y=J(r),E=new Map,N={},h={},D={};for(const e of g){const n=X(r,z,e.fragment),o=new Set;o.add(e.resolutionUniform),o.add(e.timeUniform),e.loopUniform&&o.add(e.loopUniform);for(const a of e.params)o.add(a.uniform);const t=j(r,n,Array.from(o));E.set(e.id,{program:n,uniforms:t});const i={},l={};for(const a of e.params)i[a.id]=a,l[a.id]=a.type==="seed"?Math.floor(Math.random()*1e6):a.value;N[e.id]=i,h[e.id]={...l},D[e.id]={...l}}let p=g[0],k={},v=!1,q=performance.now();function Z(e,n){let o=n;return e.min!==void 0&&(o=Math.max(e.min,o)),e.max!==void 0&&(o=Math.min(e.max,o)),e.type==="int"&&(o=Math.round(o)),o}function G(e,n){if(e.type==="int")return String(Math.round(n));const o=e.step??.01,t=o<1?Math.min(4,Math.max(2,Math.ceil(-Math.log10(o)))):0;return n.toFixed(t)}function ee(e){return e.length===1?e.toLowerCase():e}function ne(e){return e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement||e instanceof HTMLSelectElement}function b(e,n,o,t=!0){const i=N[e][n];if(!i)return;const l=Z(i,o);if(h[e][n]=l,e===p.id&&t){const a=k[n];a!=null&&a.range&&(a.range.value=String(l)),a!=null&&a.number&&(a.number.value=G(i,l))}}function te(e){var n;for(const o of((n=g.find(t=>t.id===e))==null?void 0:n.params)??[])o.type==="seed"&&b(e,o.id,Math.floor(Math.random()*1e6))}function oe(e){const n=D[e];for(const[o,t]of Object.entries(n))b(e,o,t,!0)}function ie(){S.innerHTML="";for(const e of g){const n=document.createElement("button");n.className="scene-button",n.textContent=e.name,n.dataset.scene=e.id,n.addEventListener("click",()=>F(e.id)),S.appendChild(n)}}function ae(e){x.innerHTML="";const n=document.createElement("button");if(n.className="ghost small",n.textContent="Reset",n.addEventListener("click",()=>oe(e.id)),x.appendChild(n),e.params.some(t=>t.type==="seed")){const t=document.createElement("button");t.className="ghost small",t.textContent="Reseed",t.addEventListener("click",()=>te(e.id)),x.appendChild(t)}}function le(e){w.innerHTML="",k={};for(const n of e.params){if(n.type==="seed")continue;const o=document.createElement("div");o.className="control";const t=document.createElement("div");t.className="control-header";const i=document.createElement("label");if(i.textContent=n.label,t.appendChild(i),n.key){const c=document.createElement("span");c.className="key-cap",c.textContent=`${n.key.inc.toUpperCase()}/${n.key.dec.toUpperCase()}`,t.appendChild(c)}const l=document.createElement("div");l.className="control-inputs";const a=document.createElement("input");a.type="range",a.min=String(n.min??0),a.max=String(n.max??1),a.step=String(n.step??(n.type==="int"?1:.01)),a.value=String(h[e.id][n.id]),a.addEventListener("input",c=>{const m=Number(c.target.value);Number.isNaN(m)||b(e.id,n.id,m)});const u=document.createElement("input");u.type="number",u.min=a.min,u.max=a.max,u.step=a.step,u.value=G(n,h[e.id][n.id]),u.addEventListener("input",c=>{const m=Number(c.target.value);Number.isNaN(m)||b(e.id,n.id,m)}),l.appendChild(a),l.appendChild(u),o.appendChild(t),o.appendChild(l),w.appendChild(o),k[n.id]={range:a,number:u}}}function re(e){y.innerHTML="";for(const n of e.params){if(!n.key||n.type==="seed")continue;const o=document.createElement("div");o.className="key-row",o.textContent=`${n.key.inc.toUpperCase()}/${n.key.dec.toUpperCase()}  ${n.label}`,y.appendChild(o)}y.childElementCount||(y.textContent="No mapped keys for this scene.")}function F(e){const n=g.find(o=>o.id===e);n&&(p=n,B.textContent=n.name,U.textContent=n.description,ae(n),le(n),re(n),S.querySelectorAll(".scene-button").forEach(o=>{const t=o;t.classList.toggle("active",t.dataset.scene===n.id)}))}function se(e){if(ne(e.target))return;const n=ee(e.key),o=p.params;for(const t of o){if(!t.key||t.type==="seed")continue;const i=n===t.key.inc,l=n===t.key.dec;if(!i&&!l)continue;const a=e.shiftKey&&t.key.shiftStep?t.key.shiftStep:t.key.step,c=h[p.id][t.id]+a*(i?1:-1);b(p.id,t.id,c),e.preventDefault();break}}function ce(e,n,o,t,i){const l=h[e.id],a=n.uniforms,u=a[e.resolutionUniform];u&&r.uniform2f(u,t,i);const c=a[e.timeUniform];if(c)if(e.timeMode==="phase"){const s=e.loopDuration??8,f=o%s/s;r.uniform1f(c,f)}else if(e.timeMode==="looped"){const s=e.loopDuration??8,f=o%s;if(r.uniform1f(c,f),e.loopUniform){const d=a[e.loopUniform];d&&r.uniform1f(d,s)}}else r.uniform1f(c,o);const m={};for(const s of e.params){const f=a[s.uniform],d=l[s.id];if(s.component!==void 0){const I=m[s.uniform]??[0,0,0];I[s.component]=d,m[s.uniform]=I;continue}f&&(s.type==="int"?r.uniform1i(f,Math.round(d)):r.uniform1f(f,d))}for(const[s,f]of Object.entries(m)){const d=a[s];d&&r.uniform3f(d,f[0],f[1],f[2])}}function O(e){const n=(e-q)/1e3,{width:o,height:t}=Q(C),i=E.get(p.id);i&&(r.viewport(0,0,o,t),r.clearColor(0,0,0,1),r.clear(r.COLOR_BUFFER_BIT),r.useProgram(i.program),r.bindVertexArray(Y),ce(p,i,n,o,t),r.drawArrays(r.TRIANGLES,0,3),requestAnimationFrame(O))}function H(e){v=e,document.body.classList.toggle("ui-hidden",v),T.textContent=v?"Show UI":"Hide UI",R.textContent=v?"Show UI":"Hide UI"}T.addEventListener("click",()=>H(!v));R.addEventListener("click",()=>H(!v));M.addEventListener("click",async()=>{document.fullscreenElement?await document.exitFullscreen():await P.requestFullscreen()});document.addEventListener("keydown",se);document.addEventListener("fullscreenchange",()=>{M.textContent=document.fullscreenElement?"Exit Fullscreen":"Fullscreen"});document.addEventListener("visibilitychange",()=>{document.hidden||(q=performance.now())});ie();F(p.id);requestAnimationFrame(O);
