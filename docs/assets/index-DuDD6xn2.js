(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))o(a);new MutationObserver(a=>{for(const r of a)if(r.type==="childList")for(const l of r.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&o(l)}).observe(document,{childList:!0,subtree:!0});function n(a){const r={};return a.integrity&&(r.integrity=a.integrity),a.referrerPolicy&&(r.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?r.credentials="include":a.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function o(a){if(a.ep)return;a.ep=!0;const r=n(a);fetch(a.href,r)}})();const mn=`#version 300 es
precision highp float;

const vec2 verts[3] = vec2[](
  vec2(-1.0, -1.0),
  vec2(3.0, -1.0),
  vec2(-1.0, 3.0)
);

void main() {
  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
`,dn=`#version 300 es
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
`,pn=`#version 300 es
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
`,xn=`#version 300 es\r
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
}\r
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
}\r
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
`,Rn=`#version 300 es
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
}\r
`,Bn=`#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uCoreRadius;
uniform float uCoreNoiseScale;
uniform float uCoreNoiseAmp;
uniform float uCoreIntensity;
uniform float uBoltLengthMin;
uniform float uBoltLengthMax;
uniform float uBoltWidth;
uniform float uBoltWiggle;
uniform float uBoltNoiseScale;
uniform float uBoltNoiseSpeed;
uniform float uBoltSecondaryScale;
uniform float uBoltIntensity;
uniform float uFlickerSpeed;
uniform float uAngleJitter;
uniform float uTwist;
uniform float uSeed;
uniform int uBoltCount;
uniform int uNoiseOctaves;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;
uniform vec3 uColorAccent;

const float TAU = 6.28318530718;

mat2 Rotate(float angle) {
  return mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
}

float CircleSDF(vec2 p, float r) {
  return length(p) - r;
}

float LineSDF(vec2 p, vec2 a, vec2 b, float s) {
  vec2 pa = a - p;
  vec2 ba = a - b;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - s;
}

float RandomFloat(vec2 seed) {
  seed = sin(seed * vec2(123.45, 546.23)) * 345.21 + 12.57;
  return fract(seed.x * seed.y);
}

float SimpleNoise(vec2 uv, int octaves) {
  float sn = 0.0;
  float amplitude = 1.0;
  float deno = 0.0;
  int count = clamp(octaves, 1, 6);
  for (int i = 1; i <= 6; i++) {
    if (i > count) {
      break;
    }
    vec2 grid = smoothstep(0.0, 1.0, fract(uv));
    vec2 id = floor(uv);
    vec2 offs = vec2(0.0, 1.0);
    float bl = RandomFloat(id);
    float br = RandomFloat(id + offs.yx);
    float tl = RandomFloat(id + offs);
    float tr = RandomFloat(id + offs.yy);
    sn += mix(mix(bl, br, grid.x), mix(tl, tr, grid.x), grid.y) * amplitude;
    deno += amplitude;
    uv *= 3.5;
    amplitude *= 0.5;
  }
  return sn / max(1e-4, deno);
}

vec3 Bolt(vec2 uv, float len, float ind, float time) {
  vec2 travel = vec2(0.0, mod(time, 200.0) * uBoltNoiseSpeed);

  float sn = SimpleNoise(
    uv * uBoltNoiseScale - travel + vec2(ind * 1.5 + uSeed * 0.01, 0.0),
    uNoiseOctaves
  ) * 2.0 - 1.0;
  uv.x += sn * uBoltWiggle * smoothstep(0.0, 0.2, abs(uv.y));

  vec3 l = vec3(LineSDF(uv, vec2(0.0), vec2(0.0, len), uBoltWidth));
  l = uBoltIntensity / max(vec3(0.0), l) * uColorSecondary;
  l = clamp(1.0 - exp(l * -0.02), 0.0, 1.0) * smoothstep(len - 0.01, 0.0, abs(uv.y));
  vec3 bolt = l;

  uv = Rotate(TAU * uTwist) * uv;
  sn = SimpleNoise(
    uv * (uBoltNoiseScale * 1.25) - travel * 1.2 + vec2(ind * 2.3 + uSeed * 0.03, 0.0),
    uNoiseOctaves
  ) * 2.0 - 1.0;
  uv.x += sn * uv.y * uBoltSecondaryScale * smoothstep(0.1, 0.25, len);
  len *= 0.5;

  l = vec3(LineSDF(uv, vec2(0.0), vec2(0.0, len), uBoltWidth * 0.8));
  l = uBoltIntensity * 0.7 / max(vec3(0.0), l) * uColorAccent;
  l = clamp(1.0 - exp(l * -0.03), 0.0, 1.0) * smoothstep(len * 0.7, 0.0, abs(uv.y));
  bolt += l;

  float hz = uFlickerSpeed * time * TAU;
  float r = RandomFloat(vec2(ind + uSeed * 0.1)) * 0.5 * TAU;
  float flicker = sin(hz + r) * 0.5 + 0.5;
  return bolt * smoothstep(0.5, 0.0, flicker);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
  uv *= uZoom;

  float time = uTime * uTimeScale;
  vec3 col = vec3(0.0);

  float coreNoise = SimpleNoise(
    uv * uCoreNoiseScale - vec2(0.0, mod(time, 200.0) * uCoreNoiseScale * 0.2),
    uNoiseOctaves
  );
  float r = uCoreRadius + uCoreNoiseAmp * (coreNoise * 2.0 - 1.0);
  vec3 core = uCoreIntensity / max(0.0, CircleSDF(uv, r)) * uColorPrimary;
  core = 1.0 - exp(core * -0.05);
  col = core;

  int count = max(uBoltCount, 1);
  for (int i = 0; i < 12; i++) {
    if (i >= count) {
      break;
    }
    float fi = float(i);
    float angle = fi * TAU / float(count);
    angle += (RandomFloat(vec2(float(count) + floor(time * 5.0 + fi) + uSeed)) - 0.5)
      * uAngleJitter;
    float len = mix(
      uBoltLengthMin,
      uBoltLengthMax,
      RandomFloat(vec2(angle + uSeed, fi * 1.7))
    );
    col += Bolt(Rotate(angle) * uv, len, fi, time);
  }

  outColor = vec4(col, 1.0);
}
`,An=`#version 300 es
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
`,In=`#version 300 es
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
`,En=`#version 300 es\r
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
`,ve=[{id:"neon",name:"Neon Isoclines",description:"Electric contour bands driven by seeded radial harmonics.",fragment:dn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"components",label:"Components",uniform:"uComponents",type:"int",value:64,min:1,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:10}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:8,min:1,max:64,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:10}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.25,min:.01,max:.75,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.05}},{id:"noiseAmount",label:"Noise Amount",uniform:"uNoiseAmount",type:"float",value:2.5,min:0,max:5,step:.05,key:{inc:"r",dec:"f",step:.1,shiftStep:.25}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tanh-terrain",name:"Tanh Terrain Isoclines",description:"Tanh warped contours with bubbling noise and topo glow.",fragment:pn,resolutionUniform:"uResolution",timeUniform:"uPhase",timeMode:"phase",loopDuration:8,params:[{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:2.1,min:.1,max:6,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"octaves",label:"Octaves",uniform:"uOctaves",type:"int",value:4,min:1,max:12,step:1,key:{inc:"3",dec:"4",step:1}},{id:"lacunarity",label:"Lacunarity",uniform:"uLacunarity",type:"float",value:1.4,min:1.01,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"gain",label:"Gain",uniform:"uGain",type:"float",value:.5,min:.01,max:.99,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"isoBands",label:"Iso Bands",uniform:"uIsoBands",type:"int",value:16,min:1,max:96,step:1,key:{inc:"q",dec:"a",step:4,shiftStep:12}},{id:"lineThickness",label:"Line Thickness",uniform:"uLineThickness",type:"float",value:.2,min:.02,max:.75,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"bubbleAmp",label:"Bubble Amp",uniform:"uBubbleAmp",type:"float",value:.26,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02,shiftStep:.08}},{id:"bubbleFreq",label:"Bubble Freq",uniform:"uBubbleFreq",type:"float",value:2,min:0,max:6,step:.05,key:{inc:"r",dec:"f",step:.25,shiftStep:.75}},{id:"bubbleDetail",label:"Bubble Detail",uniform:"uBubbleDetail",type:"float",value:1.2,min:.1,max:3,step:.05,key:{inc:"t",dec:"g",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"tunnel",name:"Brownian Loop Tunnel",description:"Looped tunnel with Brownian noise, fog, and hue spin.",fragment:hn,resolutionUniform:"iResolution",timeUniform:"iTime",timeMode:"looped",loopDuration:8,loopUniform:"uLoopDuration",params:[{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:1,min:.1,max:4,step:.05,key:{inc:"1",dec:"2",step:.1,shiftStep:.5}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:4,min:0,max:10,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"noiseScale",label:"Noise Scale",uniform:"uNoiseScale",type:"float",value:1.9,min:.1,max:4,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.5,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"colorCycle",label:"Color Cycle",uniform:"uColorCycle",type:"float",value:1,min:0,max:4,step:.05,key:{inc:"q",dec:"a",step:.1,shiftStep:.5}},{id:"fogDensity",label:"Fog Density",uniform:"uFogDensity",type:"float",value:2,min:.1,max:6,step:.05,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"baseRed",label:"Base Red",uniform:"uBaseColor",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:0},{id:"baseGreen",label:"Base Green",uniform:"uBaseColor",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:1},{id:"baseBlue",label:"Base Blue",uniform:"uBaseColor",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:2}]},{id:"prismatic-fold",name:"Prismatic Fold Raymarch",description:"Rotating folded planes with prismatic glow and controllable depth.",fragment:xn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:24,step:1,key:{inc:"1",dec:"2",step:1,shiftStep:4}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.2,min:-1.5,max:1.5,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"foldOffset",label:"Fold Offset",uniform:"uFoldOffset",type:"float",value:.5,min:.1,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:5,min:1.5,max:10,step:.1,key:{inc:"7",dec:"8",step:.2,shiftStep:.6}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"cameraDistance",label:"Camera Distance",uniform:"uCameraDistance",type:"float",value:50,min:10,max:120,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:5}},{id:"cameraSpin",label:"Camera Spin",uniform:"uCameraSpin",type:"float",value:1,min:-3,max:3,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.4}},{id:"colorMix",label:"Color Mix",uniform:"uColorMix",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"alphaGain",label:"Alpha Gain",uniform:"uAlphaGain",type:"float",value:1,min:.3,max:2,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:.9,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"[",dec:"]",step:.05},component:2}]},{id:"koch",name:"Koch Snowflake",description:"Iterative snowflake edges with neon glow mixing.",fragment:vn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:4,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:.8,min:.1,max:2,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:.2,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:2,min:0,max:5,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.05},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.3,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.05},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.6,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.05},component:2}]},{id:"quasi",name:"Quasi Snowflake",description:"Quasicrystal warp with a drifting snowflake outline.",fragment:yn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:6,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"scale",label:"Scale",uniform:"uScale",type:"float",value:1.1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"rotation",label:"Rotation",uniform:"uRotation",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"5",dec:"6",step:.1,shiftStep:.5}},{id:"glowIntensity",label:"Glow",uniform:"uGlowIntensity",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"7",dec:"8",step:.1,shiftStep:.5}},{id:"primRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.2,min:0,max:1,step:.01,key:{inc:"q",dec:"a",step:.05},component:0},{id:"primGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.8,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.05},component:1},{id:"primBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.05},component:2},{id:"secRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.02,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:0},{id:"secGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.03,min:0,max:1,step:.01,key:{inc:"t",dec:"g",step:.02},component:1},{id:"secBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.05,min:0,max:1,step:.01,key:{inc:"y",dec:"h",step:.02},component:2}]},{id:"tileable-water-plus",name:"Tileable Water Plus",description:"Tileable water ripples with tunable speed, scale, and tint.",fragment:Cn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"tileScale",label:"Tile Scale",uniform:"uTileScale",type:"float",value:1,min:.5,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.2,max:2.5,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"contrast",label:"Contrast",uniform:"uContrast",type:"float",value:1.2,min:.3,max:2.5,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"waveShift",label:"Wave Shift",uniform:"uWaveShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"tintRed",label:"Tint Red",uniform:"uTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"w",dec:"s",step:.02},component:0},{id:"tintGreen",label:"Tint Green",uniform:"uTint",type:"float",value:.35,min:0,max:1,step:.01,key:{inc:"e",dec:"d",step:.02},component:1},{id:"tintBlue",label:"Tint Blue",uniform:"uTint",type:"float",value:.5,min:0,max:1,step:.01,key:{inc:"r",dec:"f",step:.02},component:2}]},{id:"seascape",name:"Seascape Plus",description:"Raymarched ocean with tunable swell and camera drift.",fragment:wn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.3,min:0,max:1.5,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"seaHeight",label:"Sea Height",uniform:"uSeaHeight",type:"float",value:.6,min:.1,max:1.5,step:.02,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"seaChoppy",label:"Sea Choppy",uniform:"uSeaChoppy",type:"float",value:4,min:1,max:7,step:.1,key:{inc:"5",dec:"6",step:.1,shiftStep:.4}},{id:"seaFreq",label:"Sea Freq",uniform:"uSeaFreq",type:"float",value:.16,min:.05,max:.4,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.04}},{id:"seaSpeed",label:"Sea Speed",uniform:"uSeaSpeed",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:3.5,min:1,max:8,step:.1,key:{inc:"w",dec:"s",step:.1,shiftStep:.5}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:1,max:10,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.5}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:0,min:-1,max:1,step:.02,key:{inc:"r",dec:"f",step:.02,shiftStep:.08}},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:0,min:-.5,max:.5,step:.02,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1.1,min:.6,max:1.6,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"waterBrightness",label:"Water Bright",uniform:"uWaterBrightness",type:"float",value:.6,min:.2,max:1.2,step:.02,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"waterRed",label:"Water Red",uniform:"uWaterTint",type:"float",value:0,min:0,max:1,step:.01,key:{inc:"i",dec:"k",step:.02},component:0},{id:"waterGreen",label:"Water Green",uniform:"uWaterTint",type:"float",value:.09,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02},component:1},{id:"waterBlue",label:"Water Blue",uniform:"uWaterTint",type:"float",value:.18,min:0,max:1,step:.01,key:{inc:"p",dec:";",step:.02},component:2}]},{id:"sunset-plus",name:"Sunset Plus",description:"Volumetric sunset clouds with tunable turbulence and hue drift.",fragment:Tn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:2,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"cloudHeight",label:"Cloud Height",uniform:"uCloudHeight",type:"float",value:.3,min:.05,max:1,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.08}},{id:"stepBase",label:"Step Base",uniform:"uStepBase",type:"float",value:.005,min:.001,max:.02,step:.001,key:{inc:"7",dec:"8",step:.001,shiftStep:.004}},{id:"stepScale",label:"Step Scale",uniform:"uStepScale",type:"float",value:4,min:2,max:10,step:.2,key:{inc:"q",dec:"a",step:.2,shiftStep:.8}},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-3,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"hueSpeed",label:"Hue Speed",uniform:"uHueSpeed",type:"float",value:.4,min:-2,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"intensity",label:"Intensity",uniform:"uIntensity",type:"float",value:1,min:.4,max:2.5,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}}]},{id:"diff-chromatic",name:"Chromatic Flow",description:"Two-channel diffusion with hue-as-angle and drifting color pulses.",fragment:gn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.998,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"rotate",label:"Rotate",uniform:"uRotate",type:"float",value:.02,min:-.2,max:.2,step:.005,key:{inc:"7",dec:"8",step:.01,shiftStep:.03}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.35,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"w",dec:"s",step:.01,shiftStep:.03}},{id:"valueGain",label:"Value Gain",uniform:"uValueGain",type:"float",value:2.2,min:.2,max:6,step:.05,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"zippy-zaps",name:"Zippy Zaps Plus",description:"Tanh-warped chromatic flow with twistable energy and glow.",fragment:kn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.2,min:.05,max:.5,step:.005,key:{inc:"1",dec:"2",step:.01,shiftStep:.03}},{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:.1,max:3,step:.05,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1,min:.2,max:2,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:1,min:0,max:2,step:.05,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1,min:0,max:2.5,step:.05,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"iterLimit",label:"Iter Limit",uniform:"uIterLimit",type:"float",value:19,min:4,max:19,step:1,key:{inc:"w",dec:"s",step:1,shiftStep:3}},{id:"glow",label:"Glow",uniform:"uGlow",type:"float",value:1,min:.4,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"offsetX",label:"Offset X",uniform:"uOffsetX",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"t",dec:"g",step:.02,shiftStep:.08}},{id:"offsetY",label:"Offset Y",uniform:"uOffsetY",type:"float",value:0,min:-.5,max:.5,step:.01,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}}]},{id:"space-lightning-plus",name:"Space Lightning Plus",description:"Funky ion bolts with twistable arcs, palette waves, and core glow.",fragment:Rn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:.35,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.02,shiftStep:.1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.9,min:.3,max:1.8,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"spin",label:"Spin",uniform:"uSpin",type:"float",value:.4,min:-2,max:2,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:1.2,min:0,max:3,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"warp",label:"Warp",uniform:"uWarp",type:"float",value:.9,min:0,max:2,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.2}},{id:"pulse",label:"Pulse",uniform:"uPulse",type:"float",value:1.1,min:0,max:3,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"boltDensity",label:"Bolt Density",uniform:"uBoltDensity",type:"float",value:6.5,min:1,max:20,step:.1,key:{inc:"e",dec:"d",step:.1,shiftStep:.3}},{id:"boltSharpness",label:"Bolt Sharpness",uniform:"uBoltSharpness",type:"float",value:.9,min:.1,max:2.5,step:.02,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:1.2,min:.2,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"arcSteps",label:"Arc Steps",uniform:"uArcSteps",type:"float",value:40,min:6,max:80,step:1,key:{inc:"y",dec:"h",step:1,shiftStep:5}},{id:"coreSize",label:"Core Size",uniform:"uCoreSize",type:"float",value:.25,min:0,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.08}},{id:"coreGlow",label:"Core Glow",uniform:"uCoreGlow",type:"float",value:.8,min:0,max:2,step:.02,key:{inc:"i",dec:"k",step:.05,shiftStep:.2}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.25,min:0,max:1,step:.01,key:{inc:"o",dec:"l",step:.02,shiftStep:.08}},{id:"paletteShift",label:"Palette Shift",uniform:"uPaletteShift",type:"float",value:0,min:0,max:6.283,step:.05,key:{inc:"p",dec:";",step:.05,shiftStep:.2}},{id:"primaryRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.08,min:0,max:1,step:.01,component:0},{id:"primaryGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.7,min:0,max:1,step:.01,component:1},{id:"primaryBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"secondaryRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.9,min:0,max:1,step:.01,component:0},{id:"secondaryGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"secondaryBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.8,min:0,max:1,step:.01,component:2},{id:"accentRed",label:"Accent Red",uniform:"uColorAccent",type:"float",value:1,min:0,max:1,step:.01,component:0},{id:"accentGreen",label:"Accent Green",uniform:"uColorAccent",type:"float",value:.8,min:0,max:1,step:.01,component:1},{id:"accentBlue",label:"Accent Blue",uniform:"uColorAccent",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"lightning-blade-plus",name:"Lightning Blade Plus",description:"Flaring core with jittered blades and controllable noise flicker.",fragment:Bn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:.4,min:.1,max:1.2,step:.01,key:{inc:"3",dec:"4",step:.02,shiftStep:.1}},{id:"coreRadius",label:"Core Radius",uniform:"uCoreRadius",type:"float",value:.02,min:.005,max:.1,step:.001,key:{inc:"5",dec:"6",step:.002,shiftStep:.01}},{id:"coreNoiseScale",label:"Core Noise Scale",uniform:"uCoreNoiseScale",type:"float",value:50,min:5,max:120,step:.5,key:{inc:"7",dec:"8",step:1,shiftStep:5}},{id:"coreNoiseAmp",label:"Core Noise Amp",uniform:"uCoreNoiseAmp",type:"float",value:.02,min:0,max:.08,step:.001,key:{inc:"q",dec:"a",step:.002,shiftStep:.01}},{id:"coreIntensity",label:"Core Intensity",uniform:"uCoreIntensity",type:"float",value:.6,min:.1,max:2,step:.02,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"boltCount",label:"Bolt Count",uniform:"uBoltCount",type:"int",value:4,min:1,max:10,step:1,key:{inc:"e",dec:"d",step:1,shiftStep:2}},{id:"boltLengthMin",label:"Bolt Length Min",uniform:"uBoltLengthMin",type:"float",value:.12,min:.05,max:.4,step:.01,key:{inc:"r",dec:"f",step:.01,shiftStep:.05}},{id:"boltLengthMax",label:"Bolt Length Max",uniform:"uBoltLengthMax",type:"float",value:.35,min:.1,max:.7,step:.01,key:{inc:"t",dec:"g",step:.01,shiftStep:.05}},{id:"boltWidth",label:"Bolt Width",uniform:"uBoltWidth",type:"float",value:6e-4,min:1e-4,max:.004,step:1e-4,key:{inc:"y",dec:"h",step:2e-4,shiftStep:.001}},{id:"boltWiggle",label:"Bolt Wiggle",uniform:"uBoltWiggle",type:"float",value:.03,min:0,max:.15,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"boltNoiseScale",label:"Bolt Noise Scale",uniform:"uBoltNoiseScale",type:"float",value:20,min:5,max:60,step:.5,key:{inc:"i",dec:"k",step:1,shiftStep:3}},{id:"boltNoiseSpeed",label:"Bolt Noise Speed",uniform:"uBoltNoiseSpeed",type:"float",value:2,min:0,max:8,step:.05,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"boltSecondaryScale",label:"Bolt Secondary",uniform:"uBoltSecondaryScale",type:"float",value:.8,min:0,max:1.5,step:.02,key:{inc:"p",dec:";",step:.02,shiftStep:.1}},{id:"boltIntensity",label:"Bolt Intensity",uniform:"uBoltIntensity",type:"float",value:.25,min:.05,max:1.2,step:.02},{id:"flickerSpeed",label:"Flicker Speed",uniform:"uFlickerSpeed",type:"float",value:4,min:0,max:12,step:.1},{id:"angleJitter",label:"Angle Jitter",uniform:"uAngleJitter",type:"float",value:.5,min:0,max:2,step:.02},{id:"twist",label:"Twist",uniform:"uTwist",type:"float",value:.125,min:0,max:.5,step:.005},{id:"noiseOctaves",label:"Noise Octaves",uniform:"uNoiseOctaves",type:"int",value:3,min:1,max:6,step:1},{id:"primaryRed",label:"Primary Red",uniform:"uColorPrimary",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"primaryGreen",label:"Primary Green",uniform:"uColorPrimary",type:"float",value:.1,min:0,max:1,step:.01,component:1},{id:"primaryBlue",label:"Primary Blue",uniform:"uColorPrimary",type:"float",value:.5,min:0,max:1,step:.01,component:2},{id:"secondaryRed",label:"Secondary Red",uniform:"uColorSecondary",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"secondaryGreen",label:"Secondary Green",uniform:"uColorSecondary",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"secondaryBlue",label:"Secondary Blue",uniform:"uColorSecondary",type:"float",value:.8,min:0,max:1,step:.01,component:2},{id:"accentRed",label:"Accent Red",uniform:"uColorAccent",type:"float",value:.8,min:0,max:1,step:.01,component:0},{id:"accentGreen",label:"Accent Green",uniform:"uColorAccent",type:"float",value:.3,min:0,max:1,step:.01,component:1},{id:"accentBlue",label:"Accent Blue",uniform:"uColorAccent",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"mandelbulb-inside-plus",name:"Inside the Mandelbulb Plus",description:"Raymarched mandelbulb interior with tunable optics and palette glow.",fragment:An,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"power",label:"Power",uniform:"uPower",type:"float",value:8,min:2,max:12,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"bulbSpin",label:"Bulb Spin",uniform:"uBulbSpin",type:"float",value:.2,min:0,max:1.5,step:.01,key:{inc:"5",dec:"6",step:.02,shiftStep:.1}},{id:"loops",label:"Loops",uniform:"uLoops",type:"int",value:2,min:1,max:6,step:1,key:{inc:"7",dec:"8",step:1,shiftStep:1}},{id:"rayMarches",label:"Ray Marches",uniform:"uRayMarches",type:"int",value:60,min:20,max:96,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:5}},{id:"maxRayLength",label:"Max Ray Length",uniform:"uMaxRayLength",type:"float",value:20,min:5,max:40,step:.5,key:{inc:"w",dec:"s",step:.5,shiftStep:2}},{id:"tolerance",label:"Tolerance",uniform:"uTolerance",type:"float",value:1e-4,min:1e-5,max:.001,step:1e-5,key:{inc:"e",dec:"d",step:2e-5,shiftStep:1e-4}},{id:"normOffset",label:"Normal Offset",uniform:"uNormOffset",type:"float",value:.005,min:.001,max:.02,step:5e-4,key:{inc:"r",dec:"f",step:5e-4,shiftStep:.002}},{id:"bounces",label:"Bounces",uniform:"uBounces",type:"int",value:5,min:1,max:5,step:1,key:{inc:"t",dec:"g",step:1,shiftStep:1}},{id:"initStep",label:"Init Step",uniform:"uInitStep",type:"float",value:.1,min:.01,max:.3,step:.01,key:{inc:"y",dec:"h",step:.01,shiftStep:.05}},{id:"rotSpeedX",label:"Rot Speed X",uniform:"uRotSpeedX",type:"float",value:.2,min:-1,max:1,step:.01,key:{inc:"u",dec:"j",step:.02,shiftStep:.1}},{id:"rotSpeedY",label:"Rot Speed Y",uniform:"uRotSpeedY",type:"float",value:.3,min:-1,max:1,step:.01,key:{inc:"i",dec:"k",step:.02,shiftStep:.1}},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:5,min:2,max:10,step:.1,key:{inc:"o",dec:"l",step:.1,shiftStep:.5}},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:2,min:.5,max:5,step:.1,key:{inc:"p",dec:";",step:.1,shiftStep:.5}},{id:"fov",label:"FOV",uniform:"uFov",type:"float",value:.523,min:.3,max:1.2,step:.01},{id:"skyBoost",label:"Sky Boost",uniform:"uSkyBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"glowBoost",label:"Glow Boost",uniform:"uGlowBoost",type:"float",value:1.2,min:0,max:4,step:.05},{id:"glowFalloff",label:"Glow Falloff",uniform:"uGlowFalloff",type:"float",value:.06,min:.01,max:.2,step:.005},{id:"diffuseBoost",label:"Diffuse Boost",uniform:"uDiffuseBoost",type:"float",value:1,min:.2,max:3,step:.05},{id:"matTransmit",label:"Mat Transmit",uniform:"uMatTransmit",type:"float",value:.8,min:0,max:1,step:.01},{id:"matReflect",label:"Mat Reflect",uniform:"uMatReflect",type:"float",value:.5,min:0,max:1,step:.01},{id:"refractIndex",label:"Refract Index",uniform:"uRefractIndex",type:"float",value:1.05,min:1,max:2,step:.01},{id:"hueShift",label:"Hue Shift",uniform:"uHueShift",type:"float",value:0,min:-1,max:1,step:.01},{id:"glowHueOffset",label:"Glow Hue Offset",uniform:"uGlowHueOffset",type:"float",value:.065,min:-.5,max:.5,step:.005},{id:"nebulaMix",label:"Nebula Mix",uniform:"uNebulaMix",type:"float",value:0,min:0,max:1,step:.01},{id:"nebulaHueShift",label:"Nebula Hue",uniform:"uNebulaHueShift",type:"float",value:.12,min:-1,max:1,step:.01},{id:"nebulaSat",label:"Nebula Sat",uniform:"uNebulaSat",type:"float",value:.9,min:0,max:1,step:.01},{id:"nebulaVal",label:"Nebula Val",uniform:"uNebulaVal",type:"float",value:1.6,min:.2,max:3,step:.02},{id:"nebulaGlowHue",label:"Nebula Glow Hue",uniform:"uNebulaGlowHue",type:"float",value:.35,min:-1,max:1,step:.01},{id:"nebulaGlowBoost",label:"Nebula Glow",uniform:"uNebulaGlowBoost",type:"float",value:1.6,min:0,max:4,step:.05},{id:"skySat",label:"Sky Saturation",uniform:"uSkySat",type:"float",value:.86,min:0,max:1,step:.01},{id:"skyVal",label:"Sky Value",uniform:"uSkyVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"glowSat",label:"Glow Saturation",uniform:"uGlowSat",type:"float",value:.8,min:0,max:1,step:.01},{id:"glowVal",label:"Glow Value",uniform:"uGlowVal",type:"float",value:6,min:.5,max:8,step:.1},{id:"diffuseSat",label:"Diffuse Saturation",uniform:"uDiffuseSat",type:"float",value:.85,min:0,max:1,step:.01},{id:"diffuseVal",label:"Diffuse Value",uniform:"uDiffuseVal",type:"float",value:1,min:.2,max:2,step:.02},{id:"beerRed",label:"Beer Red",uniform:"uBeerColor",type:"float",value:.02,min:0,max:.2,step:.005,component:0},{id:"beerGreen",label:"Beer Green",uniform:"uBeerColor",type:"float",value:.08,min:0,max:.2,step:.005,component:1},{id:"beerBlue",label:"Beer Blue",uniform:"uBeerColor",type:"float",value:.12,min:0,max:.2,step:.005,component:2},{id:"lightX",label:"Light X",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:0},{id:"lightY",label:"Light Y",uniform:"uLightPos",type:"float",value:10,min:-5,max:25,step:.5,component:1},{id:"lightZ",label:"Light Z",uniform:"uLightPos",type:"float",value:0,min:-20,max:20,step:.5,component:2}]},{id:"auroras-plus",name:"Auroras Plus",description:"Volumetric auroras with tunable trails, palette waves, and sky glare.",fragment:In,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"timeScale",label:"Time Scale",uniform:"uTimeScale",type:"float",value:1,min:0,max:3,step:.02,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"auroraSpeed",label:"Aurora Speed",uniform:"uAuroraSpeed",type:"float",value:.06,min:0,max:.2,step:.005,key:{inc:"3",dec:"4",step:.005,shiftStep:.02}},{id:"auroraScale",label:"Aurora Scale",uniform:"uAuroraScale",type:"float",value:1,min:.2,max:3,step:.05,key:{inc:"5",dec:"6",step:.05,shiftStep:.2}},{id:"auroraWarp",label:"Aurora Warp",uniform:"uAuroraWarp",type:"float",value:.35,min:0,max:1,step:.02,key:{inc:"7",dec:"8",step:.02,shiftStep:.08}},{id:"auroraSteps",label:"Aurora Steps",uniform:"uAuroraSteps",type:"int",value:50,min:8,max:64,step:1,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"auroraBase",label:"Aurora Base",uniform:"uAuroraBase",type:"float",value:.8,min:.2,max:1.6,step:.02,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"auroraStride",label:"Aurora Stride",uniform:"uAuroraStride",type:"float",value:.002,min:2e-4,max:.01,step:2e-4,key:{inc:"e",dec:"d",step:2e-4,shiftStep:.001}},{id:"auroraCurve",label:"Aurora Curve",uniform:"uAuroraCurve",type:"float",value:1.4,min:.8,max:2.2,step:.05,key:{inc:"r",dec:"f",step:.05,shiftStep:.2}},{id:"auroraIntensity",label:"Aurora Intensity",uniform:"uAuroraIntensity",type:"float",value:1.8,min:.2,max:4,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"trailBlend",label:"Trail Blend",uniform:"uTrailBlend",type:"float",value:.5,min:.1,max:.9,step:.02,key:{inc:"y",dec:"h",step:.02,shiftStep:.08}},{id:"trailFalloff",label:"Trail Falloff",uniform:"uTrailFalloff",type:"float",value:.065,min:.01,max:.15,step:.005,key:{inc:"u",dec:"j",step:.005,shiftStep:.02}},{id:"trailFade",label:"Trail Fade",uniform:"uTrailFade",type:"float",value:2.5,min:.5,max:5,step:.1,key:{inc:"i",dec:"k",step:.1,shiftStep:.4}},{id:"ditherStrength",label:"Dither Strength",uniform:"uDitherStrength",type:"float",value:.006,min:0,max:.02,step:5e-4,key:{inc:"o",dec:"l",step:5e-4,shiftStep:.002}},{id:"horizonFade",label:"Horizon Fade",uniform:"uHorizonFade",type:"float",value:.01,min:.001,max:.05,step:.001,key:{inc:"p",dec:";",step:.001,shiftStep:.005}},{id:"camYaw",label:"Cam Yaw",uniform:"uCamYaw",type:"float",value:-.1,min:-1,max:1,step:.01},{id:"camPitch",label:"Cam Pitch",uniform:"uCamPitch",type:"float",value:.1,min:-1,max:1,step:.01},{id:"camWobble",label:"Cam Wobble",uniform:"uCamWobble",type:"float",value:.2,min:0,max:.6,step:.01},{id:"camDistance",label:"Cam Distance",uniform:"uCamDistance",type:"float",value:6.7,min:4,max:12,step:.1},{id:"camHeight",label:"Cam Height",uniform:"uCamHeight",type:"float",value:0,min:-1,max:2,step:.05},{id:"skyStrength",label:"Sky Strength",uniform:"uSkyStrength",type:"float",value:.63,min:.1,max:2,step:.02},{id:"starDensity",label:"Star Density",uniform:"uStarDensity",type:"float",value:5e-4,min:0,max:.005,step:1e-4},{id:"starIntensity",label:"Star Intensity",uniform:"uStarIntensity",type:"float",value:.8,min:0,max:2,step:.05},{id:"reflectionStrength",label:"Reflection Strength",uniform:"uReflectionStrength",type:"float",value:.6,min:0,max:1.5,step:.05},{id:"reflectionTint",label:"Reflection Tint",uniform:"uReflectionTint",type:"float",value:1,min:0,max:2,step:.05},{id:"reflectionFog",label:"Reflection Fog",uniform:"uReflectionFog",type:"float",value:2,min:0,max:6,step:.1},{id:"colorBand",label:"Color Band",uniform:"uColorBand",type:"float",value:.043,min:0,max:.2,step:.002},{id:"colorSpeed",label:"Color Speed",uniform:"uColorSpeed",type:"float",value:0,min:-1,max:1,step:.01},{id:"auroraRedA",label:"Aurora Red A",uniform:"uAuroraColorA",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenA",label:"Aurora Green A",uniform:"uAuroraColorA",type:"float",value:.9,min:0,max:1,step:.01,component:1},{id:"auroraBlueA",label:"Aurora Blue A",uniform:"uAuroraColorA",type:"float",value:.6,min:0,max:1,step:.01,component:2},{id:"auroraRedB",label:"Aurora Red B",uniform:"uAuroraColorB",type:"float",value:.6,min:0,max:1,step:.01,component:0},{id:"auroraGreenB",label:"Aurora Green B",uniform:"uAuroraColorB",type:"float",value:.2,min:0,max:1,step:.01,component:1},{id:"auroraBlueB",label:"Aurora Blue B",uniform:"uAuroraColorB",type:"float",value:1,min:0,max:1,step:.01,component:2},{id:"auroraRedC",label:"Aurora Red C",uniform:"uAuroraColorC",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"auroraGreenC",label:"Aurora Green C",uniform:"uAuroraColorC",type:"float",value:.6,min:0,max:1,step:.01,component:1},{id:"auroraBlueC",label:"Aurora Blue C",uniform:"uAuroraColorC",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedA",label:"BG Red A",uniform:"uBgColorA",type:"float",value:.05,min:0,max:1,step:.01,component:0},{id:"bgGreenA",label:"BG Green A",uniform:"uBgColorA",type:"float",value:.1,min:0,max:1,step:.01,component:1},{id:"bgBlueA",label:"BG Blue A",uniform:"uBgColorA",type:"float",value:.2,min:0,max:1,step:.01,component:2},{id:"bgRedB",label:"BG Red B",uniform:"uBgColorB",type:"float",value:.1,min:0,max:1,step:.01,component:0},{id:"bgGreenB",label:"BG Green B",uniform:"uBgColorB",type:"float",value:.05,min:0,max:1,step:.01,component:1},{id:"bgBlueB",label:"BG Blue B",uniform:"uBgColorB",type:"float",value:.2,min:0,max:1,step:.01,component:2}]},{id:"diff-edge-flow",name:"Edge Flow Vectors",description:"Diffusive scalar field rendered as glowing edge-flow vectors.",fragment:bn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:384,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.6,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.996,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"blobAmp",label:"Blob Amp",uniform:"uBlobAmp",type:"float",value:.5,min:0,max:2,step:.02,key:{inc:"7",dec:"8",step:.05,shiftStep:.2}},{id:"blobRadius",label:"Blob Radius",uniform:"uBlobRadius",type:"float",value:.07,min:.01,max:.25,step:.005,key:{inc:"q",dec:"a",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.8,min:0,max:3,step:.05,key:{inc:"w",dec:"s",step:.05,shiftStep:.2}},{id:"flowGain",label:"Flow Gain",uniform:"uFlowGain",type:"float",value:3,min:.2,max:8,step:.1,key:{inc:"e",dec:"d",step:.2,shiftStep:.6}},{id:"flowThreshold",label:"Flow Threshold",uniform:"uFlowThreshold",type:"float",value:.02,min:0,max:.2,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"diff-threshold",name:"Threshold Feedback",description:"Diffusion with nonlinear feedback for digital fungus crackle.",fragment:Sn,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",stateful:!0,bufferSize:192,params:[{id:"selfWeight",label:"Self Weight",uniform:"uSelfWeight",type:"float",value:.5,min:0,max:2,step:.01,key:{inc:"1",dec:"2",step:.05,shiftStep:.2}},{id:"neighborWeight",label:"Neighbor Weight",uniform:"uNeighborWeight",type:"float",value:1,min:.05,max:2,step:.01,key:{inc:"3",dec:"4",step:.05,shiftStep:.2}},{id:"decay",label:"Decay",uniform:"uDecay",type:"float",value:.995,min:.9,max:.9999,step:5e-4,key:{inc:"5",dec:"6",step:.001,shiftStep:.005}},{id:"threshold",label:"Threshold",uniform:"uThreshold",type:"float",value:.5,min:.1,max:.9,step:.01,key:{inc:"7",dec:"8",step:.02,shiftStep:.06}},{id:"sharpness",label:"Sharpness",uniform:"uSharpness",type:"float",value:18,min:1,max:40,step:.5,key:{inc:"q",dec:"a",step:1,shiftStep:4}},{id:"noiseAmp",label:"Noise Amp",uniform:"uNoiseAmp",type:"float",value:.08,min:0,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.06}},{id:"turbulence",label:"Turbulence",uniform:"uTurbulence",type:"float",value:.8,min:0,max:2,step:.05,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectAmp",label:"Inject Amp",uniform:"uInjectAmp",type:"float",value:.2,min:0,max:1.5,step:.02,key:{inc:"e",dec:"d",step:.05,shiftStep:.2}},{id:"injectRadius",label:"Inject Radius",uniform:"uInjectRadius",type:"float",value:.06,min:.01,max:.25,step:.005,key:{inc:"r",dec:"f",step:.01,shiftStep:.03}},{id:"speed",label:"Speed",uniform:"uSpeed",type:"float",value:.9,min:0,max:3,step:.05,key:{inc:"t",dec:"g",step:.05,shiftStep:.2}},{id:"seed",label:"Seed",uniform:"uSeed",type:"seed",value:0}]},{id:"fractal-fold",name:"Fractal Fold Raymarch",description:"Recursive box-folding fractal with prismatic lighting and IQ palette.",fragment:En,resolutionUniform:"uResolution",timeUniform:"uTime",timeMode:"seconds",params:[{id:"iterations",label:"Iterations",uniform:"uIterations",type:"int",value:8,min:1,max:8,step:1,key:{inc:"1",dec:"2",step:1}},{id:"zoom",label:"Zoom",uniform:"uZoom",type:"float",value:1,min:.5,max:5,step:.1,key:{inc:"3",dec:"4",step:.1,shiftStep:.5}},{id:"distort",label:"Distort",uniform:"uDistort",type:"float",value:2.5,min:1.5,max:4,step:.02,key:{inc:"5",dec:"6",step:.05,shiftStep:.15}},{id:"colorShift",label:"Color Shift",uniform:"uColorShift",type:"float",value:0,min:0,max:1,step:.02,key:{inc:"q",dec:"a",step:.05,shiftStep:.15}},{id:"rotateSpeed",label:"Rotate Speed",uniform:"uRotateSpeed",type:"float",value:.1,min:-.5,max:.5,step:.01,key:{inc:"w",dec:"s",step:.02,shiftStep:.08}},{id:"maxSteps",label:"Max Steps",uniform:"uMaxSteps",type:"float",value:100,min:20,max:200,step:10,key:{inc:"e",dec:"d",step:10,shiftStep:30}}]}];function Rt(e,t,n){const o=e.createShader(t);if(!o)throw new Error("Failed to create shader");if(e.shaderSource(o,n),e.compileShader(o),!e.getShaderParameter(o,e.COMPILE_STATUS)){const a=e.getShaderInfoLog(o)||"Unknown shader error";throw e.deleteShader(o),new Error(a)}return o}function Mn(e,t,n){const o=Rt(e,e.VERTEX_SHADER,t),a=Rt(e,e.FRAGMENT_SHADER,n),r=e.createProgram();if(!r)throw new Error("Failed to create program");if(e.attachShader(r,o),e.attachShader(r,a),e.linkProgram(r),e.deleteShader(o),e.deleteShader(a),!e.getProgramParameter(r,e.LINK_STATUS)){const l=e.getProgramInfoLog(r)||"Unknown program error";throw e.deleteProgram(r),new Error(l)}return r}function zn(e,t,n){const o={};for(const a of n)o[a]=e.getUniformLocation(t,a);return o}function Fn(e,t=2){const n=Math.min(window.devicePixelRatio||1,t),o=Math.max(1,Math.floor(e.clientWidth*n)),a=Math.max(1,Math.floor(e.clientHeight*n));return(e.width!==o||e.height!==a)&&(e.width=o,e.height=a),{width:o,height:a,dpr:n}}function Dn(e){const t=e.createVertexArray();if(!t)throw new Error("Failed to create VAO");return e.bindVertexArray(t),t}var rt=(e,t,n)=>{if(!t.has(e))throw TypeError("Cannot "+n)},i=(e,t,n)=>(rt(e,t,"read from private field"),n?n.call(e):t.get(e)),p=(e,t,n)=>{if(t.has(e))throw TypeError("Cannot add the same private member more than once");t instanceof WeakSet?t.add(e):t.set(e,n)},A=(e,t,n,o)=>(rt(e,t,"write to private field"),t.set(e,n),n),Pn=(e,t,n,o)=>({set _(a){A(e,t,a)},get _(){return i(e,t,o)}}),h=(e,t,n)=>(rt(e,t,"access private method"),n),v=new Uint8Array(8),_=new DataView(v.buffer),R=e=>[(e%256+256)%256],b=e=>(_.setUint16(0,e,!1),[v[0],v[1]]),Nn=e=>(_.setInt16(0,e,!1),[v[0],v[1]]),zt=e=>(_.setUint32(0,e,!1),[v[1],v[2],v[3]]),f=e=>(_.setUint32(0,e,!1),[v[0],v[1],v[2],v[3]]),_n=e=>(_.setInt32(0,e,!1),[v[0],v[1],v[2],v[3]]),Y=e=>(_.setUint32(0,Math.floor(e/2**32),!1),_.setUint32(4,e,!1),[v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]]),lt=e=>(_.setInt16(0,2**8*e,!1),[v[0],v[1]]),P=e=>(_.setInt32(0,2**16*e,!1),[v[0],v[1],v[2],v[3]]),He=e=>(_.setInt32(0,2**30*e,!1),[v[0],v[1],v[2],v[3]]),z=(e,t=!1)=>{let n=Array(e.length).fill(null).map((o,a)=>e.charCodeAt(a));return t&&n.push(0),n},Me=e=>e&&e[e.length-1],st=e=>{let t;for(let n of e)(!t||n.presentationTimestamp>t.presentationTimestamp)&&(t=n);return t},N=(e,t,n=!0)=>{let o=e*t;return n?Math.round(o):o},Ft=e=>{let t=e*(Math.PI/180),n=Math.cos(t),o=Math.sin(t);return[n,o,0,-o,n,0,0,0,1]},Dt=Ft(0),Pt=e=>[P(e[0]),P(e[1]),He(e[2]),P(e[3]),P(e[4]),He(e[5]),P(e[6]),P(e[7]),He(e[8])],se=e=>!e||typeof e!="object"?e:Array.isArray(e)?e.map(se):Object.fromEntries(Object.entries(e).map(([t,n])=>[t,se(n)])),ne=e=>e>=0&&e<2**32,T=(e,t,n)=>({type:e,contents:t&&new Uint8Array(t.flat(10)),children:n}),x=(e,t,n,o,a)=>T(e,[R(t),zt(n),o??[]],a),Un=e=>{let t=512;return e.fragmented?T("ftyp",[z("iso5"),f(t),z("iso5"),z("iso6"),z("mp41")]):T("ftyp",[z("isom"),f(t),z("isom"),e.holdsAvc?z("avc1"):[],z("mp41")])},Xe=e=>({type:"mdat",largeSize:e}),Gn=e=>({type:"free",size:e}),Te=(e,t,n=!1)=>T("moov",null,[Ln(t,e),...e.map(o=>Wn(o,t)),n?gi(e):null]),Ln=(e,t)=>{let n=N(Math.max(0,...t.filter(l=>l.samples.length>0).map(l=>{const u=st(l.samples);return u.presentationTimestamp+u.duration})),Ze),o=Math.max(...t.map(l=>l.id))+1,a=!ne(e)||!ne(n),r=a?Y:f;return x("mvhd",+a,0,[r(e),r(e),f(Ze),r(n),P(1),lt(1),Array(10).fill(0),Pt(Dt),Array(24).fill(0),f(o)])},Wn=(e,t)=>T("trak",null,[On(e,t),Hn(e,t)]),On=(e,t)=>{let n=st(e.samples),o=N(n?n.presentationTimestamp+n.duration:0,Ze),a=!ne(t)||!ne(o),r=a?Y:f,l;return e.info.type==="video"?l=typeof e.info.rotation=="number"?Ft(e.info.rotation):e.info.rotation:l=Dt,x("tkhd",+a,3,[r(t),r(t),f(e.id),f(0),r(o),Array(8).fill(0),b(0),b(0),lt(e.info.type==="audio"?1:0),b(0),Pt(l),P(e.info.type==="video"?e.info.width:0),P(e.info.type==="video"?e.info.height:0)])},Hn=(e,t)=>T("mdia",null,[Vn(e,t),qn(e.info.type==="video"?"vide":"soun"),jn(e)]),Vn=(e,t)=>{let n=st(e.samples),o=N(n?n.presentationTimestamp+n.duration:0,e.timescale),a=!ne(t)||!ne(o),r=a?Y:f;return x("mdhd",+a,0,[r(t),r(t),f(e.timescale),r(o),b(21956),b(0)])},qn=e=>x("hdlr",0,0,[z("mhlr"),z(e),f(0),f(0),f(0),z("mp4-muxer-hdlr",!0)]),jn=e=>T("minf",null,[e.info.type==="video"?Xn():$n(),Yn(),Qn(e)]),Xn=()=>x("vmhd",0,1,[b(0),b(0),b(0),b(0)]),$n=()=>x("smhd",0,0,[b(0),b(0)]),Yn=()=>T("dinf",null,[Zn()]),Zn=()=>x("dref",0,0,[f(1)],[Kn()]),Kn=()=>x("url ",0,1),Qn=e=>{const t=e.compositionTimeOffsetTable.length>1||e.compositionTimeOffsetTable.some(n=>n.sampleCompositionTimeOffset!==0);return T("stbl",null,[Jn(e),mi(e),di(e),pi(e),hi(e),vi(e),t?yi(e):null])},Jn=e=>x("stsd",0,0,[f(1)],[e.info.type==="video"?ei(Ai[e.info.codec],e):ui(Ei[e.info.codec],e)]),ei=(e,t)=>T(e,[Array(6).fill(0),b(1),b(0),b(0),Array(12).fill(0),b(t.info.width),b(t.info.height),f(4718592),f(4718592),f(0),b(1),Array(32).fill(0),b(24),Nn(65535)],[Ii[t.info.codec](t),t.info.decoderConfig.colorSpace?oi(t):null]),ti={bt709:1,bt470bg:5,smpte170m:6},ni={bt709:1,smpte170m:6,"iec61966-2-1":13},ii={rgb:0,bt709:1,bt470bg:5,smpte170m:6},oi=e=>T("colr",[z("nclx"),b(ti[e.info.decoderConfig.colorSpace.primaries]),b(ni[e.info.decoderConfig.colorSpace.transfer]),b(ii[e.info.decoderConfig.colorSpace.matrix]),R((e.info.decoderConfig.colorSpace.fullRange?1:0)<<7)]),ai=e=>e.info.decoderConfig&&T("avcC",[...new Uint8Array(e.info.decoderConfig.description)]),ri=e=>e.info.decoderConfig&&T("hvcC",[...new Uint8Array(e.info.decoderConfig.description)]),li=e=>{if(!e.info.decoderConfig)return null;let t=e.info.decoderConfig;if(!t.colorSpace)throw new Error("'colorSpace' is required in the decoder config for VP9.");let n=t.codec.split("."),o=Number(n[1]),a=Number(n[2]),u=(Number(n[3])<<4)+(0<<1)+Number(t.colorSpace.fullRange);return x("vpcC",1,0,[R(o),R(a),R(u),R(2),R(2),R(2),b(0)])},si=()=>{let n=(1<<7)+1;return T("av1C",[n,0,0,0])},ui=(e,t)=>T(e,[Array(6).fill(0),b(1),b(0),b(0),f(0),b(t.info.numberOfChannels),b(16),b(0),b(0),P(t.info.sampleRate)],[Mi[t.info.codec](t)]),fi=e=>{let t=new Uint8Array(e.info.decoderConfig.description);return x("esds",0,0,[f(58753152),R(32+t.byteLength),b(1),R(0),f(75530368),R(18+t.byteLength),R(64),R(21),zt(0),f(130071),f(130071),f(92307584),R(t.byteLength),...t,f(109084800),R(1),R(2)])},ci=e=>{var a;let t=3840,n=0;const o=(a=e.info.decoderConfig)==null?void 0:a.description;if(o){if(o.byteLength<18)throw new TypeError("Invalid decoder description provided for Opus; must be at least 18 bytes long.");const r=ArrayBuffer.isView(o)?new DataView(o.buffer,o.byteOffset,o.byteLength):new DataView(o);t=r.getUint16(10,!0),n=r.getInt16(14,!0)}return T("dOps",[R(0),R(e.info.numberOfChannels),b(t),f(e.info.sampleRate),lt(n),R(0)])},mi=e=>x("stts",0,0,[f(e.timeToSampleTable.length),e.timeToSampleTable.map(t=>[f(t.sampleCount),f(t.sampleDelta)])]),di=e=>{if(e.samples.every(n=>n.type==="key"))return null;let t=[...e.samples.entries()].filter(([,n])=>n.type==="key");return x("stss",0,0,[f(t.length),t.map(([n])=>f(n+1))])},pi=e=>x("stsc",0,0,[f(e.compactlyCodedChunkTable.length),e.compactlyCodedChunkTable.map(t=>[f(t.firstChunk),f(t.samplesPerChunk),f(1)])]),hi=e=>x("stsz",0,0,[f(0),f(e.samples.length),e.samples.map(t=>f(t.size))]),vi=e=>e.finalizedChunks.length>0&&Me(e.finalizedChunks).offset>=2**32?x("co64",0,0,[f(e.finalizedChunks.length),e.finalizedChunks.map(t=>Y(t.offset))]):x("stco",0,0,[f(e.finalizedChunks.length),e.finalizedChunks.map(t=>f(t.offset))]),yi=e=>x("ctts",0,0,[f(e.compositionTimeOffsetTable.length),e.compositionTimeOffsetTable.map(t=>[f(t.sampleCount),f(t.sampleCompositionTimeOffset)])]),gi=e=>T("mvex",null,e.map(bi)),bi=e=>x("trex",0,0,[f(e.id),f(1),f(0),f(0),f(0)]),Bt=(e,t)=>T("moof",null,[Si(e),...t.map(xi)]),Si=e=>x("mfhd",0,0,[f(e)]),Nt=e=>{let t=0,n=0,o=0,a=0,r=e.type==="delta";return n|=+r,r?t|=1:t|=2,t<<24|n<<16|o<<8|a},xi=e=>T("traf",null,[wi(e),Ci(e),Ti(e)]),wi=e=>{let t=0;t|=8,t|=16,t|=32,t|=131072;let n=e.currentChunk.samples[1]??e.currentChunk.samples[0],o={duration:n.timescaleUnitsToNextSample,size:n.size,flags:Nt(n)};return x("tfhd",0,t,[f(e.id),f(o.duration),f(o.size),f(o.flags)])},Ci=e=>x("tfdt",1,0,[Y(N(e.currentChunk.startTimestamp,e.timescale))]),Ti=e=>{let t=e.currentChunk.samples.map(F=>F.timescaleUnitsToNextSample),n=e.currentChunk.samples.map(F=>F.size),o=e.currentChunk.samples.map(Nt),a=e.currentChunk.samples.map(F=>N(F.presentationTimestamp-F.decodeTimestamp,e.timescale)),r=new Set(t),l=new Set(n),u=new Set(o),d=new Set(a),y=u.size===2&&o[0]!==o[1],g=r.size>1,w=l.size>1,S=!y&&u.size>1,O=d.size>1||[...d].some(F=>F!==0),U=0;return U|=1,U|=4*+y,U|=256*+g,U|=512*+w,U|=1024*+S,U|=2048*+O,x("trun",1,U,[f(e.currentChunk.samples.length),f(e.currentChunk.offset-e.currentChunk.moofOffset||0),y?f(o[0]):[],e.currentChunk.samples.map((F,Z)=>[g?f(t[Z]):[],w?f(n[Z]):[],S?f(o[Z]):[],O?_n(a[Z]):[]])])},ki=e=>T("mfra",null,[...e.map(Ri),Bi()]),Ri=(e,t)=>x("tfra",1,0,[f(e.id),f(63),f(e.finalizedChunks.length),e.finalizedChunks.map(o=>[Y(N(o.startTimestamp,e.timescale)),Y(o.moofOffset),f(t+1),f(1),f(1)])]),Bi=()=>x("mfro",0,0,[f(0)]),Ai={avc:"avc1",hevc:"hvc1",vp9:"vp09",av1:"av01"},Ii={avc:ai,hevc:ri,vp9:li,av1:si},Ei={aac:"mp4a",opus:"Opus"},Mi={aac:fi,opus:ci},Ge=class{},zi=class extends Ge{constructor(){super(...arguments),this.buffer=null}},_t=class extends Ge{constructor(e){if(super(),this.options=e,typeof e!="object")throw new TypeError("StreamTarget requires an options object to be passed to its constructor.");if(e.onData){if(typeof e.onData!="function")throw new TypeError("options.onData, when provided, must be a function.");if(e.onData.length<2)throw new TypeError("options.onData, when provided, must be a function that takes in at least two arguments (data and position). Ignoring the position argument, which specifies the byte offset at which the data is to be written, can lead to broken outputs.")}if(e.chunked!==void 0&&typeof e.chunked!="boolean")throw new TypeError("options.chunked, when provided, must be a boolean.");if(e.chunkSize!==void 0&&(!Number.isInteger(e.chunkSize)||e.chunkSize<1024))throw new TypeError("options.chunkSize, when provided, must be an integer and not smaller than 1024.")}},Ut=class extends Ge{constructor(e,t){if(super(),this.stream=e,this.options=t,!(e instanceof FileSystemWritableFileStream))throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");if(t!==void 0&&typeof t!="object")throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");if(t&&t.chunkSize!==void 0&&(!Number.isInteger(t.chunkSize)||t.chunkSize<=0))throw new TypeError("options.chunkSize, when provided, must be a positive integer")}},H,Q,Gt=class{constructor(){this.pos=0,p(this,H,new Uint8Array(8)),p(this,Q,new DataView(i(this,H).buffer)),this.offsets=new WeakMap}seek(e){this.pos=e}writeU32(e){i(this,Q).setUint32(0,e,!1),this.write(i(this,H).subarray(0,4))}writeU64(e){i(this,Q).setUint32(0,Math.floor(e/2**32),!1),i(this,Q).setUint32(4,e,!1),this.write(i(this,H).subarray(0,8))}writeAscii(e){for(let t=0;t<e.length;t++)i(this,Q).setUint8(t%8,e.charCodeAt(t)),t%8===7&&this.write(i(this,H));e.length%8!==0&&this.write(i(this,H).subarray(0,e.length%8))}writeBox(e){if(this.offsets.set(e,this.pos),e.contents&&!e.children)this.writeBoxHeader(e,e.size??e.contents.byteLength+8),this.write(e.contents);else{let t=this.pos;if(this.writeBoxHeader(e,0),e.contents&&this.write(e.contents),e.children)for(let a of e.children)a&&this.writeBox(a);let n=this.pos,o=e.size??n-t;this.seek(t),this.writeBoxHeader(e,o),this.seek(n)}}writeBoxHeader(e,t){this.writeU32(e.largeSize?1:t),this.writeAscii(e.type),e.largeSize&&this.writeU64(t)}measureBoxHeader(e){return 8+(e.largeSize?8:0)}patchBox(e){let t=this.pos;this.seek(this.offsets.get(e)),this.writeBox(e),this.seek(t)}measureBox(e){if(e.contents&&!e.children)return this.measureBoxHeader(e)+e.contents.byteLength;{let t=this.measureBoxHeader(e);if(e.contents&&(t+=e.contents.byteLength),e.children)for(let n of e.children)n&&(t+=this.measureBox(n));return t}}};H=new WeakMap;Q=new WeakMap;var ke,$,de,ae,Re,$e,Fi=class extends Gt{constructor(e){super(),p(this,Re),p(this,ke,void 0),p(this,$,new ArrayBuffer(2**16)),p(this,de,new Uint8Array(i(this,$))),p(this,ae,0),A(this,ke,e)}write(e){h(this,Re,$e).call(this,this.pos+e.byteLength),i(this,de).set(e,this.pos),this.pos+=e.byteLength,A(this,ae,Math.max(i(this,ae),this.pos))}finalize(){h(this,Re,$e).call(this,this.pos),i(this,ke).buffer=i(this,$).slice(0,Math.max(i(this,ae),this.pos))}};ke=new WeakMap;$=new WeakMap;de=new WeakMap;ae=new WeakMap;Re=new WeakSet;$e=function(e){let t=i(this,$).byteLength;for(;t<e;)t*=2;if(t===i(this,$).byteLength)return;let n=new ArrayBuffer(t),o=new Uint8Array(n);o.set(i(this,de),0),A(this,$,n),A(this,de,o)};var Di=2**24,Pi=2,ue,V,re,L,M,ze,Ye,ut,Lt,ft,Wt,fe,Fe,ct=class extends Gt{constructor(e){var t,n;super(),p(this,ze),p(this,ut),p(this,ft),p(this,fe),p(this,ue,void 0),p(this,V,[]),p(this,re,void 0),p(this,L,void 0),p(this,M,[]),A(this,ue,e),A(this,re,((t=e.options)==null?void 0:t.chunked)??!1),A(this,L,((n=e.options)==null?void 0:n.chunkSize)??Di)}write(e){i(this,V).push({data:e.slice(),start:this.pos}),this.pos+=e.byteLength}flush(){var n,o;if(i(this,V).length===0)return;let e=[],t=[...i(this,V)].sort((a,r)=>a.start-r.start);e.push({start:t[0].start,size:t[0].data.byteLength});for(let a=1;a<t.length;a++){let r=e[e.length-1],l=t[a];l.start<=r.start+r.size?r.size=Math.max(r.size,l.start+l.data.byteLength-r.start):e.push({start:l.start,size:l.data.byteLength})}for(let a of e){a.data=new Uint8Array(a.size);for(let r of i(this,V))a.start<=r.start&&r.start<a.start+a.size&&a.data.set(r.data,r.start-a.start);i(this,re)?(h(this,ze,Ye).call(this,a.data,a.start),h(this,fe,Fe).call(this)):(o=(n=i(this,ue).options).onData)==null||o.call(n,a.data,a.start)}i(this,V).length=0}finalize(){i(this,re)&&h(this,fe,Fe).call(this,!0)}};ue=new WeakMap;V=new WeakMap;re=new WeakMap;L=new WeakMap;M=new WeakMap;ze=new WeakSet;Ye=function(e,t){let n=i(this,M).findIndex(u=>u.start<=t&&t<u.start+i(this,L));n===-1&&(n=h(this,ft,Wt).call(this,t));let o=i(this,M)[n],a=t-o.start,r=e.subarray(0,Math.min(i(this,L)-a,e.byteLength));o.data.set(r,a);let l={start:a,end:a+r.byteLength};if(h(this,ut,Lt).call(this,o,l),o.written[0].start===0&&o.written[0].end===i(this,L)&&(o.shouldFlush=!0),i(this,M).length>Pi){for(let u=0;u<i(this,M).length-1;u++)i(this,M)[u].shouldFlush=!0;h(this,fe,Fe).call(this)}r.byteLength<e.byteLength&&h(this,ze,Ye).call(this,e.subarray(r.byteLength),t+r.byteLength)};ut=new WeakSet;Lt=function(e,t){let n=0,o=e.written.length-1,a=-1;for(;n<=o;){let r=Math.floor(n+(o-n+1)/2);e.written[r].start<=t.start?(n=r+1,a=r):o=r-1}for(e.written.splice(a+1,0,t),(a===-1||e.written[a].end<t.start)&&a++;a<e.written.length-1&&e.written[a].end>=e.written[a+1].start;)e.written[a].end=Math.max(e.written[a].end,e.written[a+1].end),e.written.splice(a+1,1)};ft=new WeakSet;Wt=function(e){let n={start:Math.floor(e/i(this,L))*i(this,L),data:new Uint8Array(i(this,L)),written:[],shouldFlush:!1};return i(this,M).push(n),i(this,M).sort((o,a)=>o.start-a.start),i(this,M).indexOf(n)};fe=new WeakSet;Fe=function(e=!1){var t,n;for(let o=0;o<i(this,M).length;o++){let a=i(this,M)[o];if(!(!a.shouldFlush&&!e)){for(let r of a.written)(n=(t=i(this,ue).options).onData)==null||n.call(t,a.data.subarray(r.start,r.end),a.start+r.start);i(this,M).splice(o--,1)}}};var Ni=class extends ct{constructor(e){var t;super(new _t({onData:(n,o)=>e.stream.write({type:"write",data:n,position:o}),chunked:!0,chunkSize:(t=e.options)==null?void 0:t.chunkSize}))}},Ze=1e3,_i=["avc","hevc","vp9","av1"],Ui=["aac","opus"],Gi=2082844800,Li=["strict","offset","cross-track-offset"],c,m,De,E,I,B,J,te,mt,q,j,ce,Ke,Ot,Qe,Ht,dt,Vt,Je,qt,pt,jt,Be,et,D,G,ht,Xt,me,Pe,Ne,vt,ie,ye,Ae,tt,Wi=class{constructor(e){if(p(this,Ke),p(this,Qe),p(this,dt),p(this,Je),p(this,pt),p(this,Be),p(this,D),p(this,ht),p(this,me),p(this,Ne),p(this,ie),p(this,Ae),p(this,c,void 0),p(this,m,void 0),p(this,De,void 0),p(this,E,void 0),p(this,I,null),p(this,B,null),p(this,J,Math.floor(Date.now()/1e3)+Gi),p(this,te,[]),p(this,mt,1),p(this,q,[]),p(this,j,[]),p(this,ce,!1),h(this,Ke,Ot).call(this,e),e.video=se(e.video),e.audio=se(e.audio),e.fastStart=se(e.fastStart),this.target=e.target,A(this,c,{firstTimestampBehavior:"strict",...e}),e.target instanceof zi)A(this,m,new Fi(e.target));else if(e.target instanceof _t)A(this,m,new ct(e.target));else if(e.target instanceof Ut)A(this,m,new Ni(e.target));else throw new Error(`Invalid target: ${e.target}`);h(this,Je,qt).call(this),h(this,Qe,Ht).call(this)}addVideoChunk(e,t,n,o){if(!(e instanceof EncodedVideoChunk))throw new TypeError("addVideoChunk's first argument (sample) must be of type EncodedVideoChunk.");if(t&&typeof t!="object")throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");if(n!==void 0&&(!Number.isFinite(n)||n<0))throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");if(o!==void 0&&!Number.isFinite(o))throw new TypeError("addVideoChunk's fourth argument (compositionTimeOffset), when provided, must be a real number.");let a=new Uint8Array(e.byteLength);e.copyTo(a),this.addVideoChunkRaw(a,e.type,n??e.timestamp,e.duration,t,o)}addVideoChunkRaw(e,t,n,o,a,r){if(!(e instanceof Uint8Array))throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");if(t!=="key"&&t!=="delta")throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(n)||n<0)throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(o)||o<0)throw new TypeError("addVideoChunkRaw's fourth argument (duration) must be a non-negative real number.");if(a&&typeof a!="object")throw new TypeError("addVideoChunkRaw's fifth argument (meta), when provided, must be an object.");if(r!==void 0&&!Number.isFinite(r))throw new TypeError("addVideoChunkRaw's sixth argument (compositionTimeOffset), when provided, must be a real number.");if(h(this,Ae,tt).call(this),!i(this,c).video)throw new Error("No video track declared.");if(typeof i(this,c).fastStart=="object"&&i(this,I).samples.length===i(this,c).fastStart.expectedVideoChunks)throw new Error(`Cannot add more video chunks than specified in 'fastStart' (${i(this,c).fastStart.expectedVideoChunks}).`);let l=h(this,Be,et).call(this,i(this,I),e,t,n,o,a,r);if(i(this,c).fastStart==="fragmented"&&i(this,B)){for(;i(this,j).length>0&&i(this,j)[0].decodeTimestamp<=l.decodeTimestamp;){let u=i(this,j).shift();h(this,D,G).call(this,i(this,B),u)}l.decodeTimestamp<=i(this,B).lastDecodeTimestamp?h(this,D,G).call(this,i(this,I),l):i(this,q).push(l)}else h(this,D,G).call(this,i(this,I),l)}addAudioChunk(e,t,n){if(!(e instanceof EncodedAudioChunk))throw new TypeError("addAudioChunk's first argument (sample) must be of type EncodedAudioChunk.");if(t&&typeof t!="object")throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");if(n!==void 0&&(!Number.isFinite(n)||n<0))throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");let o=new Uint8Array(e.byteLength);e.copyTo(o),this.addAudioChunkRaw(o,e.type,n??e.timestamp,e.duration,t)}addAudioChunkRaw(e,t,n,o,a){if(!(e instanceof Uint8Array))throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");if(t!=="key"&&t!=="delta")throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(n)||n<0)throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(o)||o<0)throw new TypeError("addAudioChunkRaw's fourth argument (duration) must be a non-negative real number.");if(a&&typeof a!="object")throw new TypeError("addAudioChunkRaw's fifth argument (meta), when provided, must be an object.");if(h(this,Ae,tt).call(this),!i(this,c).audio)throw new Error("No audio track declared.");if(typeof i(this,c).fastStart=="object"&&i(this,B).samples.length===i(this,c).fastStart.expectedAudioChunks)throw new Error(`Cannot add more audio chunks than specified in 'fastStart' (${i(this,c).fastStart.expectedAudioChunks}).`);let r=h(this,Be,et).call(this,i(this,B),e,t,n,o,a);if(i(this,c).fastStart==="fragmented"&&i(this,I)){for(;i(this,q).length>0&&i(this,q)[0].decodeTimestamp<=r.decodeTimestamp;){let l=i(this,q).shift();h(this,D,G).call(this,i(this,I),l)}r.decodeTimestamp<=i(this,I).lastDecodeTimestamp?h(this,D,G).call(this,i(this,B),r):i(this,j).push(r)}else h(this,D,G).call(this,i(this,B),r)}finalize(){if(i(this,ce))throw new Error("Cannot finalize a muxer more than once.");if(i(this,c).fastStart==="fragmented"){for(let t of i(this,q))h(this,D,G).call(this,i(this,I),t);for(let t of i(this,j))h(this,D,G).call(this,i(this,B),t);h(this,Ne,vt).call(this,!1)}else i(this,I)&&h(this,me,Pe).call(this,i(this,I)),i(this,B)&&h(this,me,Pe).call(this,i(this,B));let e=[i(this,I),i(this,B)].filter(Boolean);if(i(this,c).fastStart==="in-memory"){let t;for(let o=0;o<2;o++){let a=Te(e,i(this,J)),r=i(this,m).measureBox(a);t=i(this,m).measureBox(i(this,E));let l=i(this,m).pos+r+t;for(let u of i(this,te)){u.offset=l;for(let{data:d}of u.samples)l+=d.byteLength,t+=d.byteLength}if(l<2**32)break;t>=2**32&&(i(this,E).largeSize=!0)}let n=Te(e,i(this,J));i(this,m).writeBox(n),i(this,E).size=t,i(this,m).writeBox(i(this,E));for(let o of i(this,te))for(let a of o.samples)i(this,m).write(a.data),a.data=null}else if(i(this,c).fastStart==="fragmented"){let t=i(this,m).pos,n=ki(e);i(this,m).writeBox(n);let o=i(this,m).pos-t;i(this,m).seek(i(this,m).pos-4),i(this,m).writeU32(o)}else{let t=i(this,m).offsets.get(i(this,E)),n=i(this,m).pos-t;i(this,E).size=n,i(this,E).largeSize=n>=2**32,i(this,m).patchBox(i(this,E));let o=Te(e,i(this,J));if(typeof i(this,c).fastStart=="object"){i(this,m).seek(i(this,De)),i(this,m).writeBox(o);let a=t-i(this,m).pos;i(this,m).writeBox(Gn(a))}else i(this,m).writeBox(o)}h(this,ie,ye).call(this),i(this,m).finalize(),A(this,ce,!0)}};c=new WeakMap;m=new WeakMap;De=new WeakMap;E=new WeakMap;I=new WeakMap;B=new WeakMap;J=new WeakMap;te=new WeakMap;mt=new WeakMap;q=new WeakMap;j=new WeakMap;ce=new WeakMap;Ke=new WeakSet;Ot=function(e){if(typeof e!="object")throw new TypeError("The muxer requires an options object to be passed to its constructor.");if(!(e.target instanceof Ge))throw new TypeError("The target must be provided and an instance of Target.");if(e.video){if(!_i.includes(e.video.codec))throw new TypeError(`Unsupported video codec: ${e.video.codec}`);if(!Number.isInteger(e.video.width)||e.video.width<=0)throw new TypeError(`Invalid video width: ${e.video.width}. Must be a positive integer.`);if(!Number.isInteger(e.video.height)||e.video.height<=0)throw new TypeError(`Invalid video height: ${e.video.height}. Must be a positive integer.`);const t=e.video.rotation;if(typeof t=="number"&&![0,90,180,270].includes(t))throw new TypeError(`Invalid video rotation: ${t}. Has to be 0, 90, 180 or 270.`);if(Array.isArray(t)&&(t.length!==9||t.some(n=>typeof n!="number")))throw new TypeError(`Invalid video transformation matrix: ${t.join()}`);if(e.video.frameRate!==void 0&&(!Number.isInteger(e.video.frameRate)||e.video.frameRate<=0))throw new TypeError(`Invalid video frame rate: ${e.video.frameRate}. Must be a positive integer.`)}if(e.audio){if(!Ui.includes(e.audio.codec))throw new TypeError(`Unsupported audio codec: ${e.audio.codec}`);if(!Number.isInteger(e.audio.numberOfChannels)||e.audio.numberOfChannels<=0)throw new TypeError(`Invalid number of audio channels: ${e.audio.numberOfChannels}. Must be a positive integer.`);if(!Number.isInteger(e.audio.sampleRate)||e.audio.sampleRate<=0)throw new TypeError(`Invalid audio sample rate: ${e.audio.sampleRate}. Must be a positive integer.`)}if(e.firstTimestampBehavior&&!Li.includes(e.firstTimestampBehavior))throw new TypeError(`Invalid first timestamp behavior: ${e.firstTimestampBehavior}`);if(typeof e.fastStart=="object"){if(e.video){if(e.fastStart.expectedVideoChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedVideoChunks'.");if(!Number.isInteger(e.fastStart.expectedVideoChunks)||e.fastStart.expectedVideoChunks<0)throw new TypeError("'expectedVideoChunks' must be a non-negative integer.")}if(e.audio){if(e.fastStart.expectedAudioChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedAudioChunks'.");if(!Number.isInteger(e.fastStart.expectedAudioChunks)||e.fastStart.expectedAudioChunks<0)throw new TypeError("'expectedAudioChunks' must be a non-negative integer.")}}else if(![!1,"in-memory","fragmented"].includes(e.fastStart))throw new TypeError("'fastStart' option must be false, 'in-memory', 'fragmented' or an object.");if(e.minFragmentDuration!==void 0&&(!Number.isFinite(e.minFragmentDuration)||e.minFragmentDuration<0))throw new TypeError("'minFragmentDuration' must be a non-negative number.")};Qe=new WeakSet;Ht=function(){var e;if(i(this,m).writeBox(Un({holdsAvc:((e=i(this,c).video)==null?void 0:e.codec)==="avc",fragmented:i(this,c).fastStart==="fragmented"})),A(this,De,i(this,m).pos),i(this,c).fastStart==="in-memory")A(this,E,Xe(!1));else if(i(this,c).fastStart!=="fragmented"){if(typeof i(this,c).fastStart=="object"){let t=h(this,dt,Vt).call(this);i(this,m).seek(i(this,m).pos+t)}A(this,E,Xe(!0)),i(this,m).writeBox(i(this,E))}h(this,ie,ye).call(this)};dt=new WeakSet;Vt=function(){if(typeof i(this,c).fastStart!="object")return;let e=0,t=[i(this,c).fastStart.expectedVideoChunks,i(this,c).fastStart.expectedAudioChunks];for(let n of t)n&&(e+=8*Math.ceil(2/3*n),e+=4*n,e+=12*Math.ceil(2/3*n),e+=4*n,e+=8*n);return e+=4096,e};Je=new WeakSet;qt=function(){if(i(this,c).video&&A(this,I,{id:1,info:{type:"video",codec:i(this,c).video.codec,width:i(this,c).video.width,height:i(this,c).video.height,rotation:i(this,c).video.rotation??0,decoderConfig:null},timescale:i(this,c).video.frameRate??57600,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,c).audio&&(A(this,B,{id:i(this,c).video?2:1,info:{type:"audio",codec:i(this,c).audio.codec,numberOfChannels:i(this,c).audio.numberOfChannels,sampleRate:i(this,c).audio.sampleRate,decoderConfig:null},timescale:i(this,c).audio.sampleRate,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,c).audio.codec==="aac")){let e=h(this,pt,jt).call(this,2,i(this,c).audio.sampleRate,i(this,c).audio.numberOfChannels);i(this,B).info.decoderConfig={codec:i(this,c).audio.codec,description:e,numberOfChannels:i(this,c).audio.numberOfChannels,sampleRate:i(this,c).audio.sampleRate}}};pt=new WeakSet;jt=function(e,t,n){let a=[96e3,88200,64e3,48e3,44100,32e3,24e3,22050,16e3,12e3,11025,8e3,7350].indexOf(t),r=n,l="";l+=e.toString(2).padStart(5,"0"),l+=a.toString(2).padStart(4,"0"),a===15&&(l+=t.toString(2).padStart(24,"0")),l+=r.toString(2).padStart(4,"0");let u=Math.ceil(l.length/8)*8;l=l.padEnd(u,"0");let d=new Uint8Array(l.length/8);for(let y=0;y<l.length;y+=8)d[y/8]=parseInt(l.slice(y,y+8),2);return d};Be=new WeakSet;et=function(e,t,n,o,a,r,l){let u=o/1e6,d=(o-(l??0))/1e6,y=a/1e6,g=h(this,ht,Xt).call(this,u,d,e);return u=g.presentationTimestamp,d=g.decodeTimestamp,r!=null&&r.decoderConfig&&(e.info.decoderConfig===null?e.info.decoderConfig=r.decoderConfig:Object.assign(e.info.decoderConfig,r.decoderConfig)),{presentationTimestamp:u,decodeTimestamp:d,duration:y,data:t,size:t.byteLength,type:n,timescaleUnitsToNextSample:N(y,e.timescale)}};D=new WeakSet;G=function(e,t){i(this,c).fastStart!=="fragmented"&&e.samples.push(t);const n=N(t.presentationTimestamp-t.decodeTimestamp,e.timescale);if(e.lastTimescaleUnits!==null){let a=N(t.decodeTimestamp,e.timescale,!1),r=Math.round(a-e.lastTimescaleUnits);if(e.lastTimescaleUnits+=r,e.lastSample.timescaleUnitsToNextSample=r,i(this,c).fastStart!=="fragmented"){let l=Me(e.timeToSampleTable);l.sampleCount===1?(l.sampleDelta=r,l.sampleCount++):l.sampleDelta===r?l.sampleCount++:(l.sampleCount--,e.timeToSampleTable.push({sampleCount:2,sampleDelta:r}));const u=Me(e.compositionTimeOffsetTable);u.sampleCompositionTimeOffset===n?u.sampleCount++:e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:n})}}else e.lastTimescaleUnits=0,i(this,c).fastStart!=="fragmented"&&(e.timeToSampleTable.push({sampleCount:1,sampleDelta:N(t.duration,e.timescale)}),e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:n}));e.lastSample=t;let o=!1;if(!e.currentChunk)o=!0;else{let a=t.presentationTimestamp-e.currentChunk.startTimestamp;if(i(this,c).fastStart==="fragmented"){let r=i(this,I)??i(this,B);const l=i(this,c).minFragmentDuration??1;e===r&&t.type==="key"&&a>=l&&(o=!0,h(this,Ne,vt).call(this))}else o=a>=.5}o&&(e.currentChunk&&h(this,me,Pe).call(this,e),e.currentChunk={startTimestamp:t.presentationTimestamp,samples:[]}),e.currentChunk.samples.push(t)};ht=new WeakSet;Xt=function(e,t,n){var l,u;const o=i(this,c).firstTimestampBehavior==="strict",a=n.lastDecodeTimestamp===-1;if(o&&a&&t!==0)throw new Error(`The first chunk for your media track must have a timestamp of 0 (received DTS=${t}).Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of thedocument, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
`);if(i(this,c).firstTimestampBehavior==="offset"||i(this,c).firstTimestampBehavior==="cross-track-offset"){n.firstDecodeTimestamp===void 0&&(n.firstDecodeTimestamp=t);let d;i(this,c).firstTimestampBehavior==="offset"?d=n.firstDecodeTimestamp:d=Math.min(((l=i(this,I))==null?void 0:l.firstDecodeTimestamp)??1/0,((u=i(this,B))==null?void 0:u.firstDecodeTimestamp)??1/0),t-=d,e-=d}if(t<n.lastDecodeTimestamp)throw new Error(`Timestamps must be monotonically increasing (DTS went from ${n.lastDecodeTimestamp*1e6} to ${t*1e6}).`);return n.lastDecodeTimestamp=t,{presentationTimestamp:e,decodeTimestamp:t}};me=new WeakSet;Pe=function(e){if(i(this,c).fastStart==="fragmented")throw new Error("Can't finalize individual chunks if 'fastStart' is set to 'fragmented'.");if(e.currentChunk){if(e.finalizedChunks.push(e.currentChunk),i(this,te).push(e.currentChunk),(e.compactlyCodedChunkTable.length===0||Me(e.compactlyCodedChunkTable).samplesPerChunk!==e.currentChunk.samples.length)&&e.compactlyCodedChunkTable.push({firstChunk:e.finalizedChunks.length,samplesPerChunk:e.currentChunk.samples.length}),i(this,c).fastStart==="in-memory"){e.currentChunk.offset=0;return}e.currentChunk.offset=i(this,m).pos;for(let t of e.currentChunk.samples)i(this,m).write(t.data),t.data=null;h(this,ie,ye).call(this)}};Ne=new WeakSet;vt=function(e=!0){if(i(this,c).fastStart!=="fragmented")throw new Error("Can't finalize a fragment unless 'fastStart' is set to 'fragmented'.");let t=[i(this,I),i(this,B)].filter(u=>u&&u.currentChunk);if(t.length===0)return;let n=Pn(this,mt)._++;if(n===1){let u=Te(t,i(this,J),!0);i(this,m).writeBox(u)}let o=i(this,m).pos,a=Bt(n,t);i(this,m).writeBox(a);{let u=Xe(!1),d=0;for(let g of t)for(let w of g.currentChunk.samples)d+=w.size;let y=i(this,m).measureBox(u)+d;y>=2**32&&(u.largeSize=!0,y=i(this,m).measureBox(u)+d),u.size=y,i(this,m).writeBox(u)}for(let u of t){u.currentChunk.offset=i(this,m).pos,u.currentChunk.moofOffset=o;for(let d of u.currentChunk.samples)i(this,m).write(d.data),d.data=null}let r=i(this,m).pos;i(this,m).seek(i(this,m).offsets.get(a));let l=Bt(n,t);i(this,m).writeBox(l),i(this,m).seek(r);for(let u of t)u.finalizedChunks.push(u.currentChunk),i(this,te).push(u.currentChunk),u.currentChunk=null;e&&h(this,ie,ye).call(this)};ie=new WeakSet;ye=function(){i(this,m)instanceof ct&&i(this,m).flush()};Ae=new WeakSet;tt=function(){if(i(this,ce))throw new Error("Cannot add new video or audio chunks after the file has been finalized.")};const $t=document.querySelector("#app");if(!$t)throw new Error("Missing #app root element");$t.innerHTML=`
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
`;const k=document.querySelector("#gl-canvas"),ee=document.querySelector("#scene-list"),nt=document.querySelector("#control-list"),Ie=document.querySelector("#panel-actions"),le=document.querySelector("#key-help"),_e=document.querySelector("#hud-title"),Yt=document.querySelector("#hud-desc"),Le=document.querySelector("[data-action='toggle-sidebar']"),yt=document.querySelector(".stage");if(!k||!ee||!nt||!Ie||!le||!_e||!Yt||!Le||!yt)throw new Error("Missing required UI elements");const s=k.getContext("webgl2",{antialias:!0});if(!s)throw yt.innerHTML=`
    <div class="fallback">
      <h2>WebGL2 unavailable</h2>
      <p>Your browser or GPU does not expose WebGL2. Try a different browser.</p>
    </div>
  `,new Error("WebGL2 unavailable");const Oi=s.getExtension("EXT_color_buffer_float"),Hi=!!Oi;s.disable(s.DEPTH_TEST);s.disable(s.BLEND);const At=Dn(s),Zt=new Map,Kt={},oe={},Qt={},it=new Map;for(const e of ve){const t=Mn(s,mn,e.fragment),n=new Set;n.add(e.resolutionUniform),n.add(e.timeUniform),e.loopUniform&&n.add(e.loopUniform),e.stateful&&(n.add(e.passUniform??"uPass"),n.add(e.stateUniform??"uState"),n.add(e.gridUniform??"uGridSize"));for(const l of e.params)n.add(l.uniform);const o=zn(s,t,Array.from(n));Zt.set(e.id,{program:t,uniforms:o});const a={},r={};for(const l of e.params)a[l.id]=l,r[l.id]=l.type==="seed"?Math.floor(Math.random()*1e6):l.value;Kt[e.id]=a,oe[e.id]={...r},Qt[e.id]={...r}}let W=ve[0],ot={},gt=performance.now(),Ve=null,qe=null,X=null,we=[],Jt=0,Ee=null,ge=!1;function Vi(){const e=["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm","video/mp4"];for(const t of e)if(MediaRecorder.isTypeSupported(t))return t;return""}function qi(e){return e.startsWith("video/mp4")?"mp4":"webm"}function ji(e){const t=Math.floor(e/1e3),n=String(Math.floor(t/60)).padStart(2,"0"),o=String(t%60).padStart(2,"0");return`${n}:${o}`}function Xi(){const e=document.getElementById("rec-badge");!e||!ge||(e.textContent=`⏺ ${ji(performance.now()-Jt)}`)}function $i(){const e=Vi();if(!e){alert("Recording is not supported in this browser.");return}const t=k.captureStream(60);we=[],X=new MediaRecorder(t,{mimeType:e,videoBitsPerSecond:16e6}),X.ondataavailable=n=>{n.data.size>0&&we.push(n.data)},X.onstop=()=>{const n=qi(e),o=new Blob(we,{type:e}),a=URL.createObjectURL(o),r=document.createElement("a");r.href=a,r.download=`${W.id}-${Date.now()}.${n}`,r.click(),URL.revokeObjectURL(a),we=[]},X.start(500),ge=!0,Jt=performance.now(),en(),Ee=window.setInterval(Xi,250)}function Yi(){X&&X.state!=="inactive"&&X.stop(),ge=!1,Ee!==null&&(clearInterval(Ee),Ee=null),en()}function Zi(){ge?Yi():$i()}function en(){const e=document.getElementById("rec-btn"),t=document.getElementById("rec-badge");!e||!t||(ge?(e.textContent="Stop",e.classList.add("recording"),t.classList.remove("hidden")):(e.textContent="Record",e.classList.remove("recording"),t.classList.add("hidden"),t.textContent=""))}let Ue=!1;function at(e){Le.classList.toggle("hidden",e)}function Ki(){at(!1),qe!==null&&window.clearTimeout(qe),qe=window.setTimeout(()=>{at(!0)},2500)}function tn(){const e=document.body.classList.contains("sidebar-collapsed");Le.textContent=e?">>":"<<"}function Qi(){var e;(e=_e.parentElement)==null||e.classList.remove("hidden"),Ve!==null&&window.clearTimeout(Ve),Ve=window.setTimeout(()=>{var t;(t=_e.parentElement)==null||t.classList.add("hidden")},1e4)}function It(e){const t=s.createTexture();if(!t)throw new Error("Failed to create state texture");return s.bindTexture(s.TEXTURE_2D,t),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_MIN_FILTER,s.NEAREST),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_MAG_FILTER,s.NEAREST),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_S,s.REPEAT),s.texParameteri(s.TEXTURE_2D,s.TEXTURE_WRAP_T,s.REPEAT),s.texImage2D(s.TEXTURE_2D,0,s.RGBA16F,e,e,0,s.RGBA,s.HALF_FLOAT,null),s.bindTexture(s.TEXTURE_2D,null),t}function Ji(e){const t=It(e),n=It(e),o=s.createFramebuffer(),a=s.createFramebuffer();if(!o||!a)throw new Error("Failed to create framebuffer");return s.bindFramebuffer(s.FRAMEBUFFER,o),s.framebufferTexture2D(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,t,0),s.bindFramebuffer(s.FRAMEBUFFER,a),s.framebufferTexture2D(s.FRAMEBUFFER,s.COLOR_ATTACHMENT0,s.TEXTURE_2D,n,0),s.bindFramebuffer(s.FRAMEBUFFER,null),{size:e,textures:[t,n],fbos:[o,a],index:0,needsInit:!0}}function bt(e){if(!e.stateful)return null;let t=it.get(e.id);const n=e.bufferSize??192;return(!t||t.size!==n)&&(t=Ji(n),it.set(e.id,t)),t}function pe(e){const t=it.get(e);t&&(t.needsInit=!0)}function eo(e,t){let n=t;return e.min!==void 0&&(n=Math.max(e.min,n)),e.max!==void 0&&(n=Math.min(e.max,n)),e.type==="int"&&(n=Math.round(n)),n}function nn(e,t){if(e.type==="int")return String(Math.round(t));const n=e.step??.01,o=n<1?Math.min(4,Math.max(2,Math.ceil(-Math.log10(n)))):0;return t.toFixed(o)}function to(e){return e.length===1?e.toLowerCase():e}function no(e){return e instanceof HTMLInputElement||e instanceof HTMLTextAreaElement||e instanceof HTMLSelectElement}function he(e,t,n,o=!0){const a=Kt[e][t];if(!a)return;const r=eo(a,n);if(oe[e][t]=r,e===W.id&&o){const l=ot[t];l!=null&&l.range&&(l.range.value=String(r)),l!=null&&l.number&&(l.number.value=nn(a,r))}}function io(e){var t;for(const n of((t=ve.find(o=>o.id===e))==null?void 0:t.params)??[])n.type==="seed"&&he(e,n.id,Math.floor(Math.random()*1e6));pe(e)}function oo(e){const t=Qt[e];for(const[n,o]of Object.entries(t))he(e,n,o,!0);pe(e)}function ao(){ee.innerHTML="";for(const e of ve){const t=document.createElement("option");t.value=e.id,t.textContent=e.name,ee.appendChild(t)}ee.addEventListener("change",()=>{on(ee.value)})}function ro(e){Ie.innerHTML="";const t=document.createElement("button");if(t.className="ghost small",t.textContent="Reset",t.addEventListener("click",()=>oo(e.id)),Ie.appendChild(t),e.params.some(o=>o.type==="seed")){const o=document.createElement("button");o.className="ghost small",o.textContent="Reseed",o.addEventListener("click",()=>io(e.id)),Ie.appendChild(o)}}function lo(e){nt.innerHTML="",ot={};for(const t of e.params){if(t.type==="seed")continue;const n=document.createElement("div");n.className="control";const o=document.createElement("div");o.className="control-header";const a=document.createElement("label");if(a.textContent=t.label,o.appendChild(a),t.key){const d=document.createElement("span");d.className="key-cap",d.textContent=`${t.key.inc.toUpperCase()}/${t.key.dec.toUpperCase()}`,o.appendChild(d)}const r=document.createElement("div");r.className="control-inputs";const l=document.createElement("input");l.type="range",l.min=String(t.min??0),l.max=String(t.max??1),l.step=String(t.step??(t.type==="int"?1:.01)),l.value=String(oe[e.id][t.id]),l.addEventListener("input",d=>{const y=Number(d.target.value);Number.isNaN(y)||he(e.id,t.id,y)});const u=document.createElement("input");u.type="number",u.min=l.min,u.max=l.max,u.step=l.step,u.value=nn(t,oe[e.id][t.id]),u.addEventListener("input",d=>{const y=Number(d.target.value);Number.isNaN(y)||he(e.id,t.id,y)}),r.appendChild(l),r.appendChild(u),n.appendChild(o),n.appendChild(r),nt.appendChild(n),ot[t.id]={range:l,number:u}}}function so(e){le.innerHTML="";for(const t of e.params){if(!t.key||t.type==="seed")continue;const n=document.createElement("div");n.className="key-row",n.textContent=`${t.key.inc.toUpperCase()}/${t.key.dec.toUpperCase()}  ${t.label}`,le.appendChild(n)}le.childElementCount||(le.textContent="No mapped keys for this scene.")}function on(e){const t=ve.find(n=>n.id===e);t&&(W=t,t.stateful&&(bt(t),pe(t.id)),_e.textContent=t.name,Yt.textContent=t.description,Qi(),ro(t),lo(t),so(t),ee.value=t.id)}function uo(e){if(no(e.target))return;const t=to(e.key),n=W.params;for(const o of n){if(!o.key||o.type==="seed")continue;const a=t===o.key.inc,r=t===o.key.dec;if(!a&&!r)continue;const l=e.shiftKey&&o.key.shiftStep?o.key.shiftStep:o.key.step,d=oe[W.id][o.id]+l*(a?1:-1);he(W.id,o.id,d),e.preventDefault();break}}function Ce(e,t,n,o,a){const r=oe[e.id],l=t.uniforms,u=l[e.resolutionUniform];u&&s.uniform2f(u,o,a);const d=l[e.timeUniform];if(d)if(e.timeMode==="phase"){const g=e.loopDuration??8,w=n%g/g;s.uniform1f(d,w)}else if(e.timeMode==="looped"){const g=e.loopDuration??8,w=n%g;if(s.uniform1f(d,w),e.loopUniform){const S=l[e.loopUniform];S&&s.uniform1f(S,g)}}else s.uniform1f(d,n);const y={};for(const g of e.params){const w=l[g.uniform],S=r[g.id];if(g.component!==void 0){const O=y[g.uniform]??[0,0,0];O[g.component]=S,y[g.uniform]=O;continue}w&&(g.type==="int"?s.uniform1i(w,Math.round(S)):s.uniform1f(w,S))}for(const[g,w]of Object.entries(y)){const S=l[g];S&&s.uniform3f(S,w[0],w[1],w[2])}}function je(e,t,n,o){const a=t.uniforms[e.passUniform??"uPass"];a&&s.uniform1i(a,o);const r=t.uniforms[e.gridUniform??"uGridSize"];r&&s.uniform2f(r,n.size,n.size);const l=t.uniforms[e.stateUniform??"uState"];l&&s.uniform1i(l,0)}function an(e,t,n,o){const a=Zt.get(e.id);if(a){if(e.stateful){if(!Hi)return;const r=bt(e);if(!r)return;const l=()=>r.textures[r.index],u=()=>r.fbos[(r.index+1)%2];s.useProgram(a.program),s.bindVertexArray(At),r.needsInit&&(s.bindFramebuffer(s.FRAMEBUFFER,u()),s.viewport(0,0,r.size,r.size),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,l()),Ce(e,a,t,n,o),je(e,a,r,2),s.drawArrays(s.TRIANGLES,0,3),r.index=(r.index+1)%2,r.needsInit=!1),s.bindFramebuffer(s.FRAMEBUFFER,u()),s.viewport(0,0,r.size,r.size),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,l()),Ce(e,a,t,n,o),je(e,a,r,0),s.drawArrays(s.TRIANGLES,0,3),r.index=(r.index+1)%2,s.bindFramebuffer(s.FRAMEBUFFER,null),s.viewport(0,0,n,o),s.clearColor(0,0,0,1),s.clear(s.COLOR_BUFFER_BIT),s.activeTexture(s.TEXTURE0),s.bindTexture(s.TEXTURE_2D,l()),Ce(e,a,t,n,o),je(e,a,r,1),s.drawArrays(s.TRIANGLES,0,3);return}s.viewport(0,0,n,o),s.clearColor(0,0,0,1),s.clear(s.COLOR_BUFFER_BIT),s.useProgram(a.program),s.bindVertexArray(At),Ce(e,a,t,n,o),s.drawArrays(s.TRIANGLES,0,3)}}function St(e){if(Ue)return;const t=(e-gt)/1e3,{width:n,height:o}=Fn(k);an(W,t,n,o),requestAnimationFrame(St)}function fo(){return new Promise(e=>requestAnimationFrame(()=>e()))}async function co(e,t,n,o,a){if(typeof VideoEncoder>"u"||typeof VideoFrame>"u"){alert("Offline rendering requires the WebCodecs API (Chrome/Edge 94+, Safari 16.4+).");return}const r={codec:"avc1.640028",width:o,height:a,bitrate:16e6,framerate:n};if(!(await VideoEncoder.isConfigSupported(r)).supported){alert("H.264 video encoding is not supported on this device.");return}let u;try{u=await window.showSaveFilePicker({suggestedName:`${e.id}-${o}x${a}-${n}fps-${t}s.mp4`,types:[{description:"MP4 Video",accept:{"video/mp4":[".mp4"]}}]})}catch{return}const d=await u.createWritable(),y=document.getElementById("offline-btn"),g=document.getElementById("offline-progress"),w=document.getElementById("offline-bar"),S=document.getElementById("offline-status");y&&(y.disabled=!0),g==null||g.classList.remove("hidden"),S==null||S.classList.remove("hidden"),Ue=!0,e.stateful&&(bt(e),pe(e.id));const O=k.width,U=k.height,F=k.style.width,Z=k.style.height;k.width=o,k.height=a,k.style.width=`${o}px`,k.style.height=`${a}px`;const rn=new Ut(d),xt=new Wi({target:rn,video:{codec:"avc",width:o,height:a},fastStart:!1});let be=null;const K=new VideoEncoder({output:(C,Oe)=>xt.addVideoChunk(C,Oe??void 0),error:C=>{be=C}});K.configure(r);const Se=Math.ceil(t*n),wt=Math.round(1e6/n),ln=n*2,sn=n*30,un=performance.now();let xe=!1;const Ct=C=>{C.preventDefault(),xe=!0};k.addEventListener("webglcontextlost",Ct);for(let C=0;C<Se&&!(be||xe);C++){const Oe=C/n;an(e,Oe,o,a),s.finish();const Tt=new VideoFrame(k,{timestamp:C*wt,duration:wt});K.encode(Tt,{keyFrame:C%ln===0}),Tt.close(),C>0&&C%sn===0&&await K.flush();const fn=(C+1)/Se*100;if(w&&(w.style.width=`${fn}%`),C%30===0){const cn=(performance.now()-un)/1e3/(C+1)*(Se-C-1);S&&(S.textContent=`Frame ${C+1}/${Se}  —  ~${Math.ceil(cn)}s left`)}K.encodeQueueSize>10&&await new Promise(kt=>setTimeout(kt,1)),await fo()}k.removeEventListener("webglcontextlost",Ct);try{await K.flush(),K.close(),xt.finalize(),await d.close()}catch(C){console.error("Finalization failed:",C);try{await d.close()}catch{}}k.style.width=F,k.style.height=Z,k.width=O,k.height=U,e.stateful&&pe(e.id),Ue=!1,gt=performance.now(),requestAnimationFrame(St),y&&(y.disabled=!1);const We=be?be.message:null;S&&(xe?S.textContent="Context lost — partial video saved.":We?S.textContent=`Error: ${We}`:S.textContent="Done!"),setTimeout(()=>{g==null||g.classList.add("hidden"),S==null||S.classList.add("hidden"),w&&(w.style.width="0%")},xe||!!We?8e3:3e3)}Le.addEventListener("click",()=>{document.body.classList.toggle("sidebar-collapsed"),tn()});yt.addEventListener("mousemove",()=>{Ki()});document.addEventListener("keydown",uo);document.addEventListener("visibilitychange",()=>{document.hidden||(gt=performance.now())});var Et;(Et=document.getElementById("rec-btn"))==null||Et.addEventListener("click",Zi);var Mt;(Mt=document.getElementById("offline-btn"))==null||Mt.addEventListener("click",()=>{if(Ue)return;const e=document.getElementById("offline-duration"),t=document.getElementById("offline-fps"),n=document.getElementById("offline-res"),o=Math.max(1,Math.min(3600,Number((e==null?void 0:e.value)??10))),a=Number((t==null?void 0:t.value)??60),[r,l]=((n==null?void 0:n.value)??"1920x1080").split("x").map(Number);co(W,o,a,r,l)});ao();on(W.id);tn();at(!0);requestAnimationFrame(St);
