#version 300 es
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

float hash1(float n) {
  return fract(sin(n * 127.1) * 43758.5453123);
}

// Smooth value noise (C1 continuous, no seams).
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

const mat2 ROT = mat2(0.8, 0.6, -0.6, 0.8);

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p = ROT * p * 2.03 + vec2(11.7, 5.1);
    a *= 0.5;
  }
  return v;
}

vec3 palette(float t) {
  return uColorPrimary + uColorSecondary * cos(TAU * (uColorAccent * t + uPaletteShift));
}

// Smallest signed periodic angular difference, continuous everywhere.
float angDiff(float a, float b) {
  return sin((a - b) * 0.5);
}

void main() {
  vec2 uv = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;
  uv *= uZoom;

  float t = uTime * uTimeScale;
  float seed = uSeed * 0.013;

  float spin = t * uSpin * 0.35;
  mat2 rot = mat2(cos(spin), -sin(spin), sin(spin), cos(spin));
  uv = rot * uv;

  // Gentle continuous swirl instead of a hard warp.
  float swirl = uTwist * 0.35 * exp(-dot(uv, uv) * 0.6);
  float sa = swirl * sin(t * 0.4);
  uv = mat2(cos(sa), -sin(sa), sin(sa), cos(sa)) * uv;

  float radius = length(uv) + 1e-4;
  float angle = atan(uv.y, uv.x);

  // ------------------------------------------------------------------
  // Nebula: domain-warped fbm gas clouds diffusing in every direction.
  // ------------------------------------------------------------------
  vec2 p = uv * 1.6 + seed;
  vec2 q = vec2(
    fbm(p + vec2(0.0, 0.0) + t * 0.10),
    fbm(p + vec2(5.2, 1.3) - t * 0.08)
  );
  vec2 r2 = vec2(
    fbm(p + (1.2 + uWarp) * q + vec2(1.7, 9.2) + t * 0.15),
    fbm(p + (1.2 + uWarp) * q + vec2(8.3, 2.8) - t * 0.12)
  );
  float f = fbm(p + (1.5 + uWarp) * r2);

  // High-contrast shaping: deep space stays dark, filaments pop.
  float density = smoothstep(0.32, 0.78, f);
  density *= density;

  vec3 gasA = palette(f * 1.4 + radius * 0.25 + t * 0.03);
  vec3 gasB = palette(length(q) * 0.9 + 0.33 + t * 0.02);
  vec3 gasC = palette(length(r2) * 0.7 + 0.66);
  vec3 nebula = mix(gasA, gasB, clamp(length(q) * 1.2 - 0.2, 0.0, 1.0));
  nebula = mix(nebula, gasC, clamp(r2.x * r2.x * 1.4 - 0.15, 0.0, 1.0));
  nebula = max(nebula, 0.0) * density;

  // Sparse starfield twinkle riding on the gas.
  float sparkle = fbm(uv * 9.0 - t * 0.5 + seed);
  nebula += uNoiseAmp * 0.7 * palette(sparkle) * pow(max(sparkle - 0.62, 0.0) * 2.6, 2.0);

  // ------------------------------------------------------------------
  // Lightning: jagged bolts bursting from the core in all directions.
  // ------------------------------------------------------------------
  float bolts = 0.0;
  float boltCount = clamp(uArcSteps * 0.3, 2.0, 24.0);
  float wiggleFreq = 1.5 + uBoltDensity * 0.45;
  for (int i = 0; i < 24; i++) {
    float fi = float(i);
    float mask = step(fi + 0.5, boltCount);
    if (mask < 0.5) continue;

    float hs = fi * 7.31 + seed;
    // Stratified around the full circle so bolts burst in all directions,
    // each ray slowly roaming around its slot.
    float baseAng = (fi + 0.5) / boltCount * TAU + (hash1(hs) - 0.5) * 1.2
      + t * (0.15 + 0.35 * hash1(hs + 3.7)) * (hash1(hs + 9.1) > 0.5 ? 1.0 : -1.0)
      + sin(t * (0.6 + uPulse * 0.3) + fi * 1.7) * 0.25;

    // Jagged path: the ray's angle wiggles with radius (continuous noise).
    float wig = noise(vec2(radius * wiggleFreq - t * (1.5 + uPulse), hs)) - 0.5;
    float wig2 = noise(vec2(radius * wiggleFreq * 2.7 + t * 2.0, hs + 40.0)) - 0.5;
    float ang = baseAng + (wig * 0.9 + wig2 * 0.45) * (0.35 + 0.4 * uWarp) / (0.25 + radius);

    // Perpendicular chord distance to the bolt (screen space, seam-free).
    float d = abs(angDiff(angle, ang)) * 2.0 * radius;
    float w = 0.014 / (0.35 + uBoltSharpness);
    float ray = exp(-pow(d / w, 2.0));
    float glow = exp(-pow(d / (w * 10.0), 2.0)) * 0.12;

    // Bursting strobe: bolts flare violently then fade.
    float burst = 0.5 + 0.5 * sin(t * (2.0 + uPulse * 2.5) + fi * 2.71 + hash1(hs + 1.3) * TAU);
    burst = pow(burst, 5.0);
    burst = 0.12 + 0.88 * burst;

    // Finite reach with a soft, continuous tip; fade near the core where
    // all rays would otherwise overlap into a white blob.
    float len = 0.45 + hash1(hs + 5.5) * 1.5;
    float reach = smoothstep(len, len * 0.25, radius) * smoothstep(0.02, 0.3, radius);

    bolts += (ray + glow) * burst * reach;
  }
  bolts *= uBoltIntensity * 2.2;

  // ------------------------------------------------------------------
  // Core: a hot plasma heart the bolts erupt from.
  // ------------------------------------------------------------------
  float coreR = max(uCoreSize * 0.5, 0.04);
  float core = uCoreGlow * 0.9 * exp(-pow(radius / coreR, 2.0) * 2.0);
  core += uCoreGlow * 0.25 * exp(-pow(radius / (coreR * 2.5), 2.0))
    * (0.7 + 0.3 * sin(t * (3.0 + uPulse * 2.0)));

  // ------------------------------------------------------------------
  // Composite.
  // ------------------------------------------------------------------
  float vignette = smoothstep(2.4, 0.4, radius);

  vec3 boltColor = mix(palette(0.1 + t * 0.02), vec3(1.0), 0.55);
  vec3 col = nebula * (0.55 + 0.45 * vignette);
  col += boltColor * bolts * (0.35 + 0.65 * vignette);
  col += mix(palette(0.05), vec3(1.0), 0.4) * core;
  // Bolts electrify the gas they pass through.
  col += nebula * bolts * 0.8;

  // Filmic-ish exposure keeps highlights crisp and blacks deep.
  col = 1.0 - exp(-col * 1.35);
  col = pow(max(col, 0.0), vec3(1.25));
  outColor = vec4(col, 1.0);
}
