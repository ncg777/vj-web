#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;

uniform int uLoopCount;
uniform float uCycleMin;
uniform float uCycleMax;
uniform float uRadiusMin;
uniform float uRadiusMax;
uniform float uCenterDrift;
uniform float uCenterDriftSpeed;
uniform int uHarmonicCount;
uniform float uWobbleAmount;
uniform float uWobbleFalloff;
uniform float uMorphSpeed;
uniform float uVibrationAmount;
uniform int uVibrationFreq;
uniform float uVibrationSpeed;
uniform float uStrokeWidthMin;
uniform float uStrokeWidthMax;
uniform float uWidthMod;
uniform float uInkTexture;
uniform float uDrawSoftness;
uniform float uBleedStrength;
uniform float uBleedSpread;
uniform float uSoakStrength;
uniform float uDryMix;
uniform float uWetMix;
uniform float uAbsorbStrength;
uniform float uHueShift;
uniform float uSaturationBoost;
uniform float uValueBoost;
uniform float uPastelMix;
uniform float uPaperGrainScale;
uniform float uPaperGrainAmount;
uniform float uPaperPulpAmount;
uniform float uPaperRingDensity;
uniform float uPaperRingWobble;
uniform float uPaperRingAmount;
uniform float uPaperFleckAmount;
uniform float uPaperBlotchAmount;
uniform float uVignetteStrength;

const float TAU = 6.28318530718;
const float EPSILON = 0.001;
const int MAX_LOOP_COUNT = 24;
const int MAX_HARMONICS = 8;

float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

float hash21(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

mat2 rot(float a) {
  float s = sin(a);
  float c = cos(a);
  return mat2(c, -s, s, c);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);

  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));

  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
  float value = 0.0;
  float amplitude = 0.5;
  for (int i = 0; i < 5; i++) {
    value += amplitude * noise(p);
    p = rot(0.35) * p * 2.02 + vec2(11.7, 4.3);
    amplitude *= 0.52;
  }
  return value;
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

// Smooth wobbly closed-loop radius built from integer harmonics of the
// polar angle. Because every term is a whole multiple of theta the curve
// always closes perfectly and stays C-infinity smooth, while stacked
// harmonics plus a high-frequency vibration term keep it rich in detail.
float loopRadius(float theta, float fi, float time) {
  float r = 1.0;
  int harmonics = clamp(uHarmonicCount, 0, MAX_HARMONICS);
  for (int k = 0; k < MAX_HARMONICS; k++) {
    if (k >= harmonics) {
      break;
    }
    float n = float(k + 2);
    float amp = uWobbleAmount * (hash11(fi * 7.77 + n * 13.13) - 0.5) * 2.0 / pow(n, max(uWobbleFalloff, 0.1));
    float phase = hash11(fi * 3.33 + n * 21.7) * TAU;
    float drift = time * uMorphSpeed * (0.3 + hash11(fi * 5.11 + n * 9.19));
    r += amp * sin(n * theta + phase + drift);
  }
  int vibFreq = max(uVibrationFreq, 1);
  float vibPhase = hash11(fi * 91.3) * TAU;
  r += uVibrationAmount * sin(float(vibFreq) * theta + vibPhase + time * uVibrationSpeed);
  r += uVibrationAmount * 0.45 * sin(float(vibFreq * 2 + 1) * theta - vibPhase * 1.7 + time * uVibrationSpeed * 1.31);
  return r;
}

// Wobbly closed-ring paper texture: concentric harmonic loops around a
// handful of scattered centers replace the old straight linear fibers.
float paperRings(vec2 uv, float aspect) {
  float rings = 0.0;
  for (int i = 0; i < 3; i++) {
    float fi = float(i) + 1.0;
    vec2 center = vec2(hash11(fi * 27.7), hash11(fi * 63.1));
    vec2 d = (uv - center) * vec2(aspect, 1.0);
    float theta = atan(d.y, d.x);
    float wobble = uPaperRingWobble * (
      sin(theta * 3.0 + fi * 2.3) * 0.5 +
      sin(theta * 7.0 - fi * 4.1) * 0.3 +
      sin(theta * 13.0 + fi * 7.9) * 0.2
    );
    float band = sin((length(d) * (1.0 + wobble)) * uPaperRingDensity * TAU + fi * 11.0);
    rings += smoothstep(0.55, 0.95, band) / 3.0;
  }
  return rings;
}

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  float aspect = uResolution.x / max(uResolution.y, 1.0);
  float time = uTime * uTimeScale;

  float grain = fbm(uv * vec2(18.0, 24.0) * max(uPaperGrainScale, 0.05));
  float pulp = fbm(uv * vec2(7.0, 10.0) + vec2(3.1, 8.7));
  float rings = paperRings(uv, aspect);
  float flecks = fbm(uv * vec2(90.0, 115.0) + vec2(12.0, 4.0));
  float blotches = fbm(uv * 4.0 + vec2(0.0, 7.3));
  float vignette = smoothstep(1.15, 0.15, length((uv - 0.5) * vec2(1.1, 0.95)));

  vec3 paper = vec3(0.955, 0.925, 0.875);
  paper += uPaperGrainAmount * (grain - 0.5);
  paper += vec3(0.035, 0.032, 0.026) * rings * uPaperRingAmount;
  paper += uPaperPulpAmount * (pulp - 0.5);
  paper -= uPaperFleckAmount * smoothstep(0.7, 0.92, flecks);
  paper -= uPaperBlotchAmount * smoothstep(0.45, 0.95, blotches);
  paper *= 1.0 - uVignetteStrength + uVignetteStrength * vignette;

  vec3 color = paper;

  int loopCount = clamp(uLoopCount, 1, MAX_LOOP_COUNT);

  for (int i = 0; i < MAX_LOOP_COUNT; i++) {
    if (i >= loopCount) {
      break;
    }

    float fi = float(i) + 1.0;
    vec2 center = vec2(
      mix(0.14, 0.86, hash11(fi * 7.13)),
      mix(0.14, 0.86, hash11(fi * 11.9))
    );
    float driftPhase = hash11(fi * 17.3) * TAU;
    center += uCenterDrift * vec2(
      sin(time * uCenterDriftSpeed * (0.5 + hash11(fi * 19.7) * 0.8) + driftPhase),
      cos(time * uCenterDriftSpeed * (0.4 + hash11(fi * 23.1) * 0.9) + driftPhase * 1.9)
    );

    vec2 d = (uv - center) * vec2(aspect, 1.0);
    float dist = length(d);
    float theta = atan(d.y, d.x);

    float baseR = mix(uRadiusMin, uRadiusMax, hash11(fi * 29.7));
    float radius = baseR * loopRadius(theta, fi, time);
    float signedDist = dist - radius;

    // Progressive painting: each stroke is drawn around its loop over a
    // cycle, lingers, then washes out before being repainted.
    float cycle = mix(uCycleMin, uCycleMax, hash11(fi * 2.71));
    float phase = fract(time / max(cycle, EPSILON) + hash11(fi * 41.3));
    float paintOn = smoothstep(0.0, 0.06, phase) * (1.0 - smoothstep(0.86, 0.99, phase));
    float travel = smoothstep(0.02, 0.62, phase);
    float thetaNorm = fract(theta / TAU + hash11(fi * 47.9));
    float soft = max(uDrawSoftness, EPSILON);
    float reveal = smoothstep(travel, travel - soft, thetaNorm);

    float width = baseR * mix(uStrokeWidthMin, uStrokeWidthMax, hash11(fi * 31.7));
    width *= 1.0 + uWidthMod * sin(theta * 3.0 + hash11(fi * 53.3) * TAU + time * uMorphSpeed * 0.4);
    width = max(width, EPSILON);

    float ink = mix(1.0 - uInkTexture, 1.0, fbm(vec2(thetaNorm * 6.0 + fi * 0.7, fi * 1.3 + signedDist * 18.0)));
    float strokeCore = smoothstep(width * 1.5, width * 0.35, abs(signedDist));
    float strokeMask = strokeCore * reveal * paintOn * ink;

    float bleedSpread = max(width * uBleedSpread, EPSILON);
    float bleed =
      exp(-pow(signedDist / bleedSpread, 2.0)) *
      (0.45 + 0.55 * fbm(vec2(fi * 0.7, theta * 2.2 + dist * 9.0))) *
      reveal * paintOn * uBleedStrength;

    float soak =
      smoothstep(0.0, -baseR * 0.6, signedDist) *
      (0.35 + 0.65 * grain + 0.18 * pulp) *
      (0.3 + 0.7 * paintOn) *
      uSoakStrength;

    float dryGhost = strokeCore * ink * 0.6 * (1.0 - paintOn);

    float dryMask = clamp(dryGhost + bleed * 0.18 + soak * 0.2, 0.0, 1.0);
    float wetMask = clamp(strokeMask + bleed * 0.4 + soak * 0.5, 0.0, 1.0);

    float hue = fract(hash11(fi * 13.7) * 0.9 + 0.08 * sin(fi * 1.7) + uHueShift);
    float saturation = clamp(mix(0.45, 0.78, hash11(fi * 53.9)) + uSaturationBoost, 0.0, 1.0);
    float value = clamp(mix(0.56, 0.84, hash11(fi * 59.2)) + uValueBoost, 0.0, 1.0);
    vec3 pigment = hsv2rgb(vec3(hue, saturation, value));
    pigment = mix(pigment, vec3(0.98, 0.96, 0.93), clamp(uPastelMix, 0.0, 1.0));

    float pooling = clamp(strokeMask * 0.55 + soak * 0.5 + dryMask * 0.24, 0.0, 1.0);
    float absorb = (0.72 + 0.6 * grain + 0.18 * pulp) * uAbsorbStrength;
    vec3 dryTint = mix(pigment, paper, 0.54 + 0.16 * blotches + 0.06 * grain);
    vec3 edgeTint = mix(pigment, paper, 0.3 + 0.22 * blotches + 0.1 * grain);
    vec3 seepTint = mix(pigment, paper, 0.62 + 0.16 * grain);
    vec3 dryWash = mix(dryTint, pigment * 0.72, clamp(dryMask * 0.62 + bleed * 0.14, 0.0, 1.0));
    vec3 soakWash = mix(seepTint, pigment * 0.8, clamp(soak * 0.72 + bleed * 0.18, 0.0, 1.0));
    vec3 wash = mix(edgeTint, pigment * 0.9, pooling);
    color = mix(color, dryWash, dryMask * absorb * 0.5 * uDryMix);
    color = mix(color, soakWash, clamp(soak, 0.0, 1.0) * absorb * 0.32 * uWetMix);
    color = mix(color, wash, wetMask * absorb * 0.78 * uWetMix);
  }

  color *= 0.98 + 0.02 * vignette;
  outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
