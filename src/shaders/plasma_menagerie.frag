#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uPlasmaScale;
uniform float uWarp;
uniform int uFold;
uniform float uMorph;
uniform int uShapeCount;
uniform float uOrbit;
uniform float uFilament;
uniform float uGlow;
uniform float uColorCycle;
uniform float uSeed;

const float TAU = 6.28318530718;

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

float hash11(float value) {
  return fract(sin(value * 127.1 + uSeed * 0.000017) * 43758.5453123);
}

float hash21(vec2 point) {
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33 + fract(uSeed * 0.000001));
  return fract((p.x + p.y) * p.z);
}

float noise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  local = local * local * (3.0 - 2.0 * local);
  float a = hash21(cell);
  float b = hash21(cell + vec2(1.0, 0.0));
  float c = hash21(cell + vec2(0.0, 1.0));
  float d = hash21(cell + vec2(1.0));
  return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
}

float fbm(vec2 point) {
  float value = 0.0;
  float amplitude = 0.5;
  mat2 turn = rotate2d(0.57);
  for (int octave = 0; octave < 5; octave++) {
    value += amplitude * noise(point);
    point = turn * point * 2.03 + vec2(1.7, -2.4);
    amplitude *= 0.5;
  }
  return value;
}

vec2 kaleidoscope(vec2 point, float folds) {
  float radius = length(point);
  float sector = TAU / max(1.0, floor(folds + 0.5));
  float angle = abs(mod(atan(point.y, point.x) + sector * 0.5, sector) - sector * 0.5);
  return radius * vec2(cos(angle), sin(angle));
}

vec3 plasmaPalette(float phase) {
  vec3 base = vec3(0.50, 0.48, 0.52);
  vec3 amplitude = vec3(0.50, 0.46, 0.48);
  vec3 frequency = vec3(1.0, 0.72, 0.47);
  vec3 offset = vec3(0.02, 0.24, 0.58);
  return base + amplitude * cos(TAU * (frequency * phase + offset));
}

void main() {
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  point *= uZoom;
  float time = uTime * uTimeScale;

  vec2 folded = kaleidoscope(point, float(uFold));
  vec2 noiseDrift = vec2(time * 0.08, -time * 0.055);
  vec2 warpField = vec2(
    fbm(folded * 1.35 + noiseDrift),
    fbm(folded * 1.35 + vec2(5.2, -3.7) - noiseDrift.yx)
  ) - 0.5;
  vec2 plasmaPoint = mix(point, folded, 0.28) + warpField * uWarp;

  float plasma = sin(plasmaPoint.x * uPlasmaScale + time * 1.3);
  plasma += sin(plasmaPoint.y * uPlasmaScale * 1.17 - time * 1.05);
  plasma += sin((plasmaPoint.x + plasmaPoint.y) * uPlasmaScale * 0.73 + time * 0.61);
  plasma += sin(length(plasmaPoint + warpField) * uPlasmaScale * 1.85 - time * 1.7);
  plasma *= 0.25;

  float filamentFrequency = mix(3.0, 12.0, clamp(uFilament, 0.0, 1.0));
  float filaments = pow(1.0 - abs(sin(plasma * filamentFrequency)), 5.0);
  float cloud = smoothstep(-0.7, 0.9, plasma + (fbm(plasmaPoint * 2.1 - time * 0.04) - 0.5));

  float shapeBody = 0.0;
  float shapeEdge = 0.0;
  float eyes = 0.0;
  float tendrils = 0.0;
  float shapePhase = 0.0;

  for (int index = 0; index < 8; index++) {
    if (index >= uShapeCount) {
      break;
    }

    float fi = float(index);
    float randomA = hash11(fi + 2.0);
    float randomB = hash11(fi + 19.0);
    float phase = randomA * TAU;
    float speed = 0.23 + randomB * 0.24;
    vec2 center = vec2(
      sin(time * speed + phase),
      cos(time * speed * 0.73 + phase * 1.61)
    ) * uOrbit * vec2(0.72, 0.46);

    vec2 creature = point - center;
    creature = rotate2d(-phase * 0.35 - sin(time * 0.19 + phase) * 0.8) * creature;
    creature += uMorph * 0.045 * vec2(
      sin(creature.y * 9.0 + time * 1.2 + phase),
      cos(creature.x * 8.0 - time * 0.9 + phase)
    );

    float angle = atan(creature.y, creature.x);
    float radius = length(creature);
    float lobes = 3.0 + floor(randomA * 5.0);
    float membrane = 0.13 + randomB * 0.075;
    membrane += uMorph * 0.035 * sin(angle * lobes + time * (0.8 + randomA) + phase);
    membrane += uMorph * 0.018 * sin(angle * (lobes + 3.0) - time * 1.6);
    float signedShape = radius - membrane;
    float antialias = max(fwidth(signedShape), 0.001);
    float body = 1.0 - smoothstep(-antialias, antialias, signedShape);
    float edge = exp(-max(abs(signedShape) - antialias, 0.0) * mix(70.0, 24.0, uGlow));

    vec2 eyePoint = creature;
    eyePoint.x += membrane * 0.15;
    float eyeDistance = length(eyePoint * vec2(0.8, 1.9));
    float eyeRing = exp(-abs(eyeDistance - membrane * 0.34) * 95.0) * body;
    float pupil = exp(-eyeDistance * 75.0) * body;

    float tailGate = smoothstep(0.06, membrane * 1.4, creature.x)
      * (1.0 - smoothstep(membrane * 1.4, membrane * 4.0, creature.x));
    float tailPath = abs(
      creature.y - sin(creature.x * (12.0 + randomA * 8.0) - time * 2.0 + phase)
      * (0.025 + uMorph * 0.035)
    );
    float tail = exp(-tailPath * 100.0) * tailGate;

    shapeBody = max(shapeBody, body * (0.35 + 0.65 * sin(angle * lobes + phase) * 0.5 + 0.325));
    shapeEdge += edge;
    eyes += eyeRing + pupil * 1.8;
    tendrils += tail;
    shapePhase += (edge + eyeRing) * (randomA - 0.5);
  }

  float pulse = 0.82 + 0.18 * sin(time * 3.0 + plasma * 4.0);
  vec3 background = vec3(0.004, 0.006, 0.018);
  vec3 plasmaColor = plasmaPalette(plasma * 0.26 + time * uColorCycle * 0.08);
  vec3 creatureColor = plasmaPalette(0.62 + shapePhase * 0.16 - time * uColorCycle * 0.055);
  vec3 eyeColor = plasmaPalette(0.12 - time * 0.025);

  vec3 color = background;
  color += plasmaColor * (filaments * 0.72 + cloud * 0.12);
  color += creatureColor * shapeBody * 0.22;
  color += creatureColor * shapeEdge * uGlow * 1.35 * pulse;
  color += eyeColor * eyes * uGlow * 1.8;
  color += plasmaPalette(0.88 + plasma * 0.1) * tendrils * uGlow;

  float bloom = filaments * shapeEdge + eyes * 0.45 + tendrils * 0.35;
  color += vec3(0.55, 0.72, 1.0) * bloom * uGlow * 0.45;
  color *= 1.0 - 0.28 * smoothstep(0.45, 1.45, length(point));
  color = 1.0 - exp(-color * (0.9 + uGlow * 0.5));
  color = pow(max(color, 0.0), vec3(0.82));

  outColor = vec4(color, 1.0);
}