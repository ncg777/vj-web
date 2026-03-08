#version 300 es
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

vec3 Bolt(vec2 uv, float len, float ind, float eventSeed) {
  vec2 travel = vec2(0.0, eventSeed * uBoltNoiseSpeed * 3.0);

  float sn = SimpleNoise(
    uv * uBoltNoiseScale - travel + vec2(ind * 1.5 + uSeed * 0.01 + eventSeed * 7.3, 0.0),
    uNoiseOctaves
  ) * 2.0 - 1.0;
  uv.x += sn * uBoltWiggle * smoothstep(0.0, 0.2, abs(uv.y));

  vec3 l = vec3(LineSDF(uv, vec2(0.0), vec2(0.0, len), uBoltWidth));
  l = uBoltIntensity / max(vec3(0.0), l) * uColorSecondary;
  l = clamp(1.0 - exp(l * -0.02), 0.0, 1.0) * smoothstep(len - 0.01, 0.0, abs(uv.y));
  vec3 bolt = l;

  uv = Rotate(TAU * uTwist) * uv;
  sn = SimpleNoise(
    uv * (uBoltNoiseScale * 1.25) - travel * 1.2 + vec2(ind * 2.3 + uSeed * 0.03 + eventSeed * 5.1, 0.0),
    uNoiseOctaves
  ) * 2.0 - 1.0;
  uv.x += sn * uv.y * uBoltSecondaryScale * smoothstep(0.1, 0.25, len);
  len *= 0.5;

  l = vec3(LineSDF(uv, vec2(0.0), vec2(0.0, len), uBoltWidth * 0.8));
  l = uBoltIntensity * 0.7 / max(vec3(0.0), l) * uColorAccent;
  l = clamp(1.0 - exp(l * -0.03), 0.0, 1.0) * smoothstep(len * 0.7, 0.0, abs(uv.y));
  bolt += l;

  return bolt;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
  uv *= uZoom;

  float aspect = uResolution.x / uResolution.y;
  float time = uTime * uTimeScale;
  vec3 col = vec3(0.0);

  int count = max(uBoltCount, 1);
  for (int i = 0; i < 12; i++) {
    if (i >= count) {
      break;
    }
    float fi = float(i);

    // Per-lane timing for erratic spawning
    float laneRate = uFlickerSpeed * (0.7 + 0.6 * RandomFloat(vec2(fi + uSeed, 7.3)));
    float lanePhase = RandomFloat(vec2(fi + uSeed * 0.5, 13.1)) * 20.0;
    float laneTime = time * laneRate + lanePhase;
    float eventId = floor(laneTime);
    float localT = fract(laneTime);

    // Erratic spawn: not every event produces a flash
    float spawnChance = RandomFloat(vec2(eventId + fi * 3.7, 17.0 + uSeed));
    if (spawnChance < 0.3) continue;

    // Flash envelope: quick attack, smooth decay
    float flash = smoothstep(0.0, 0.05, localT) * (1.0 - smoothstep(0.1, 0.6, localT));

    // Random anchor position seeded by event (spans entire screen)
    vec2 anchor = vec2(
      RandomFloat(vec2(eventId + fi, 21.0 + uSeed)) * 2.0 - 1.0,
      RandomFloat(vec2(eventId + fi, 25.0 + uSeed)) * 2.0 - 1.0
    ) * vec2(aspect * 0.5, 0.5);

    // Random angle seeded by event
    float angle = RandomFloat(vec2(eventId + fi, 29.0 + uSeed)) * TAU;
    angle += (RandomFloat(vec2(eventId + fi, 33.0 + uSeed)) - 0.5) * uAngleJitter;

    // Random length seeded by event
    float len = mix(
      uBoltLengthMin,
      uBoltLengthMax,
      RandomFloat(vec2(eventId + fi, 37.0 + uSeed))
    );

    // Transform UV relative to anchor and angle
    vec2 localUV = Rotate(angle) * (uv - anchor);

    // Event seed for shape variation (shape changes every flash)
    float eventSeed = RandomFloat(vec2(eventId, fi + 41.0 + uSeed));

    col += Bolt(localUV, len, fi, eventSeed) * flash;
  }

  outColor = vec4(col, 1.0);
}
