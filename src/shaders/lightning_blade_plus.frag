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

float BoltMask(vec2 p, vec2 a, vec2 b, float width, float intensity) {
  float dist = max(LineSDF(p, a, b, width), 1e-4);
  return clamp(1.0 - exp(-(intensity / dist) * 0.03), 0.0, 1.0);
}

vec3 FlashBolt(vec2 uv, float lane, float eventId, float localT) {
  float flashSeed = lane * 19.73 + eventId * 31.17 + uSeed * 0.11;
  vec2 span = vec2(uResolution.x / uResolution.y, 1.0) * 1.35;
  vec2 anchor = vec2(
    RandomFloat(vec2(flashSeed, 1.7)),
    RandomFloat(vec2(flashSeed, 6.1))
  ) * 2.0 - 1.0;
  anchor *= span;

  float angle = RandomFloat(vec2(flashSeed, 9.9)) * TAU;
  angle += (RandomFloat(vec2(flashSeed, 11.3)) - 0.5) * uAngleJitter;
  vec2 tangent = vec2(cos(angle), sin(angle));
  vec2 normal = vec2(-tangent.y, tangent.x);

  float len = mix(
    uBoltLengthMin,
    uBoltLengthMax,
    RandomFloat(vec2(flashSeed, 14.7))
  ) * 6.0 + 0.8;
  len *= mix(0.85, 1.35, RandomFloat(vec2(flashSeed, 18.2)));

  vec2 delta = uv - anchor;
  vec2 local = vec2(dot(delta, tangent), dot(delta, normal));
  float sweep = clamp(local.x / max(len, 1e-4), -0.5, 0.5);

  vec2 noisePos = vec2(
    (sweep + 0.5) * uBoltNoiseScale * 5.0 + flashSeed * 0.23,
    localT * (2.0 + uBoltNoiseSpeed * 1.3) + flashSeed * 0.07
  );
  float bend = SimpleNoise(noisePos, uNoiseOctaves) * 2.0 - 1.0;
  float detail = SimpleNoise(noisePos * vec2(1.8, 1.3) + vec2(5.2, 11.7), uNoiseOctaves) * 2.0 - 1.0;
  float bodyShape = mix(bend, detail, 0.45);
  float taper = 1.0 - smoothstep(0.15, 0.55, abs(sweep));
  local.y += bodyShape * (uBoltWiggle + uCoreNoiseAmp * 2.5) * (0.35 + 0.65 * taper);

  float width = uBoltWidth * mix(1.4, 3.0, RandomFloat(vec2(flashSeed, 21.6))) + 0.00015;
  float mainBolt = BoltMask(
    local,
    vec2(-0.5 * len, 0.0),
    vec2(0.5 * len, 0.0),
    width,
    uBoltIntensity * 1.7
  );

  float branchPos = mix(-0.28, 0.22, RandomFloat(vec2(flashSeed, 24.1))) * len;
  vec2 branchOrigin = vec2(branchPos, bodyShape * (uBoltWiggle * 0.7 + 0.01));
  vec2 branchDir = normalize(vec2(
    0.35 + RandomFloat(vec2(flashSeed, 27.4)) * 0.65,
    mix(-1.0, 1.0, RandomFloat(vec2(flashSeed, 28.8))) + uTwist * 2.0
  ));
  float branchLen = len * mix(0.15, 0.42, RandomFloat(vec2(flashSeed, 30.5))) * (0.5 + 0.5 * uBoltSecondaryScale);
  float branchBolt = BoltMask(
    local,
    branchOrigin,
    branchOrigin + branchDir * branchLen,
    width * 0.8,
    uBoltIntensity * (0.7 + 0.6 * uBoltSecondaryScale)
  );

  float hotspotPos = mix(-0.3, 0.3, RandomFloat(vec2(flashSeed, 33.2))) * len;
  vec2 hotspot = anchor + tangent * hotspotPos;
  float sparkRadius = max(0.012, uCoreRadius * (5.0 + 6.0 * RandomFloat(vec2(flashSeed, 36.1))));
  float spark = clamp(1.0 - exp(-(uCoreIntensity / max(CircleSDF(uv - hotspot, sparkRadius), 1e-4)) * 0.03), 0.0, 1.0);

  float attack = smoothstep(0.01, 0.12, localT);
  float sustain = 1.0 - smoothstep(0.16, 0.55, localT);
  float crackle = 0.65 + 0.35 * sin(localT * TAU * (3.0 + RandomFloat(vec2(flashSeed, 39.4)) * 4.0));
  float energy = attack * sustain * crackle;

  vec3 col = vec3(0.0);
  col += mainBolt * (uColorSecondary + vec3(0.28));
  col += branchBolt * (uColorAccent + vec3(0.16));
  col += spark * (uColorPrimary + vec3(0.25));

  return col * energy;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y;
  uv *= uZoom;

  float time = uTime * uTimeScale;
  float cadence = 0.35 + max(uFlickerSpeed, 0.05) * 0.55;
  vec3 col = vec3(0.0);

  float ambient = SimpleNoise(
    uv * (uCoreNoiseScale * 0.18) + vec2(time * 0.07, -time * 0.05),
    uNoiseOctaves
  );
  col += (0.015 + ambient * 0.02) * uColorPrimary;

  int count = max(uBoltCount, 1);
  for (int i = 0; i < 12; i++) {
    if (i >= count) {
      break;
    }
    float lane = float(i);
    float laneRate = mix(0.45, 1.45, RandomFloat(vec2(lane + uSeed, 41.0)));
    float laneOffset = RandomFloat(vec2(lane + uSeed * 0.3, 44.0)) * 20.0;
    float laneTime = time * cadence * laneRate + laneOffset;
    float eventId = floor(laneTime);
    float localT = fract(laneTime);
    float spawn = step(0.25, RandomFloat(vec2(eventId + lane * 3.7, 47.0 + uSeed)));
    col += FlashBolt(uv, lane, eventId, localT) * spawn;
  }

  col = 1.0 - exp(-col * 1.2);
  outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
