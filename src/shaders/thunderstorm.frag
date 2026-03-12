#version 300 es
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
