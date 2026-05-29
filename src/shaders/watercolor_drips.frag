#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;

const float TAU = 6.28318530718;
const int DRIP_COUNT = 18;

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

void main() {
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  vec2 aspectUv = uv;
  aspectUv.x *= uResolution.x / uResolution.y;

  float grain = fbm(uv * vec2(18.0, 24.0));
  float pulp = fbm(uv * vec2(7.0, 10.0) + vec2(3.1, 8.7));
  float fibers = fbm(vec2(uv.x * 2.5 + pulp * 0.4, uv.y * 55.0));
  float flecks = fbm(uv * vec2(90.0, 115.0) + vec2(12.0, 4.0));
  float blotches = fbm(uv * 4.0 + vec2(0.0, 7.3));
  float vignette = smoothstep(1.15, 0.15, length((uv - 0.5) * vec2(1.1, 0.95)));

  vec3 paper = vec3(0.955, 0.925, 0.875);
  paper += 0.08 * (grain - 0.5);
  paper += vec3(0.035, 0.032, 0.026) * smoothstep(0.52, 0.9, fibers);
  paper += 0.03 * (pulp - 0.5);
  paper -= 0.018 * smoothstep(0.7, 0.92, flecks);
  paper -= 0.06 * smoothstep(0.45, 0.95, blotches);
  paper *= 0.93 + 0.07 * vignette;

  vec3 color = paper;

  for (int i = 0; i < DRIP_COUNT; i++) {
    float fi = float(i) + 1.0;
    float lane = mix(0.08, 0.92, hash11(fi * 7.13));
    float cycle = mix(16.0, 30.0, hash11(fi * 2.71));
    float phase = fract(uTime / cycle + hash11(fi * 11.9));
    float dripActive = smoothstep(0.01, 0.08, phase) * (1.0 - smoothstep(0.8, 0.98, phase));
    float travel = pow(phase, mix(1.15, 1.55, hash11(fi * 5.37)));
    float headY = mix(-0.24, 1.16, travel);
    float radius = mix(0.018, 0.04, hash11(fi * 17.2));
    float trailLength = mix(0.14, 0.42, hash11(fi * 19.4)) * smoothstep(0.03, 0.35, phase);
    float sway = (hash11(fi * 23.1) - 0.5) * 0.05;
    float staticSpine =
      lane +
      sway * sin(uv.y * 3.4 + fi * 1.9) +
      0.012 * sin(uv.y * 10.0 + fi * 4.1 + hash11(fi * 29.3) * TAU);
    float headSpine =
      lane +
      sway * sin(headY * 3.4 + fi * 1.9) +
      0.012 * sin(headY * 10.0 + fi * 4.1 + uTime * mix(0.1, 0.24, hash11(fi * 29.3)));

    float dx = abs(uv.x - staticSpine);
    float headDx = abs(uv.x - headSpine);
    float trailWidth = radius * mix(0.35, 0.65, hash11(fi * 31.7));
    float driedCore = smoothstep(trailWidth * 2.8, trailWidth * 0.55, dx);
    float driedMask =
      driedCore *
      smoothstep(-0.05, 0.05, uv.y) *
      (1.0 - smoothstep(0.96, 1.12, uv.y)) *
      mix(0.5, 1.0, fbm(vec2(fi * 0.71, uv.y * 6.0 + dx * 14.0)));
    float trailCore = smoothstep(trailWidth * 1.9, trailWidth * 0.48, headDx);
    float trailMask =
      trailCore *
      smoothstep(headY - trailLength - 0.05, headY - trailLength + 0.02, uv.y) *
      (1.0 - smoothstep(headY - radius * 0.4, headY + radius * 0.35, uv.y)) *
      dripActive;

    vec2 headOffset = vec2((uv.x - headSpine) / (radius * 0.9), (uv.y - headY) / (radius * 1.2));
    float headMask = 1.0 - smoothstep(0.7, 1.25, length(headOffset));

    float bleed =
      exp(-pow(dx / (radius * 2.8), 2.0)) *
      smoothstep(-0.05, headY + radius * 0.8, uv.y) *
      (0.35 + 0.65 * fbm(vec2(fi * 0.7, uv.y * 7.0)));

    float satellites = 0.0;
    for (int j = 0; j < 2; j++) {
      float fj = float(j) + 1.0;
      float lag = fract(phase + 0.23 * fj + hash11(fi * 41.3 + fj));
      float satY = headY - trailLength * (0.3 + 0.45 * fj) - lag * 0.08;
      float satX = headSpine + (hash11(fi * 43.7 + fj) - 0.5) * trailWidth * 4.0;
      float satR = radius * mix(0.18, 0.32, hash11(fi * 47.1 + fj));
      vec2 satOffset = vec2((uv.x - satX) / satR, (uv.y - satY) / (satR * 1.2));
      satellites += (1.0 - smoothstep(0.8, 1.4, length(satOffset))) * dripActive;
    }

    float dryMask = clamp(driedMask * 0.95 + bleed * 0.16, 0.0, 1.0);
    float wetMask = clamp(headMask + trailMask * 0.95 + bleed * 0.24 + satellites * 0.35, 0.0, 1.0);
    wetMask *= dripActive;

    float hue = fract(hash11(fi * 13.7) * 0.9 + 0.08 * sin(fi * 1.7));
    float saturation = mix(0.45, 0.78, hash11(fi * 53.9));
    float value = mix(0.56, 0.84, hash11(fi * 59.2));
    vec3 pigment = hsv2rgb(vec3(hue, saturation, value));
    pigment = mix(pigment, vec3(0.98, 0.96, 0.93), 0.18);

    float pooling = clamp(headMask * 0.85 + trailMask * 0.35 + dryMask * 0.2, 0.0, 1.0);
    float absorb = 0.55 + 0.45 * grain;
    vec3 dryTint = mix(pigment, paper, 0.48 + 0.18 * blotches);
    vec3 edgeTint = mix(pigment, paper, 0.28 + 0.24 * blotches);
    vec3 dryWash = mix(dryTint, pigment * 0.72, clamp(dryMask * 0.65 + bleed * 0.1, 0.0, 1.0));
    vec3 wash = mix(edgeTint, pigment * 0.9, pooling);
    color = mix(color, dryWash, dryMask * absorb * 0.5);
    color = mix(color, wash, wetMask * absorb * 0.78);
  }

  color *= 0.98 + 0.02 * vignette;
  outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
