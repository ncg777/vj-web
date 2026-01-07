#version 300 es
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
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec2 lissa(float t, vec2 freq, vec2 phase, float radius) {
  return 0.5 + radius * vec2(sin(t * freq.x + phase.x), cos(t * freq.y + phase.y));
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
    float t = uTime * uSpeed;
    vec2 freq1 = vec2(mix(0.6, 1.4, hash11(seed + 1.0)), mix(0.6, 1.7, hash11(seed + 2.0)));
    vec2 freq2 = vec2(mix(0.7, 1.6, hash11(seed + 3.0)), mix(0.8, 1.9, hash11(seed + 4.0)));
    vec2 phase1 = vec2(hash11(seed + 5.0), hash11(seed + 6.0)) * 6.2831853;
    vec2 phase2 = vec2(hash11(seed + 7.0), hash11(seed + 8.0)) * 6.2831853;

    vec2 c1 = lissa(t, freq1, phase1, 0.35);
    vec2 c2 = lissa(t, freq2, phase2, 0.33);
    float sigma = max(1e-4, uBlobRadius);
    float d1 = distance(uv, c1);
    float d2 = distance(uv, c2);
    float g1 = exp(-d1 * d1 / (sigma * sigma));
    float g2 = exp(-d2 * d2 / (sigma * sigma));
    x += uBlobAmp * (g1 + 0.8 * g2);

    x = clamp(x, 0.0, 1.0);
    outColor = vec4(x, 0.0, 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  float x = texture(uState, uv).r;
  float tone = 1.0 - exp(-x * 1.6);
  outColor = vec4(vec3(tone), 1.0);
}
