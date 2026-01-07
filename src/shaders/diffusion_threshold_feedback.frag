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
uniform float uThreshold;
uniform float uSharpness;
uniform float uNoiseAmp;
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
    float x = diffuseScalar(uv, texel);

    x = sigmoid(uSharpness * (x - uThreshold));
    x *= uDecay;

    float seed = uSeed * 0.001;
    float t = uTime * uSpeed + seed * 4.0;
    vec2 pos = 0.5 + 0.33 * vec2(sin(t * 1.1 + seed), cos(t * 1.4 + seed * 1.9));
    float sigma = max(1e-4, uInjectRadius);
    float g = exp(-distance(uv, pos) * distance(uv, pos) / (sigma * sigma));
    x += uInjectAmp * g;

    float noise = (hash21(uv * uGridSize + uTime * 2.0 + seed) - 0.5) * uNoiseAmp;
    x = clamp(x + noise, 0.0, 1.0);
    outColor = vec4(x, 0.0, 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  float x = texture(uState, uv).r;
  outColor = vec4(vec3(x), 1.0);
}
