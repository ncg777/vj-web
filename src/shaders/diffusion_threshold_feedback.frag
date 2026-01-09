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
