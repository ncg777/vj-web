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
uniform float uFlowGain;
uniform float uFlowThreshold;
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
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
    float t = uTime * uSpeed + seed * 2.0;
    vec2 c1 = 0.5 + 0.34 * vec2(sin(t * 1.2 + seed), cos(t * 1.6 + seed * 1.3));
    vec2 c2 = 0.5 + 0.30 * vec2(sin(t * 0.8 + seed * 2.1), cos(t * 1.1 + seed * 0.7));
    float sigma = max(1e-4, uBlobRadius);
    float g1 = exp(-distance(uv, c1) * distance(uv, c1) / (sigma * sigma));
    float g2 = exp(-distance(uv, c2) * distance(uv, c2) / (sigma * sigma));
    x += uBlobAmp * (g1 + 0.8 * g2);

    x = clamp(x, 0.0, 1.0);
    outColor = vec4(x, 0.0, 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  vec2 texel = 1.0 / uGridSize;
  float xL = texture(uState, uv - vec2(texel.x, 0.0)).r;
  float xR = texture(uState, uv + vec2(texel.x, 0.0)).r;
  float xD = texture(uState, uv - vec2(0.0, texel.y)).r;
  float xU = texture(uState, uv + vec2(0.0, texel.y)).r;

  vec2 grad = vec2(xR - xL, xU - xD);
  float mag = length(grad) * uFlowGain;
  float threshold = max(0.0, uFlowThreshold);
  float edge = smoothstep(threshold, threshold + 0.05, mag);

  float hue = (atan(grad.y, grad.x) + 3.14159265) / 6.2831853;
  float value = clamp(mag, 0.0, 1.0) * edge;
  vec3 rgb = hsv2rgb(vec3(hue, 0.9, value));
  outColor = vec4(rgb, 1.0);
}
