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
uniform float uDu;
uniform float uDv;
uniform float uFeed;
uniform float uKill;
uniform float uValueGain;
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec2 diffuseUV(vec2 uv, vec2 texel) {
  vec2 c = texture(uState, uv).rg;
  vec2 sum = c * uSelfWeight;
  sum += texture(uState, uv + vec2(texel.x, 0.0)).rg * uNeighborWeight;
  sum += texture(uState, uv - vec2(texel.x, 0.0)).rg * uNeighborWeight;
  sum += texture(uState, uv + vec2(0.0, texel.y)).rg * uNeighborWeight;
  sum += texture(uState, uv - vec2(0.0, texel.y)).rg * uNeighborWeight;
  float norm = max(1e-4, uSelfWeight + 4.0 * uNeighborWeight);
  return sum / norm;
}

void main() {
  if (uPass == 2) {
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    float u = 1.0;
    float v = 0.0;
    float seed = uSeed * 0.001;
    float radius = 0.06;
    for (int i = 0; i < 3; ++i) {
      float fi = float(i);
      vec2 pos = vec2(hash11(seed + fi * 2.3 + 1.0), hash11(seed + fi * 3.9 + 2.0));
      float d = distance(uv, pos);
      float g = exp(-d * d / (radius * radius));
      v += g;
      u -= 0.5 * g;
    }
    outColor = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), 0.0, 1.0);
    return;
  }

  if (uPass == 0) {
    vec2 texel = 1.0 / uGridSize;
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    vec2 current = texture(uState, uv).rg;
    vec2 diffused = diffuseUV(uv, texel);

    float uOld = current.r;
    float vOld = current.g;
    float uD = diffused.r;
    float vD = diffused.g;

    float uvv = uOld * vOld * vOld;
    float du = uDu * (uD - uOld) - uvv + uFeed * (1.0 - uOld);
    float dv = uDv * (vD - vOld) + uvv - (uFeed + uKill) * vOld;
    float u = uOld + du;
    float v = vOld + dv;

    outColor = vec4(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0), 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  float v = texture(uState, uv).g;
  float intensity = clamp(v * uValueGain, 0.0, 1.0);
  vec3 base = vec3(0.04, 0.06, 0.08);
  vec3 glow = vec3(0.85, 0.92, 0.78);
  vec3 rgb = mix(base, glow, intensity);
  outColor = vec4(rgb, 1.0);
}
