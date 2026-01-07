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
uniform float uRotate;
uniform float uInjectAmp;
uniform float uInjectRadius;
uniform float uValueGain;
uniform float uSeed;

float hash11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

vec2 diffuseVec2(vec2 uv, vec2 texel) {
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
    vec2 ab = vec2(0.0);
    float seed = uSeed * 0.001;
    float radius = 0.08;
    for (int i = 0; i < 3; ++i) {
      float fi = float(i);
      vec2 pos = vec2(hash11(seed + fi * 3.1 + 1.0), hash11(seed + fi * 4.7 + 2.0));
      float ang = hash11(seed + fi * 5.3 + 3.0) * 6.2831853;
      float d = distance(uv, pos);
      float g = exp(-d * d / (radius * radius));
      ab += g * vec2(cos(ang), sin(ang));
    }
    outColor = vec4(ab, 0.0, 1.0);
    return;
  }

  if (uPass == 0) {
    vec2 texel = 1.0 / uGridSize;
    vec2 uv = (gl_FragCoord.xy + 0.5) / uGridSize;
    vec2 ab = diffuseVec2(uv, texel);

    float ang = uRotate;
    float ca = cos(ang);
    float sa = sin(ang);
    ab = mat2(ca, -sa, sa, ca) * ab;
    ab *= uDecay;

    float seed = uSeed * 0.001;
    float t = uTime * 0.6 + seed * 3.0;
    vec2 pos = 0.5 + 0.32 * vec2(sin(t * 1.1 + seed), cos(t * 1.4 + seed * 1.7));
    float injectAng = t * 1.7 + seed * 5.0;
    float dist = distance(uv, pos);
    float sigma = max(1e-4, uInjectRadius);
    float g = exp(-dist * dist / (sigma * sigma));
    ab += uInjectAmp * g * vec2(cos(injectAng), sin(injectAng));

    outColor = vec4(ab, 0.0, 1.0);
    return;
  }

  vec2 uv = gl_FragCoord.xy / uResolution;
  vec2 ab = texture(uState, uv).rg;
  float angle = atan(ab.y, ab.x);
  float mag = length(ab);
  float hue = (angle + 3.14159265) / 6.2831853;
  float value = clamp(mag * uValueGain, 0.0, 1.0);
  vec3 rgb = hsv2rgb(vec3(hue, 1.0, value));
  outColor = vec4(rgb, 1.0);
}
