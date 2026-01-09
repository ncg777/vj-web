#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uTileScale;
uniform float uIntensity;
uniform float uContrast;
uniform float uWaveShift;
uniform vec3 uTint;

const float TAU = 6.28318530718;
const int MAX_ITER = 5;

void main() {
  float time = uTime * uTimeScale + 23.0;
  vec2 uv = gl_FragCoord.xy / uResolution.xy;
  vec2 p = mod(uv * TAU * uTileScale, TAU) - 250.0;
  vec2 i = p;
  float c = 1.0;
  float inten = 0.005;

  for (int n = 0; n < MAX_ITER; n++) {
    float t = time * (1.0 - (3.5 / float(n + 1))) + uWaveShift;
    i = p + vec2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));
    vec2 denom = vec2(p.x / (sin(i.x + t) / inten), p.y / (cos(i.y + t) / inten));
    c += 1.0 / length(denom);
  }

  c /= float(MAX_ITER);
  c = 1.17 - pow(c, 1.4);
  float cAdj = pow(clamp(c, 0.0, 1.0), max(0.1, uContrast));
  vec3 color = vec3(pow(abs(cAdj), 8.0)) * uIntensity;
  color = clamp(color + uTint, 0.0, 1.0);

  outColor = vec4(color, 1.0);
}
