#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uZoom;
uniform float uTimeScale;
uniform float uTwist;
uniform float uWarp;
uniform float uPulse;
uniform float uIterLimit;
uniform float uGlow;
uniform float uOffsetX;
uniform float uOffsetY;
uniform float uColorShift;

void main() {
  vec2 frag = gl_FragCoord.xy;
  vec2 res = uResolution.xy;
  vec2 uv = frag;
  vec2 v = res;
  vec2 offset = vec2(uOffsetX, uOffsetY) * res;

  uv = uZoom * (uv + uv - v + offset) / v.y;

  vec4 z = vec4(1.0, 2.0, 3.0, 0.0);
  vec4 o = z;
  float a = 0.5;
  float t = uTime * uTimeScale;

  for (int i = 0; i < 19; ++i) {
    float fi = float(i) + 1.0;
    float mask = step(fi, uIterLimit);
    float denom = length(
      (1.0 + fi * dot(v, v))
        * sin(1.5 * uv / (0.5 - dot(uv, uv)) - uTwist * 9.0 * uv.yx + t)
    );
    o += mask * (1.0 + cos(z + t + uColorShift)) / max(1e-3, denom);

    a += 0.03;
    float ap = pow(a, fi);
    t += 1.0;
    v = cos(t - uPulse * 7.0 * uv * ap) - 5.0 * uv;

    uv *= mat2(cos(fi + 0.02 * t - vec4(0.0, 11.0, 33.0, 0.0)));
    vec2 warp = tanh(uWarp * 40.0 * dot(uv, uv) * cos(100.0 * uv.yx + t)) / 200.0;
    uv += warp
      + 0.2 * a * uv
      + cos(4.0 / exp(dot(o, o) / 100.0) + t) / 300.0;
  }

  vec4 mapped = (25.6 * uGlow) / (min(o, 13.0) + 164.0 / o);
  mapped -= dot(uv, uv) / 250.0;
  outColor = vec4(mapped.rgb, 1.0);
}
