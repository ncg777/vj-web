#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uTurbulence;
uniform float uCloudHeight;
uniform float uStepBase;
uniform float uStepScale;
uniform float uHueShift;
uniform float uHueSpeed;
uniform float uIntensity;

void main() {
  vec2 I = gl_FragCoord.xy;
  float t = uTime * uTimeScale;
  float i = 0.0;
  float z = 0.0;
  float d = 0.0;
  float s = 0.0;
  vec4 O = vec4(0.0);

  for (O *= i; i++ < 100.0;) {
    vec3 p = z * normalize(vec3(I + I, 0.0) - uResolution.xyy);

    for (d = 5.0; d < 200.0; d += d) {
      p += uTurbulence * 0.6 * sin(p.yzx * d - 0.2 * t) / d;
    }

    float height = max(0.05, uCloudHeight);
    s = height - abs(p.y);
    z += d = uStepBase + max(s, -s * 0.2) / uStepScale;

    vec4 phase = vec4(3.0, 4.0, 5.0, 0.0) + uHueShift + uHueSpeed * t;
    O += (cos(s / 0.07 + p.x + 0.5 * t - phase) + 1.5) * exp(s / 0.1) / d;
  }

  O = tanh(O * O / 4e8);
  outColor = vec4(O.rgb * uIntensity, 1.0);
}
