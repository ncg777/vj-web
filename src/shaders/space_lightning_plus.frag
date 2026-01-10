#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uSpin;
uniform float uTwist;
uniform float uWarp;
uniform float uPulse;
uniform float uBoltDensity;
uniform float uBoltSharpness;
uniform float uBoltIntensity;
uniform float uArcSteps;
uniform float uCoreSize;
uniform float uCoreGlow;
uniform float uNoiseAmp;
uniform float uPaletteShift;
uniform float uSeed;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;
uniform vec3 uColorAccent;

const float TAU = 6.28318530718;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 palette(float t) {
  return uColorPrimary + uColorSecondary * cos(TAU * (uColorAccent * t + uPaletteShift));
}

void main() {
  vec2 uv = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;
  uv *= uZoom;

  float t = uTime * uTimeScale;
  float spin = t * uSpin;
  mat2 rot = mat2(cos(spin), -sin(spin), sin(spin), cos(spin));
  uv = rot * uv;

  vec2 twist = vec2(
    sin(uv.y * (2.0 + uTwist) + t),
    cos(uv.x * (2.5 + uTwist) - t)
  );
  uv += twist * 0.15 * uWarp;

  float radius = length(uv) + 1e-4;
  float angle = atan(uv.y, uv.x);

  float bolts = 0.0;
  for (int i = 0; i < 80; i++) {
    float fi = float(i) + 1.0;
    float mask = step(fi, uArcSteps);
    float phase = t * (0.8 + uPulse) + fi * 0.37 + uSeed * 0.001;
    float wave = sin(angle * (uBoltDensity + fi * 0.12) + phase)
      + cos(radius * (uBoltDensity * 1.7) - phase);
    float d = abs(wave) + 0.12 + radius * uBoltSharpness * 0.35;
    float contribution = uBoltIntensity / (d * d);
    bolts += mask * contribution;
  }

  vec2 q = uv;
  float spark = 0.0;
  for (int i = 0; i < 7; i++) {
    float fi = float(i) + 1.0;
    float denom = max(0.25, dot(q, q));
    q = abs(q) / denom - vec2(0.55, 0.35) * uWarp;
    spark += exp(-fi) * (0.5 + 0.5 * sin(fi * 2.3 + t + q.x * 4.0 + q.y * 5.0));
  }

  float grain = hash(uv * (12.0 + uSeed * 0.001) + t);
  spark += uNoiseAmp * (grain - 0.5);

  float core = uCoreGlow / (abs(radius - uCoreSize) + 0.02);

  float energy = bolts * 0.03 + spark * 0.9 + core * 0.6;
  energy *= smoothstep(1.8, 0.2, radius);

  vec3 col = palette(radius + spark * 0.35 + t * 0.1);
  col *= energy;
  col += vec3(bolts * 0.02);
  col += vec3(core * 0.15);

  col = pow(max(col, 0.0), vec3(0.75));
  outColor = vec4(col, 1.0);
}
