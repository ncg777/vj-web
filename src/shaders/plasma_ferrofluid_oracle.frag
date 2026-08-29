#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform int uPoleCount;
uniform float uFieldLines;
uniform float uViscosity;
uniform float uBlobSize;
uniform float uPoleOrbit;
uniform float uEyeStrength;
uniform float uGlow;
uniform float uHue;
uniform float uSeed;

const float TAU = 6.28318530718;

float hash11(float value) {
  return fract(sin(value * 117.13 + uSeed * 0.000019) * 43758.5453);
}

vec3 palette(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.67, 0.91, 1.13) + vec3(0.03, 0.26, 0.59)));
}

void main() {
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  point *= uZoom;
  float time = uTime * uTimeScale;

  vec2 field = vec2(0.0);
  float potential = 0.0;
  float fluid = 0.0;
  float eyes = 0.0;
  float poleGlow = 0.0;

  for (int index = 0; index < 7; index++) {
    if (index >= uPoleCount) {
      break;
    }
    float fi = float(index);
    float randomA = hash11(fi + 4.0);
    float randomB = hash11(fi + 29.0);
    float phase = randomA * TAU;
    float polarity = mod(fi, 2.0) < 1.0 ? 1.0 : -1.0;
    vec2 pole = vec2(
      sin(time * (0.13 + randomA * 0.12) + phase),
      cos(time * (0.11 + randomB * 0.1) + phase * 1.31)
    ) * uPoleOrbit * vec2(0.7, 0.48);
    vec2 delta = point - pole;
    float distanceSquared = dot(delta, delta) + 0.018;
    float inverseDistance = inversesqrt(distanceSquared);
    field += polarity * delta / distanceSquared;
    potential += polarity * log(distanceSquared) * 0.5;

    float blobRadius = uBlobSize * (0.11 + randomB * 0.055);
    float wobble = 1.0 + uViscosity * 0.16 * sin(atan(delta.y, delta.x) * (4.0 + floor(randomA * 5.0)) + time + phase);
    float signedBlob = length(delta) - blobRadius * wobble;
    fluid += 1.0 - smoothstep(-0.01, 0.025, signedBlob);
    poleGlow += exp(-abs(signedBlob) * 70.0);

    vec2 eyePoint = delta * vec2(0.8, 1.65);
    float eyeRadius = length(eyePoint);
    float eye = exp(-abs(eyeRadius - blobRadius * 0.34) * 105.0);
    float pupil = exp(-eyeRadius * 80.0);
    eyes += (eye + pupil * 1.6) * (1.0 - smoothstep(blobRadius, blobRadius * 1.3, length(delta)));
    fluid += inverseDistance * 0.006 * uViscosity;
  }

  float fieldAngle = atan(field.y, field.x);
  float linePhase = potential * uFieldLines * 3.2 + fieldAngle * 1.7 - time * 0.36;
  float magneticLines = pow(1.0 - abs(sin(linePhase)), 10.0);
  magneticLines *= smoothstep(0.25, 2.2, length(field));

  float ridge = pow(1.0 - abs(sin(length(field) * 0.18 - time * 0.7)), 8.0);
  ridge *= smoothstep(0.3, 5.0, length(field)) * 0.55;
  float silhouette = smoothstep(0.15, 1.15, fluid);
  float oily = 0.5 + 0.5 * sin(potential * 5.0 + time * 0.5);

  vec3 lineColor = palette(uHue + potential * 0.08 + time * 0.016);
  vec3 fluidColor = palette(uHue + 0.46 + oily * 0.18);
  vec3 eyeColor = palette(uHue + 0.82 - time * 0.025);

  vec3 color = vec3(0.003, 0.006, 0.012);
  color += lineColor * magneticLines * uGlow * 1.05;
  color += lineColor * ridge * uGlow * 0.62;
  color += fluidColor * silhouette * (0.12 + oily * 0.2);
  color += fluidColor * poleGlow * uGlow * 0.95;
  color += eyeColor * eyes * uEyeStrength * uGlow * 1.4;
  color += vec3(0.55, 0.72, 1.0) * magneticLines * poleGlow * uGlow;

  color *= 1.0 - smoothstep(0.65, 1.65, length(point)) * 0.38;
  color = 1.0 - exp(-color * 1.5);
  color = pow(max(color, 0.0), vec3(0.83));
  outColor = vec4(color, 1.0);
}