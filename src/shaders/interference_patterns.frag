#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform int uStyle;
uniform float uSpeed;
uniform float uFrequency;
uniform float uScale;
uniform float uSeparation;
uniform float uAngle;
uniform float uMismatch;
uniform int uSources;
uniform int uModeX;
uniform int uModeY;
uniform float uPhase;
uniform float uMotion;
uniform float uContrast;
uniform float uHue;
uniform float uGlow;
uniform float uSeed;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

mat2 rotate(float angle) {
  float c = cos(angle), s = sin(angle);
  return mat2(c, -s, s, c);
}

// Filter each wave before superposition to suppress subpixel grating shimmer.
vec2 wave(float phase) {
  vec2 gradient = vec2(dFdx(phase), dFdy(phase));
  float visibility = exp(-0.5 * dot(gradient, gradient));
  return vec2(cos(phase), sin(phase)) * visibility;
}

vec3 spectrum(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase + vec3(0.02, 0.35, 0.68)));
}

void main() {
  vec2 point = (2.0 * gl_FragCoord.xy - uResolution) / min(uResolution.x, uResolution.y);
  point *= uScale;
  float time = uTime * uSpeed;
  // Keep seed arithmetic small so neighboring seeds remain distinct at highp precision.
  float seed = fract(sin(uSeed * 0.01371 + 0.123) * 43758.5453);
  float orientation = seed * TAU;
  float motion = time * uMotion;
  float k = TAU * uFrequency;
  float carrierTime = time * 2.4;
  float intensity;
  float carrier;
  float colorPhase;
  float nodes = 0.0;

  if (uStyle == 0) {
    vec2 axis = rotate(orientation + 0.16 * motion) * vec2(uSeparation, 0.0);
    vec2 drift = 0.12 * uMotion * vec2(sin(time * 0.31 + orientation), cos(time * 0.27));
    float distanceA = length(point - axis - drift);
    float distanceB = length(point + axis - drift);
    float phaseA = k * distanceA - carrierTime;
    float phaseB = k * distanceB - carrierTime + uPhase;
    vec2 amplitude = 0.5 * (wave(phaseA) + wave(phaseB));
    // Squaring the summed complex amplitudes preserves true cancellation fringes.
    intensity = dot(amplitude, amplitude);
    carrier = 0.5 + 0.5 * amplitude.x;
    colorPhase = (distanceA - distanceB) * uFrequency * 0.12 + length(point) * 0.13;
  } else if (uStyle == 1) {
    float angle = radians(uAngle) + 0.12 * sin(motion * 0.19 + orientation);
    vec2 directionA = vec2(cos(orientation + motion * 0.035), sin(orientation + motion * 0.035));
    vec2 directionB = rotate(angle) * directionA;
    float phaseA = k * dot(point, directionA) - carrierTime * 0.4;
    float phaseB = k * (1.0 + uMismatch) * dot(point, directionB) + uPhase + carrierTime * 0.28;
    vec2 amplitude = 0.5 * (wave(phaseA) + wave(phaseB));
    intensity = dot(amplitude, amplitude);
    carrier = 0.5 + 0.5 * amplitude.x;
    colorPhase = (phaseA - phaseB) / TAU * 0.16 + dot(point, directionA) * 0.08;
  } else if (uStyle == 2) {
    vec2 p = rotate((seed - 0.5) * 0.35 + 0.06 * sin(motion * 0.3)) * point;
    vec2 amplitude = vec2(0.0);
    for (int slit = 0; slit < 8; slit++) {
      if (slit >= uSources) break;
      float offset = float(slit) - 0.5 * float(uSources - 1);
      vec2 source = vec2(offset * uSeparation, -1.15);
      float distance = length(p - source);
      float phase = k * distance - carrierTime + offset * (uPhase + 0.7 * sin(motion * 0.4));
      amplitude += wave(phase);
    }
    amplitude /= float(uSources);
    intensity = dot(amplitude, amplitude);
    carrier = 0.5 + 0.5 * amplitude.x;
    colorPhase = p.x * 0.22 + length(p - vec2(0.0, -1.15)) * 0.17 + seed * 0.2;
  } else {
    vec2 p = rotate((seed - 0.5) * 0.4 + 0.08 * sin(motion * 0.2)) * point;
    float x = k * p.x * 0.25;
    float y = k * p.y * 0.25;
    // Independent even/odd modes also avoid a blank field when mode numbers match.
    float modeA = wave(float(uModeX) * x).x * wave(float(uModeY) * y).x;
    float modeB = wave(float(uModeY) * x).y * wave(float(uModeX) * y).y;
    float blend = 0.65 * sin(motion * 0.55 + orientation) + uPhase;
    float field = modeA * cos(blend) + modeB * sin(blend);
    intensity = field * field;
    carrier = 0.5 + 0.5 * field * cos(carrierTime);
    float width = max(fwidth(field), 0.012);
    nodes = 1.0 - smoothstep(width, width * 2.5 + 0.025, abs(field));
    // Fade nodal lines when the modes are too fine for the output resolution.
    nodes *= smoothstep(0.005, 0.08, abs(modeA) + abs(modeB));
    colorPhase = (modeA - modeB) * 0.27 + length(p) * 0.15;
  }

  float energy = pow(clamp(intensity, 0.0, 1.0), uContrast);
  float crest = pow(clamp(carrier, 0.0, 1.0), 4.0);
  vec3 pigment = spectrum(colorPhase + uHue + seed * 0.25 + time * 0.012);
  vec3 complementary = spectrum(colorPhase + uHue + seed * 0.25 + 0.21);
  vec3 color = vec3(0.004, 0.007, 0.018);
  color += pigment * energy * (0.38 + 1.7 * crest);
  color += complementary * energy * energy * uGlow * 0.65;
  color += mix(pigment, vec3(0.65, 0.9, 1.0), 0.6) * nodes * (0.12 + uGlow * 0.65);
  float vignette = 1.0 - 0.22 * smoothstep(0.3, 1.5, length(point / uScale));
  color = 1.0 - exp(-color * vignette * 1.65);
  outColor = vec4(pow(max(color, 0.0), vec3(0.9)), 1.0);
}
