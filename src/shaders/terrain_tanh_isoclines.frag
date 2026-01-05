#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uPhase;
uniform float uScale;
uniform int uOctaves;
uniform float uLacunarity;
uniform float uGain;
uniform int uIsoBands;
uniform float uLineThickness;
uniform float uSeed;
uniform float uBubbleAmp;
uniform float uBubbleFreq;
uniform float uBubbleDetail;

const float PI = 3.14159265358979323846;
const float TAU = 6.28318530717958647692;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

float h11(float n) {
  return fract(sin(n) * 43758.5453123);
}

vec2 h21(float n) {
  return vec2(h11(n * 19.0 + 0.73), h11(n * 23.0 + 1.91));
}

float fbm(vec2 p, int octaves, float lac, float gain) {
  float sum = 0.0;
  float amp = 0.5;
  float norm = 0.0;
  vec2 pp = p;
  for (int i = 0; i < 12; ++i) {
    if (i >= octaves) {
      break;
    }
    sum += amp * noise(pp);
    norm += amp;
    pp *= lac;
    amp *= gain;
  }
  return (norm > 1e-6) ? sum / norm : 0.0;
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 6.0, 4.0 / 6.0)) * 6.0 - 3.0);
  vec3 rgb = clamp(p - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), rgb, c.y);
}

void main() {
  float minDim = min(uResolution.x, uResolution.y);
  vec2 p = (gl_FragCoord.xy - 0.5 * uResolution) / minDim;

  vec2 seedShift = (h21(uSeed * 0.137) - 0.5) * 1024.0;
  vec2 timeShift = vec2(cos(TAU * uPhase), sin(TAU * uPhase)) * (0.75 * uScale);
  vec2 world = p * uScale + seedShift + timeShift;

  vec2 warpOff = vec2(cos(TAU * (uPhase + 0.27)), sin(TAU * (uPhase + 0.27))) * (0.33 * uScale);
  float base0 = fbm(world + warpOff, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));
  float signed0 = base0 * 2.0 - 1.0;
  float tanh0 = tanh(1.35 * signed0);

  vec2 swirl = vec2(-p.y, p.x);
  vec2 warp = (0.18 * uScale) * (swirl * tanh0)
    + (0.12 * uScale) * vec2(sin(world.y * 0.8), cos(world.x * 0.8)) * tanh0;
  vec2 world2 = world + warp;

  float base1 = fbm(world2 + warpOff * 0.6, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));
  float signed1 = base1 * 2.0 - 1.0;
  float h = 0.5 + 0.5 * tanh(1.25 * signed1);
  float hCurve = h * h * (3.0 - 2.0 * h);
  float hFinal = mix(h, hCurve, 0.6);

  float bubbleDet = max(0.25, uBubbleDetail);
  vec2 bubbleTimeShift = vec2(cos(TAU * (uPhase + 0.43)), sin(TAU * (uPhase + 0.43))) * (0.55 * bubbleDet);
  float bubbleNoise = fbm(world2 * bubbleDet + bubbleTimeShift, max(1, uOctaves), max(1.01, uLacunarity), clamp(uGain, 0.01, 0.99));
  float bubbleWave = sin(TAU * (uBubbleFreq * uPhase) + bubbleNoise * PI + 1.5 * tanh0);
  float hBubbled = hFinal + uBubbleAmp * bubbleWave * (0.35 + 0.65 * bubbleNoise);
  hBubbled = clamp(hBubbled, 0.0, 1.0);

  float e = 1.25 / minDim;
  float hx = fbm(world2 + vec2(e, 0.0), uOctaves, uLacunarity, uGain)
    - fbm(world2 - vec2(e, 0.0), uOctaves, uLacunarity, uGain);
  float hy = fbm(world2 + vec2(0.0, e), uOctaves, uLacunarity, uGain)
    - fbm(world2 - vec2(0.0, e), uOctaves, uLacunarity, uGain);
  float slope = length(vec2(hx, hy));

  int bands = max(1, uIsoBands);
  float line = abs(sin(PI * float(bands) * hBubbled));
  float lt = clamp(uLineThickness, 0.02, 0.75);

  float core = pow(max(0.0, 1.0 - (line / lt)), 1.35);
  float glow = pow(max(0.0, 1.0 - (line / (lt * 3.0))), 2.2);
  float intensity = clamp(core + 0.5 * glow, 0.0, 1.0);

  float r = length(p) / 0.9;
  float vignette = smoothstep(1.0, 0.6, r);
  intensity *= vignette;

  float hue = fract(0.62 * hBubbled + 0.18 * slope + 0.1 * sin(TAU * uPhase));
  float sat = mix(0.65, 1.0, intensity);
  float bri = mix(0.12, 1.0, intensity);
  hue = fract(hue + 0.05 * tanh0 + 0.04 * sin(TAU * (uPhase + hBubbled)));

  vec3 rgb = hsv2rgb(vec3(hue, sat, bri));
  outColor = vec4(rgb, 1.0);
}
