#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform int uBranches;
uniform float uGrowth;
uniform float uCurl;
uniform float uPulse;
uniform float uArcDensity;
uniform float uGlow;
uniform float uHue;
uniform float uSeed;

const float TAU = 6.28318530718;

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

float hash11(float value) {
  return fract(sin(value * 127.17 + uSeed * 0.000013) * 43758.5453);
}

float hash21(vec2 point) {
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33 + fract(uSeed * 0.000001));
  return fract((p.x + p.y) * p.z);
}

float noise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  local = local * local * (3.0 - 2.0 * local);
  return mix(
    mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), local.x),
    mix(hash21(cell + vec2(0.0, 1.0)), hash21(cell + vec2(1.0)), local.x),
    local.y
  );
}

float fbm(vec2 point) {
  float value = 0.0;
  float amplitude = 0.52;
  for (int octave = 0; octave < 5; octave++) {
    value += amplitude * noise(point);
    point = rotate2d(0.68) * point * 2.04 + vec2(2.3, -1.6);
    amplitude *= 0.49;
  }
  return value;
}

vec3 palette(float phase) {
  return 0.48 + 0.52 * cos(TAU * (phase * vec3(1.0, 0.78, 0.55) + vec3(0.02, 0.28, 0.62)));
}

void main() {
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  point *= uZoom;
  float time = uTime * uTimeScale;

  vec2 smokePoint = point * 2.2;
  smokePoint += vec2(
    fbm(point * 1.4 + vec2(time * 0.08, 0.0)),
    fbm(point * 1.4 - vec2(0.0, time * 0.07))
  ) * 0.8;
  float smoke = fbm(smokePoint - time * 0.035);
  float arcField = 1.0 - abs(sin((smoke + point.x * 0.35 - point.y * 0.22) * uArcDensity * 8.0));
  arcField = pow(arcField, 7.0);

  float coral = 0.0;
  float buds = 0.0;
  float membranes = 0.0;
  float choirPhase = 0.0;
  float branchCount = float(uBranches);

  for (int colony = 0; colony < 4; colony++) {
    float fi = float(colony);
    float randomA = hash11(fi + 5.0);
    float randomB = hash11(fi + 31.0);
    float phase = randomA * TAU;
    vec2 center = vec2(
      sin(time * (0.11 + randomA * 0.08) + phase),
      cos(time * (0.09 + randomB * 0.07) + phase * 1.4)
    ) * vec2(0.64, 0.42);
    vec2 local = rotate2d(phase + time * (randomA - 0.5) * 0.16) * (point - center);
    float radius = length(local);
    float angle = atan(local.y, local.x);

    float curledAngle = angle * branchCount
      + uCurl * sin(radius * (8.0 + 5.0 * randomB) - time * 1.2 + phase)
      + sin(radius * 21.0 - time * 0.7) * 0.35;
    float branchDistance = abs(sin(curledAngle));
    float trunk = exp(-branchDistance * (19.0 - min(uGrowth, 2.0) * 3.0));
    float reach = 1.0 - smoothstep(0.08, 0.48 + uGrowth * 0.16, radius);
    float hollow = smoothstep(0.018, 0.075, radius);
    float forkPulse = 0.58 + 0.42 * sin(radius * 34.0 - time * uPulse * 3.0 + phase);
    float branch = trunk * reach * hollow * (0.48 + 0.52 * forkPulse);

    float shellRadius = 0.12 + randomB * 0.06 + sin(time * uPulse + phase) * 0.018;
    float shell = exp(-abs(radius - shellRadius) * 75.0);
    float budBand = abs(fract(radius * (4.0 + uGrowth * 2.0) - time * uPulse * 0.23 + randomA) - 0.5);
    float bud = exp(-budBand * 24.0) * pow(trunk, 2.0) * reach;

    coral += branch;
    membranes += shell;
    buds += bud;
    choirPhase += (branch + shell) * (randomA - 0.5);
  }

  float heartbeat = 0.78 + 0.22 * sin(time * uPulse * 2.0 + smoke * 5.0);
  vec3 color = vec3(0.004, 0.007, 0.015);
  vec3 smokeColor = palette(uHue + smoke * 0.24 + time * 0.018);
  vec3 coralColor = palette(uHue + 0.42 + choirPhase * 0.035 - time * 0.012);
  vec3 budColor = palette(uHue + 0.78 - smoke * 0.12);

  color += smokeColor * smoke * 0.10;
  color += smokeColor * arcField * (0.18 + uGlow * 0.24);
  color += coralColor * coral * uGlow * 0.92 * heartbeat;
  color += budColor * buds * uGlow * 1.35;
  color += mix(coralColor, vec3(0.85, 0.95, 1.0), 0.55) * membranes * uGlow;
  color += vec3(0.25, 0.55, 1.0) * coral * arcField * uGlow * 0.8;

  color *= 1.0 - smoothstep(0.55, 1.5, length(point)) * 0.45;
  color = 1.0 - exp(-color * 1.45);
  color = pow(max(color, 0.0), vec3(0.86));
  outColor = vec4(color, 1.0);
}