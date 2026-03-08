#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 iResolution;
uniform float iTime;

uniform float uLoopDuration;
uniform float uSpeed;
uniform float uTwist;
uniform float uNoiseScale;
uniform float uNoiseAmp;
uniform float uColorCycle;
uniform float uFogDensity;
uniform vec3 uBaseColor;

const float TAU = 6.28318530718;

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

float fbm(vec2 p) {
  float sum = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    sum += amp * noise(p);
    p *= 2.0;
    amp *= 0.5;
  }
  return sum;
}

vec3 tunnelPalette(float t) {
  return 0.5 + 0.5 * cos(TAU * (t + vec3(0.0, 0.17, 0.36)));
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

  float phase = mod(iTime, uLoopDuration) / max(uLoopDuration, 0.001);
  float theta = phase * TAU;
  vec2 loopA = vec2(cos(theta), sin(theta));
  vec2 loopB = vec2(cos(2.0 * theta + 0.7), sin(3.0 * theta - 0.5));

  float baseR = length(uv);
  float a = atan(uv.y, uv.x);
  float twist = uTwist * (0.65 + 0.35 * loopA.x);
  a += twist * baseR + 0.25 * loopB.y;

  vec2 dir = vec2(cos(a), sin(a));
  vec2 flow = loopA * (0.7 + 0.3 * uSpeed) + loopB * (0.25 + 0.2 * uSpeed);
  vec2 np = dir * (0.9 * uNoiseScale) + uv * (2.2 * uNoiseScale) + flow * (0.75 * uNoiseScale);
  float n = fbm(np);
  float n2 = fbm(np * 1.9 + loopB * 3.0 + loopA.yx * 1.7);

  float r = baseR + uNoiseAmp * ((n - 0.5) * 1.4 + (n2 - 0.5) * 0.8);
  float lane = 0.5 + 0.5 * sin(a * 9.0 + n * 4.0 + 2.8 * loopB.x);
  float rings = 0.5 + 0.5 * cos(r * 22.0 - n2 * 5.5 + 2.2 * loopA.y);
  float tunnel = smoothstep(0.28, 0.92, lane * 0.75 + rings * 0.95);

  float huePhase = a / TAU + 0.25 * n + 0.12 * n2 + 0.12 * uColorCycle * loopA.y;
  vec3 dynamicHue = tunnelPalette(huePhase);
  vec3 oilHue = tunnelPalette(huePhase + 0.12 * loopB.x + 0.08 * loopA.y);
  vec3 col = mix(uBaseColor, dynamicHue, 0.45);
  col = mix(col, oilHue, 0.45 + 0.25 * lane);
  col = mix(col, vec3(1.0), 0.55 * tunnel);

  float fogBase = exp(-baseR * uFogDensity);
  float glowBase = pow(fogBase, 1.8);
  float e = 0.003 * max(0.5, uNoiseScale);
  vec2 grad;
  grad.x = fbm(np + vec2(e, 0.0)) - fbm(np - vec2(e, 0.0));
  grad.y = fbm(np + vec2(0.0, e)) - fbm(np - vec2(0.0, e));
  vec2 normal2D = normalize(grad + vec2(1e-6));
  vec2 uvR = uv + normal2D * (0.028 + 0.02 * glowBase);

  float rR = length(uvR);
  float aR = atan(uvR.y, uvR.x) + twist * rR + 0.25 * loopB.x;
  vec2 dirR = vec2(cos(aR), sin(aR));
  vec2 npR = dirR * (0.9 * uNoiseScale) + uvR * (2.2 * uNoiseScale) + flow * (0.75 * uNoiseScale);
  float nR = fbm(npR);
  float nR2 = fbm(npR * 1.9 + loopB * 3.0 + loopA.yx * 1.7);
  float laneR = 0.5 + 0.5 * sin(aR * 9.0 + nR * 4.0 + 2.8 * loopA.x);
  float ringsR = 0.5 + 0.5 * cos(rR * 22.0 - nR2 * 5.5 + 2.2 * loopB.y);
  vec3 colR = mix(uBaseColor, tunnelPalette(aR / TAU + 0.25 * nR + 0.12 * nR2 + 0.12 * uColorCycle * loopB.x), 0.45);
  colR = mix(colR, tunnelPalette(aR / TAU + 0.14 * loopA.x + 0.08 * nR), 0.45 + 0.25 * laneR);
  colR = mix(colR, vec3(1.0), 0.45 * smoothstep(0.28, 0.92, laneR * 0.75 + ringsR * 0.95));

  col = mix(col, colR, 0.58);
  col *= mix(0.55, 1.7, fogBase);
  col += glowBase * 0.32 * (0.5 * dynamicHue + 0.5 * oilHue);

  col = clamp(col, 0.0, 1.0);
  outColor = vec4(col, 1.0);
}
