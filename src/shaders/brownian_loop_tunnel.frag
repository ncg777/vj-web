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

float loopTime(float t, float duration) {
  float phase = mod(t, duration) / duration;
  return phase * TAU;
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

  float t = loopTime(iTime * uSpeed, uLoopDuration);
  float phase = t / TAU;

  float r = length(uv);
  float a = atan(uv.y, uv.x);
  a += uTwist * r;

  vec2 dir = vec2(cos(a), sin(a));
  vec2 tOff = vec2(cos(t), sin(t)) * (0.6 * uNoiseScale);
  vec2 np = dir * (0.75 * uNoiseScale) + tOff + r * (2.0 * uNoiseScale);
  float n = fbm(np);
  r += uNoiseAmp * n;
  a += uNoiseAmp * 0.5 * n;

  float stripePhase = fract(phase + r * 0.5);
  float stripe = smoothstep(0.3, 0.5, sin(a * 8.0 + stripePhase * TAU));

  vec3 baseHue = uBaseColor;
  vec3 dynamicHue = 0.5 + 0.5 * cos(vec3(0.0, 0.6, 1.2) + a * 2.0 + n * 2.0 + uColorCycle * t);
  vec3 col = mix(baseHue, dynamicHue, 0.7);

  float stripeMask = stripe;
  col = mix(col, vec3(1.0), 0.6 * stripeMask);

  float fogBase = exp(-r * uFogDensity);
  float glowBase = pow(fogBase, 2.0);
  float e = 0.003 * max(0.5, uNoiseScale);
  vec2 grad;
  grad.x = fbm(np + vec2(e, 0.0)) - fbm(np - vec2(e, 0.0));
  grad.y = fbm(np + vec2(0.0, e)) - fbm(np - vec2(0.0, e));
  vec2 normal2D = normalize(grad + vec2(1e-6));
  float refractStrength = 0.03;
  vec2 uvR = uv + normal2D * refractStrength * (0.3 + 0.7 * glowBase);

  float rR = length(uvR);
  float aR = atan(uvR.y, uvR.x);
  aR += uTwist * rR;
  vec2 dirR = vec2(cos(aR), sin(aR));
  vec2 npR = dirR * (0.75 * uNoiseScale) + tOff + rR * (2.0 * uNoiseScale);
  float nR = fbm(npR);
  float stripePhaseR = fract(phase + rR * 0.5);
  float stripeR = smoothstep(0.3, 0.5, sin(aR * 8.0 + stripePhaseR * TAU));
  vec3 dynamicHueR = 0.5 + 0.5 * cos(vec3(0.0, 0.6, 1.2) + aR * 2.0 + nR * 2.0 + uColorCycle * t);
  vec3 colR = mix(baseHue, dynamicHueR, 0.7);
  colR = mix(colR, vec3(1.0), 0.6 * stripeR);

  col = mix(col, colR, 0.6);

  float fog = fogBase;
  float glow = glowBase;
  col *= mix(0.6, 1.6, fog);
  col += glow * 0.35 * (0.6 * baseHue + 0.4 * dynamicHue);

  col = clamp(col, 0.0, 1.0);
  outColor = vec4(col, 1.0);
}
