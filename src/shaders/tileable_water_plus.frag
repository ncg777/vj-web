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
uniform float uSwirlStrength;
uniform float uSwirlGridScale;
uniform float uSwirlRadius;
uniform float uSwirlWobble;
uniform float uSwirlWobbleSpeed;
uniform float uSwirlSpin;
uniform float uSwirlDesync;
uniform float uSwirlPulse;
uniform float uSwirlColorTwist;
uniform float uRainbowStrength;
uniform float uRainbowScale;
uniform float uRainbowSpeed;
uniform float uRainbowContrast;

const float TAU = 6.28318530718;
const int MAX_ITER = 5;

vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

mat2 rot(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat2(c, -s, s, c);
}

// Loose wobbly lattice of out-of-sync whirlpools: each cell hosts a
// breathing vortex whose radius pulses and whose spin waxes and wanes on
// its own clock. Warps `uv` in place and returns the net swirl amount.
float whirlpoolField(inout vec2 uv, float time) {
  float grid = max(uSwirlGridScale, 0.01);
  vec2 gp = uv * grid;
  vec2 cell = floor(gp);
  float swirlSum = 0.0;
  vec2 warped = uv;
  for (int oy = -1; oy <= 1; oy++) {
    for (int ox = -1; ox <= 1; ox++) {
      vec2 id = cell + vec2(float(ox), float(oy));
      vec2 rnd = hash22(id + 41.7);
      float phase = (rnd.x + rnd.y) * TAU * uSwirlDesync;
      vec2 wobble = vec2(
        cos(time * uSwirlWobbleSpeed * (0.5 + rnd.y) + phase),
        sin(time * uSwirlWobbleSpeed * (0.6 + rnd.x) + phase * 2.3)
      ) * uSwirlWobble;
      vec2 center = (id + 0.5 + (rnd - 0.5) * 0.9 + wobble) / grid;
      vec2 d = warped - center;
      float dist = length(d);
      // Breathing radius: each whirlpool inhales and exhales out of sync.
      float breath = 1.0 + uSwirlPulse * sin(time * (0.4 + rnd.x * 0.7) + phase);
      float radius = max(uSwirlRadius, 0.01) * breath / grid;
      float falloff = smoothstep(radius, 0.0, dist);
      falloff *= falloff * (3.0 - 2.0 * falloff);
      // Alternate spin direction on a checkerboard for counter-rotation.
      float dir = mod(id.x + id.y, 2.0) < 1.0 ? 1.0 : -1.0;
      float spin = sin(time * uSwirlSpin * (0.8 + rnd.y * 0.5) + phase) * 0.6 + 0.7;
      float angle = uSwirlStrength * dir * spin * falloff;
      warped = center + rot(angle) * (warped - center);
      swirlSum += angle;
    }
  }
  uv = warped;
  return swirlSum;
}

void main() {
  float time = uTime * uTimeScale + 23.0;
  vec2 uv = gl_FragCoord.xy / uResolution.xy;

  float swirl = whirlpoolField(uv, time);

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

  // Twist hue channels inside the whirlpools.
  if (abs(uSwirlColorTwist) > 0.0001) {
    float twist = swirl * uSwirlColorTwist;
    color.rg = rot(twist) * color.rg;
    color.gb = rot(twist * 0.7) * color.gb;
    color = abs(color);
  }

  // Oil-slick interference sheen: film thickness rides the water caustic
  // brightness and the swirl field, refracting into a spectral gradient.
  if (uRainbowStrength > 0.0001) {
    float thickness = cAdj * uRainbowScale + swirl * 2.0 + time * uRainbowSpeed * 0.1;
    thickness += sin(uv.x * 9.0 + time * 0.3) * 0.3 + cos(uv.y * 7.0 - time * 0.23) * 0.3;
    vec3 sheen = 0.5 + 0.5 * cos(TAU * (thickness * vec3(1.0, 1.3, 1.7) + vec3(0.0, 0.33, 0.67)));
    sheen = pow(sheen, vec3(max(uRainbowContrast, 0.1)));
    float lum = clamp(dot(color, vec3(0.299, 0.587, 0.114)) * 1.8, 0.0, 1.0);
    color = mix(color, color * (0.4 + 1.2 * sheen) + sheen * lum * 0.5, uRainbowStrength);
  }

  outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
