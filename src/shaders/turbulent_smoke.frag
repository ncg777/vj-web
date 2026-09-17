#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;
uniform vec2 uResolution;
uniform vec2 uGridSize;
uniform sampler2D uState;
uniform int uPass;
uniform int uStyle;
uniform float uTime;
uniform float uSpeed;
uniform float uTurbulence;
uniform float uEddyScale;
uniform float uBuoyancy;
uniform float uWind;
uniform float uEmission;
uniform float uWidth;
uniform float uPersistence;
uniform float uDensity;
uniform float uHue;
uniform float uGlow;
uniform float uSeed;

const float TAU = 6.28318530718;

// Saturated emissive pigments; RGB is transported premultiplied by density.
vec3 palette(float h) {
  vec3 c = 0.5 + 0.5 * cos(TAU * (h + uHue + vec3(0.0, 0.333, 0.667)));
  return 0.025 + 0.975 * c * c;
}

// Analytic curl of a multiscale stream function produces smooth eddies.
// Prescribed stirring plus density buoyancy is a visual gas approximation,
// not a pressure-projected Navier-Stokes solver.
vec2 velocity(vec2 p, float t, float density) {
  vec2 curl = vec2(0);
  float frequency = 1.0;
  float amplitude = 1.0;
  float seed = mod(uSeed, 997.0) * 0.071;
  for (int i = 0; i < 5; ++i) {
    float f = float(i);
    float a = f * 1.618 + seed;
    vec2 k = vec2(cos(a), sin(a));
    vec2 l = vec2(-k.y, k.x);
    vec2 q = vec2(dot(p, k), dot(p, l)) * (TAU * frequency * uEddyScale);
    q += vec2(sin(t * 0.29 + f + seed), cos(t * 0.23 - f));
    vec2 gradient = k * cos(q.x) * cos(q.y) - l * sin(q.x) * sin(q.y);
    curl += amplitude * vec2(gradient.y, -gradient.x);
    frequency *= 2.03;
    amplitude *= 0.56;
  }
  vec2 v = curl * (0.055 * uTurbulence);
  v += vec2(uWind * 0.12, uBuoyancy * (0.065 + 0.055 * min(density, 2.0)));
  vec2 d = p - 0.5;
  if (uStyle == 1) {
    v.x += -0.24 * tanh(d.x * 7.0) * exp(-d.y * d.y * 16.0);
    v.y += 0.16 * tanh(d.y * 10.0) * exp(-d.x * d.x * 22.0);
  }
  if (uStyle == 2) {
    v += vec2(-d.y, d.x) * 0.65 * exp(-dot(d, d) * 3.0) - d * 0.025;
  }
  if (uStyle == 3) {
    v.x += 0.13 + 0.055 * sin(p.y * 18.0 + t * 0.3);
    v.y += 0.025 * sin(p.x * 12.0 - t * 0.5);
  }
  if (uStyle == 4) v += vec2(d.y, -d.x) * 0.12;
  return v;
}

// Manual bilinear interpolation supports nearest-filtered floating targets.
// Open boundaries let smoke escape instead of wrapping to the other edge.
vec4 sampleGas(vec2 uv) {
  if (any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)))) return vec4(0);
  uv = clamp(uv, 0.5 / uGridSize, 1.0 - 0.5 / uGridSize);
  vec2 p = uv * uGridSize - 0.5;
  vec2 f = fract(p);
  vec2 pixel = 1.0 / uGridSize;
  vec2 a = (floor(p) + 0.5) * pixel;
  return mix(mix(texture(uState, a), texture(uState, a + vec2(pixel.x, 0)), f.x),
    mix(texture(uState, a + vec2(0, pixel.y)), texture(uState, a + pixel), f.x), f.y);
}

vec4 source(vec2 p, float t) {
  vec4 gas = vec4(0);
  float seed = mod(uSeed, 997.0) * 0.017;
  for (int i = 0; i < 6; ++i) {
    float f = float(i);
    float angle = TAU * f / 6.0 + seed;
    vec2 center = vec2(0.12 + f * 0.152 + 0.015 * sin(t + f), 0.085);
    vec2 size = vec2(0.022, 0.027);
    if (uStyle == 1) {
      center = vec2(i < 3 ? 0.08 : 0.92, 0.3 + mod(f, 3.0) * 0.2);
      size = vec2(0.028, 0.036);
    }
    if (uStyle == 2) {
      center = 0.5 + 0.32 * vec2(cos(angle + t * 0.08), sin(angle + t * 0.08));
      size = vec2(0.038);
    }
    if (uStyle == 3) {
      center = vec2(0.07 + f * 0.15, 0.16 + 0.055 * sin(t * 0.5 + f));
      size = vec2(0.05, 0.025);
    }
    if (uStyle == 4) {
      angle += t * 0.35;
      center = 0.5 + (0.23 + 0.07 * sin(t * 0.4 + f)) * vec2(cos(angle), sin(angle));
      size = vec2(0.018);
    }
    vec2 d = (p - center) / (size * uWidth);
    float amount = exp(-dot(d, d) * 0.5) * (0.75 + 0.25 * sin(t * 1.7 + f * 2.3));
    gas += vec4(palette(f / 6.0 + seed + t * 0.012), 1) * amount;
  }
  return gas * uEmission;
}

void main() {
  float t = uTime * uSpeed;
  if (uPass == 2) {
    // Small emitter puffs appear immediately; the rest of the domain is clear.
    outColor = source(gl_FragCoord.xy / uGridSize, t) * 0.8;
    return;
  }
  if (uPass == 0) {
    vec2 p = gl_FragCoord.xy / uGridSize;
    float dt = uSpeed / 60.0;
    float density = texture(uState, p).a;
    vec2 midpoint = p - 0.5 * dt * velocity(p, t, density);
    vec2 back = p - dt * velocity(midpoint, t, density);
    vec4 gas = sampleGas(back);
    vec2 pixel = 1.0 / uGridSize;
    vec4 neighbors = (sampleGas(back + vec2(pixel.x, 0)) + sampleGas(back - vec2(pixel.x, 0))
      + sampleGas(back + vec2(0, pixel.y)) + sampleGas(back - vec2(0, pixel.y))) * 0.25;
    gas = mix(gas, neighbors, 0.025 * min(uSpeed, 1.0));
    gas *= pow(uPersistence, uSpeed);
    gas += source(p, t) * (0.065 * uSpeed);
    outColor = clamp(gas, 0.0, 6.0);
    return;
  }

  // Fill the screen; rendering and transport use the same normalized domain.
  vec2 uv = gl_FragCoord.xy / uResolution;
  vec4 gas = sampleGas(uv);
  vec3 pigment = gas.rgb / max(gas.a, 0.0001);
  vec2 pixel = 1.5 / uGridSize;
  float dx = sampleGas(uv + vec2(pixel.x, 0)).a - sampleGas(uv - vec2(pixel.x, 0)).a;
  float dy = sampleGas(uv + vec2(0, pixel.y)).a - sampleGas(uv - vec2(0, pixel.y)).a;
  vec3 normal = normalize(vec3(-dx * 5.0, -dy * 5.0, 0.65));
  float light = 0.55 + 0.65 * max(dot(normal, normalize(vec3(-0.5, 0.7, 1))), 0.0);
  float opacity = 1.0 - exp(-gas.a * uDensity * 1.7);
  float wisps = pow(max(gas.a, 0.0), 0.55) * exp(-gas.a * 0.8);
  vec3 color = pigment * (opacity * light * 1.7 + wisps * uGlow * 0.65);
  vec3 background = vec3(0.006, 0.009, 0.022);
  color += background * (1.0 - opacity);
  color = 1.0 - exp(-color * 1.4);
  outColor = vec4(pow(color, vec3(0.85)), 1);
}
