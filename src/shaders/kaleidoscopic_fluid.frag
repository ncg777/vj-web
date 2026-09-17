#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;
uniform vec2 uResolution;
uniform vec2 uGridSize;
uniform sampler2D uState;
uniform int uPass;
uniform int uStyle;
uniform int uSymmetry;
uniform int uDetail;
uniform float uTime;
uniform float uSpeed;
uniform float uTurbulence;
uniform float uScale;
uniform float uTwist;
uniform float uPersistence;
uniform float uInjection;
uniform float uHue;
uniform float uGlow;
uniform float uSeed;

const float TAU = 6.28318530718;
mat2 rotate(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

vec3 palette(float h) {
  return 0.52 + 0.48 * cos(TAU * (h + vec3(0.0, 0.33, 0.67)));
}

// A periodic multiscale stream function. Its perpendicular gradient gives
// divergence-free stirring without a separate pressure texture/solver.
float stream(vec2 p, float t) {
  float field = 0.0;
  float amplitude = 1.0;
  float frequency = 1.0;
  float seed = mod(uSeed, 997.0) * 0.031;
  for (int i = 0; i < 5; ++i) {
    float f = float(i);
    vec2 q = TAU * p * frequency;
    field += amplitude * sin(q.x + sin(t * 0.31 + f + seed))
      * cos(q.y + cos(t * 0.27 - f + seed));
    frequency *= 2.0;
    amplitude *= 0.32;
  }
  if (uStyle == 1) field += 0.7 * sin(TAU * p.y + t * 0.4);
  if (uStyle == 2) field += 0.45 * sin(TAU * p.x * 3.0) * sin(TAU * p.y * 3.0);
  return field;
}

vec2 velocity(vec2 p, float t) {
  float e = 0.002;
  vec2 gradient = vec2(stream(p + vec2(e, 0), t) - stream(p - vec2(e, 0), t),
    stream(p + vec2(0, e), t) - stream(p - vec2(0, e), t)) / (2.0 * e);
  vec2 flow = vec2(gradient.y, -gradient.x) * (0.0015 * uTurbulence);
  if (uStyle == 0 || uStyle == 3) {
    vec2 d = p - 0.5;
    flow += vec2(-d.y, d.x) * (uStyle == 3 ? 0.08 : 0.035)
      * exp(-dot(d, d) * 5.0);
  }
  return flow;
}

// Float targets use nearest filtering; interpolate explicitly for smooth
// semi-Lagrangian dye transport and full-resolution display.
vec4 sampleDye(vec2 uv) {
  vec2 p = uv * uGridSize - 0.5;
  vec2 base = floor(p);
  vec2 f = fract(p);
  vec2 texel = 1.0 / uGridSize;
  vec2 a = (base + 0.5) * texel;
  return mix(mix(texture(uState, a), texture(uState, a + vec2(texel.x, 0)), f.x),
    mix(texture(uState, a + vec2(0, texel.y)), texture(uState, a + texel), f.x), f.y);
}

vec4 source(vec2 p, float t) {
  vec4 dye = vec4(0);
  float seed = mod(uSeed, 997.0) * 0.017;
  for (int i = 0; i < 6; ++i) {
    float f = float(i);
    float angle = TAU * f / 6.0 + t * 0.19 + seed;
    float radius = 0.25 + 0.08 * sin(t * 0.37 + f * 2.0);
    vec2 center = 0.5 + radius * vec2(cos(angle), sin(angle));
    if (uStyle == 1) center = vec2(fract(f / 6.0 + t * 0.025), 0.5 + 0.28 * sin(f + t * 0.3));
    vec2 d = p - center;
    d -= floor(d + 0.5);
    float ink = exp(-dot(d, d) / (uStyle == 2 ? 0.0018 : 0.0035));
    dye += vec4(palette(f / 6.0 + t * 0.025 + seed), 1.0) * ink;
  }
  return dye;
}

void main() {
  float t = uTime * uSpeed;
  if (uPass == 2) {
    vec2 p = gl_FragCoord.xy / uGridSize;
    vec3 base = palette(p.x + p.y + mod(uSeed, 997.0) * 0.01);
    outColor = vec4(base * 0.18, 0.18) + source(p, t) * 0.7;
    return;
  }
  if (uPass == 0) {
    vec2 p = gl_FragCoord.xy / uGridSize;
    // Fixed simulation step matches the application's one update per frame.
    float dt = uSpeed / 60.0;
    vec2 v = velocity(p, t);
    vec2 midpoint = p - 0.5 * dt * v;
    vec2 back = p - dt * velocity(midpoint, t);
    vec4 dye = sampleDye(back);
    vec2 pixel = 1.0 / uGridSize;
    vec4 neighbors = (sampleDye(back + vec2(pixel.x, 0)) + sampleDye(back - vec2(pixel.x, 0))
      + sampleDye(back + vec2(0, pixel.y)) + sampleDye(back - vec2(0, pixel.y))) * 0.25;
    dye = mix(dye, neighbors, 0.035 * min(uSpeed, 1.0));
    dye *= pow(uPersistence, uSpeed);
    dye += source(p, t) * (0.045 * uInjection * uSpeed);
    outColor = clamp(dye, 0.0, 4.0);
    return;
  }

  vec2 p = (2.0 * gl_FragCoord.xy - uResolution) / min(uResolution.x, uResolution.y);
  float radius = length(p);
  float angle = atan(p.y, p.x) + t * 0.045 + uTwist * radius;
  float wedge = TAU / float(uSymmetry);
  angle = abs(mod(angle + 0.5 * wedge, wedge) - 0.5 * wedge);
  vec2 q = radius * vec2(cos(angle), sin(angle)) * uScale;
  // Recursive folds make nested fluid petals while retaining mirror symmetry.
  float weight = 0.3;
  for (int i = 0; i < 5; ++i) {
    if (i >= uDetail) break;
    q += weight * sin(rotate(float(i) * 0.73) * q * 3.1 + vec2(t * 0.08, -t * 0.06));
    q = abs(q) - vec2(0.28, 0.19);
    weight *= 0.55;
  }
  vec4 dye = sampleDye(q * 0.37 + 0.5);
  vec3 pigment = dye.rgb / max(dye.a, 0.08);
  // Rotate RGB around its neutral axis for a continuous hue control.
  vec3 axis = normalize(vec3(1));
  float hue = TAU * uHue;
  pigment = pigment * cos(hue) + cross(axis, pigment) * sin(hue)
    + axis * dot(axis, pigment) * (1.0 - cos(hue));
  pigment = max(pigment, 0.0);
  float density = 1.0 - exp(-dye.a * 2.3);
  vec3 color = pigment * (0.2 + 1.2 * density);
  float veins = pow(0.5 + 0.5 * sin(dye.a * 22.0 + dot(dye.rgb, vec3(8, 13, 21))), 12.0);
  color += uGlow * veins * pigment * density * 0.4;
  color = 1.0 - exp(-color * 1.7);
  outColor = vec4(pow(color, vec3(0.85)), 1.0);
}
