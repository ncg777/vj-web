#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeed;

uniform float uSeaHeight;
uniform float uSeaChoppy;
uniform float uSeaSpeed;
uniform float uSeaFreq;
uniform float uCamHeight;
uniform float uCamDistance;
uniform float uCamYaw;
uniform float uCamPitch;
uniform float uReflection;

uniform float uWaterScale;
uniform float uWaterSpeed;
uniform int uFractalIters;
uniform float uFoldScale;
uniform float uFoldOffset;
uniform float uRotSpeed;
uniform float uDetailLevel;
uniform float uSmoothBlend;
uniform float uWaterHue;
uniform float uHueSpeed;
uniform float uSaturation;
uniform float uWaterBrightness;
uniform float uWaterContrast;
uniform float uWaterGlow;

uniform float uSkyScale;
uniform float uSkySpeed;
uniform float uSkyBoost;
uniform float uFilmThickness;
uniform float uDiffraction;
uniform float uFluidWarp;
uniform float uDropletScale;
uniform float uPlasmaDensity;
uniform float uDischargeSpeed;
uniform float uSpectralContrast;
uniform float uPlasmaGlow;
uniform float uSkyHue;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;
const mat2 OCTAVE_MATRIX = mat2(1.6, 1.2, -1.2, 1.6);

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

float seaHash(vec2 point) {
  float value = dot(point + fract(uSeed * 0.000001) * 19.17, vec2(127.1, 311.7));
  return fract(sin(value) * 43758.5453123);
}

float seaNoise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  vec2 curve = local * local * (3.0 - 2.0 * local);
  return -1.0 + 2.0 * mix(
    mix(seaHash(cell), seaHash(cell + vec2(1.0, 0.0)), curve.x),
    mix(seaHash(cell + vec2(0.0, 1.0)), seaHash(cell + vec2(1.0)), curve.x),
    curve.y
  );
}

float seaOctave(vec2 point, float choppy) {
  point += seaNoise(point);
  vec2 wave = 1.0 - abs(sin(point));
  wave = mix(wave, abs(cos(point)), wave);
  return pow(max(1.0 - pow(wave.x * wave.y, 0.65), 0.0), choppy);
}

// The same multiscale wave surface used by Seascape and Acidscape.
float seaMap(vec3 point, float seaTime, bool detailed) {
  vec2 uv = point.xz * vec2(0.75, 1.0);
  float frequency = uSeaFreq;
  float amplitude = uSeaHeight;
  float choppy = uSeaChoppy;
  float height = 0.0;
  for (int octave = 0; octave < 5; octave++) {
    if (!detailed && octave >= 3) break;
    float wave = seaOctave((uv + seaTime) * frequency, choppy);
    wave += seaOctave((uv - seaTime) * frequency, choppy);
    height += wave * amplitude;
    uv *= OCTAVE_MATRIX;
    frequency *= 1.9;
    amplitude *= 0.22;
    choppy = mix(choppy, 1.0, 0.2);
  }
  return point.y - height;
}

bool traceSea(vec3 origin, vec3 direction, float seaTime, out vec3 point) {
  float nearDistance = 0.0;
  float farDistance = 450.0;
  point = origin + direction * farDistance;
  float nearHeight = seaMap(origin, seaTime, false);
  float farHeight = seaMap(point, seaTime, false);
  if (farHeight > 0.0) return false;

  for (int stepIndex = 0; stepIndex < 32; stepIndex++) {
    float fraction = nearHeight / max(nearHeight - farHeight, 0.00001);
    float midpoint = mix(nearDistance, farDistance, fraction);
    point = origin + direction * midpoint;
    float midpointHeight = seaMap(point, seaTime, false);
    if (abs(midpointHeight) < 0.001) return true;
    if (midpointHeight < 0.0) {
      farDistance = midpoint;
      farHeight = midpointHeight;
    } else {
      nearDistance = midpoint;
      nearHeight = midpointHeight;
    }
  }
  return true;
}

vec3 seaNormal(vec3 point, float distanceSquared, float seaTime) {
  float epsilon = clamp(distanceSquared * 0.1 / uResolution.x, 0.002, 35.0);
  float height = seaMap(point, seaTime, true);
  return normalize(vec3(
    seaMap(point + vec3(epsilon, 0.0, 0.0), seaTime, true) - height,
    epsilon,
    seaMap(point + vec3(0.0, 0.0, epsilon), seaTime, true) - height
  ));
}

// Fractal Reverie's noise, nested domain warps, and four color palettes.
float reverieHash(vec2 point) {
  point = fract(point * vec2(234.34, 435.45));
  point += dot(point, point + 34.23 + fract(uSeed * 0.000001));
  return fract(point.x * point.y);
}

float reverieNoise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  vec2 curve = local * local * local * (local * (local * 6.0 - 15.0) + 10.0);
  return mix(
    mix(reverieHash(cell), reverieHash(cell + vec2(1.0, 0.0)), curve.x),
    mix(reverieHash(cell + vec2(0.0, 1.0)), reverieHash(cell + vec2(1.0)), curve.x),
    curve.y
  );
}

float reverieFbm(vec2 point, float footprint) {
  float value = 0.0;
  float amplitude = 0.55;
  float normalization = 0.0;
  int octaves = clamp(uFractalIters + 3, 3, 8);
  for (int octave = 0; octave < 8; octave++) {
    if (octave >= octaves) break;
    // Blend unresolved octaves to their mean instead of aliasing near the horizon.
    float detail = 1.0 - smoothstep(0.25, 0.8, footprint);
    value += amplitude * mix(0.5, reverieNoise(point), detail);
    normalization += amplitude;
    point = rotate2d(0.45 + 0.03 * float(octave)) * point * 2.02 + vec2(0.31, -0.27);
    footprint *= 2.02;
    amplitude *= mix(0.46, 0.60, uSmoothBlend);
  }
  return value / max(normalization, 0.0001);
}

vec3 palNeon(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(1.0, 0.8, 0.6) + vec3(0.00, 0.33, 0.67)));
}

vec3 palLava(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.7, 0.9, 1.3) + vec3(0.00, 0.18, 0.55)));
}

vec3 palFire(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(1.3, 0.5, 0.8) + vec3(0.25, 0.00, 0.55)));
}

vec3 palIce(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.6, 0.9, 1.2) + vec3(0.55, 0.70, 0.85)));
}

vec2 reverieWarp(vec2 point, float time, float footprint) {
  float drift = time * (0.05 + uRotSpeed * 0.12);
  float scale = 0.75 + uFoldScale * 0.35;
  vec2 q = vec2(
    reverieFbm(point * scale + vec2(0.0, drift), footprint * scale),
    reverieFbm(point * scale + vec2(4.8, -drift * 0.8), footprint * scale)
  );
  vec2 r = vec2(
    reverieFbm(point * (scale + 0.6) + (q - 0.5) * 2.0 + vec2(1.7, -2.6) + drift * 0.4, footprint * (scale + 0.6)),
    reverieFbm(point * (scale + 0.4) + (q - 0.5) * 2.0 + vec2(-3.1, 0.9) - drift * 0.3, footprint * (scale + 0.4))
  );
  return point + (q + r - 1.0) * (0.22 + uFoldOffset * 0.25);
}

vec3 reverieWater(vec3 point, vec3 normal, float time, float footprint) {
  vec2 uv = point.xz * uWaterScale;
  vec2 p = uv * 2.174 + vec2(time * 0.0108, 0.066);
  // A small normal offset lets the color fields bend over the wave crests.
  p += normal.xz * 0.12;
  vec2 q = reverieWarp(p, time, footprint);
  vec2 r = reverieWarp(q * 1.35 + vec2(2.4, -1.8), time * 0.7, footprint * 1.35);
  float field = reverieFbm(q, footprint);
  float filaments = reverieFbm(r * 1.7 + vec2(time * 0.03, -time * 0.025), footprint * 2.295);
  float mist = reverieFbm(p * 0.65 - vec2(time * 0.02, time * 0.015), footprint * 0.65);
  float ridgeScale = 1.4 + 0.2 * uDetailLevel;
  float ridges = 1.0 - abs(2.0 * reverieFbm(q * ridgeScale, footprint * ridgeScale) - 1.0);
  float glowMask = smoothstep(0.30, 0.88, mix(field, ridges, 0.55));
  float veil = smoothstep(0.18, 0.82, mix(mist, filaments, 0.4));
  float colorTime = uWaterHue + time * uHueSpeed * 0.25 + fract(uSeed * 0.000001);
  float horizon = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
  // Bounded radial variation avoids rapid palette flicker in distant water.
  float radial = length(uv) / (1.0 + length(uv));
  vec3 cool = mix(
    palIce(colorTime + field * 0.45 + mist * 0.2),
    palNeon(colorTime + filaments * 0.35 + horizon * 0.15),
    0.45 + 0.25 * horizon
  );
  vec3 warm = mix(
    palLava(colorTime + ridges * 0.50 + 0.08 * radial),
    palFire(colorTime + glowMask * 0.65 + filaments * 0.15),
    glowMask
  );
  vec3 color = mix(cool * (0.22 + 0.18 * horizon), warm, veil);
  color += palFire(colorTime + filaments + radial * 0.1) * glowMask * glowMask
    * (0.10 + 0.05 * uWaterGlow);
  color += palNeon(colorTime + mist * 0.6 + 0.12 * radial)
    * pow(max(1.0 - radial * 0.75, 0.0), 2.0) * 0.095;
  float haze = smoothstep(0.25, 1.35, radial + (1.0 - horizon) * 0.35);
  color = mix(color, cool * 0.35, haze * 0.35);
  color *= 0.75 + 0.25 * uWaterBrightness;
  float luma = dot(color, vec3(0.299, 0.587, 0.114));
  color = mix(vec3(luma), color, uSaturation);
  color = max(mix(vec3(0.5), color, uWaterContrast), 0.0);
  return color / (color + 0.35);
}

// Plasma Oil Diffraction projected onto a sky dome, also sampled by reflections.
float oilHash(vec2 point) {
  point += fract(uSeed * 0.000001) * 17.73;
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

vec2 oilHash22(vec2 point) {
  float value = oilHash(point);
  return vec2(value, oilHash(point + value + 19.19));
}

float oilNoise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  local = local * local * (3.0 - 2.0 * local);
  return mix(
    mix(oilHash(cell), oilHash(cell + vec2(1.0, 0.0)), local.x),
    mix(oilHash(cell + vec2(0.0, 1.0)), oilHash(cell + vec2(1.0)), local.x),
    local.y
  );
}

float oilFbm(vec2 point) {
  float value = 0.0;
  float amplitude = 0.52;
  for (int octave = 0; octave < 6; octave++) {
    value += amplitude * oilNoise(point);
    point = rotate2d(0.58) * point * 2.03 + vec2(1.7, -2.4);
    amplitude *= 0.49;
  }
  return value;
}

float cellularEdge(vec2 point, float time, out float cellDistance) {
  vec2 base = floor(point);
  vec2 local = fract(point);
  float nearest = 10.0;
  float secondNearest = 10.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 offset = vec2(float(x), float(y));
      vec2 random = oilHash22(base + offset);
      vec2 center = offset + 0.5
        + 0.34 * sin(time * vec2(0.23, 0.19) + random * TAU) - local;
      float distanceSquared = dot(center, center);
      if (distanceSquared < nearest) {
        secondNearest = nearest;
        nearest = distanceSquared;
      } else if (distanceSquared < secondNearest) {
        secondNearest = distanceSquared;
      }
    }
  }
  cellDistance = sqrt(nearest);
  return sqrt(secondNearest) - sqrt(nearest);
}

vec3 thinFilm(float thickness, float contrast) {
  vec3 phase = TAU * (thickness * vec3(1.0, 1.29, 1.63) * uDiffraction
    + uSkyHue * vec3(1.0, 0.83, 1.17));
  vec3 reflected = 0.5 + 0.5 * cos(phase + vec3(0.0, 0.45, 0.9));
  reflected = smoothstep(vec3(0.08), vec3(0.92), reflected);
  return pow(max(reflected, 0.0), vec3(max(contrast, 0.1)));
}

vec3 plasmaSky(vec3 direction, float time) {
  vec2 point = direction.xz / (0.65 + max(direction.y, 0.0));
  point *= uSkyScale;
  vec2 drift = vec2(time * 0.055, -time * 0.041);
  vec2 warpA = vec2(
    oilFbm(point * 1.15 + drift),
    oilFbm(point * 1.15 + vec2(5.4, -3.1) - drift.yx)
  ) - 0.5;
  vec2 warpB = vec2(
    oilFbm(point * 2.05 + warpA * 1.7 - drift * 0.6),
    oilFbm(point * 2.05 - warpA.yx * 1.5 + drift * 0.45 + 8.2)
  ) - 0.5;
  vec2 fluidPoint = point + (warpA * 0.75 + warpB * 0.35) * uFluidWarp;
  float cellDistance;
  float cellGap = cellularEdge(fluidPoint * uDropletScale, time, cellDistance);
  float oilBoundary = exp(-cellGap * 38.0);
  float dropletInterior = 1.0 - smoothstep(0.08, 0.62, cellDistance);
  float broadFilm = oilFbm(fluidPoint * 1.7 - drift * 0.8);
  float fineFilm = oilFbm(fluidPoint * 4.6 + warpB * 2.1 + drift);
  float radialFilm = sin(length(fluidPoint + warpA * 0.3) * 8.0 - time * 0.36) * 0.5 + 0.5;
  float thickness = uFilmThickness * (
    broadFilm * 2.8 + fineFilm * 0.85 + radialFilm * 0.4 + cellDistance * 1.25
  );
  vec3 diffractionColor = thinFilm(thickness, uSpectralContrast);
  vec3 shiftedFilm = thinFilm(thickness + cellGap * 2.5 + 0.11, uSpectralContrast * 0.85);
  float diffractionRings = pow(
    1.0 - abs(sin((thickness + fineFilm * 0.3) * uDiffraction * TAU)), 7.0
  );
  float gradientField = oilFbm(fluidPoint * 3.2 + warpA * 2.0 - time * 0.09);
  float plasmaPhase = gradientField * uPlasmaDensity * 8.0
    + broadFilm * 5.0 + cellGap * 7.0 - time * uDischargeSpeed;
  float plasma = pow(1.0 - abs(sin(plasmaPhase)), 12.0);
  float boundaryPlasma = oilBoundary * pow(
    1.0 - abs(sin(thickness * 4.0 - time * uDischargeSpeed * 1.3)), 5.0
  );
  float sparks = pow(
    1.0 - abs(sin((fluidPoint.x - fluidPoint.y) * 11.0 + warpB.x * 9.0 - time * 1.7)), 18.0
  ) * plasma;
  vec3 color = vec3(0.004, 0.006, 0.012);
  color += diffractionColor * (0.13 + dropletInterior * 0.24);
  color += shiftedFilm * oilBoundary * 0.42;
  color += diffractionColor * diffractionRings * (0.2 + uPlasmaGlow * 0.18);
  color += mix(diffractionColor, vec3(0.72, 0.88, 1.0), 0.62) * plasma * uPlasmaGlow * 0.92;
  color += mix(shiftedFilm, vec3(1.0), 0.7) * boundaryPlasma * uPlasmaGlow * 1.25;
  color += vec3(0.72, 0.86, 1.0) * sparks * uPlasmaGlow * 1.4;
  float oilySpecular = pow(max(0.0, 1.0 - length(warpB) * 1.35), 6.0);
  color += shiftedFilm * oilySpecular * dropletInterior * 0.24;
  return (1.0 - exp(-color * 1.42)) * uSkyBoost;
}

void main() {
  float time = uTime * uTimeScale;
  float seaTime = 1.0 + time * uSeaSpeed;
  float skyTime = time * uSkySpeed;
  vec2 uv = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  vec3 origin = vec3(0.0, max(uCamHeight, uSeaHeight * 2.6 + 0.2), time * uCamDistance);
  vec3 direction = normalize(vec3(uv, -2.0));
  direction.z += length(uv) * 0.14;
  direction.yz = rotate2d(uCamPitch + sin(time * 0.16) * 0.018) * direction.yz;
  direction.xy = rotate2d(sin(time * 0.11) * 0.012) * direction.xy;
  direction.xz = rotate2d(uCamYaw + time * 0.035) * direction.xz;
  direction = normalize(direction);
  vec3 sky = plasmaSky(direction, skyTime);
  vec3 color = sky;
  vec3 point;
  bool hitSea = traceSea(origin, direction, seaTime, point);
  // Evaluate derivatives before the branch so neighboring sky pixels stay defined.
  float footprint = max(length(dFdx(point.xz)), length(dFdy(point.xz))) * uWaterScale * 2.174;
  if (hitSea) {
    vec3 distance = point - origin;
    float distanceSquared = max(dot(distance, distance), 0.001);
    vec3 normal = seaNormal(point, distanceSquared, seaTime);
    vec3 light = normalize(vec3(0.0, 1.0, 0.8));
    vec3 water = reverieWater(point, normal, time * uWaterSpeed, footprint);
    water *= 0.55 + 0.45 * max(dot(normal, light), 0.0);
    float fresnel = pow(clamp(1.0 - dot(normal, -direction), 0.0, 1.0), 3.0);
    vec3 reflected = plasmaSky(reflect(direction, normal), skyTime);
    water = mix(water, reflected, fresnel * uReflection);
    float crest = pow(max(1.0 - normal.y, 0.0), 2.0);
    water += palNeon(uWaterHue + point.y * 0.2) * crest * uWaterGlow * 0.045;
    float shininess = 520.0 * inversesqrt(distanceSquared);
    float specular = pow(max(dot(reflect(direction, normal), light), 0.0), shininess);
    water += vec3(0.86, 0.94, 1.0) * specular * (shininess + 8.0) / (PI * 8.0) * 0.35;
    float haze = 1.0 - exp(-sqrt(distanceSquared) * 0.012);
    water = mix(water, sky, haze * 0.72);
    float horizonBlend = pow(1.0 - smoothstep(-0.025, 0.0, direction.y), 0.2);
    color = mix(sky, water, horizonBlend);
  }
  outColor = vec4(pow(max(color, 0.0), vec3(0.85)), 1.0);
}
