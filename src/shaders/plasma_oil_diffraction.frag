#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uFilmThickness;
uniform float uDiffraction;
uniform float uFluidWarp;
uniform float uDropletScale;
uniform float uPlasmaDensity;
uniform float uDischargeSpeed;
uniform float uSpectralContrast;
uniform float uGlow;
uniform float uHueShift;
uniform float uSeed;

const float TAU = 6.28318530718;

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

float hash21(vec2 point) {
  point += fract(uSeed * 0.000001) * 17.73;
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

vec2 hash22(vec2 point) {
  float value = hash21(point);
  return vec2(value, hash21(point + value + 19.19));
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
  for (int octave = 0; octave < 6; octave++) {
    value += amplitude * noise(point);
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
      vec2 random = hash22(base + offset);
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
  vec3 wavelengths = vec3(1.0, 1.29, 1.63);
  vec3 phase = TAU * (thickness * wavelengths * uDiffraction + uHueShift * vec3(1.0, 0.83, 1.17));
  vec3 reflected = 0.5 + 0.5 * cos(phase + vec3(0.0, 0.45, 0.9));
  reflected = smoothstep(vec3(0.08), vec3(0.92), reflected);
  return pow(max(reflected, 0.0), vec3(max(contrast, 0.1)));
}

void main() {
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  point *= uZoom;
  float time = uTime * uTimeScale;

  vec2 drift = vec2(time * 0.055, -time * 0.041);
  vec2 warpA = vec2(
    fbm(point * 1.15 + drift),
    fbm(point * 1.15 + vec2(5.4, -3.1) - drift.yx)
  ) - 0.5;
  vec2 warpB = vec2(
    fbm(point * 2.05 + warpA * 1.7 - drift * 0.6),
    fbm(point * 2.05 - warpA.yx * 1.5 + drift * 0.45 + 8.2)
  ) - 0.5;
  vec2 fluidPoint = point + (warpA * 0.75 + warpB * 0.35) * uFluidWarp;

  float cellDistance;
  float cellGap = cellularEdge(fluidPoint * uDropletScale, time, cellDistance);
  float oilBoundary = exp(-cellGap * 38.0);
  float dropletInterior = smoothstep(0.62, 0.08, cellDistance);

  float broadFilm = fbm(fluidPoint * 1.7 - drift * 0.8);
  float fineFilm = fbm(fluidPoint * 4.6 + warpB * 2.1 + drift);
  float radialFilm = sin(length(fluidPoint + warpA * 0.3) * 8.0 - time * 0.36) * 0.5 + 0.5;
  float thickness = uFilmThickness * (
    broadFilm * 2.8
      + fineFilm * 0.85
      + radialFilm * 0.4
      + cellDistance * 1.25
  );

  vec3 diffractionColor = thinFilm(thickness, uSpectralContrast);
  vec3 shiftedFilm = thinFilm(thickness + cellGap * 2.5 + 0.11, uSpectralContrast * 0.85);
  float diffractionRings = pow(
    1.0 - abs(sin((thickness + fineFilm * 0.3) * uDiffraction * TAU)),
    7.0
  );

  float gradientField = fbm(fluidPoint * 3.2 + warpA * 2.0 - time * 0.09);
  float plasmaPhase = gradientField * uPlasmaDensity * 8.0
    + broadFilm * 5.0
    + cellGap * 7.0
    - time * uDischargeSpeed;
  float plasma = pow(1.0 - abs(sin(plasmaPhase)), 12.0);
  float boundaryPlasma = oilBoundary * pow(
    1.0 - abs(sin(thickness * 4.0 - time * uDischargeSpeed * 1.3)),
    5.0
  );
  float sparks = pow(
    1.0 - abs(sin((fluidPoint.x - fluidPoint.y) * 11.0 + warpB.x * 9.0 - time * 1.7)),
    18.0
  ) * plasma;

  vec3 color = vec3(0.004, 0.006, 0.012);
  color += diffractionColor * (0.13 + dropletInterior * 0.24);
  color += shiftedFilm * oilBoundary * 0.42;
  color += diffractionColor * diffractionRings * (0.2 + uGlow * 0.18);
  color += mix(diffractionColor, vec3(0.72, 0.88, 1.0), 0.62) * plasma * uGlow * 0.92;
  color += mix(shiftedFilm, vec3(1.0), 0.7) * boundaryPlasma * uGlow * 1.25;
  color += vec3(0.72, 0.86, 1.0) * sparks * uGlow * 1.4;

  float oilySpecular = pow(max(0.0, 1.0 - length(warpB) * 1.35), 6.0);
  color += shiftedFilm * oilySpecular * dropletInterior * 0.24;
  color *= 1.0 - smoothstep(0.65, 1.7, length(point)) * 0.34;
  color = 1.0 - exp(-color * 1.42);
  color = pow(max(color, 0.0), vec3(0.84));
  outColor = vec4(color, 1.0);
}