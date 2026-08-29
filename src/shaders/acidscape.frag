#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeaHeight;
uniform float uSeaChoppy;
uniform float uSeaSpeed;
uniform float uSeaFreq;
uniform float uCamHeight;
uniform float uCamDistance;
uniform float uCamYaw;
uniform float uCamPitch;
uniform float uSkyBoost;
uniform float uWaterBrightness;
uniform float uFilmThickness;
uniform float uDiffraction;
uniform float uOilScale;
uniform float uOilWarp;
uniform float uPlasmaDensity;
uniform float uDischargeSpeed;
uniform float uSpectralContrast;
uniform float uPlasmaGlow;
uniform float uAcidSky;
uniform float uHueShift;
uniform float uSeed;

const int NUM_STEPS = 32;
const int ITER_GEOMETRY = 3;
const int ITER_FRAGMENT = 5;
const float PI = 3.141592;
const float TAU = 6.28318530718;
const float EPSILON = 1e-3;
#define EPSILON_NRM (0.1 / uResolution.x)
const mat2 octave_m = mat2(1.6, 1.2, -1.2, 1.6);

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

mat3 fromEuler(vec3 angle) {
  vec2 a1 = vec2(sin(angle.x), cos(angle.x));
  vec2 a2 = vec2(sin(angle.y), cos(angle.y));
  vec2 a3 = vec2(sin(angle.z), cos(angle.z));
  mat3 matrix;
  matrix[0] = vec3(a1.y * a3.y + a1.x * a2.x * a3.x, a1.y * a2.x * a3.x + a3.y * a1.x, -a2.y * a3.x);
  matrix[1] = vec3(-a2.y * a1.x, a1.y * a2.y, a2.x);
  matrix[2] = vec3(a3.y * a1.x * a2.x + a1.y * a3.x, a1.x * a3.x - a1.y * a3.y * a2.x, a2.y * a3.y);
  return matrix;
}

float hash(vec2 point) {
  float value = dot(point + fract(uSeed * 0.000001) * 19.17, vec2(127.1, 311.7));
  return fract(sin(value) * 43758.5453123);
}

float signedNoise(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  vec2 curve = local * local * (3.0 - 2.0 * local);
  return -1.0 + 2.0 * mix(
    mix(hash(cell), hash(cell + vec2(1.0, 0.0)), curve.x),
    mix(hash(cell + vec2(0.0, 1.0)), hash(cell + vec2(1.0)), curve.x),
    curve.y
  );
}

float fbm(vec2 point) {
  float value = 0.0;
  float amplitude = 0.52;
  for (int octave = 0; octave < 5; octave++) {
    value += amplitude * signedNoise(point);
    point = rotate2d(0.57) * point * 2.03 + vec2(1.7, -2.4);
    amplitude *= 0.49;
  }
  return value;
}

float diffuse(vec3 normal, vec3 light, float power) {
  return pow(dot(normal, light) * 0.4 + 0.6, power);
}

float specular(vec3 normal, vec3 light, vec3 eye, float shininess) {
  float normalization = (shininess + 8.0) / (PI * 8.0);
  return pow(max(dot(reflect(eye, normal), light), 0.0), shininess) * normalization;
}

vec3 spectralPalette(float phase) {
  return 0.5 + 0.5 * cos(TAU * (
    phase * vec3(1.0, 1.29, 1.63)
      + vec3(0.02, 0.31, 0.68)
      + uHueShift * vec3(1.0, 0.83, 1.17)
  ));
}

vec3 getSkyColor(vec3 eye) {
  eye.y = (max(eye.y, 0.0) * 0.8 + 0.2) * 0.8;
  float horizon = 1.0 - eye.y;
  vec3 naturalSky = vec3(horizon * horizon, horizon, 0.6 + horizon * 0.4);
  vec3 acidSky = mix(
    spectralPalette(eye.y * 0.35 + 0.08),
    spectralPalette(0.62 - eye.y * 0.22),
    horizon
  ) * vec3(0.62, 0.72, 0.82);
  return mix(naturalSky, acidSky, uAcidSky) * uSkyBoost;
}

float seaOctave(vec2 uv, float choppy) {
  uv += signedNoise(uv);
  vec2 wave = 1.0 - abs(sin(uv));
  vec2 smoothWave = abs(cos(uv));
  wave = mix(wave, smoothWave, wave);
  return pow(1.0 - pow(wave.x * wave.y, 0.65), choppy);
}

float map(vec3 point, float seaTime) {
  float frequency = uSeaFreq;
  float amplitude = uSeaHeight;
  float choppy = uSeaChoppy;
  vec2 uv = point.xz;
  uv.x *= 0.75;
  float height = 0.0;

  for (int octave = 0; octave < ITER_GEOMETRY; octave++) {
    float wave = seaOctave((uv + seaTime) * frequency, choppy);
    wave += seaOctave((uv - seaTime) * frequency, choppy);
    height += wave * amplitude;
    uv *= octave_m;
    frequency *= 1.9;
    amplitude *= 0.22;
    choppy = mix(choppy, 1.0, 0.2);
  }
  return point.y - height;
}

float mapDetailed(vec3 point, float seaTime) {
  float frequency = uSeaFreq;
  float amplitude = uSeaHeight;
  float choppy = uSeaChoppy;
  vec2 uv = point.xz;
  uv.x *= 0.75;
  float height = 0.0;

  for (int octave = 0; octave < ITER_FRAGMENT; octave++) {
    float wave = seaOctave((uv + seaTime) * frequency, choppy);
    wave += seaOctave((uv - seaTime) * frequency, choppy);
    height += wave * amplitude;
    uv *= octave_m;
    frequency *= 1.9;
    amplitude *= 0.22;
    choppy = mix(choppy, 1.0, 0.2);
  }
  return point.y - height;
}

vec3 getAcidSeaColor(
  vec3 point,
  vec3 normal,
  vec3 light,
  vec3 eye,
  vec3 distance,
  float time
) {
  float fresnel = clamp(1.0 - dot(normal, -eye), 0.0, 1.0);
  fresnel = min(fresnel * fresnel * fresnel, 0.58);

  vec2 oilPoint = point.xz * uOilScale;
  vec2 drift = vec2(time * 0.045, -time * 0.037);
  vec2 warpA = vec2(
    fbm(oilPoint * 0.72 + drift),
    fbm(oilPoint * 0.72 + vec2(5.2, -3.4) - drift.yx)
  );
  vec2 warpB = vec2(
    fbm(oilPoint * 1.65 + warpA * 1.4 - drift * 0.6),
    fbm(oilPoint * 1.65 - warpA.yx * 1.3 + drift * 0.5 + 8.7)
  );
  vec2 fluidPoint = oilPoint + (warpA * 0.72 + warpB * 0.28) * uOilWarp;

  float broadFilm = fbm(fluidPoint * 0.85 - drift * 0.7);
  float fineFilm = fbm(fluidPoint * 2.6 + warpB * 1.8 + drift);
  float crestFilm = dot(normal.xz, vec2(0.7, -0.5)) * 0.65;
  float thickness = uFilmThickness * (
    broadFilm * 2.5 + fineFilm * 0.72 + crestFilm + point.y * 0.22
  );

  vec3 film = spectralPalette(thickness * uDiffraction);
  film = smoothstep(vec3(0.06), vec3(0.94), film);
  film = pow(max(film, 0.0), vec3(max(uSpectralContrast, 0.1)));
  vec3 shiftedFilm = spectralPalette((thickness + fineFilm * 0.34) * uDiffraction + 0.17);

  float diffractionRidge = pow(
    1.0 - abs(sin((thickness + broadFilm * 0.28) * uDiffraction * TAU)),
    8.0
  );
  float plasmaPhase = fineFilm * uPlasmaDensity * 7.5
    + broadFilm * 4.0
    + dot(normal.xz, vec2(2.7, -2.1))
    - time * uDischargeSpeed;
  float plasmaVein = pow(1.0 - abs(sin(plasmaPhase)), 13.0);
  float crest = pow(max(0.0, 1.0 - normal.y), 2.2);
  float travelingCharge = pow(
    1.0 - abs(sin(point.x * 0.11 + point.z * 0.08 - time * uDischargeSpeed * 1.6)),
    16.0
  );

  vec3 reflected = getSkyColor(reflect(eye, normal));
  vec3 acidBase = film * (0.14 + 0.26 * uWaterBrightness);
  acidBase += shiftedFilm * diffractionRidge * (0.16 + 0.18 * uWaterBrightness);
  acidBase += diffuse(normal, light, 70.0) * film * 0.09;
  vec3 color = mix(acidBase, reflected * (0.7 + film * 0.45), fresnel);

  float attenuation = max(1.0 - dot(distance, distance) * 0.001, 0.0);
  color += film * (point.y - uSeaHeight) * 0.12 * attenuation;
  color += mix(film, vec3(0.78, 0.9, 1.0), 0.68)
    * plasmaVein * uPlasmaGlow * (0.34 + crest * 0.9);
  color += shiftedFilm * travelingCharge * plasmaVein * uPlasmaGlow * 0.75;
  color += vec3(0.92, 0.98, 1.0)
    * specular(normal, light, eye, 520.0 * inversesqrt(dot(distance, distance)));
  return color;
}

vec3 getNormal(vec3 point, float epsilon, float seaTime) {
  vec3 normal;
  normal.y = mapDetailed(point, seaTime);
  normal.x = mapDetailed(vec3(point.x + epsilon, point.y, point.z), seaTime) - normal.y;
  normal.z = mapDetailed(vec3(point.x, point.y, point.z + epsilon), seaTime) - normal.y;
  normal.y = epsilon;
  return normalize(normal);
}

float heightMapTracing(vec3 origin, vec3 direction, out vec3 point, float seaTime) {
  float nearDistance = 0.0;
  float farDistance = 1000.0;
  float farHeight = map(origin + direction * farDistance, seaTime);
  if (farHeight > 0.0) {
    point = origin + direction * farDistance;
    return farDistance;
  }
  float nearHeight = map(origin, seaTime);
  for (int stepIndex = 0; stepIndex < NUM_STEPS; stepIndex++) {
    float midpoint = mix(nearDistance, farDistance, nearHeight / (nearHeight - farHeight));
    point = origin + direction * midpoint;
    float midpointHeight = map(point, seaTime);
    if (midpointHeight < 0.0) {
      farDistance = midpoint;
      farHeight = midpointHeight;
    } else {
      nearDistance = midpoint;
      nearHeight = midpointHeight;
    }
    if (abs(midpointHeight) < EPSILON) {
      break;
    }
  }
  return mix(nearDistance, farDistance, nearHeight / (nearHeight - farHeight));
}

vec3 getPixel(vec2 coordinate, float time, float seaTime) {
  vec2 uv = coordinate / uResolution.xy;
  uv = uv * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;

  vec3 angle = vec3(sin(time * 3.0) * 0.1 + uCamPitch, sin(time) * 0.2 + 0.3, time + uCamYaw);
  vec3 origin = vec3(0.0, uCamHeight, time * uCamDistance);
  vec3 direction = normalize(vec3(uv.xy, -2.0));
  direction.z += length(uv) * 0.14;
  direction = normalize(direction) * fromEuler(angle);

  vec3 point;
  heightMapTracing(origin, direction, point, seaTime);
  vec3 distance = point - origin;
  vec3 normal = getNormal(point, dot(distance, distance) * EPSILON_NRM, seaTime);
  vec3 light = normalize(vec3(0.0, 1.0, 0.8));

  return mix(
    getSkyColor(direction),
    getAcidSeaColor(point, normal, light, direction, distance, time),
    pow(smoothstep(0.0, -0.02, direction.y), 0.2)
  );
}

void main() {
  float time = uTime * uTimeScale;
  float seaTime = 1.0 + time * uSeaSpeed;
  vec3 color = getPixel(gl_FragCoord.xy, time, seaTime);
  outColor = vec4(pow(max(color, 0.0), vec3(0.68)), 1.0);
}