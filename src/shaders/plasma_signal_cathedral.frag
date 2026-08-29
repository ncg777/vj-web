#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform int uColumns;
uniform int uGlyphComplexity;
uniform float uProcession;
uniform float uPortalBend;
uniform float uSignalNoise;
uniform float uScanRate;
uniform float uGlow;
uniform float uHue;
uniform float uSeed;

const float TAU = 6.28318530718;

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

float hash21(vec2 point) {
  point += fract(uSeed * 0.000001) * 13.71;
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
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

vec3 palette(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(1.0, 0.73, 0.51) + vec3(0.0, 0.31, 0.64)));
}

void main() {
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  point *= uZoom;
  float time = uTime * uTimeScale;

  float columnCount = float(uColumns);
  vec2 gridPoint = point * columnCount;
  gridPoint.y += time * uProcession * 0.22;
  gridPoint.x += sin(gridPoint.y * 0.7 + time) * uPortalBend * 0.12;
  vec2 cellId = floor(gridPoint);
  vec2 local = fract(gridPoint) - 0.5;
  float random = hash21(cellId);
  float phase = random * TAU;

  local = rotate2d(sin(time * 0.37 + phase) * 0.65 + phase) * local;
  local += vec2(
    noise(cellId + time * 0.13),
    noise(cellId - time * 0.11)
  ) * uSignalNoise * 0.12 - uSignalNoise * 0.06;

  float radius = length(local);
  float angle = atan(local.y, local.x);
  float complexity = float(uGlyphComplexity);
  float runeRadius = 0.2 + 0.055 * sin(angle * complexity + time * (0.7 + random) + phase);
  float rune = exp(-abs(radius - runeRadius) * 85.0);
  float spokes = exp(-abs(sin(angle * complexity + phase)) * 24.0)
    * smoothstep(0.06, 0.17, radius)
    * (1.0 - smoothstep(0.17, 0.42, radius));
  float orbit = exp(-abs(radius - 0.34) * 95.0)
    * (0.4 + 0.6 * pow(1.0 - abs(sin(angle * 3.0 - time + phase)), 8.0));

  vec2 archPoint = vec2(local.x, local.y + 0.16);
  float arch = exp(-abs(length(archPoint) - 0.29) * 75.0) * smoothstep(-0.08, 0.04, local.y);
  float pillars = exp(-abs(abs(local.x) - 0.29) * 85.0)
    * smoothstep(-0.45, -0.05, local.y)
    * (1.0 - smoothstep(-0.05, 0.14, local.y));

  float scan = pow(1.0 - abs(sin((gridPoint.y + random) * TAU - time * uScanRate * 2.0)), 18.0);
  float glyph = rune + spokes + orbit + arch + pillars;
  glyph *= 0.72 + 0.28 * sin(time * 2.0 + phase + radius * 14.0);

  vec2 gridEdgeDistance = abs(fract(gridPoint) - 0.5);
  float gridEdge = exp(-min(0.5 - gridEdgeDistance.x, 0.5 - gridEdgeDistance.y) * 48.0);
  float transmission = pow(1.0 - abs(sin(
    point.x * columnCount * 1.7
      + noise(point * 4.0 + time * 0.08) * uSignalNoise * 5.0
      - time * uScanRate
  )), 12.0);

  float globalRadius = length(point);
  float portalRadius = 0.64 + uPortalBend * 0.08 * sin(atan(point.y, point.x) * 6.0 - time);
  float portal = exp(-abs(globalRadius - portalRadius) * 65.0);
  float portalSpokes = pow(1.0 - abs(sin(atan(point.y, point.x) * 12.0 + time)), 16.0)
    * smoothstep(0.24, 0.66, globalRadius)
    * (1.0 - smoothstep(0.66, 1.0, globalRadius));

  vec3 glyphColor = palette(uHue + random * 0.35 + time * 0.015);
  vec3 signalColor = palette(uHue + 0.58 - time * 0.022 + point.y * 0.08);
  vec3 portalColor = palette(uHue + 0.84 + time * 0.018);

  vec3 color = vec3(0.003, 0.005, 0.014);
  color += glyphColor * glyph * uGlow * 0.82;
  color += glyphColor * scan * rune * uGlow * 1.35;
  color += signalColor * transmission * (0.2 + gridEdge * 0.45) * uGlow;
  color += signalColor * gridEdge * 0.11;
  color += portalColor * (portal + portalSpokes) * uGlow * 0.9;
  color += vec3(0.55, 0.78, 1.0) * portal * glyph * uGlow;

  color *= 1.0 - smoothstep(0.7, 1.65, globalRadius) * 0.35;
  color = 1.0 - exp(-color * 1.45);
  color = pow(max(color, 0.0), vec3(0.84));
  outColor = vec4(color, 1.0);
}