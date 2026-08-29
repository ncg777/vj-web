#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uCellScale;
uniform float uJitter;
uniform float uFracture;
uniform float uCoreRadius;
uniform float uSpin;
uniform float uShockwave;
uniform float uGlow;
uniform float uHue;
uniform float uSeed;

const float TAU = 6.28318530718;

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

vec2 hash22(vec2 point) {
  point += fract(uSeed * 0.000001) * 19.17;
  vec3 p = fract(vec3(point.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yzx + 33.33);
  return fract((p.xx + p.yz) * p.zy);
}

void voronoi(vec2 point, float time, out float nearest, out float secondNearest, out vec2 cellId) {
  vec2 base = floor(point);
  vec2 local = fract(point);
  nearest = 10.0;
  secondNearest = 10.0;
  cellId = vec2(0.0);

  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 offset = vec2(float(x), float(y));
      vec2 id = base + offset;
      vec2 random = hash22(id);
      vec2 animated = 0.5 + uJitter * 0.44 * sin(time * (0.35 + random.x * 0.4) + TAU * random);
      vec2 delta = offset + animated - local;
      float distanceSquared = dot(delta, delta);
      if (distanceSquared < nearest) {
        secondNearest = nearest;
        nearest = distanceSquared;
        cellId = id;
      } else if (distanceSquared < secondNearest) {
        secondNearest = distanceSquared;
      }
    }
  }
}

vec3 palette(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.85, 1.0, 0.63) + vec3(0.08, 0.34, 0.67)));
}

void main() {
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  point *= uZoom;
  float time = uTime * uTimeScale;
  point = rotate2d(time * uSpin * 0.08) * point;

  float radius = length(point);
  float lens = 0.1 / max(0.16, radius);
  vec2 crystalPoint = point + normalize(point + vec2(0.0001)) * lens * sin(time * 0.4 + radius * 8.0);

  float nearest;
  float secondNearest;
  vec2 cellId;
  voronoi(crystalPoint * uCellScale, time, nearest, secondNearest, cellId);
  float edgeGap = sqrt(secondNearest) - sqrt(nearest);
  float fractures = exp(-edgeGap * (18.0 + uFracture * 42.0));
  float shardPulse = 0.5 + 0.5 * sin(time * 1.4 + dot(cellId, vec2(1.7, 2.3)) + sqrt(nearest) * 12.0);
  float facets = pow(max(0.0, 1.0 - sqrt(nearest)), 3.0) * (0.18 + 0.42 * shardPulse);

  float angle = atan(point.y, point.x) + time * uSpin * 0.24;
  float sides = 6.0 + floor(fract(uSeed * 0.00001) * 4.0);
  float coreShape = uCoreRadius + 0.035 * cos(angle * sides + sin(time * 0.7) * 0.7);
  float coreEdge = exp(-abs(radius - coreShape) * 95.0);
  float coreFill = 1.0 - smoothstep(coreShape * 0.25, coreShape, radius);

  float wavePhase = radius * max(uShockwave, 0.1) * 8.0 - time * 2.1;
  float shock = pow(1.0 - abs(sin(wavePhase)), 12.0);
  shock *= smoothstep(coreShape * 0.7, coreShape * 3.8, radius)
    * (1.0 - smoothstep(coreShape * 3.8, coreShape * 6.5, radius));
  shock *= 0.55 + 0.45 * cos(angle * sides * 0.5 + time);

  float flare = exp(-radius * 3.7) * (0.6 + 0.4 * sin(angle * sides - time * 1.8));
  vec3 fractureColor = palette(uHue + dot(cellId, vec2(0.037, 0.051)) + time * 0.018);
  vec3 coreColor = palette(uHue + 0.42 - time * 0.03);
  vec3 shockColor = palette(uHue + 0.78 + radius * 0.12);

  vec3 color = vec3(0.003, 0.006, 0.016);
  color += fractureColor * fractures * uGlow * (0.48 + facets);
  color += fractureColor * facets * 0.16;
  color += coreColor * coreEdge * uGlow * 1.8;
  color += mix(coreColor, vec3(0.75, 0.9, 1.0), 0.65) * coreFill * (0.25 + flare);
  color += shockColor * shock * uGlow * 0.95;
  color += vec3(0.35, 0.55, 1.0) * fractures * shock * uGlow;

  color *= 1.0 - smoothstep(0.7, 1.7, radius) * 0.42;
  color = 1.0 - exp(-color * 1.35);
  color = pow(max(color, 0.0), vec3(0.84));
  outColor = vec4(color, 1.0);
}