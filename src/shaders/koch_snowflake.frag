#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform int uIterations;
uniform float uScale;
uniform float uRotation;
uniform float uGlowIntensity;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;

float sdSegment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

vec2 rotate(vec2 p, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

float kochSegmentDistanceIter(vec2 p, vec2 a, vec2 b, int iterations) {
  vec2 ab = b - a;
  float len = max(length(ab), 1e-6);
  vec2 dir = ab / len;
  vec2 nrm = vec2(-dir.y, dir.x);
  vec2 pl = vec2(dot(p - a, dir) / len, dot(p - a, nrm) / len);

  const float c60 = 0.5;
  const float s60 = 0.8660254037844386;
  mat2 invRotPlus = mat2(c60, s60, -s60, c60);
  mat2 invRotMinus = mat2(c60, -s60, s60, c60);

  const int MAX_ITERS = 8;
  int it = min(iterations, MAX_ITERS);
  float scaleAccum = 1.0;

  for (int i = 0; i < MAX_ITERS; ++i) {
    if (i >= it) {
      break;
    }
    pl *= 3.0;
    float region = floor(pl.x);

    if (region == 1.0) {
      vec2 c = vec2(1.5, 0.0);
      vec2 pr = pl - c;
      vec2 pr1 = invRotPlus * pr;
      vec2 pr2 = invRotMinus * pr;
      vec2 p1 = pr1 + c;
      vec2 p2 = pr2 + c;
      pl = (abs(p1.y) < abs(p2.y)) ? p1 : p2;
    }

    pl.x -= region;
    scaleAccum *= (1.0 / 3.0);
  }

  float dLocal = sdSegment(pl, vec2(0.0, 0.0), vec2(1.0, 0.0));
  return dLocal * len * scaleAccum;
}

float kochSnowflakeDistance(vec2 p, float size, int iterations) {
  float h = size * sqrt(3.0) / 2.0;
  vec2 v1 = vec2(0.0, h * 2.0 / 3.0);
  vec2 v2 = vec2(-size / 2.0, -h / 3.0);
  vec2 v3 = vec2(size / 2.0, -h / 3.0);

  float d1 = kochSegmentDistanceIter(p, v1, v2, iterations);
  float d2 = kochSegmentDistanceIter(p, v2, v3, iterations);
  float d3 = kochSegmentDistanceIter(p, v3, v1, iterations);
  return min(min(d1, d2), d3);
}

float trianglePerimeterDistance(vec2 p, float size) {
  float h = size * sqrt(3.0) / 2.0;
  vec2 v1 = vec2(0.0, h * 2.0 / 3.0);
  vec2 v2 = vec2(-size / 2.0, -h / 3.0);
  vec2 v3 = vec2(size / 2.0, -h / 3.0);
  float d1 = sdSegment(p, v1, v2);
  float d2 = sdSegment(p, v2, v3);
  float d3 = sdSegment(p, v3, v1);
  return min(min(d1, d2), d3);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * uResolution.xy) / min(uResolution.x, uResolution.y);
  float angle = uTime * uRotation;
  uv = rotate(uv, angle);

  float distKoch = kochSnowflakeDistance(uv, uScale, uIterations);
  float distTri = trianglePerimeterDistance(uv, uScale);
  float dist = min(distKoch, distTri * 0.75);

  const float lineWidth = 0.004;
  const float lineOuterMult = 1.5;
  const float lineInnerMult = 0.5;
  const float distanceScale = 15.0;
  const float timeScale = 2.0;
  const float glowMix = 0.4;
  const float edgeGlowMult = 0.3;

  float line = smoothstep(lineWidth * lineOuterMult, lineWidth * lineInnerMult, dist);
  float glow = exp(-dist * distanceScale * uGlowIntensity);
  float colorMix = sin(dist * distanceScale - uTime * timeScale) * 0.5 + 0.5;
  vec3 color = mix(uColorPrimary, uColorSecondary, colorMix);

  vec3 finalColor = color * (line + glow * glowMix);
  vec3 edgeGlowColor = vec3(0.2, 0.3, 0.5);
  finalColor += edgeGlowColor * glow * uGlowIntensity * edgeGlowMult;

  outColor = vec4(finalColor, 1.0);
}
