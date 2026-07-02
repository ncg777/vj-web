#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uAuroraSpeed;
uniform float uAuroraScale;
uniform float uAuroraWarp;
uniform float uAuroraBase;
uniform float uAuroraStride;
uniform float uAuroraCurve;
uniform float uAuroraIntensity;
uniform float uTrailBlend;
uniform float uTrailFalloff;
uniform float uTrailFade;
uniform float uDitherStrength;
uniform float uHorizonFade;
uniform float uCamYaw;
uniform float uCamPitch;
uniform float uCamWobble;
uniform float uCamDistance;
uniform float uCamHeight;
uniform float uSkyStrength;
uniform float uStarDensity;
uniform float uStarIntensity;
uniform float uReflectionStrength;
uniform float uReflectionTint;
uniform float uReflectionFog;
uniform float uColorBand;
uniform float uColorSpeed;
uniform vec3 uAuroraColorA;
uniform vec3 uAuroraColorB;
uniform vec3 uAuroraColorC;
uniform vec3 uBgColorA;
uniform vec3 uBgColorB;
uniform int uAuroraSteps;
uniform float uVortexStrength;
uniform float uVortexGridScale;
uniform float uVortexRadius;
uniform float uVortexWobble;
uniform float uVortexWobbleSpeed;
uniform float uVortexSpin;
uniform float uVortexDesync;
uniform float uVortexDrift;
uniform float uVortexColorShift;
uniform float uOilStrength;
uniform float uOilScale;
uniform float uOilSpeed;
uniform float uOilContrast;

const float TAU = 6.28318530718;

mat2 mm2(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat2(c, s, -s, c);
}

mat2 m2 = mat2(0.95534, 0.29552, -0.29552, 0.95534);

float tri(float x) {
  return clamp(abs(fract(x) - 0.5), 0.01, 0.49);
}

vec2 tri2(vec2 p) {
  return vec2(tri(p.x) + tri(p.y), tri(p.y + tri(p.x)));
}

float triNoise2d(vec2 p, float spd, float time) {
  float z = 1.8;
  float z2 = 2.5;
  float rz = 0.0;
  p *= mm2(p.x * 0.06);
  vec2 bp = p;
  for (float i = 0.0; i < 5.0; i++) {
    vec2 dg = tri2(bp * 1.85) * 0.75;
    dg *= mm2(time * spd);
    p -= dg / z2;

    bp *= 1.3;
    z2 *= 0.45;
    z *= 0.42;
    p *= 1.21 + (rz - 1.0) * 0.02;

    rz += tri(p.x + tri(p.y)) * z;
    p *= -m2;
  }
  return clamp(1.0 / pow(rz * 29.0, 1.3), 0.0, 0.55);
}

float hash21(vec2 n) {
  return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec4 aurora(vec3 ro, vec3 rd, float time) {
  vec4 col = vec4(0.0);
  vec4 avgCol = vec4(0.0);
  int steps = max(uAuroraSteps, 1);

  for (int i = 0; i < 64; i++) {
    if (i >= steps) {
      break;
    }
    float fi = float(i);
    float of = uDitherStrength * hash21(gl_FragCoord.xy) * smoothstep(0.0, 15.0, fi);
    float pt = ((uAuroraBase + pow(fi, uAuroraCurve) * uAuroraStride) - ro.y) / (rd.y * 2.0 + 0.4);
    pt -= of;
    vec3 bpos = ro + pt * rd;
    vec2 p = bpos.zx;
    float rzt = triNoise2d(p * uAuroraScale, uAuroraSpeed, time);
    rzt = mix(rzt, pow(rzt, 1.0 + uAuroraWarp), uAuroraWarp);

    vec3 wave = sin(vec3(0.0, 2.1, 4.2) + fi * uColorBand + time * uColorSpeed);
    vec3 palette = mix(uAuroraColorA, uAuroraColorB, 0.5 + 0.5 * wave);
    palette = mix(palette, uAuroraColorC, rzt);

    vec4 col2 = vec4(palette * rzt * uAuroraIntensity, rzt);
    avgCol = mix(avgCol, col2, uTrailBlend);
    col += avgCol * exp2(-fi * uTrailFalloff - uTrailFade) * smoothstep(0.0, 5.0, fi);
  }

  col *= clamp(rd.y * 15.0 + 0.4, 0.0, 1.0);
  return col;
}

vec3 nmzHash33(vec3 q) {
  uvec3 p = uvec3(ivec3(q));
  p = p * uvec3(374761393U, 1103515245U, 668265263U) + p.zxy + p.yzx;
  p = p.yzx * (p.zxy ^ (p >> 3U));
  return vec3(p ^ (p >> 16U)) * (1.0 / vec3(0xffffffffU));
}

vec3 stars(vec3 p) {
  vec3 c = vec3(0.0);
  float res = uResolution.x * 1.0;

  for (float i = 0.0; i < 4.0; i++) {
    vec3 q = fract(p * (0.15 * res)) - 0.5;
    vec3 id = floor(p * (0.15 * res));
    vec2 rn = nmzHash33(id).xy;
    float c2 = 1.0 - smoothstep(0.0, 0.6, length(q));
    c2 *= step(rn.x, uStarDensity + i * i * 0.001);
    c += c2 * (mix(vec3(1.0, 0.49, 0.1), vec3(0.75, 0.9, 1.0), rn.y) * 0.1 + 0.9);
    p *= 1.3;
  }
  return c * c * uStarIntensity;
}

vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

// Loose wobbly grid of out-of-sync vortices. Warps `p` in place and
// returns the accumulated signed swirl amount for color effects.
float vortexField(inout vec2 p, float time) {
  float grid = max(uVortexGridScale, 0.01);
  vec2 gp = p * grid;
  vec2 cell = floor(gp);
  float swirlSum = 0.0;
  vec2 warped = p;
  for (int oy = -1; oy <= 1; oy++) {
    for (int ox = -1; ox <= 1; ox++) {
      vec2 id = cell + vec2(float(ox), float(oy));
      vec2 rnd = hash22(id + 17.31);
      // Each vortex wobbles around its jittered lattice point, out of sync.
      float phase = rnd.x * TAU * uVortexDesync;
      vec2 wobble = vec2(
        sin(time * uVortexWobbleSpeed * (0.6 + rnd.x * 0.8) + phase),
        cos(time * uVortexWobbleSpeed * (0.5 + rnd.y * 0.9) + phase * 1.7)
      ) * uVortexWobble;
      vec2 center = (id + 0.5 + (rnd - 0.5) * 0.8 + wobble) / grid;
      vec2 d = p - center;
      float dist = length(d);
      float radius = max(uVortexRadius, 0.01) / grid;
      float falloff = exp(-(dist * dist) / (radius * radius));
      float dir = rnd.y > 0.5 ? 1.0 : -1.0;
      float spin = sin(time * uVortexSpin * (0.7 + rnd.y * 0.6) + phase) * 0.5 + 0.75;
      float angle = uVortexStrength * dir * spin * falloff;
      angle += uVortexDrift * dir * falloff;
      warped = center + mm2(angle) * (warped - center);
      swirlSum += angle;
    }
  }
  p = warped;
  return swirlSum;
}

// Thin-film "oil on water" interference rainbow driven by film thickness.
vec3 oilFilm(vec2 p, float swirl, float time) {
  float thickness = triNoise2d(p * uOilScale + vec2(time * uOilSpeed * 0.1, -time * uOilSpeed * 0.07), 0.03, time);
  thickness = thickness * 4.0 + swirl * 1.5;
  vec3 rainbow = 0.5 + 0.5 * cos(TAU * (thickness * vec3(1.0, 1.35, 1.8) + vec3(0.0, 0.33, 0.67)));
  return pow(rainbow, vec3(max(uOilContrast, 0.1)));
}

vec3 bg(vec3 rd) {
  float sd = dot(normalize(vec3(-0.5, -0.6, 0.9)), rd) * 0.5 + 0.5;
  sd = pow(sd, 5.0);
  vec3 col = mix(uBgColorA, uBgColorB, sd);
  return col * uSkyStrength;
}

void main() {
  vec2 q = gl_FragCoord.xy / uResolution.xy;
  vec2 p = q - 0.5;
  p.x *= uResolution.x / uResolution.y;

  float time = uTime * uTimeScale;

  float swirl = vortexField(p, time);

  vec3 ro = vec3(0.0, uCamHeight, -uCamDistance);
  vec3 rd = normalize(vec3(p, 1.3));
  rd.yz *= mm2(uCamPitch + sin(time * 0.05) * uCamWobble);
  rd.xz *= mm2(uCamYaw + sin(time * 0.05) * uCamWobble);

  vec3 col = vec3(0.0);
  float fade = smoothstep(0.0, uHorizonFade, abs(rd.y)) * 0.1 + 0.9;
  col = bg(rd) * fade;

  if (rd.y > 0.0) {
    vec4 aur = smoothstep(0.0, 1.5, aurora(ro, rd, time)) * fade;
    col += stars(rd);
    col = col * (1.0 - aur.a) + aur.rgb;
  } else {
    rd.y = abs(rd.y);
    col = bg(rd) * fade * uReflectionStrength;
    vec4 aur = smoothstep(0.0, 2.5, aurora(ro, rd, time));
    col += stars(rd) * 0.1;
    col = col * (1.0 - aur.a) + aur.rgb;
    vec3 pos = ro + ((0.5 - ro.y) / rd.y) * rd;
    float nz2 = triNoise2d(pos.xz * vec2(0.5, 0.7), 0.0, time);
    vec3 waterTint = mix(vec3(0.2, 0.25, 0.5) * 0.08, vec3(0.3, 0.3, 0.5) * 0.7, nz2 * 0.4);
    col += waterTint * uReflectionTint;
    col *= mix(1.0, exp(-abs(rd.y) * uReflectionFog), uReflectionStrength);
  }

  // Twist the colors in the vortex shapes, then wash them with an
  // iridescent oil-in-water interference film.
  if (abs(uVortexColorShift) > 0.0001) {
    float hueTwist = swirl * uVortexColorShift;
    vec3 twist = vec3(
      dot(col, vec3(0.6, 0.3, 0.1)),
      dot(col, vec3(0.1, 0.6, 0.3)),
      dot(col, vec3(0.3, 0.1, 0.6))
    );
    col = mix(col, twist, clamp(abs(hueTwist), 0.0, 1.0));
    col.rb = mm2(hueTwist) * col.rb;
    col = abs(col);
  }
  if (uOilStrength > 0.0001) {
    vec3 film = oilFilm(p, swirl, time);
    float lum = clamp(dot(col, vec3(0.299, 0.587, 0.114)) * 1.6, 0.0, 1.0);
    col = mix(col, col * (0.35 + 1.3 * film) + film * lum * 0.55, uOilStrength);
  }

  outColor = vec4(col, 1.0);
}
