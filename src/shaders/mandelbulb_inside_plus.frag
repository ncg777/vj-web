#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uPower;
uniform float uBulbSpin;
uniform float uMaxRayLength;
uniform float uTolerance;
uniform float uNormOffset;
uniform float uInitStep;
uniform float uRotSpeedX;
uniform float uRotSpeedY;
uniform float uCamDistance;
uniform float uCamHeight;
uniform float uFov;
uniform float uSkyBoost;
uniform float uGlowBoost;
uniform float uGlowFalloff;
uniform float uDiffuseBoost;
uniform float uMatTransmit;
uniform float uMatReflect;
uniform float uRefractIndex;
uniform float uHueShift;
uniform float uGlowHueOffset;
uniform float uNebulaMix;
uniform float uNebulaHueShift;
uniform float uNebulaSat;
uniform float uNebulaVal;
uniform float uNebulaGlowHue;
uniform float uNebulaGlowBoost;
uniform float uSkySat;
uniform float uSkyVal;
uniform float uGlowSat;
uniform float uGlowVal;
uniform float uDiffuseSat;
uniform float uDiffuseVal;
uniform vec3 uBeerColor;
uniform vec3 uLightPos;
uniform int uLoops;
uniform int uRayMarches;
uniform int uBounces;

const float PI = 3.141592654;
const float TAU = 6.28318530718;

mat3 g_rot = mat3(1.0);

const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}

vec3 sRGB(vec3 t) {
  return mix(1.055 * pow(t, vec3(1.0 / 2.4)) - 0.055, 12.92 * t, step(t, vec3(0.0031308)));
}

vec3 aces_approx(vec3 v) {
  v = max(v, 0.0);
  v *= 0.6;
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0);
}

float boxSDF(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float rayPlane(vec3 ro, vec3 rd, vec4 p) {
  return -(dot(ro, p.xyz) + p.w) / dot(rd, p.xyz);
}

float mandelBulb(vec3 p, float time) {
  vec3 z = p;
  float r = 0.0;
  float dr = 1.0;

  for (int i = 0; i < 6; ++i) {
    if (i >= uLoops) {
      break;
    }
    r = length(z);
    if (r > 2.0) {
      break;
    }
    r = max(r, 1e-6);
    float theta = atan(z.y, z.x);
    float phi = asin(clamp(z.z / r, -1.0, 1.0)) + time * uBulbSpin;

    dr = pow(r, uPower - 1.0) * dr * uPower + 1.0;
    r = pow(r, uPower);
    theta *= uPower;
    phi *= uPower;
    z = r * vec3(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)) + p;
  }

  return 0.5 * log(max(r, 1e-6)) * r / dr;
}

mat3 rot_z(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      c, s, 0.0,
     -s, c, 0.0,
      0.0, 0.0, 1.0
    );
}

mat3 rot_y(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      c, 0.0, s,
      0.0, 1.0, 0.0,
     -s, 0.0, c
    );
}

mat3 rot_x(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      1.0, 0.0, 0.0,
      0.0, c, s,
      0.0, -s, c
    );
}

vec3 skyColor(vec3 ro, vec3 rd, vec3 skyCol) {
  vec3 col = clamp(vec3(0.0025 / abs(rd.y)) * skyCol, 0.0, 1.0);

  float tp0 = rayPlane(ro, rd, vec4(vec3(0.0, 1.0, 0.0), 4.0));
  float tp1 = rayPlane(ro, rd, vec4(vec3(0.0, -1.0, 0.0), 6.0));
  float tp = max(tp0, tp1);
  if (tp > 0.0) {
    vec3 pos = ro + tp * rd;
    vec2 pp = pos.xz;
    float db = boxSDF(pp, vec2(6.0, 9.0)) - 1.0;
    col += vec3(4.0) * skyCol * rd.y * rd.y * smoothstep(0.25, 0.0, db);
    col += vec3(0.8) * skyCol * exp(-0.5 * max(db, 0.0));
  }

  if (tp0 > 0.0) {
    vec3 pos = ro + tp0 * rd;
    vec2 pp = pos.xz;
    float ds = length(pp) - 0.5;
    col += vec3(0.25) * skyCol * exp(-0.5 * max(ds, 0.0));
  }

  return clamp(col, 0.0, 10.0);
}

float df(vec3 p, float time) {
  p *= g_rot;
  const float z1 = 2.0;
  return mandelBulb(p / z1, time) * z1;
}

vec3 normal(vec3 pos, float time) {
  vec2 eps = vec2(uNormOffset, 0.0);
  vec3 nor;
  nor.x = df(pos + eps.xyy, time) - df(pos - eps.xyy, time);
  nor.y = df(pos + eps.yxy, time) - df(pos - eps.yxy, time);
  nor.z = df(pos + eps.yyx, time) - df(pos - eps.yyx, time);
  return normalize(nor);
}

float rayMarch(vec3 ro, vec3 rd, float dfactor, float time, out int ii) {
  float t = 0.0;
  float tol = dfactor * uTolerance;
  ii = uRayMarches;
  for (int i = 0; i < 96; ++i) {
    if (i >= uRayMarches) {
      break;
    }
    if (t > uMaxRayLength) {
      t = uMaxRayLength;
      break;
    }
    float d = dfactor * df(ro + rd * t, time);
    if (d < tol) {
      ii = i;
      break;
    }
    t += d;
  }
  return t;
}

vec3 render(vec3 ro, vec3 rd, float time) {
  vec3 agg = vec3(0.0);
  vec3 ragg = vec3(1.0);

  bool isInside = df(ro, time) < 0.0;

  vec3 baseSky = hsv2rgb(vec3(uHueShift + 0.6, uSkySat, uSkyVal)) * uSkyBoost;
  vec3 baseGlow = hsv2rgb(vec3(uHueShift + uGlowHueOffset, uGlowSat, uGlowVal)) * uGlowBoost;
  vec3 baseDiffuse = hsv2rgb(vec3(uHueShift + 0.6, uDiffuseSat, uDiffuseVal)) * uDiffuseBoost;

  vec3 nebulaSky = hsv2rgb(vec3(uNebulaHueShift + 0.18, uNebulaSat, uNebulaVal)) * (uSkyBoost * 0.9);
  vec3 nebulaGlow = hsv2rgb(vec3(uNebulaGlowHue, uNebulaSat, uNebulaVal * 1.5)) * uNebulaGlowBoost;
  vec3 nebulaDiffuse = hsv2rgb(vec3(uNebulaHueShift + 0.55, uNebulaSat * 0.8, uNebulaVal)) * uDiffuseBoost;

  float nebulaMix = clamp(uNebulaMix, 0.0, 1.0);
  vec3 skyCol = mix(baseSky, nebulaSky, nebulaMix);
  vec3 glowCol = mix(baseGlow, nebulaGlow, nebulaMix);
  vec3 diffuseCol = mix(baseDiffuse, nebulaDiffuse, nebulaMix);

  for (int bounce = 0; bounce < 5; ++bounce) {
    if (bounce >= uBounces) {
      break;
    }
    float dfactor = isInside ? -1.0 : 1.0;
    float mragg = max(max(ragg.x, ragg.y), ragg.z);
    if (mragg < 0.025) {
      break;
    }
    int iter;
    float st = rayMarch(ro, rd, dfactor, time, iter);
    if (st >= uMaxRayLength) {
      agg += ragg * skyColor(ro, rd, skyCol);
      break;
    }

    vec3 sp = ro + rd * st;
    vec3 sn = dfactor * normal(sp, time);

    float fre = 1.0 + dot(rd, sn);
    fre *= fre;
    fre = mix(0.1, 1.0, fre);

    vec3 ld = normalize(uLightPos - sp);
    float dif = max(dot(ld, sn), 0.0);
    vec3 ref = reflect(rd, sn);
    float re = uRefractIndex;
    float ire = 1.0 / re;
    vec3 refr = refract(rd, sn, !isInside ? re : ire);
    vec3 rsky = skyColor(sp, ref, skyCol);

    vec3 col = vec3(0.0);
    col += diffuseCol * dif * dif * (1.0 - uMatTransmit);
    float edge = smoothstep(1.0, 0.9, fre);
    col += rsky * uMatReflect * edge;
    col += glowCol * exp(-float(iter) * uGlowFalloff);

    if (isInside) {
      ragg *= exp(-(st + uInitStep) * uBeerColor);
    }
    agg += ragg * col;

    if (refr == vec3(0.0)) {
      rd = ref;
    } else {
      ragg *= uMatTransmit;
      isInside = !isInside;
      rd = refr;
    }
    ro = sp + uInitStep * rd;
  }

  return agg;
}

vec3 effect(vec2 p, float time) {
  g_rot = rot_x(uRotSpeedX * time) * rot_y(uRotSpeedY * time);
  vec3 ro = vec3(0.0, uCamHeight, uCamDistance);
  const vec3 la = vec3(0.0);
  const vec3 up = vec3(0.0, 1.0, 0.0);

  vec3 ww = normalize(la - ro);
  vec3 uu = normalize(cross(up, ww));
  vec3 vv = cross(ww, uu);
  float fov = tan(uFov);
  vec3 rd = normalize(-p.x * uu + p.y * vv + fov * ww);

  return render(ro, rd, time);
}

void main() {
  vec2 q = gl_FragCoord.xy / uResolution.xy;
  vec2 p = -1.0 + 2.0 * q;
  p.x *= uResolution.x / uResolution.y;
  float time = uTime * uTimeScale;
  vec3 col = effect(p, time);
  col = aces_approx(col);
  col = sRGB(col);
  outColor = vec4(col, 1.0);
}
