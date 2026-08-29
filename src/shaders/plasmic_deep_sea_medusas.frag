#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform int uMedusaCount;
uniform int uBellRibs;
uniform int uTentacles;
uniform float uBellSize;
uniform float uPulse;
uniform float uRiseSpeed;
uniform float uTentacleLength;
uniform float uTentacleSway;
uniform float uTransparency;
uniform float uGlow;
uniform float uMarineSnow;
uniform float uHue;
uniform float uSeed;

const float TAU = 6.28318530718;

mat2 rotate2d(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

float hash11(float value) {
  return fract(sin(value * 127.17 + uSeed * 0.000019) * 43758.5453);
}

float hash21(vec2 point) {
  point += fract(uSeed * 0.000001) * 17.31;
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

float fbm(vec2 point) {
  float value = 0.0;
  float amplitude = 0.52;
  for (int octave = 0; octave < 5; octave++) {
    value += amplitude * noise(point);
    point = rotate2d(0.61) * point * 2.03 + vec2(1.7, -2.1);
    amplitude *= 0.49;
  }
  return value;
}

vec3 medusaPalette(float phase) {
  return 0.5 + 0.5 * cos(TAU * (phase * vec3(0.82, 1.0, 0.61) + vec3(0.03, 0.3, 0.66)));
}

void main() {
  vec2 point = (gl_FragCoord.xy * 2.0 - uResolution.xy) / uResolution.y;
  point *= uZoom;
  float time = uTime * uTimeScale;

  float depthNoise = fbm(point * 1.25 + vec2(time * 0.018, -time * 0.025));
  float verticalDepth = clamp(gl_FragCoord.y / uResolution.y, 0.0, 1.0);
  vec3 color = mix(vec3(0.001, 0.004, 0.016), vec3(0.004, 0.032, 0.055), verticalDepth);
  color += vec3(0.005, 0.025, 0.04) * depthNoise;

  float causticA = pow(1.0 - abs(sin(point.x * 2.8 + depthNoise * 5.0 + time * 0.12)), 10.0);
  float causticB = pow(1.0 - abs(sin(point.x * 4.1 - depthNoise * 4.0 - time * 0.09)), 12.0);
  float surfaceFade = smoothstep(-0.7, 1.25, point.y);
  color += vec3(0.02, 0.12, 0.16) * (causticA + causticB) * surfaceFade * 0.28;

  vec3 medusaLight = vec3(0.0);
  float medusaMask = 0.0;
  float electricContact = 0.0;

  for (int medusaIndex = 0; medusaIndex < 8; medusaIndex++) {
    if (medusaIndex >= uMedusaCount) {
      break;
    }

    float fi = float(medusaIndex);
    float randomA = hash11(fi + 3.0);
    float randomB = hash11(fi + 27.0);
    float randomC = hash11(fi + 61.0);
    float phase = randomA * TAU;
    float depth = 0.55 + randomC * 0.65;
    float travel = 2.9 + uTentacleLength * 0.35;
    float rise = time * uRiseSpeed * (0.075 + randomB * 0.055);
    vec2 center = vec2(
      (randomA - 0.5) * 1.85 + sin(time * 0.12 + phase) * 0.12,
      mod(randomB * travel + rise + travel * 0.5, travel) - travel * 0.5
    );
    float scale = uBellSize * depth;
    vec2 local = (point - center) / max(scale, 0.05);

    float breathing = sin(time * (0.9 + randomB * 0.55) * uPulse + phase);
    local.x *= 1.0 + breathing * 0.08;
    local.y *= 1.0 - breathing * 0.1;
    local = rotate2d(sin(time * 0.19 + phase) * 0.08) * local;

    float radius = length(vec2(local.x, (local.y - 0.025) * 0.9));
    float skirtY = -0.075
      + 0.018 * cos(local.x * 68.0 + phase)
      + breathing * 0.012;
    float domeDistance = radius - 0.235;
    float antialias = max(fwidth(domeDistance), 0.0015);
    float dome = 1.0 - smoothstep(-antialias, antialias, domeDistance);
    float aboveSkirt = smoothstep(skirtY - antialias, skirtY + antialias, local.y);
    float bellBody = dome * aboveSkirt;
    float domeEdge = exp(-max(abs(domeDistance) - antialias, 0.0) * 80.0) * aboveSkirt;
    float skirtEdge = exp(-abs(local.y - skirtY) * 105.0)
      * (1.0 - smoothstep(0.19, 0.245, abs(local.x)));

    float bellAngle = atan(local.y - 0.015, local.x);
    float ribs = pow(1.0 - abs(sin(bellAngle * float(uBellRibs) + phase)), 18.0);
    ribs *= bellBody * smoothstep(0.035, 0.22, radius);
    float organRing = exp(-abs(length(local * vec2(0.82, 1.55)) - 0.082) * 95.0) * bellBody;
    float organCore = exp(-length(local * vec2(0.75, 1.5)) * 52.0) * bellBody;

    float tentacleGlow = 0.0;
    float oralArms = 0.0;
    float tentacleCount = float(uTentacles);
    for (int tentacleIndex = 0; tentacleIndex < 10; tentacleIndex++) {
      if (tentacleIndex >= uTentacles) {
        break;
      }
      float fj = float(tentacleIndex);
      float anchor = (fj / max(tentacleCount - 1.0, 1.0) - 0.5) * 0.34;
      float tailProgress = clamp((-local.y - 0.06) / max(uTentacleLength - 0.06, 0.05), 0.0, 1.0);
      float strandPhase = phase + fj * 1.73;
      float strandX = anchor
        + sin(local.y * (8.0 + randomC * 5.0) - time * (1.1 + randomA) + strandPhase)
          * uTentacleSway * (0.014 + tailProgress * 0.058)
        + sin(local.y * 21.0 + time * 0.45 + fj) * uTentacleSway * 0.008;
      float verticalMask = smoothstep(-uTentacleLength, -uTentacleLength + 0.12, local.y)
        * (1.0 - smoothstep(-0.055, -0.015, local.y));
      float strand = exp(-abs(local.x - strandX) * (105.0 - tailProgress * 32.0));
      strand *= verticalMask * (1.0 - tailProgress * 0.55);
      float travelingCharge = 0.55 + 0.45 * pow(
        1.0 - abs(sin(local.y * 18.0 - time * 2.4 + strandPhase)),
        5.0
      );
      tentacleGlow += strand * travelingCharge;

      if (tentacleIndex < 3) {
        float armAnchor = (fj - 1.0) * 0.055;
        float armX = armAnchor + sin(local.y * 6.5 - time + strandPhase) * uTentacleSway * 0.045;
        oralArms += exp(-abs(local.x - armX) * 48.0) * verticalMask
          * (1.0 - smoothstep(0.15, 0.72, tailProgress));
      }
    }

    float intensity = mix(0.5, 1.15, randomC);
    vec3 bellColor = medusaPalette(uHue + randomA * 0.42 + time * 0.012);
    vec3 organColor = medusaPalette(uHue + 0.55 + randomB * 0.23 - time * 0.018);
    float membrane = bellBody * (0.08 + depthNoise * 0.12) * uTransparency;
    float brightEdge = domeEdge + skirtEdge + ribs * 0.62;

    medusaLight += bellColor * membrane * intensity;
    medusaLight += bellColor * brightEdge * uGlow * intensity;
    medusaLight += organColor * (organRing + organCore * 1.5) * uGlow * intensity;
    medusaLight += mix(bellColor, organColor, 0.45) * tentacleGlow * uGlow * 0.78 * intensity;
    medusaLight += organColor * oralArms * uGlow * 0.42 * intensity;
    medusaMask += bellBody * 0.18 + tentacleGlow * 0.06;
    electricContact += brightEdge * tentacleGlow;
  }

  vec2 snowPoint = point * vec2(15.0, 11.0) + vec2(0.0, time * (0.18 + uRiseSpeed * 0.12));
  vec2 snowCell = floor(snowPoint);
  vec2 snowLocal = fract(snowPoint) - 0.5;
  vec2 snowOffset = vec2(hash21(snowCell), hash21(snowCell + 7.13)) - 0.5;
  float snowDistance = length(snowLocal - snowOffset * 0.72);
  float snow = exp(-snowDistance * 48.0) * step(0.58, hash21(snowCell + 19.7));
  snow *= 0.55 + 0.45 * sin(time * 1.7 + hash21(snowCell) * TAU);

  color *= 1.0 - clamp(medusaMask, 0.0, 0.42);
  color += medusaLight;
  color += vec3(0.42, 0.78, 1.0) * snow * uMarineSnow * 0.65;
  color += vec3(0.75, 0.88, 1.0) * electricContact * uGlow * 0.55;
  color *= 1.0 - smoothstep(0.65, 1.75, length(point)) * 0.34;
  color = 1.0 - exp(-color * 1.3);
  color = pow(max(color, 0.0), vec3(0.86));
  outColor = vec4(color, 1.0);
}