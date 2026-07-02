#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;

// Motion & camera
uniform float uTimeScale;
uniform float uZoom;
uniform float uSpin;
uniform float uWarp;

// Gas cloud (nebula) parameters
uniform float uCloudSpeed;
uniform float uCloudDetail;
uniform float uCurlStrength;
uniform float uTurbulence;
uniform float uCloudDensity;
uniform float uCloudContrast;
uniform float uParallax;
uniform float uHueSpeed;
uniform float uPaletteShift;

// Lightning bolt parameters
uniform float uBoltCount;
uniform float uBoltReach;
uniform float uJaggedness;
uniform float uRoughness;
uniform float uBoltWidth;
uniform float uBoltIntensity;
uniform float uBranchCount;
uniform float uBranchAngle;
uniform float uBranchDecay;
uniform float uStrikeRate;
uniform float uStrikeChaos;
uniform float uForkiness;

// Core plasma
uniform float uCoreSize;
uniform float uCoreGlow;

// Light / cosmic effects
uniform float uBloomIntensity;
uniform float uBloomRadius;
uniform float uChromaticAberration;
uniform float uStarDensity;
uniform float uVignette;
uniform float uExposure;

uniform float uSeed;

uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;
uniform vec3 uColorAccent;

const float TAU = 6.28318530718;
const float STRIKE_BASELINE = 0.12;

// ---------------------------------------------------------------------
// Hashing & noise primitives.
// ---------------------------------------------------------------------
float hash1(float n) {
  return fract(sin(n * 127.1) * 43758.5453123);
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// C1-continuous value noise.
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

const mat2 ROT = mat2(0.8, 0.6, -0.6, 0.8);

float fbm4(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * noise(p);
    p = ROT * p * 2.02 + vec2(11.7, 5.1);
    a *= 0.5;
  }
  return v;
}

// ---------------------------------------------------------------------
// Fractal 1D bolt path: signed lateral displacement of the discharge
// channel as a function of the distance travelled along it. Summing a
// few octaves of value noise whose per-octave gain is the user's
// roughness gives the jagged, self-similar silhouette of a real
// electric discharge - at a tiny fraction of the cost of building and
// scanning an explicit midpoint-displacement polyline per pixel.
// ---------------------------------------------------------------------
float boltPath(float x, float seed, float gain) {
  float v = 0.0;
  float a = 0.55;
  float f = 2.4;
  for (int i = 0; i < 4; i++) {
    v += a * (noise(vec2(x * f + seed * 7.31, seed)) - 0.5);
    f *= 2.3;
    a *= gain;
  }
  return v;
}

// Cheaper 3-octave variant for secondary branches.
float branchPath(float x, float seed, float gain) {
  float v = 0.0;
  float a = 0.55;
  float f = 3.1;
  for (int i = 0; i < 3; i++) {
    v += a * (noise(vec2(x * f + seed * 3.17, seed)) - 0.5);
    f *= 2.3;
    a *= gain;
  }
  return v;
}

// A richer palette that keeps the gas vivid and chromatic while still
// honoring the user's color controls and the time-driven hue drift.
vec3 palette(float t) {
  float hue = fract(t + uPaletteShift + uTime * uTimeScale * uHueSpeed);
  float pulse = 0.55 + 0.45 * sin(hue * TAU * 2.0 + uTime * 0.7);
  vec3 warm = mix(uColorPrimary, uColorSecondary, 0.35 + 0.25 * smoothstep(0.0, 1.0, hue));
  vec3 cool = mix(uColorAccent, vec3(1.0), 0.1 + 0.2 * pulse);
  return mix(warm, cool, smoothstep(0.3, 0.75, hue)) * (0.75 + 0.55 * pulse);
}

void main() {
  vec2 uv = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
  uv.x *= uResolution.x / uResolution.y;
  uv *= uZoom;

  float t = uTime * uTimeScale;
  float seed = uSeed * 0.0137 + 1.0;

  float spin = t * uSpin * 0.3;
  mat2 rot = mat2(cos(spin), -sin(spin), sin(spin), cos(spin));
  uv = rot * uv;

  float radius = length(uv) + 1e-4;

  // ------------------------------------------------------------------
  // Nebula: parallax layers of domain-warped fbm gas. The old
  // finite-difference curl advection cost 16 fbm evaluations per layer;
  // an analytic swirl warp gives the same fluid drift for almost free.
  // ------------------------------------------------------------------
  vec3 nebula = vec3(0.0);
  const int LAYERS = 4;
  for (int layer = 0; layer < LAYERS; layer++) {
    float fl = float(layer);
    float depth = 1.0 + fl * uParallax;
    vec2 lp = uv * (0.9 + fl * 0.55) * uCloudDetail + seed * (fl + 1.0) * 3.7;

    // Analytic swirl: a rotating sinusoidal displacement field that is
    // divergence-free by construction, so the gas swirls instead of
    // scrolling, without any extra noise fetches.
    float tc = t * uCloudSpeed;
    float sw = tc * (0.16 + 0.09 * fl);
    vec2 flow = uCurlStrength * 0.28 * vec2(
      sin(lp.y * 1.3 + sw) + 0.5 * sin(lp.y * 2.9 - sw * 1.7),
      sin(lp.x * 1.4 - sw) + 0.5 * sin(lp.x * 2.6 + sw * 1.3)
    );
    vec2 warped = lp + flow + tc * (0.04 / depth) * vec2(1.0, 0.4);

    vec2 q = vec2(
      fbm4(warped),
      fbm4(warped + vec2(5.2, 1.3))
    );
    float f = fbm4(warped + (1.3 + uWarp) * q);

    // Fine turbulent detail, ridged wisps, and broad voids break the gas
    // into more chaotic filaments and chromatic pockets.
    float fine = fbm4(warped * 3.2 - tc * 0.28);
    float ridged = 1.0 - abs(2.0 * fine - 1.0);
    float veins = fbm4(warped * 1.9 + q * 2.1 + vec2(-tc * 0.18, tc * 0.14));
    float wisps = fbm4(warped * 4.7 + q.yx * 1.6 + vec2(3.7, -8.1) - tc * 0.33);
    float pockets = fbm4(warped * 0.85 - q * 1.3 + vec2(6.3, -2.4));
    float blended = mix(f, fine, clamp(uTurbulence, 0.0, 1.0) * 0.34);
    blended = mix(blended, ridged, 0.18 + 0.27 * clamp(uTurbulence, 0.0, 1.0));
    blended += (veins - 0.5) * 0.18 + (wisps - 0.5) * 0.12 - (pockets - 0.5) * 0.1;

    float density = smoothstep(uCloudDensity - 0.24, uCloudDensity + 0.26, blended);
    density *= 0.72 + 0.55 * smoothstep(0.18, 0.92, ridged);
    density *= 0.82 + 0.38 * smoothstep(0.35, 0.85, wisps);
    density = pow(clamp(density, 0.0, 1.0), mix(1.0, 3.4, clamp(uCloudContrast, 0.0, 1.0)));

    float chroma = 0.12 + 0.88 * q.y;
    float hueScatter = fl * 0.37 + (veins - pockets) * 0.28 + ridged * 0.18;
    vec3 gasA = palette(blended * 1.55 + radius * 0.19 + hueScatter);
    vec3 gasB = palette(length(q) * 1.18 + 0.32 + fl * 0.21 - wisps * 0.14);
    vec3 gasC = palette(f * 0.92 + 0.58 - fl * 0.16 + chroma * 0.34 + veins * 0.21);
    vec3 gasD = palette(ridged * 0.75 + pockets * 0.42 + hueScatter * 0.7 + 0.41);
    vec3 layerCol = mix(gasA, gasB, clamp(length(q) * 1.15 + chroma * 0.25, 0.0, 1.0));
    layerCol = mix(layerCol, gasC, clamp(f * f * 1.35 + wisps * 0.22 - 0.12, 0.0, 1.0));
    layerCol = mix(layerCol, gasD, clamp(ridged * 0.75 + veins * 0.35 - pockets * 0.18, 0.0, 1.0));
    layerCol += (gasC - gasA) * (veins - 0.5) * 0.22;
    layerCol *= 0.65 + 0.6 * density;

    nebula += max(layerCol, 0.0) * density / depth;
  }
  nebula /= float(LAYERS) * 0.75;

  // Sparse cosmic starfield twinkling behind/within the gas.
  vec2 starCell = floor(uv * (28.0 + uStarDensity * 40.0));
  vec2 starUv = fract(uv * (28.0 + uStarDensity * 40.0)) - 0.5;
  float starHash = hash(starCell + seed * 5.3);
  float starMask = step(1.0 - clamp(uStarDensity, 0.0, 1.0) * 0.12, starHash);
  float twinkle = 0.5 + 0.5 * sin(t * (2.0 + starHash * 6.0) + starHash * TAU);
  float star = starMask * exp(-dot(starUv, starUv) * 90.0) * (0.4 + 0.6 * twinkle);
  nebula += star * palette(starHash + t * 0.02);

  // ------------------------------------------------------------------
  // Lightning: jagged fractal bolts erupting radially from the core.
  // Each bolt lives in its own rotated frame where the channel runs
  // along +x and is laterally displaced by a fractal 1D path, so the
  // distance to the channel is a single subtraction - no polylines.
  // Every discrete strike re-seeds the path, so each flash is a brand
  // new discharge that flares white-hot and decays.
  // ------------------------------------------------------------------
  float boltCore = 0.0;
  float boltGlow = 0.0;
  vec3 boltGlowColor = vec3(0.0);
  const int MAX_BOLTS = 14;
  int boltCount = int(clamp(uBoltCount, 1.0, float(MAX_BOLTS)));
  float coreR = max(uCoreSize * 0.5, 0.04);
  float gain = clamp(uRoughness, 0.1, 0.9);
  int branchCount = int(clamp(uBranchCount, 0.0, 6.0));

  for (int i = 0; i < MAX_BOLTS; i++) {
    float fi = float(i);
    float hs = fi * 12.9 + seed * 91.7;

    // Discrete strike timeline: each bolt fires periodically and is
    // reshaped from scratch on every strike, then flashes bright and
    // decays quickly - the hallmark of a real electric discharge.
    float rate = max(uStrikeRate, 0.05) * (0.6 + 0.8 * hash1(hs + 1.0));
    float cycle = t * rate + hash1(hs) * 50.0;
    float strikeIndex = floor(cycle);
    float strikePhase = fract(cycle);
    float strikeSeed = fract(hash1(hs + strikeIndex * (1.0 + uStrikeChaos * 3.0))) * 61.7 + 3.0;
    float slotChance = clamp(
      (float(boltCount) + (hash1(strikeSeed + 11.7) - 0.5) * (2.0 + uStrikeChaos * 2.5)) / float(MAX_BOLTS),
      0.08,
      0.98
    );
    if (hash1(hs * 3.7 + strikeIndex * 5.1 + seed * 2.3) > slotChance) {
      continue;
    }

    // Distributed strikes stay dimmer between flashes so the frame can
    // breathe instead of reading as a permanently lit starburst.
    float flashBaseline = STRIKE_BASELINE * 0.3;
    float flash = flashBaseline + 0.92 * exp(-strikePhase * (3.6 + uStrikeChaos * 6.0));
    flash *= smoothstep(0.0, 0.03, strikePhase);
    // Rapid intra-strike flicker, like a channel re-striking.
    flash *= 0.75 + 0.25 * sin((strikePhase * 40.0 + hash1(hs + 2.0) * TAU) * (1.0 + uStrikeChaos));

    float energy = flash * uBoltIntensity * (0.75 + 0.55 * hash1(strikeSeed + 12.3));

    float baseAng = hash1(strikeSeed + 3.1) * TAU + (hash1(strikeSeed + 4.8) - 0.5) * uStrikeChaos * 1.3;
    float reach = uBoltReach * (0.35 + 0.9 * hash1(strikeSeed + 4.2));
    float originAng = hash1(strikeSeed + 5.4) * TAU;
    float originRadius = mix(coreR * 0.3, uBoltReach * (0.55 + 1.2 * hash1(strikeSeed + 6.8)), hash1(strikeSeed + 6.1));
    vec2 strikeOrigin = vec2(cos(originAng), sin(originAng)) * originRadius;
    strikeOrigin += (vec2(hash1(strikeSeed + 7.1), hash1(strikeSeed + 8.9)) - 0.5) * (0.45 + 0.75 * uStrikeChaos);
    strikeOrigin += 0.12 * vec2(
      sin(strikePhase * 19.0 + hs),
      cos(strikePhase * 17.0 - hs)
    ) * (0.2 + 0.8 * hash1(strikeSeed + 10.7));

    // Rotate into the bolt frame: channel along +x from a randomized
    // strike origin so flashes can erupt anywhere in the nebula.
    float ca = cos(baseAng);
    float sa = sin(baseAng);
    vec2 local = uv - strikeOrigin;
    vec2 p = vec2(ca * local.x + sa * local.y, -sa * local.x + ca * local.y);

    float along = clamp(p.x / reach, 0.0, 1.0);

    // Capsule-style distance: past either end of the channel the
    // longitudinal overshoot is folded into the distance, so the wide
    // glow tapers off in soft round caps instead of hard-edged strips.
    float overX = max(max(-p.x, p.x - reach), 0.0);

    // Conservative cull: pixels provably far from everything this bolt
    // (and its forks) can touch skip all the fractal path evaluation.
    float ampMax = uJaggedness * reach * 0.55;
    float gwMax = uBoltWidth * 0.004 * (14.0 + uBloomRadius * 30.0);
    float forkSpan = branchCount > 0 ? reach * 0.85 * uForkiness : 0.0;
    if (length(vec2(overX, max(abs(p.y) - ampMax, 0.0))) > forkSpan + gwMax * 14.0) {
      continue;
    }

    // Amplitude envelope: pinned at the core, widening as it travels.
    float ampEnv = ampMax * smoothstep(0.0, 0.25, along);
    float offset = boltPath(p.x, strikeSeed, gain) * ampEnv;
    float d = length(vec2(overX, p.y - offset));

    // White-hot filament plus a wide soft glow (inverse falloff reads
    // as an over-exposed electric arc far better than a gaussian).
    float w = uBoltWidth * (1.0 - 0.55 * along) * 0.004;
    float ray = w / (d + w);
    ray *= ray;
    float gw = w * (14.0 + uBloomRadius * 30.0);
    float glow = gw / (d + gw);
    float hueOffset = hash1(strikeSeed + 13.7) * 0.45;

    boltCore += ray * energy;
    boltGlow += glow * glow * energy * uBloomIntensity * 0.3;
    boltGlowColor += glow * glow * energy * palette(along * 0.6 + fi * 0.21 + hueOffset);

    // Secondary forks: thinner fractal channels splitting off the main
    // one at pseudo-random points, inheriting its exact offset there so
    // they stay attached, then decaying in width and length.
    for (int b = 0; b < 6; b++) {
      if (b >= branchCount) break;
      float fb = float(b);
      float bs = strikeSeed + fb * 33.7 + 5.0;
      float bAlong = (0.12 + 0.6 * hash1(bs + 1.0)) * reach;
      float blen = (reach - bAlong) * (0.35 + 0.5 * hash1(bs + 3.0)) * uForkiness;

      // Cheap cull before any noise: the branch lives inside a disc
      // around its (approximate) origin on the main channel.
      float bAmpMax = uJaggedness * blen * uBranchDecay * 0.6;
      float bgwMax = gwMax * uBranchDecay;
      if (length(p - vec2(bAlong, 0.0)) > ampMax + blen + bAmpMax + bgwMax * 12.0) {
        continue;
      }

      vec2 origin = vec2(bAlong, boltPath(bAlong, strikeSeed, gain) * ampMax * smoothstep(0.0, 0.25, bAlong / reach));
      float spread = (hash1(bs + 2.0) - 0.5) * 2.0 * uBranchAngle;
      float cb = cos(spread);
      float sb = sin(spread);
      vec2 q2 = p - origin;
      vec2 bp = vec2(cb * q2.x + sb * q2.y, -sb * q2.x + cb * q2.y);

      float bAlongN = clamp(bp.x / max(blen, 1e-3), 0.0, 1.0);
      float bAmp = uJaggedness * blen * uBranchDecay * 0.6 * smoothstep(0.0, 0.3, bAlongN);
      float bOffset = branchPath(bp.x, bs, gain) * bAmp;
      float bOverX = max(max(-bp.x, bp.x - blen), 0.0);
      float bd = length(vec2(bOverX, bp.y - bOffset));

      float bw = w * uBranchDecay;
      float bray = bw / (bd + bw);
      bray *= bray;
      float bgw = bw * (12.0 + uBloomRadius * 24.0);
      float bglow = bgw / (bd + bgw);
      float benergy = energy * (0.5 + 0.3 * hash1(bs + 4.0));

      boltCore += bray * benergy * 0.85;
      boltGlow += bglow * bglow * benergy * uBloomIntensity * 0.22;
      boltGlowColor += bglow * bglow * benergy * palette(bAlongN * 0.6 + fb * 0.13 + fi * 0.21 + hueOffset);
    }
  }

  // ------------------------------------------------------------------
  // Core: a hot plasma heart the bolts erupt from, with a soft
  // chromatic fringe for a more physical, lens-like light effect.
  // ------------------------------------------------------------------
  float core = uCoreGlow * 0.9 * exp(-pow(radius / coreR, 2.0) * 2.0);
  core += uCoreGlow * 0.25 * exp(-pow(radius / (coreR * 2.5), 2.0))
    * (0.7 + 0.3 * sin(t * 3.4));

  float ab = uChromaticAberration * 0.02;
  float coreR_ = uCoreGlow * 0.9 * exp(-pow((radius + ab) / coreR, 2.0) * 2.0);
  float coreB_ = uCoreGlow * 0.9 * exp(-pow((radius - ab) / coreR, 2.0) * 2.0);

  // ------------------------------------------------------------------
  // Composite.
  // ------------------------------------------------------------------
  float vignette = smoothstep(2.6, 0.3, radius * mix(1.0, 1.6, clamp(uVignette, 0.0, 1.0)));

  vec3 col = nebula * (0.5 + 0.5 * vignette);
  // Filament is near-white (over-exposed channel), glow carries color.
  col += vec3(1.0) * boltCore * 2.2;
  col += (boltGlowColor + vec3(0.25) * boltGlow) * (0.4 + 0.6 * vignette);
  col += mix(palette(0.05), vec3(1.0), 0.4) * core;
  col.r += coreR_ * 0.15;
  col.b += coreB_ * 0.15;
  // Bolts electrify the gas they pass through, lighting it from within.
  col += nebula * (boltCore * 1.6 + boltGlow) * 0.6;

  // Filmic-ish exposure keeps highlights crisp and blacks deep.
  col = 1.0 - exp(-col * max(uExposure, 0.05));
  col = pow(max(col, 0.0), vec3(1.25));
  outColor = vec4(col, 1.0);
}
