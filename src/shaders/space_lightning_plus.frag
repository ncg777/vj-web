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
const float PI = 3.14159265359;

// ---------------------------------------------------------------------
// Hashing & noise primitives.
// ---------------------------------------------------------------------
float hash1(float n) {
  return fract(sin(n * 127.1) * 43758.5453123);
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec2 hash2(vec2 p) {
  float n = sin(dot(p, vec2(41.3, 289.1)));
  return fract(vec2(262144.0, 32768.0) * n) - 0.5;
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

// Fractional Brownian motion; octave count fixed for perf, amplitude
// controlled by the caller for a variable-looking level of detail.
float fbm(vec2 p, int octaves) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 7; i++) {
    if (i >= octaves) break;
    v += a * noise(p);
    p = ROT * p * 2.02 + vec2(11.7, 5.1);
    a *= 0.5;
  }
  return v;
}

// Curl of a scalar noise potential -> divergence-free velocity field,
// used to advect the gas so it swirls and diffuses like a real fluid
// instead of just scrolling.
vec2 curlNoise(vec2 p) {
  const float e = 0.09;
  float n1 = fbm(p + vec2(0.0, e), 4);
  float n2 = fbm(p - vec2(0.0, e), 4);
  float n3 = fbm(p + vec2(e, 0.0), 4);
  float n4 = fbm(p - vec2(e, 0.0), 4);
  float dx = (n1 - n2) / (2.0 * e);
  float dy = (n3 - n4) / (2.0 * e);
  return vec2(dy, -dx);
}

// Inigo Quilez style cosine palette, continuously rotated by uHueSpeed
// so the whole spectrum drifts through the nebula and bolts over time.
vec3 palette(float t) {
  float hue = t + uPaletteShift + uTime * uTimeScale * uHueSpeed;
  return uColorPrimary + uColorSecondary * cos(TAU * (uColorAccent * hue));
}

// Smallest signed periodic angular difference, continuous everywhere.
float angDiff(float a, float b) {
  return atan(sin(a - b), cos(a - b));
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

// ---------------------------------------------------------------------
// Fractal midpoint-displacement lightning bolt.
//
// A straight segment between `start` and `end` is recursively bisected;
// at every subdivision the new midpoint is displaced perpendicular to
// the segment by an amount that shrinks with a Hurst-style roughness
// exponent each generation. This is the classic algorithm used to
// synthesize realistic fractal lightning / terrain silhouettes, and it
// naturally produces the jagged, self-similar look of a real electric
// discharge. The resulting polyline's minimum distance to `p` (and its
// normalized arc-length position) are returned so branches and glow can
// be evaluated from it.
// ---------------------------------------------------------------------
const int BOLT_DEPTH = 5;
const int BOLT_POINTS = 33; // 2^BOLT_DEPTH + 1

void buildBolt(vec2 start, vec2 end, float amp, float roughness, float seed, out vec2 pts[BOLT_POINTS]) {
  pts[0] = start;
  pts[BOLT_POINTS - 1] = end;
  int step = BOLT_POINTS - 1;
  float a = amp;
  for (int level = 0; level < BOLT_DEPTH; level++) {
    int half_ = step / 2;
    for (int i = half_; i < BOLT_POINTS - 1; i += step) {
      vec2 pa = pts[i - half_];
      vec2 pb = pts[i + half_];
      vec2 mid = (pa + pb) * 0.5;
      vec2 dir = pb - pa;
      vec2 perp = normalize(vec2(-dir.y, dir.x) + 1e-6);
      float n = hash(vec2(seed + float(i), float(level) * 17.13 + seed)) - 0.5;
      pts[i] = mid + perp * n * a;
    }
    a *= roughness;
    step = half_;
  }
}

// Distance from p to the fractal bolt polyline; also returns the
// normalized arc-length parameter (0 at the root, 1 at the tip) of the
// closest segment via `tOut`, and the index of the closest segment via
// `iOut`, so branch spawn points can be placed procedurally along it.
float boltDistance(vec2 p, vec2 pts[BOLT_POINTS], out float tOut, out int iOut) {
  float best = 1e9;
  float bestT = 0.0;
  int bestI = 0;
  for (int i = 0; i < BOLT_POINTS - 1; i++) {
    float d = sdSegment(p, pts[i], pts[i + 1]);
    if (d < best) {
      best = d;
      bestT = float(i) / float(BOLT_POINTS - 2);
      bestI = i;
    }
  }
  tOut = bestT;
  iOut = bestI;
  return best;
}

const int BRANCH_DEPTH = 3;
const int BRANCH_POINTS = 9; // 2^BRANCH_DEPTH + 1

void buildBranch(vec2 start, vec2 end, float amp, float roughness, float seed, out vec2 pts[BRANCH_POINTS]) {
  pts[0] = start;
  pts[BRANCH_POINTS - 1] = end;
  int step = BRANCH_POINTS - 1;
  float a = amp;
  for (int level = 0; level < BRANCH_DEPTH; level++) {
    int half_ = step / 2;
    for (int i = half_; i < BRANCH_POINTS - 1; i += step) {
      vec2 pa = pts[i - half_];
      vec2 pb = pts[i + half_];
      vec2 mid = (pa + pb) * 0.5;
      vec2 dir = pb - pa;
      vec2 perp = normalize(vec2(-dir.y, dir.x) + 1e-6);
      float n = hash(vec2(seed + float(i) * 3.1, float(level) * 9.7 + seed)) - 0.5;
      pts[i] = mid + perp * n * a;
    }
    a *= roughness;
    step = half_;
  }
}

float branchDistance(vec2 p, vec2 pts[BRANCH_POINTS]) {
  float best = 1e9;
  for (int i = 0; i < BRANCH_POINTS - 1; i++) {
    best = min(best, sdSegment(p, pts[i], pts[i + 1]));
  }
  return best;
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
  float angle = atan(uv.y, uv.x);

  // ------------------------------------------------------------------
  // Nebula: several parallax layers of curl-advected fbm gas, each
  // diffusing at its own depth/speed for a complex dynamic cloudscape.
  // ------------------------------------------------------------------
  vec3 nebula = vec3(0.0);
  const int LAYERS = 3;
  for (int layer = 0; layer < LAYERS; layer++) {
    float fl = float(layer);
    float depth = 1.0 + fl * uParallax;
    vec2 lp = uv * (0.9 + fl * 0.55) * uCloudDetail + seed * (fl + 1.0) * 3.7;

    // Curl-noise domain warp: divergence-free advection gives the gas a
    // genuinely fluid, swirling diffusion instead of a simple scroll.
    float tc = t * uCloudSpeed;
    vec2 flow = curlNoise(lp * 0.6 + tc * (0.05 + 0.03 * fl)) * uCurlStrength;
    vec2 warped = lp + flow + tc * (0.04 / depth) * vec2(1.0, 0.4);

    vec2 q = vec2(
      fbm(warped + vec2(0.0, 0.0), 5),
      fbm(warped + vec2(5.2, 1.3), 5)
    );
    vec2 r2 = vec2(
      fbm(warped + (1.1 + uWarp) * q + vec2(1.7, 9.2), 5),
      fbm(warped + (1.1 + uWarp) * q + vec2(8.3, 2.8), 5)
    );
    float f = fbm(warped + (1.4 + uWarp) * r2, 6);

    // Fine turbulent detail layered on top for wispy filaments.
    float fine = fbm(warped * 3.1 - t * 0.3, 4);
    f = mix(f, fine, clamp(uTurbulence, 0.0, 1.0) * 0.4);

    float density = smoothstep(uCloudDensity - 0.22, uCloudDensity + 0.22, f);
    density = pow(density, mix(1.0, 3.0, clamp(uCloudContrast, 0.0, 1.0)));

    vec3 gasA = palette(f * 1.4 + radius * 0.2 + fl * 0.31);
    vec3 gasB = palette(length(q) * 0.9 + 0.33 + fl * 0.17);
    vec3 gasC = palette(length(r2) * 0.7 + 0.66 - fl * 0.12);
    vec3 layerCol = mix(gasA, gasB, clamp(length(q) * 1.2 - 0.2, 0.0, 1.0));
    layerCol = mix(layerCol, gasC, clamp(r2.x * r2.x * 1.4 - 0.15, 0.0, 1.0));

    nebula += max(layerCol, 0.0) * density / depth;
  }
  nebula /= float(LAYERS) * 0.6;

  // Sparse cosmic starfield twinkling behind/within the gas.
  vec2 starCell = floor(uv * (28.0 + uStarDensity * 40.0));
  vec2 starUv = fract(uv * (28.0 + uStarDensity * 40.0)) - 0.5;
  float starHash = hash(starCell + seed * 5.3);
  float starMask = step(1.0 - clamp(uStarDensity, 0.0, 1.0) * 0.12, starHash);
  float twinkle = 0.5 + 0.5 * sin(t * (2.0 + starHash * 6.0) + starHash * TAU);
  float star = starMask * exp(-dot(starUv, starUv) * 90.0) * (0.4 + 0.6 * twinkle);
  nebula += star * palette(starHash + t * 0.02);

  // ------------------------------------------------------------------
  // Lightning: fractal branching bolts erupting from the core in every
  // direction, each one re-randomized on every discrete strike for a
  // true flickering, non-repeating electrical storm.
  // ------------------------------------------------------------------
  float bolts = 0.0;
  vec3 boltGlowColor = vec3(0.0);
  int boltCount = int(clamp(uBoltCount, 1.0, 10.0));
  float coreR = max(uCoreSize * 0.5, 0.04);

  for (int i = 0; i < 10; i++) {
    if (i >= boltCount) break;
    float fi = float(i);
    float hs = fi * 12.9 + seed * 91.7;

    // Discrete strike timeline: each bolt fires periodically and is
    // reshaped from scratch on every strike (unique random seed per
    // strike index), then flashes bright and decays quickly - the
    // hallmark of a real electric discharge rather than a static shape.
    float rate = max(uStrikeRate, 0.05) * (0.6 + 0.8 * hash1(hs + 1.0));
    float cycle = t * rate + hash1(hs) * 50.0;
    float strikeIndex = floor(cycle);
    float strikePhase = fract(cycle);
    float strikeSeed = hs + strikeIndex * 71.3 * (1.0 + uStrikeChaos);

    // Sudden bright flash, fast exponential-feeling decay.
    float flash = exp(-strikePhase * (5.0 + uStrikeChaos * 10.0));
    flash *= smoothstep(0.0, 0.03, strikePhase);
    if (flash < 0.01) continue;

    float baseAng = hash1(strikeSeed + 3.1) * TAU;
    float reach = uBoltReach * (0.5 + hash1(strikeSeed + 4.2));
    vec2 start = coreR * vec2(cos(baseAng), sin(baseAng)) * 0.6;
    vec2 end = start + reach * vec2(cos(baseAng + (hash1(strikeSeed + 5.3) - 0.5) * 0.8),
                                     sin(baseAng + (hash1(strikeSeed + 5.3) - 0.5) * 0.8));

    vec2 pts[BOLT_POINTS];
    float amp = uJaggedness * reach * 0.5;
    buildBolt(start, end, amp, clamp(uRoughness, 0.05, 0.95), strikeSeed, pts);

    float tParam;
    int segIdx;
    float d = boltDistance(uv, pts, tParam, segIdx);

    float w = uBoltWidth * (0.4 + 0.6 * (1.0 - tParam)) * 0.02;
    float ray = exp(-pow(d / max(w, 1e-4), 2.0));
    float glow = exp(-pow(d / (max(w, 1e-4) * (6.0 + uBloomRadius * 14.0)), 2.0));

    float energy = flash * uBoltIntensity;
    bolts += ray * energy * 2.4;
    bolts += glow * energy * uBloomIntensity * 0.5;
    boltGlowColor += glow * energy * palette(tParam * 0.6 + fi * 0.21);

    // Secondary forks branching off the main channel at pseudo-random
    // points, each a smaller fractal bolt of its own with a decayed
    // amplitude and width - real lightning branches recursively too.
    int branchCount = int(clamp(uBranchCount, 0.0, 6.0));
    for (int b = 0; b < 6; b++) {
      if (b >= branchCount) continue;
      float fb = float(b);
      float bs = strikeSeed + fb * 33.7 + 5.0;
      float along = 0.15 + 0.75 * hash1(bs + 1.0);
      int originIdx = int(along * float(BOLT_POINTS - 2));
      vec2 origin = pts[originIdx];
      vec2 tangent = normalize(pts[min(originIdx + 1, BOLT_POINTS - 1)] - pts[max(originIdx - 1, 0)] + 1e-6);
      float spread = (hash1(bs + 2.0) - 0.5) * 2.0 * uBranchAngle;
      float ca = cos(spread);
      float sa = sin(spread);
      vec2 dir = mat2(ca, -sa, sa, ca) * tangent;
      float blen = reach * (1.0 - along) * (0.4 + 0.5 * hash1(bs + 3.0)) * uForkiness;
      vec2 bend = origin + dir * blen;

      vec2 bpts[BRANCH_POINTS];
      float bamp = amp * uBranchDecay * (1.0 - along * 0.5);
      buildBranch(origin, bend, bamp, clamp(uRoughness, 0.05, 0.95), bs, bpts);

      float bd = branchDistance(uv, bpts);
      float bw = w * uBranchDecay * 1.1;
      float bray = exp(-pow(bd / max(bw, 1e-4), 2.0));
      float bglow = exp(-pow(bd / (max(bw, 1e-4) * (6.0 + uBloomRadius * 14.0)), 2.0));
      float benergy = energy * (0.55 + 0.25 * hash1(bs + 4.0));

      bolts += bray * benergy * 2.0;
      bolts += bglow * benergy * uBloomIntensity * 0.4;
      boltGlowColor += bglow * benergy * palette(along * 0.6 + fb * 0.13 + fi * 0.21);
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

  vec3 boltColor = mix(boltGlowColor, vec3(1.0), 0.5);
  vec3 col = nebula * (0.5 + 0.5 * vignette);
  col += boltColor * bolts * (0.4 + 0.6 * vignette);
  col += mix(palette(0.05), vec3(1.0), 0.4) * core;
  col.r += coreR_ * 0.15;
  col.b += coreB_ * 0.15;
  // Bolts electrify the gas they pass through, lighting it from within.
  col += nebula * bolts * 0.7;

  // Filmic-ish exposure keeps highlights crisp and blacks deep.
  col = 1.0 - exp(-col * max(uExposure, 0.05));
  col = pow(max(col, 0.0), vec3(1.25));
  outColor = vec4(col, 1.0);
}
