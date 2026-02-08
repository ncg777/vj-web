#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2  uResolution;
uniform float uPhase;

// Structure
uniform int   uOrder;       // Hadamard order: matrix size = 2^order
uniform float uRotSpeed;    // rotations per loop cycle
uniform float uZoom;        // disk scale
uniform float uRadialPow;   // power curve on radial mapping
uniform float uSpiral;      // spiral twist amount

// Cell appearance
uniform float uSmooth;      // smoothing between cells (0 = crisp)
uniform float uGap;         // visible gap between cells
uniform float uFadeStart;   // edge fade begin (fraction of radius)
uniform float uFadeWidth;   // edge fade extent
uniform float uPulse;       // radial breathing amount

// Color
uniform float uBaseR;
uniform float uBaseG;
uniform float uBaseB;
uniform float uAmpR;        // red modulation amplitude (cos)
uniform float uAmpG;        // green modulation amplitude (sin)
uniform float uAmpB;        // blue modulation amplitude (sin)
uniform float uFreqR;       // red oscillation cycles per loop
uniform float uFreqG;       // green oscillation cycles per loop
uniform float uFreqB;       // blue oscillation cycles per loop

// Extra
uniform float uGlow;        // glow halo intensity
uniform float uBgBright;    // background brightness
uniform float uSeed;        // random angular offset

#define PI  3.14159265359
#define TAU 6.28318530718

/* ── Hadamard value (Sylvester / natural order) ────────────────
   H[row][col] = (-1)^popcount(row & col)
   Returns 1.0 for +1 entries, 0.0 for -1 entries. */
int popcount(int x) {
    int c = 0;
    int v = x;
    for (int i = 0; i < 16; i++) {
        c += v & 1;
        v >>= 1;
        if (v == 0) break;
    }
    return c;
}

float hadamard(int row, int col) {
    return float(1 - (popcount(row & col) & 1));
}

/* Simple hash for seed-based offset */
float hash(float n) {
    return fract(sin(n * 127.1) * 43758.5453);
}

void main() {
    float minDim = min(uResolution.x, uResolution.y);
    vec2 uv = (2.0 * gl_FragCoord.xy - uResolution) / minDim;

    // Zoom
    uv /= max(uZoom, 0.01);

    float r     = length(uv);
    float theta = atan(uv.y, uv.x);
    float t     = uPhase;                       // 0 → 1 per loop

    // Seed-based static angular offset
    theta += hash(uSeed * 13.37) * TAU;

    // Rotation
    theta += t * uRotSpeed * TAU;

    // Spiral warp: angle shifts proportionally to radius
    theta += uSpiral * r * TAU;

    // Radial pulse (breathing)
    float rAdj = r + uPulse * 0.08 * sin(t * TAU);

    // Matrix dimensions
    int   size  = 1 << clamp(uOrder, 1, 10);
    float fSize = float(size);

    // ── Map polar coords → matrix indices ─────────────────────
    // Radius  → row  (center = row 0, edge = last row)
    float rN   = clamp(rAdj, 0.0, 0.9999);
    rN         = pow(rN, uRadialPow);
    float rowF = rN * fSize;
    int   row  = clamp(int(floor(rowF)), 0, size - 1);

    // Angle → column (wraps)
    float aN   = fract(theta / TAU);
    float colF = aN * fSize;
    int   col  = clamp(int(floor(colF)), 0, size - 1);

    // ── Compute cell value ────────────────────────────────────
    float v;
    if (uSmooth > 0.001) {
        // Bilinear interpolation between neighbouring cells
        int row2 = min(row + 1, size - 1);
        int col2 = (col + 1) % size;          // wrap angularly

        float fr = fract(rowF);
        float fc = fract(colF);

        // Smoothstep the fractional parts for a tuneable blend
        float sf = uSmooth;
        fr = smoothstep(0.5 - 0.5 * sf, 0.5 + 0.5 * sf, fr);
        fc = smoothstep(0.5 - 0.5 * sf, 0.5 + 0.5 * sf, fc);

        v = mix(
            mix(hadamard(row, col),  hadamard(row2, col),  fr),
            mix(hadamard(row, col2), hadamard(row2, col2), fr),
            fc
        );
    } else {
        v = hadamard(row, col);
    }

    // ── Cell gap ──────────────────────────────────────────────
    if (uGap > 0.001) {
        float edgeR = min(fract(rowF), 1.0 - fract(rowF));
        float edgeC = min(fract(colF), 1.0 - fract(colF));
        float edge  = min(edgeR, edgeC);
        v *= smoothstep(0.0, uGap * 0.5 + 0.005, edge);
    }

    // ── Time-varying colour (matches original Java formula) ───
    float cr = (uBaseR + uAmpR * cos(uFreqR * TAU * t)) * v;
    float cg = (uBaseG + uAmpG * sin(uFreqG * TAU * t)) * v;
    float cb = (uBaseB + uAmpB * sin(uFreqB * TAU * t)) * v;

    // ── Edge fade ─────────────────────────────────────────────
    float fadeEnd = uFadeStart + max(uFadeWidth, 0.001);
    float alpha   = smoothstep(fadeEnd, uFadeStart, r);

    // ── Glow halo around disk edge ────────────────────────────
    vec3 glowCol = vec3(
        uBaseR + uAmpR * cos(uFreqR * TAU * t),
        uBaseG + uAmpG * sin(uFreqG * TAU * t),
        uBaseB + uAmpB * sin(uFreqB * TAU * t)
    );
    float glowFactor = uGlow * 0.35 * exp(-10.0 * max(r - uFadeStart, 0.0));

    // ── Compose ───────────────────────────────────────────────
    vec3 fg    = clamp(vec3(cr, cg, cb), 0.0, 1.0);
    vec3 bg    = vec3(uBgBright);
    vec3 color = mix(bg, fg, alpha) + glowFactor * glowCol;

    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
