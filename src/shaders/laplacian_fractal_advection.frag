#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

// --- UNIFORMS ---
uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeed;

// Fractal controls
uniform int   uFractalIters;
uniform float uFoldScale;
uniform float uFoldOffset;
uniform float uRotSpeed;

// Smoke / advection controls
uniform float uSmokeDensity;
uniform float uSmokeSteps;
uniform float uTurbulence;
uniform float uAdvectionRate;
uniform float uDissipation;

// Lighting
uniform float uLightAngle;
uniform float uLightHeight;
uniform float uLightIntensity;
uniform float uAbsorption;
uniform float uScatterPower;

// Color palette
uniform float uHueShift;
uniform float uHueSpeed;
uniform float uSaturation;
uniform float uBrightness;
uniform float uGlowIntensity;
uniform float uGlowFalloff;

// Camera
uniform float uZoom;
uniform float uCamHeight;
uniform float uCamOrbit;

// ──────────────────────────────────────────────────────────────
// MATH HELPERS
// ──────────────────────────────────────────────────────────────

const float PI  = 3.14159265;
const float TAU = 6.28318530;

mat2 rot2(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

// Hash for seeded noise
float hash13(vec3 p3, float seed) {
    p3 = fract(p3 * 0.1031 + seed * 0.117);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

float hash12(vec2 p, float seed) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031 + seed * 0.117);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth 3D value noise
float noise3(vec3 p, float seed) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash13(i + vec3(0, 0, 0), seed);
    float b = hash13(i + vec3(1, 0, 0), seed);
    float c = hash13(i + vec3(0, 1, 0), seed);
    float d = hash13(i + vec3(1, 1, 0), seed);
    float e = hash13(i + vec3(0, 0, 1), seed);
    float g = hash13(i + vec3(1, 0, 1), seed);
    float h = hash13(i + vec3(0, 1, 1), seed);
    float k = hash13(i + vec3(1, 1, 1), seed);

    return mix(
        mix(mix(a, b, f.x), mix(c, d, f.x), f.y),
        mix(mix(e, g, f.x), mix(h, k, f.x), f.y),
        f.z
    );
}

// FBM (fractal Brownian motion) — 4 octaves max for performance
float fbm(vec3 p, float seed, float turbulence) {
    float val = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    float lacunarity = 2.0 + turbulence * 0.3;
    float gain = 0.5 - turbulence * 0.05;
    for (int i = 0; i < 4; i++) {
        val += amp * noise3(p * freq, seed + float(i) * 1.731);
        freq *= lacunarity;
        amp *= gain;
    }
    return val;
}

// ──────────────────────────────────────────────────────────────
// FRACTAL SDF — Mandelbox-style folding
// ──────────────────────────────────────────────────────────────

// Precomputed rotation matrices (set once per frame in main)
mat2 fracRotXY, fracRotYZ;

float fractalSDF(vec3 p) {
    float scale = uFoldScale;
    float offset = uFoldOffset;
    vec3 orig = p;
    float dr = 1.0;

    for (int i = 0; i < 8; i++) {
        if (i >= uFractalIters) break;

        // Box fold
        p = clamp(p, -1.0, 1.0) * 2.0 - p;

        // Sphere fold
        float r2 = dot(p, p);
        if (r2 < 0.25) {
            float t = 4.0; // fixedR2 / minR2 = 1.0/0.25
            p *= t;
            dr *= t;
        } else if (r2 < 1.0) {
            float t = 1.0 / r2;
            p *= t;
            dr *= t;
        }

        // Rotate (precomputed matrices)
        p.xy *= fracRotXY;
        p.yz *= fracRotYZ;

        // Scale & translate
        p = p * scale + orig * offset;
        dr = dr * abs(scale) + 1.0;
    }

    return length(p) / abs(dr) - 0.002;
}

// ──────────────────────────────────────────────────────────────
// SMOKE DENSITY — fractal surface perturbs a volumetric field
// ──────────────────────────────────────────────────────────────

// Returns density in .x, fractal distance in .y (avoids recomputing SDF)
vec2 smokeDensityAndDist(vec3 p, float time, float seed) {
    float dist = fractalSDF(p);

    // Near-surface density — sharper falloff so smoke hugs the fractal
    float falloff = 6.0 / max(uSmokeDensity, 0.01);
    float surfaceMask = exp(-abs(dist) * falloff);

    // Quick exit for points far from the surface
    if (surfaceMask < 0.01) return vec2(0.0, dist);

    // FBM turbulence advected by time
    vec3 advected = p + vec3(
        sin(time * 0.3 + p.y * 0.7) * uAdvectionRate,
        cos(time * 0.2 + p.z * 0.6) * uAdvectionRate * 0.8,
        sin(time * 0.4 + p.x * 0.5) * uAdvectionRate * 0.6
    );
    float turb = fbm(advected * 1.5, seed, uTurbulence);

    // Turbulence-driven wisps with low base — keeps gaps visible
    float density = surfaceMask * max(turb * 0.9 - 0.05, 0.0);
    density *= exp(-length(p) * uDissipation * 0.2);

    return vec2(max(density, 0.0), dist);
}

// Cheap density for light sampling — fractal SDF only, no FBM
float smokeDensityCheap(vec3 p) {
    float dist = fractalSDF(p);
    float falloff = 6.0 / max(uSmokeDensity, 0.01);
    float surfaceMask = exp(-abs(dist) * falloff);
    return surfaceMask * 0.4 * exp(-length(p) * uDissipation * 0.2);
}

// ──────────────────────────────────────────────────────────────
// COLOR PALETTE (IQ cosine gradient)
// ──────────────────────────────────────────────────────────────

vec3 palette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5) * uSaturation;
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(uHueShift, uHueShift + 0.33, uHueShift + 0.67);
    return a + b * cos(TAU * (c * t + d));
}

vec3 emissionPalette(float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.6, 0.6, 0.6);
    vec3 c = vec3(1.0, 0.7, 0.4);
    vec3 d = vec3(uHueShift + 0.1, uHueShift + 0.45, uHueShift + 0.8);
    return a + b * cos(TAU * (c * t + d));
}

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

vec3 oilPalette(float t) {
    vec3 layerA = 0.5 + 0.5 * cos(TAU * (vec3(1.0, 1.25, 1.6) * t + vec3(0.02, 0.19, 0.37)));
    vec3 layerB = 0.5 + 0.5 * cos(TAU * (vec3(0.75, 1.05, 1.45) * t + vec3(0.28, 0.47, 0.66)));
    return mix(layerA, layerB, 0.5);
}

vec3 smokyOilColor(vec3 p, vec3 rd, vec3 lightDir, float density, float fracDist, float time, float seed) {
    float shimmer = fbm(
        p * 1.8 + vec3(
            cos(time * 0.11 + p.z * 0.3),
            sin(time * 0.13 + p.x * 0.4),
            cos(time * 0.17 - p.y * 0.35)
        ),
        seed + 23.7,
        uTurbulence
    );
    float pearl = 0.5 + 0.5 * cos(
        fracDist * 24.0
        - dot(rd, lightDir) * 5.0
        + shimmer * 8.0
        + time * (0.15 + uHueSpeed)
    );
    float baseParam = length(p) * 0.16 + density * 1.4 + time * uHueSpeed;
    vec3 smokeBase = palette(baseParam);
    vec3 smokeHaze = mix(vec3(luminance(smokeBase)), smokeBase, 0.35 + 0.4 * density);
    vec3 iridescence = oilPalette(baseParam * 0.7 + shimmer * 0.45 + pearl * 0.6);
    return mix(smokeHaze, iridescence, 0.35 + 0.45 * pearl);
}

// ──────────────────────────────────────────────────────────────
// MAIN
// ──────────────────────────────────────────────────────────────

void main() {
    vec2 uv = (gl_FragCoord.xy - uResolution * 0.5) / min(uResolution.x, uResolution.y);
    uv *= uZoom;

    float time = uTime * uTimeScale;
    float seed = uSeed;

    // ── Precompute fractal rotation matrices for this frame ──
    fracRotXY = rot2(time * (0.08 + uRotSpeed * 0.22));
    fracRotYZ = rot2(time * (0.05 + uRotSpeed * 0.17));

    // ── Camera setup ──
    float orbitAngle = time * uCamOrbit;
    float camR = 3.5;
    vec3 ro = vec3(
        cos(orbitAngle) * camR,
        uCamHeight + sin(time * 0.17) * 0.3,
        sin(orbitAngle) * camR
    );
    vec3 target = vec3(0.0);
    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, fwd);
    vec3 rd = normalize(fwd + uv.x * right + uv.y * up);

    // ── Light direction ──
    vec3 lightDir = normalize(vec3(
        cos(uLightAngle),
        uLightHeight,
        sin(uLightAngle)
    ));

    // ── Volume raymarching ──
    int maxSteps = int(uSmokeSteps);
    float tNear = 0.1;
    float tFar = 12.0;
    float stepSize = (tFar - tNear) / float(maxSteps);

    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;
    float glowAccum = 0.0;

    // Jitter start to reduce banding
    float jitter = hash12(gl_FragCoord.xy, seed) * stepSize;

    for (int i = 0; i < 256; i++) {
        if (i >= maxSteps) break;
        if (accumAlpha > 0.97) break;

        float t = tNear + float(i) * stepSize + jitter;
        vec3 p = ro + rd * t;

        vec2 dd = smokeDensityAndDist(p, time, seed);
        float density = dd.x;
        float fracDist = dd.y;

        if (density > 0.001) {
            // ── Light sampling (cheap, 2 steps, SDF-only) ──
            float lightDensity = 0.0;
            lightDensity += smokeDensityCheap(p + lightDir * 0.4);
            lightDensity += smokeDensityCheap(p + lightDir * 0.8);
            float lightTransmit = exp(-lightDensity * 0.4 * uAbsorption);

            // ── Color from position & density ──
            float colorParam = length(p) * 0.2 + density * 1.5 + time * uHueSpeed;
            vec3 smokeCol = smokyOilColor(p, rd, lightDir, density, fracDist, time, seed) * uBrightness;
            float pearl = 0.5 + 0.5 * cos(
                fracDist * 30.0
                + density * 7.0
                - dot(rd, lightDir) * 4.5
                + time * (0.12 + uHueSpeed)
            );
            vec3 oilSheen = oilPalette(colorParam * 0.75 + pearl * 0.55);
            smokeCol = mix(smokeCol, oilSheen * (0.75 + 0.55 * pearl), 0.2 + 0.4 * density);
            smokeCol = mix(vec3(luminance(smokeCol)), smokeCol, clamp(0.55 + 0.35 * uSaturation, 0.0, 1.4));

            // ── Emission near fractal surface (reuse fracDist) ──
            float emission = exp(-abs(fracDist) * 8.0) * uGlowIntensity;
            vec3 emitCol = mix(
                emissionPalette(colorParam + 0.3),
                oilPalette(colorParam + 0.25 + pearl * 0.35),
                0.55
            ) * emission * 2.2;

            // ── Forward scattering (Henyey-Greenstein-like) ──
            float cosTheta = dot(rd, lightDir);
            float scatter = pow(max(0.5 + 0.5 * cosTheta, 0.0), uScatterPower);

            // ── Combine ──
            vec3 radiance = smokeCol * lightTransmit * uLightIntensity * (1.0 + scatter * 2.0) + emitCol;

            // ── Beer-Lambert absorption (softer for translucency) ──
            float alpha = (1.0 - exp(-density * stepSize * uAbsorption));
            accumColor += radiance * alpha * (1.0 - accumAlpha);
            accumAlpha += alpha * (1.0 - accumAlpha);

            // Glow accumulation for post-effect
            glowAccum += emission * stepSize / (1.0 + t * uGlowFalloff);
        }
    }

    // ── Background gradient ──
    float bgGrad = 0.5 + 0.5 * rd.y;
    vec3 bgColor = mix(
        vec3(0.01, 0.005, 0.02),
        vec3(0.03, 0.01, 0.05),
        bgGrad
    );
    vec3 col = mix(bgColor, accumColor, accumAlpha);

    // ── Add volumetric glow halo ──
    col += glowAccum * palette(time * uHueSpeed + 0.5) * 0.3;

    // ── Tone mapping (ACES-ish) ──
    col = col / (col + 0.5);

    // ── Vignette ──
    vec2 vUv = gl_FragCoord.xy / uResolution;
    col *= 1.0 - length(vUv - 0.5) * 0.6;

    // ── Final gamma ──
    col = pow(max(col, 0.0), vec3(0.9));

    outColor = vec4(col, 1.0);
}
