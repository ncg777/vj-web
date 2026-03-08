#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

// ─── UNIFORMS ───────────────────────────────────────────────
uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeed;

// Fractal geometry
uniform int   uFractalIters;
uniform float uFoldScale;
uniform float uFoldOffset;
uniform float uRotSpeed;
uniform float uKaleidoFolds;

// Rendering
uniform float uRaySteps;
uniform float uDetailLevel;

// Lighting
uniform float uLightIntensity;
uniform float uSpecPower;

// Color
uniform float uHueShift;
uniform float uHueSpeed;
uniform float uSaturation;
uniform float uBrightness;
uniform float uContrast;
uniform float uGlowIntensity;
uniform float uChromaShift;
uniform float uSmoothBlend;

// Camera
uniform float uZoom;
uniform float uCamHeight;
uniform float uCamOrbit;

// ─── CONSTANTS ──────────────────────────────────────────────
const float PI  = 3.14159265;
const float TAU = 6.28318530;

// ─── MATH HELPERS ───────────────────────────────────────────
mat2 rot2(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

float smin(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * h * k * (1.0 / 6.0);
}

// ─── PSYCHEDELIC COLOR PALETTES ─────────────────────────────
vec3 pal_neon(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.0, 0.8, 0.6)
         + vec3(0.00, 0.33, 0.67)));
}

// Warm lava-lamp tones: amber → magenta → deep orange
vec3 pal_lava(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.7, 0.9, 1.3)
         + vec3(0.00, 0.18, 0.55)));
}

vec3 pal_fire(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.3, 0.5, 0.8)
         + vec3(0.25, 0.00, 0.55)));
}

vec3 pal_ice(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.6, 0.9, 1.2)
         + vec3(0.55, 0.70, 0.85)));
}

// ─── ORBIT TRAP STATE ───────────────────────────────────────
vec4 orbitTrap;

// ─── FRACTAL DISTANCE ESTIMATOR ─────────────────────────────
// Hybrid Mandelbox / Kaleidoscopic IFS with orbit traps
float fractalDE(vec3 p, float time) {
    vec3 offset = p;
    float dr = 1.0;
    float scale = uFoldScale;

    orbitTrap = vec4(1e10);

    mat2 rXY = rot2(time * (0.01 + uRotSpeed * 0.03));
    mat2 rYZ = rot2(time * (0.006 + uRotSpeed * 0.022));
    mat2 rXZ = rot2(time * (0.008 + uRotSpeed * 0.026));

    float kalAngle = PI / max(uKaleidoFolds, 1.5);

    for (int i = 0; i < 16; i++) {
        if (i >= uFractalIters) break;

        // ── Kaleidoscopic fold ──
        p = abs(p);
        float angle = atan(p.y, p.x);
        angle = mod(angle + kalAngle * 0.5, kalAngle) - kalAngle * 0.5;
        float r = length(p.xy);
        p.xy = vec2(cos(angle), sin(angle)) * r;

        // ── Menger-style conditional folds ──
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;

        // ── Box fold – smooth near boundary for lava-lamp softness ──
        float sk = max(uSmoothBlend * 0.5, 0.01);
        vec3 wt = smoothstep(vec3(1.0 - sk), vec3(1.0 + sk), abs(p));
        p = mix(p, sign(p) * (2.0 - abs(p)), wt);

        // ── Sphere fold – enlarged radii so blobs are big and merge ──
        float r2 = dot(p, p);
        float minR2  = 0.45 + uSmoothBlend * 0.35;   // inner: ~0.45-0.80
        float fixedR2 = 1.6 + uSmoothBlend * 0.60;   // outer: ~1.60-2.20
        if (r2 < minR2) {
            float t = fixedR2 / minR2;
            p *= t;  dr *= t;
        } else if (r2 < fixedR2) {
            float t = fixedR2 / r2;
            p *= t;  dr *= t;
        }

        // ── Accumulate orbit traps for coloring ──
        orbitTrap = min(orbitTrap, vec4(
            abs(p.x),
            length(p.xz),
            dot(p, p),
            abs(p.z - p.x)
        ));

        // ── Rotate ──
        p.xy *= rXY;
        p.yz *= rYZ;
        p.xz *= rXZ;

        // ── Scale & translate ──
        p = p * scale + offset * uFoldOffset;
        dr = dr * abs(scale) + 1.0;
    }

    return length(p) / abs(dr);
}

// ─── NORMAL ESTIMATION (tetrahedron technique) ──────────────
vec3 calcNormal(vec3 p, float time) {
    float h = 0.0005 * uDetailLevel;
    vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * fractalDE(p + k.xyy * h, time) +
        k.yyx * fractalDE(p + k.yyx * h, time) +
        k.yxy * fractalDE(p + k.yxy * h, time) +
        k.xxx * fractalDE(p + k.xxx * h, time)
    );
}

// ─── AMBIENT OCCLUSION ──────────────────────────────────────
float calcAO(vec3 p, vec3 n, float time) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 3; i++) {
        float h = 0.01 + 0.15 * float(i) / 2.0;
        float d = fractalDE(p + n * h, time);
        occ += (h - d) * sca;
        sca *= 0.9;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

// ─── RAYMARCH ────────────────────────────────────────────────
float march(vec3 ro, vec3 rd, float time, int maxSteps,
            out vec3 hitPos, out vec3 hitNorm,
            out vec4 hitTrap, out int hitSteps) {
    float t = 0.0;
    float d = 0.0;
    hitSteps = 0;

    for (int i = 0; i < 200; i++) {
        if (i >= maxSteps) break;
        vec3 p = ro + rd * t;
        d = fractalDE(p, time);
        if (d < 0.0004 * (1.0 + t * 0.5)) {
            hitPos   = p;
            hitNorm  = calcNormal(p, time);
            hitTrap  = orbitTrap;
            hitSteps = i;
            return t;
        }
        t += d * 0.8;
        hitSteps = i;
        if (t > 20.0) break;
    }
    hitPos  = ro + rd * t;
    hitNorm = vec3(0.0);
    hitTrap = vec4(1e10);
    return -1.0;
}

// ─── SHADE A HIT POINT ─────────────────────────────────────
vec3 shade(vec3 p, vec3 n, vec3 rd, vec4 trap,
           int steps, int maxSteps, float time) {

    float colorT = time * uHueSpeed + uHueShift;

    // ── Orbit-trap psychedelic coloring ──
    vec3 c1 = pal_neon(trap.x * 2.0 + colorT);
    vec3 c2 = pal_lava(trap.y * 1.5 + colorT + 0.33);
    vec3 c3 = pal_fire(sqrt(trap.z) * 0.5 + colorT + 0.67);
    vec3 c4 = pal_ice(trap.w * 1.8 + colorT * 1.5 + 0.5);

    vec3 baseColor = c1 * 0.30 + c2 * 0.30 + c3 * 0.20 + c4 * 0.20;
    baseColor = pow(baseColor, vec3(0.8));
    baseColor = mix(vec3(dot(baseColor, vec3(0.299, 0.587, 0.114))),
                    baseColor, uSaturation);

    // ── Two-point lighting ──
    vec3 l1 = normalize(vec3(cos(time * 0.1) * 0.7, 0.8,
                             sin(time * 0.13) * 0.7));
    vec3 l2 = normalize(vec3(-0.5, -0.3, 0.8));

    float diff1 = max(dot(n, l1), 0.0);
    float diff2 = max(dot(n, l2), 0.0) * 0.3;

    // ── Specular highlights ──
    vec3 h1 = normalize(l1 - rd);
    float spec = pow(max(dot(n, h1), 0.0), uSpecPower);

    // ── Rim light ──
    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    vec3 rimCol = pal_lava(colorT + trap.x) * rim * 1.5;

    // ── AO + step-based AO ──
    float ao     = calcAO(p, n, time);
    float stepAO = 1.0 - float(steps) / float(maxSteps);

    // ── Edge glow from orbit trap ──
    float edgeGlow = exp(-trap.x * 8.0) * uGlowIntensity;
    vec3 glowCol = pal_fire(colorT + trap.y * 2.0) * edgeGlow;

    // ── Combine ──
    vec3 col = baseColor * (diff1 + diff2 + 0.15) * ao * stepAO
             * uBrightness * uLightIntensity;
    col += spec * vec3(1.0, 0.9, 0.8) * 0.6;
    col += rimCol;
    col += glowCol;

    // ── Distance fog with color ──
    float dist = length(p);
    float fog = 1.0 - exp(-dist * 0.06);
    vec3 fogCol = pal_neon(colorT + 0.5) * 0.08;
    col = mix(col, fogCol, fog);

    return col;
}

// ─── BACKGROUND ─────────────────────────────────────────────
vec3 background(vec2 uv, vec3 rd, float time, int steps, int maxSteps) {
    float colorT = time * uHueSpeed * 0.3 + uHueShift;
    float bgParam = length(uv) * 0.5 + colorT;
    vec3 bg = pal_neon(bgParam) * 0.03 + pal_lava(bgParam + 0.3) * 0.02;

    float glow = float(steps) / float(max(maxSteps, 1));
    bg += pal_fire(time * uHueSpeed + glow) * glow * glow
        * uGlowIntensity * 0.3;

    return bg;
}

// ─── MAIN ───────────────────────────────────────────────────
void main() {
    vec2 uv = (gl_FragCoord.xy - uResolution * 0.5)
            / min(uResolution.x, uResolution.y);
    uv *= uZoom;

    float time = uTime * uTimeScale;
    int maxSteps = int(uRaySteps);

    // ── Camera ──
    float orbitAngle = time * uCamOrbit;
    float camR = 3.5;
    vec3 ro = vec3(cos(orbitAngle) * camR,
                   uCamHeight + sin(time * 0.17) * 0.4,
                   sin(orbitAngle) * camR);
    vec3 fwd   = normalize(-ro);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up    = cross(right, fwd);

    // ── Single-ray march ──
    vec3 rd = normalize(fwd + uv.x * right + uv.y * up);

    vec3 hitPos, hitNorm;
    vec4 hitTrap;
    int hitSteps;
    float d = march(ro, rd, time, maxSteps, hitPos, hitNorm, hitTrap, hitSteps);

    vec3 col;
    if (d > 0.0) {
        col = shade(hitPos, hitNorm, rd, hitTrap, hitSteps, maxSteps, time);
    } else {
        col = background(uv, rd, time, hitSteps, maxSteps);
    }

    // ── Screen-space chromatic aberration ──
    float ca = uChromaShift * 0.003;
    float r2 = dot(uv, uv);
    col.r *= 1.0 + ca * r2 * 2.0;
    col.b *= 1.0 - ca * r2 * 2.0;

    // ── Contrast ──
    col = mix(vec3(0.5), col, uContrast);
    col = max(col, 0.0);

    // ── Tone mapping (filmic) ──
    col = col / (col + 0.35);

    // ── Vignette ──
    vec2 vUv = gl_FragCoord.xy / uResolution;
    col *= 1.0 - length(vUv - 0.5) * 0.5;

    // ── Gamma ──
    col = pow(max(col, 0.0), vec3(0.85));

    outColor = vec4(col, 1.0);
}
