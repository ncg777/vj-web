#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2 uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uSeed;

uniform int uFractalIters;
uniform float uFoldScale;
uniform float uFoldOffset;
uniform float uRotSpeed;

uniform float uRaySteps;
uniform float uDetailLevel;

uniform float uLightIntensity;
uniform float uSpecPower;

uniform float uHueShift;
uniform float uHueSpeed;
uniform float uSaturation;
uniform float uBrightness;
uniform float uContrast;
uniform float uGlowIntensity;
uniform float uChromaShift;
uniform float uSmoothBlend;

uniform float uZoom;
uniform float uCamHeight;
uniform float uCamOrbit;

const float TAU = 6.28318530718;

mat2 rot2(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.45));
    p += dot(p, p + 34.23 + fract(uSeed * 0.000001));
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.55;
    float norm = 0.0;
    int octaves = clamp(uFractalIters + 3, 3, 8);
    float persistence = mix(0.46, 0.60, uSmoothBlend);

    for (int i = 0; i < 8; i++) {
        if (i >= octaves) {
            break;
        }
        sum += amp * noise(p);
        norm += amp;
        p = rot2(0.45 + 0.03 * float(i)) * p * 2.02 + vec2(0.31, -0.27);
        amp *= persistence;
    }

    return sum / max(norm, 0.0001);
}

vec3 palNeon(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.0, 0.8, 0.6) + vec3(0.00, 0.33, 0.67)));
}

vec3 palLava(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.7, 0.9, 1.3) + vec3(0.00, 0.18, 0.55)));
}

vec3 palFire(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(1.3, 0.5, 0.8) + vec3(0.25, 0.00, 0.55)));
}

vec3 palIce(float t) {
    return 0.5 + 0.5 * cos(TAU * (t * vec3(0.6, 0.9, 1.2) + vec3(0.55, 0.70, 0.85)));
}

vec2 warp(vec2 p, float time) {
    float drift = time * (0.05 + uRotSpeed * 0.12);
    float warpScale = 0.75 + uFoldScale * 0.35;
    float warpGain = 0.22 + uFoldOffset * 0.25;

    vec2 q = vec2(
        fbm(p * warpScale + vec2(0.0, drift)),
        fbm(p * warpScale + vec2(4.8, -drift * 0.8))
    );

    vec2 r = vec2(
        fbm(p * (warpScale + 0.6) + (q - 0.5) * 2.0 + vec2(1.7, -2.6) + drift * 0.4),
        fbm(p * (warpScale + 0.4) + (q - 0.5) * 2.0 + vec2(-3.1, 0.9) - drift * 0.3)
    );

    return p + (q + r - 1.0) * warpGain;
}

vec3 background(vec2 uv, vec3 rd, float time) {
    float colorT = uHueShift + time * uHueSpeed * 0.25 + fract(uSeed * 0.000001);
    vec2 p = uv * (1.8 + uZoom * 0.55);
    p += vec2(time * uCamOrbit * 0.6, uCamHeight * 0.22);

    vec2 q = warp(p, time);
    vec2 r = warp(q * 1.35 + vec2(2.4, -1.8), time * 0.7);

    float field = fbm(q);
    float filaments = fbm(r * 1.7 + vec2(time * 0.03, -time * 0.025));
    float mist = fbm(p * 0.65 - vec2(time * 0.02, time * 0.015));
    float ridges = 1.0 - abs(2.0 * fbm(q * (1.4 + 0.2 * uDetailLevel)) - 1.0);
    float glowMask = smoothstep(0.30, 0.88, mix(field, ridges, 0.55));
    float veil = smoothstep(0.18, 0.82, mix(mist, filaments, 0.4));

    float horizon = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    float radial = length(uv);

    vec3 cool = mix(
        palIce(colorT + field * 0.45 + mist * 0.2),
        palNeon(colorT + filaments * 0.35 + horizon * 0.15),
        0.45 + 0.25 * horizon
    );

    vec3 warm = mix(
        palLava(colorT + ridges * 0.50 + 0.08 * radial),
        palFire(colorT + glowMask * 0.65 + filaments * 0.15),
        glowMask
    );

    vec3 col = mix(cool * (0.22 + 0.18 * horizon), warm, veil);
    col += palFire(colorT + filaments + radial * 0.1) * glowMask * glowMask
        * (0.10 + 0.05 * uGlowIntensity);
    col += palNeon(colorT + mist * 0.6 + 0.12 * radial)
        * pow(max(1.0 - radial * 0.75, 0.0), 2.0)
        * (0.06 + 0.03 * uLightIntensity);

    float haze = smoothstep(0.25, 1.35, radial + (1.0 - horizon) * 0.35);
    col = mix(col, cool * 0.35, haze * 0.35);

    return col * (0.75 + 0.25 * uBrightness);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - uResolution * 0.5) / min(uResolution.x, uResolution.y);
    float time = uTime * uTimeScale;
    vec3 rd = normalize(vec3(uv, 1.9 + 0.25 * uZoom));

    vec3 col = background(uv, rd, time);

    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(luma), col, uSaturation);

    float ca = uChromaShift * 0.003;
    float r2 = dot(uv, uv);
    col.r *= 1.0 + ca * r2 * 1.7;
    col.b *= 1.0 - ca * r2 * 1.7;

    col = mix(vec3(0.5), col, uContrast);
    col = max(col, 0.0);
    col = col / (col + 0.35);

    vec2 vUv = gl_FragCoord.xy / uResolution;
    col *= 1.0 - length(vUv - 0.5) * 0.5;
    col = pow(max(col, 0.0), vec3(0.85));

    outColor = vec4(col, 1.0);
}
