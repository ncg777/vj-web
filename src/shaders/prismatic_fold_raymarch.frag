#version 330 core

out vec4 FragColor;

uniform float uTime;
uniform vec2 uResolution;
uniform int uIterations;
uniform float uRotateSpeed;
uniform float uFoldOffset;
uniform float uStepScale;
uniform float uGlow;
uniform float uCameraDistance;
uniform float uCameraSpin;
uniform vec3 uColorPrimary;
uniform vec3 uColorSecondary;
uniform float uColorMix;
uniform float uAlphaGain;

vec3 palette(float d) {
    vec3 base = mix(uColorPrimary, uColorSecondary, clamp(d, 0.0, 1.0));
    return mix(base, base * base, uColorMix);
}

vec2 rotate2D(vec2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return mat2(c, s, -s, c) * p;
}

float mapFunc(vec3 p) {
    float t = uTime * uRotateSpeed;
    for (int i = 0; i < 64; ++i) {
        if (i >= uIterations) break;
        p.xz = rotate2D(p.xz, t);
        p.xy = rotate2D(p.xy, t * 1.89);
        p.xz = abs(p.xz);
        p.xz -= vec2(uFoldOffset);
    }
    return dot(sign(p), p) / uStepScale;
}

vec4 rm(vec3 ro, vec3 rd) {
    float t = 0.0;
    vec3 col = vec3(0.0);
    float d = 1.0;

    for (int i = 0; i < 72; ++i) {
        vec3 p = ro + rd * t;
        d = mapFunc(p) * 0.5;

        if (d < 0.02) break;
        if (d > 120.0) break;

        float shade = length(p) * 0.08;
        col += palette(shade) * uGlow / (400.0 * d);
        t += d;
    }

    float alpha = 1.0 / (max(d, 0.01) * 100.0);
    return vec4(col, clamp(alpha * uAlphaGain, 0.0, 1.0));
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec2 uv = (fragCoord - (uResolution * 0.5)) / uResolution.x;

    vec3 ro = vec3(0.0, 0.0, -uCameraDistance);
    ro.xz = rotate2D(ro.xz, uTime * uCameraSpin);

    vec3 cf = normalize(-ro);
    vec3 cs = normalize(cross(cf, vec3(0.0, 1.0, 0.0)));
    vec3 cu = normalize(cross(cf, cs));

    vec3 uuv = ro + cf * 3.0 + uv.x * cs + uv.y * cu;
    vec3 rd = normalize(uuv - ro);

    FragColor = rm(ro, rd);
}
