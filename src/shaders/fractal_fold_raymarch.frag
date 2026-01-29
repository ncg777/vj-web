#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

// --- UNIFORMS ---
uniform float uTime;
uniform vec2 uResolution;
uniform float uZoom;
uniform float uColorShift;
uniform int uIterations;
uniform float uDistort;
uniform float uRotateSpeed;
uniform float uMaxSteps;

// --- MATH HELPERS ---

// Rotation Matrix
mat2 rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

// Palette function for coloring (IQ style)
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

// The SDF (Signed Distance Function) - The Core Math
// This function returns the distance from point 'p' to the fractal surface
float map(vec3 p, float time, float distort) {
    float scale = 1.0;
    float offset = 1.0;
    
    // Recursive Folding Loop - Menger-like fractal
    for (int i = 0; i < 8; i++) {
        if (i >= uIterations) break;
        
        // Rotate space
        p.xy *= rot(time);
        p.yz *= rot(time * 0.7);
        
        // Folding - creates symmetry
        p = abs(p);
        
        // Menger fold
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;
        
        // Scale and translate
        p = p * distort - offset * (distort - 1.0);
        scale *= distort;
    }
    
    // Return distance to a box, scaled back
    float d = (length(p) - 1.5) / scale;
    return d;
}

void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    
    // 1. Setup Camera
    vec2 uv = (fragCoord - uResolution * 0.5) / min(uResolution.x, uResolution.y);
    uv *= uZoom;

    vec3 ro = vec3(0.0, 0.0, -3.0); // Ray Origin
    vec3 rd = normalize(vec3(uv, 1.0)); // Ray Direction

    float time = uTime * uRotateSpeed;

    // 2. Raymarching Loop
    float t = 0.0; // Total distance traveled
    float d = 0.0; // Distance to surface
    int maxSteps = int(uMaxSteps);

    vec3 col = vec3(0.0);
    vec3 p = ro;
    float glow = 0.0;

    for (int i = 0; i < 200; i++) {
        if (i >= maxSteps) break;

        p = ro + rd * t;
        d = map(p, time, uDistort); // Get distance to fractal

        // Accumulate glow based on proximity
        glow += 0.02 / (0.1 + abs(d));

        // If we hit the surface
        if (abs(d) < 0.001) {
            // Calculate Normal
            vec2 e = vec2(0.001, 0.0);
            vec3 n = normalize(vec3(
                map(p + e.xyy, time, uDistort) - map(p - e.xyy, time, uDistort),
                map(p + e.yxy, time, uDistort) - map(p - e.yxy, time, uDistort),
                map(p + e.yyx, time, uDistort) - map(p - e.yyx, time, uDistort)
            ));

            // Lighting
            vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
            float diff = max(dot(n, lightDir), 0.0);
            float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 16.0);

            // Coloring based on position and normal
            float fresnel = pow(1.0 + dot(rd, n), 3.0);
            
            // Dynamic Palette
            vec3 paletteColor = palette(
                length(p) * 0.4 + uTime * 0.1 + uColorShift, 
                vec3(0.5), 
                vec3(0.5), 
                vec3(1.0), 
                vec3(0.263, 0.416, 0.557)
            );

            col = paletteColor * (diff * 0.8 + 0.2) + vec3(1.0) * spec * 0.5;
            col = mix(col, vec3(1.0), fresnel * 0.3);
            break;
        }

        // Move ray forward
        t += d * 0.5; // Use smaller steps for safety
        
        // Stop if too far
        if (t > 20.0) break;
    }

    // Add glow effect for missed rays
    col += glow * 0.02 * palette(
        uTime * 0.05 + uColorShift,
        vec3(0.5), 
        vec3(0.5), 
        vec3(1.0), 
        vec3(0.263, 0.416, 0.557)
    );

    // 3. Post-Processing (Vignette)
    vec2 vUv = fragCoord / uResolution;
    col *= 1.0 - length(vUv - 0.5) * 0.5;

    outColor = vec4(col, 1.0);
}
