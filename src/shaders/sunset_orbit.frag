#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uTurbulence;
uniform float uCloudHeight;
uniform float uStepBase;
uniform float uStepScale;
uniform float uHueShift;
uniform float uHueSpeed;
uniform float uIntensity;

// --- new degrees of freedom ---
uniform float uWarpAmp;       // amplitude of horizon sine warp
uniform float uWarpFreq;      // spatial frequency of the warp
uniform float uWarpSpeed;     // temporal speed of the warp animation
uniform float uWarpHarmonics; // add harmonic overtones for richer shape
uniform float uOrbitRadius;   // radius of circular horizon orbit
uniform float uOrbitSpeed;    // angular speed of the orbit
uniform float uOrbitEcc;      // eccentricity (ellipse stretch)
uniform float uTiltAngle;     // tilt the horizon plane
uniform float uCloudDensity;  // per-step density multiplier
uniform float uFogFalloff;    // controls exponential fog rolloff
uniform float uColorSep;      // chromatic separation between RGB channels

/*
   Smooth periodic horizon function.
   The horizon is no longer a single flat plane at y=0.
   Instead it follows a warped surface whose height varies with (x,z)
   and is additionally orbited in the (y, z) plane over time.
*/
float horizonHeight(vec3 p, float t) {
    // Base warp: sum of harmonics of a spatial sine wave
    float h = 0.0;
    float amp = uWarpAmp;
    float freq = uWarpFreq;
    for (float k = 1.0; k <= 5.0; k += 1.0) {
        if (k > uWarpHarmonics) break;
        // phase varies with time, direction alternates each harmonic
        float phase = uWarpSpeed * t * (0.7 + 0.3 * k) + k * 1.37;
        h += amp * sin(freq * p.x + phase)
           * cos(freq * 0.6 * p.z + phase * 0.8);
        amp  *= 0.55;   // each harmonic is weaker
        freq *= 1.8;    // and higher frequency
    }

    // Orbit: translate the center of the horizon along an elliptical path
    float orbitAngle = uOrbitSpeed * t;
    float oy = uOrbitRadius * sin(orbitAngle);
    float oz = uOrbitRadius * uOrbitEcc * cos(orbitAngle);

    // Tilt: rotate horizon normal by uTiltAngle around the x-axis
    float ct = cos(uTiltAngle);
    float st = sin(uTiltAngle);
    float tiltedY = ct * (p.y - oy) - st * (p.z - oz);

    return h - tiltedY;   // positive = inside clouds
}

void main() {
    vec2 I = gl_FragCoord.xy;
    float t = uTime * uTimeScale;
    float z = 0.0;
    float d = 0.0;
    float s = 0.0;
    vec4 O = vec4(0.0);

    vec3 rd = normalize(vec3(I + I, 0.0) - uResolution.xyy);

    for (float i = 0.0; i < 100.0; i++) {
        vec3 p = z * rd;

        // Volumetric turbulence (same family as sunset_plus)
        for (d = 5.0; d < 200.0; d += d) {
            p += uTurbulence * 0.6 * sin(p.yzx * d - 0.2 * t) / d;
        }

        float height = max(0.05, uCloudHeight);

        // Sample the periodic horizon surface
        s = horizonHeight(p, t);
        s = height - abs(s);            // cloud shell thickness

        // Adaptive ray step
        z += d = uStepBase + max(s, -s * 0.2) / uStepScale;

        // Colour with per-channel separation for richer sunsets
        vec4 phaseR = vec4(3.0, 4.0, 5.0, 0.0) + uHueShift + uHueSpeed * t;
        vec4 phaseG = phaseR + uColorSep;
        vec4 phaseB = phaseR - uColorSep;

        float envelope = exp(s / 0.1) / d;
        float density  = uCloudDensity * envelope;

        float fogAtt = exp(-z * uFogFalloff);

        vec4 cR = (cos(s / 0.07 + p.x + 0.5 * t - phaseR) + 1.5) * density;
        vec4 cG = (cos(s / 0.07 + p.x + 0.5 * t - phaseG) + 1.5) * density;
        vec4 cB = (cos(s / 0.07 + p.x + 0.5 * t - phaseB) + 1.5) * density;

        O.r += cR.r * fogAtt;
        O.g += cG.g * fogAtt;
        O.b += cB.b * fogAtt;
    }

    O = tanh(O * O / 4e8);
    outColor = vec4(O.rgb * uIntensity, 1.0);
}
