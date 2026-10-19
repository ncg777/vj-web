#version 300 es
precision highp float;
uniform sampler2D uScene, uHistory, uSpectrum;
uniform vec2 uResolution;
uniform float uTime, uAmount, uScale, uColor, uPersistence, uSensitivity, uDelta;
uniform int uMode, uHasAudio;
out vec4 fragColor;
float band(float x) { return min(1.5, texture(uSpectrum, vec2(x, 0.5)).r * uSensitivity); }
vec2 mirrorUV(vec2 uv) { return 1.0 - abs(mod(uv, 2.0) - 1.0); }
void main() {
  vec2 uv = gl_FragCoord.xy / uResolution;
  vec2 aspect = vec2(uResolution.x / uResolution.y, 1.0);
  vec2 p = (uv - 0.5) * aspect;
  float radius = length(p), angle = atan(p.y, p.x);
  float bass = (band(0.08) + band(0.18)) * 0.5;
  float energy = (bass + band(0.45) + band(0.75)) / 3.0;
  float drive = uHasAudio == 1 ? energy : 0.45;
  float spectral = uHasAudio == 1 ? band(fract(radius * 0.6 + angle / 6.28318)) : 0.3;
  float t = uTime;
  vec2 flow = vec2(sin(p.y * uScale + t * 0.7 + spectral * 5.0), cos(p.x * uScale - t * 0.6 + bass * 4.0));
  float ripple = sin(radius * uScale * 8.0 - t * 3.0 + spectral * 8.0);
  vec2 warped = uv + (flow * 0.055 * drive + normalize(p + 0.0001) * ripple * 0.025 * spectral) * uAmount / aspect;
  if (uMode == 1) {
    float sectors = max(3.0, floor(uScale * 2.0));
    float wedge = 6.28318 / sectors;
    float a = abs(mod(angle + t * 0.08 + bass * 0.4, wedge) - wedge * 0.5);
    vec2 folded = vec2(cos(a), sin(a)) * radius * (1.0 + drive * 0.15);
    warped = mix(warped, folded / aspect + 0.5, uAmount);
  }
  vec2 split = (flow + p) * uColor * uAmount * (0.002 + drive * 0.018) / aspect;
  vec3 color = vec3(texture(uScene, mirrorUV(warped + split)).r, texture(uScene, mirrorUV(warped)).g, texture(uScene, mirrorUV(warped - split)).b);
  if (uMode == 2) {
    float rotation = uDelta * (0.12 + bass * 0.3);
    mat2 rot = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
    vec2 historyUV = rot * p * (1.0 - uDelta * (0.12 + drive * 0.2)) / aspect + 0.5;
    vec4 history = texture(uHistory, mirrorUV(historyUV));
    float retention = pow(uPersistence * 0.97, max(0.001, uDelta) * 60.0);
    color = mix(color, history.rgb, retention * uAmount * history.a);
  }
  fragColor = vec4(color, 1.0);
}
