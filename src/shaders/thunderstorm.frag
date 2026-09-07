#version 300 es
precision highp float;
precision highp int;

out vec4 outColor;

uniform vec2  uResolution;
uniform float uTime;
uniform float uTimeScale;
uniform float uZoom;
uniform float uCloudScale;
uniform float uCloudSpeed;
uniform float uCloudDensity;
uniform float uCloudDetail;
uniform float uBoltLengthMin;
uniform float uBoltLengthMax;
uniform float uBoltWidth;
uniform float uBoltWiggle;
uniform float uBoltNoiseScale;
uniform float uBoltNoiseSpeed;
uniform float uBoltBranching;
uniform float uBoltIntensity;
uniform float uFlickerSpeed;
uniform float uCloudIllumination;
uniform float uSeed;
uniform int   uBoltCount;
uniform int   uNoiseOctaves;
uniform vec3  uCloudColor;
uniform vec3  uLightningColor;

const float TAU = 6.28318530718;

float hash(vec2 point) {
  vec3 hashed = fract(vec3(point.xyx) * 0.1031);
  hashed += dot(hashed, hashed.yzx + 33.33);
  return fract((hashed.x + hashed.y) * hashed.z);
}

float noise(vec2 point) {
  vec2 cell = floor(point);
  vec2 blend = fract(point);
  blend = blend * blend * (3.0 - 2.0 * blend);
  return mix(mix(hash(cell), hash(cell + vec2(1.0, 0.0)), blend.x),
             mix(hash(cell + vec2(0.0, 1.0)), hash(cell + 1.0), blend.x), blend.y);
}

float fbm(vec2 point, int octaves) {
  float value = 0.0;
  float amplitude = 0.5;
  float total = 0.0;
  for (int octave = 0; octave < 8; octave++) {
    if (octave >= octaves) break;
    value += noise(point) * amplitude;
    total += amplitude;
    point = mat2(1.6, 1.2, -1.2, 1.6) * point + vec2(1.7, 9.2);
    amplitude *= 0.5;
  }
  return value / max(total, 0.001);
}

float segmentDistance(vec2 point, vec2 start, vec2 end) {
  vec2 direction = end - start;
  float along = clamp(dot(point - start, direction) / max(dot(direction, direction), 1e-10), 0.0, 1.0);
  return length(point - start - direction * along);
}

float jaggedNoise(float position, float seed) {
  float cell = floor(position);
  return mix(hash(vec2(cell, seed)), hash(vec2(cell + 1.0, seed)), fract(position)) * 2.0 - 1.0;
}

vec2 channelPoint(vec2 start, vec2 end, float progress, float seed) {
  vec2 direction = end - start;
  vec2 perpendicular = vec2(-direction.y, direction.x);
  float frequency = uBoltNoiseScale;
  float displacement = jaggedNoise(progress * 3.0, seed) * 0.55
                     + jaggedNoise(progress * frequency, seed + 13.0) * 0.30
                     + jaggedNoise(progress * frequency * 2.7, seed + 29.0) * 0.15;
  float envelope = pow(max(sin(progress * 3.14159265), 0.0), 0.45);
  return mix(start, end, progress) + perpendicular * displacement * uBoltWiggle * 4.0 * envelope;
}

vec3 channelLight(float distanceToChannel, float width, float energy, float pixelSize) {
  float core = (1.0 - smoothstep(width * 0.35, width + pixelSize, distanceToChannel))
             * width / max(width, pixelSize * 0.65);
  float corona = exp(-distanceToChannel / max(width * 3.0, pixelSize)) * 0.32;
  float bloom = exp(-distanceToChannel / max(width * 18.0, pixelSize * 2.0)) * 0.035;
  vec3 hotColor = mix(uLightningColor, vec3(max(max(uLightningColor.r, uLightningColor.g), uLightningColor.b)), 0.88);
  return energy * (hotColor * core * 4.5 + uLightningColor * (corona + bloom));
}

vec3 discharge(vec2 point, vec2 start, vec2 end, float seed, float growth,
               float energy, float branchEnergy, float pixelSize, out float channelDistance) {
  channelDistance = 100.0;
  float nearestProgress = 0.0;
  vec2 previous = start;
  for (int segment = 1; segment <= 8; segment++) {
    float progress = min(float(segment) / 8.0, growth);
    vec2 current = channelPoint(start, end, progress, seed);
    channelDistance = min(channelDistance, segmentDistance(point, previous, current));
    previous = current;
    if (float(segment) / 8.0 >= growth) break;
  }

  vec3 light = vec3(0.0);
  vec2 direction = end - start;
  vec2 perpendicular = vec2(-direction.y, direction.x);
  float haloRadius = max(uBoltWidth * 120.0, pixelSize * 12.0);
  if (segmentDistance(point, start, end) < length(direction) * uBoltWiggle * 4.0 + haloRadius) {
    float coreDistance = 100.0;
    previous = start;
    for (int segment = 1; segment <= 64; segment++) {
      float progress = min(float(segment) / 64.0, growth);
      vec2 current = channelPoint(start, end, progress, seed);
      float distanceToSegment = segmentDistance(point, previous, current);
      if (distanceToSegment < coreDistance) {
        coreDistance = distanceToSegment;
        nearestProgress = progress;
      }
      previous = current;
      if (float(segment) / 64.0 >= growth) break;
    }
    float width = uBoltWidth * mix(1.15, 0.55, nearestProgress);
    light = channelLight(coreDistance, width, energy, pixelSize);
  }

  for (int branch = 0; branch < 7; branch++) {
    float branchIndex = float(branch);
    float branchSeed = seed + branchIndex * 17.31 + 71.0;
    if (hash(vec2(branchSeed, 4.0)) >= uBoltBranching) continue;
    float attachment = 0.12 + branchIndex * 0.105 + hash(vec2(branchSeed, 5.0)) * 0.06;
    attachment = floor(attachment * 64.0) / 64.0;
    if (growth <= attachment) continue;
    float side = hash(vec2(branchSeed, 6.0)) < 0.5 ? -1.0 : 1.0;
    float extent = mix(0.16, 0.44, hash(vec2(branchSeed, 7.0))) * (1.0 - attachment * 0.5);
    vec2 branchStart = channelPoint(start, end, attachment, seed);
    vec2 branchEnd = branchStart + (direction + perpendicular * side * mix(0.5, 1.5, hash(vec2(branchSeed, 8.0)))) * extent;
    float branchGrowth = clamp((growth - attachment) / extent, 0.0, 1.0);
    vec2 forkStart = channelPoint(branchStart, branchEnd, 0.5, branchSeed);
    vec2 forkEnd = forkStart + (direction * 0.5 - perpendicular * side * 0.7) * extent * 0.5;
    float branchBound = length(branchEnd - branchStart) * uBoltWiggle * 4.0 + haloRadius;
    float forkBound = length(forkEnd - forkStart) * uBoltWiggle * 4.0 + haloRadius;
    if (segmentDistance(point, branchStart, branchEnd) > branchBound
      && segmentDistance(point, forkStart, forkEnd) > forkBound) continue;
    float branchDistance = 100.0;
    float branchProgress = 0.0;
    previous = branchStart;
    for (int segment = 1; segment <= 16; segment++) {
      float progress = min(float(segment) / 16.0, branchGrowth);
      vec2 current = channelPoint(branchStart, branchEnd, progress, branchSeed);
      float distanceToSegment = segmentDistance(point, previous, current);
      if (distanceToSegment < branchDistance) {
        branchDistance = distanceToSegment;
        branchProgress = progress;
      }
      previous = current;
      if (float(segment) / 16.0 >= branchGrowth) break;
    }
    float branchWidth = uBoltWidth * mix(0.55, 0.08, branchProgress);
    light += channelLight(branchDistance, branchWidth, branchEnergy * (1.0 - branchProgress * 0.75), pixelSize);

    if (hash(vec2(branchSeed, 9.0)) < uBoltBranching && branchGrowth > 0.5) {
      float forkGrowth = clamp((branchGrowth - 0.5) * 2.5, 0.0, 1.0);
      float forkDistance = 100.0;
      previous = forkStart;
      for (int segment = 1; segment <= 8; segment++) {
        float progress = min(float(segment) / 8.0, forkGrowth);
        vec2 current = channelPoint(forkStart, forkEnd, progress, branchSeed + 53.0);
        forkDistance = min(forkDistance, segmentDistance(point, previous, current));
        previous = current;
        if (float(segment) / 8.0 >= forkGrowth) break;
      }
      light += channelLight(forkDistance, uBoltWidth * 0.18, branchEnergy * 0.22, pixelSize);
    }
  }
  return light;
}

float strokePulse(float age, float onset, float decay) {
  float elapsed = age - onset;
  return smoothstep(0.0, 0.002, elapsed) * exp(-max(elapsed, 0.0) / decay);
}

void main() {
  vec2 point = (gl_FragCoord.xy - 0.5 * uResolution.xy) / uResolution.y * uZoom;
  float pixelSize = uZoom / uResolution.y;
  float aspect = uResolution.x / uResolution.y;
  float time = uTime * uTimeScale;
  vec2 cloudUV = point * uCloudScale + time * uCloudSpeed * vec2(0.03, 0.02);
  float cloudBase = fbm(cloudUV, uNoiseOctaves);
  float cloudDetail = fbm(cloudUV * 3.1 + cloudBase * 2.0, max(uNoiseOctaves - 1, 1));
  float cloudMass = mix(cloudBase, cloudBase * 0.65 + cloudDetail * 0.35, uCloudDetail);
  float clouds = smoothstep(0.68 - uCloudDensity * 0.48, 0.88 - uCloudDensity * 0.30, cloudMass);
  float relief = fbm(cloudUV + vec2(0.07, 0.11), max(uNoiseOctaves - 1, 1)) - cloudBase;
  vec3 color = uCloudColor * (0.10 + clouds * 0.65) * clamp(0.75 + relief * 3.0, 0.3, 1.2);
  vec3 lightning = vec3(0.0);

  float rate = max(0.08, uFlickerSpeed * 0.6);
  float baseEvent = floor(time * rate);
  for (int lane = 0; lane < 12; lane++) {
    if (lane >= uBoltCount) break;
    float laneSeed = float(lane) * 19.19 + mod(uSeed, 10000.0) * 0.137;
    for (int candidate = 0; candidate < 4; candidate++) {
      float event = baseEvent - float(candidate);
      float eventSeed = hash(vec2(event, laneSeed + 11.0)) * 937.0;
      float eventTime = (event + hash(vec2(event, laneSeed + 23.0))) / rate;
      float age = time - eventTime;
      if (age < 0.0 || age > 0.52 || hash(vec2(event, laneSeed + 37.0)) < 0.52) continue;

      float leaderDuration = mix(0.018, 0.045, hash(vec2(eventSeed, 1.0)));
      float strokeAge = age - leaderDuration;
      float spacing = mix(0.11, 0.042, clamp(uBoltNoiseSpeed / 8.0, 0.0, 1.0));
      float primary = strokePulse(strokeAge, 0.0, 0.018);
      float secondary = strokePulse(strokeAge, spacing, 0.012) * 0.65;
      float tertiary = strokePulse(strokeAge, spacing * 2.3, 0.016) * 0.4;
      secondary *= step(0.3, hash(vec2(eventSeed, 2.0)));
      tertiary *= step(0.55, hash(vec2(eventSeed, 3.0)));
      float afterglow = strokePulse(strokeAge, 0.0, 0.065) * 0.08;
      float leader = strokeAge < 0.0 ? 0.045 : 0.0;
      float flash = primary + secondary + tertiary + afterglow;
      float energy = (flash + leader) * uBoltIntensity * 4.0;
      if (energy < 0.001) continue;

      vec2 start = (vec2(hash(vec2(eventSeed, 21.0)), hash(vec2(eventSeed, 25.0))) - 0.5)
                 * vec2(aspect * 0.85, 0.85);
      float channelLength = mix(min(uBoltLengthMin, uBoltLengthMax), max(uBoltLengthMin, uBoltLengthMax), hash(vec2(eventSeed, 37.0)));
      float angle = hash(vec2(eventSeed, 29.0)) * TAU;
      vec2 end = start + vec2(cos(angle), sin(angle)) * channelLength;
      float growth = strokeAge < 0.0 ? floor(clamp(age / leaderDuration, 0.0, 1.0) * 24.0) / 24.0 : 1.0;
      float branchEnergy = (primary + secondary * 0.22 + tertiary * 0.12 + afterglow + leader) * uBoltIntensity * 3.2;
      float channelDistance;
      vec3 channel = discharge(point, start, end, eventSeed, growth, energy, branchEnergy, pixelSize, channelDistance);
      float veil = smoothstep(0.40, 0.78, cloudMass) * 0.55;
      lightning += channel * (1.0 - veil);
      float scatter = exp(-channelDistance / (0.045 + clouds * 0.12));
      color += mix(uLightningColor, vec3(1.0), 0.3) * scatter * flash * uCloudIllumination
             * uBoltIntensity * (0.12 + clouds * 1.8) * (0.65 + cloudDetail * 0.7);
    }
  }

  color += lightning;
  color = vec3(1.0) - exp(-color);
  outColor = vec4(pow(max(color, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}