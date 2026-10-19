import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const source = await readFile(new URL('../src/audio.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.ESNext } });
const { analyze, Soundtrack } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`);
function signal(frequency = 0) {
  const sampleRate = 48000, duration = 0.5;
  const data = Float32Array.from({ length: sampleRate * duration }, (_, i) => frequency ? Math.sin(i * 2 * Math.PI * frequency / sampleRate) * 0.7 : 0);
  return { sampleRate, duration, numberOfChannels: 1, getChannelData: () => data };
}

test('audio analysis distinguishes bass and treble and leaves silence at zero', async () => {
  const bass = await analyze(signal(110));
  const treble = await analyze(signal(8000));
  const silent = await analyze(signal());
  const peak = values => {
    const frame = values.slice(10 * 64, 11 * 64);
    return frame.indexOf(Math.max(...frame));
  };
  assert.ok(peak(bass) < 20);
  assert.ok(peak(treble) > 45);
  assert.ok(Math.max(...bass) > 0.5);
  assert.ok(silent.every(value => value === 0));
  assert.deepEqual(await analyze(signal(110)), bass);
});

test('feature sampling interpolates, loops and stops at the same track boundaries as audio', () => {
  globalThis.Audio = class { loop = false; };
  const sound = new Soundtrack();
  sound.buffer = { duration: 1 };
  sound.frames = Float32Array.from({ length: 60 * 64 }, (_, i) => Math.floor(i / 64) / 60);
  assert.ok(Math.abs(sound.sample(0.125)[0] - 0.125) < 1e-6);
  assert.deepEqual(sound.sample(1.25).slice(), sound.sample(0.25).slice());
  sound.player.loop = false;
  assert.ok(sound.sample(1).every(value => value === 0));
});
