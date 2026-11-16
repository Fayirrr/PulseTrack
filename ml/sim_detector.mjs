// Node simulation of the browser detector (src/ml/crashDetector.js) with a fake AudioContext and fake fetch.
// Feeds the saved check windows as 48 kHz audio in 4096 sample chunks, like a phone would.
// Run from the repo root after ml/check_export.py:   node ml/sim_detector.mjs
// Checks: predictions arrive every 0.5 s of audio and the number of tensors stays flat.
import fs from 'node:fs';
import path from 'node:path';
const root = process.cwd();
globalThis.fetch = async (url) => {
  const p = path.join(root, 'public', String(url).replace(/^\//, ''));
  const buf = fs.readFileSync(p);
  return new Response(buf, { status: 200, headers: { 'content-type': p.endsWith('.json') ? 'application/json' : 'application/octet-stream' } });
};
const SR = 48000;
let proc;
const node = () => ({ connect() {}, disconnect() {}, gain: { value: 1 } });
const audioContext = { sampleRate: SR, destination: {}, createGain: node,
  createScriptProcessor: () => { proc = { connect() {}, disconnect() {}, onaudioprocess: null }; return proc; } };
const { createCrashDetector } = await import('../src/ml/crashDetector.js');
const preds = [];
const det = await createCrashDetector({ audioContext, sourceNode: node(), onPrediction: (p) => preds.push(p) });
// feed 6 s: check windows upsampled x3 (16k -> 48k by linear interpolation)
const raw = fs.readFileSync(path.join(root, 'ml/export/check_windows.bin'));
const i16 = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
const exp = JSON.parse(fs.readFileSync(path.join(root, 'ml/export/check_windows.json'), 'utf8'));
const secs = 6; const sig = new Float32Array(SR * secs);
for (let s = 0; s < secs; s++) for (let j = 0; j < SR; j++) {
  const pos = j / 3, a = Math.floor(pos), b = Math.min(a + 1, 15999);
  sig[s * SR + j] = (i16[s * 16000 + a] * (1 - (pos - a)) + i16[s * 16000 + b] * (pos - a)) / 32768;
}
const t0 = det.numTensors(); const start = Date.now();
for (let i = 0; i + 4096 <= sig.length; i += 4096) proc.onaudioprocess({ inputBuffer: { getChannelData: () => sig.subarray(i, i + 4096) } });
console.log('predictions', preds.length, 'ms per prediction', ((Date.now() - start) / preds.length).toFixed(1), 'tensor growth', det.numTensors() - t0);
console.log('version', det.modelVersion, 'synthetic', det.synthetic);
preds.forEach((p, i) => console.log(i, p.label, p.confidence.toFixed(3), 'crashProb', p.crashProb.toFixed(3)));
console.log('expected labels per second', exp.labels.slice(0, secs).join(', '));
det.stop();
