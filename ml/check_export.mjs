// Phase 5 check, Node side. Run through ml/check_export.py (it prepares the input files):
//     ml\.venv\Scripts\python.exe ml\check_export.py
// or directly:  node ml/check_export.mjs
//
// Loads public/model with @tensorflow/tfjs, computes the features with the SAME code the
// browser uses (src/ml/features.js, tf.signal.stft + mel.json) on the 10 saved windows and
// compares the probabilities with the ones Keras gave. They must match within 1e-3.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { logMel } from '../src/ml/features.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const modelDir = process.env.PT_MODEL_DIR || path.join(here, '..', 'public', 'model'); // PT_MODEL_DIR: test another export
const exportDir = path.join(here, 'export');
const TOLERANCE = 1e-3;

const cfg = JSON.parse(fs.readFileSync(path.join(modelDir, 'mel.json'), 'utf8'));
const expected = JSON.parse(fs.readFileSync(path.join(exportDir, 'check_windows.json'), 'utf8'));
const raw = fs.readFileSync(path.join(exportDir, 'check_windows.bin'));
const int16 = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);

// load the layers model from disk (plain tfjs has no file loader in Node)
const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, 'model.json'), 'utf8'));
const weightSpecs = modelJson.weightsManifest.flatMap((g) => g.weights);
const buffers = modelJson.weightsManifest.flatMap((g) => g.paths.map((p) => fs.readFileSync(path.join(modelDir, p))));
const merged = Buffer.concat(buffers);
const weightData = merged.buffer.slice(merged.byteOffset, merged.byteOffset + merged.byteLength);
const model = await tf.loadLayersModel({
  load: async () => ({ modelTopology: modelJson.modelTopology, weightSpecs, weightData }),
});

const melMatrix = tf.tensor2d(cfg.melMatrix, cfg.melMatrixShape, 'float32');
const n = expected.kerasProbs.length;
const tfjsProbs = [];
let maxDiff = 0;
const tensorsBefore = tf.memory().numTensors;
for (let i = 0; i < n; i++) {
  const wave = new Float32Array(cfg.windowSamples);
  for (let j = 0; j < cfg.windowSamples; j++) wave[j] = int16[i * cfg.windowSamples + j] / 32768;
  const probs = Array.from(tf.tidy(() => model.predict(logMel(tf, wave, melMatrix, cfg)).dataSync()));
  tfjsProbs.push(probs);
  probs.forEach((p, c) => { maxDiff = Math.max(maxDiff, Math.abs(p - expected.kerasProbs[i][c])); });
}
const leaked = tf.memory().numTensors - tensorsBefore;

fs.writeFileSync(path.join(exportDir, 'check_tfjs_probs.json'), JSON.stringify({ tfjsProbs, maxDiff, leakedTensors: leaked }));
console.log(`tfjs ${tf.version.tfjs}, backend ${tf.getBackend()}, windows ${n}`);
console.log(`max |keras - tfjs| probability difference: ${maxDiff.toExponential(3)} (tolerance ${TOLERANCE})`);
console.log(`tensors leaked over ${n} predictions: ${leaked}`);
if (!(maxDiff <= TOLERANCE) || leaked !== 0) {
  console.error('CHECK FAILED');
  process.exit(1);
}
console.log('CHECK PASSED');
