// Live crash sound detector for the Victim screen.
//
// Loads the team trained CNN (public/model/model.json) and the feature settings
// (public/model/mel.json) once, keeps the last 1 second of microphone audio, and every
// 0.5 seconds resamples it to 16 kHz, builds the log Mel spectrogram and runs the model.
//
// This file only reports class probabilities. The decision to start the cancel countdown
// is made in VictimUI.jsx and never comes from a single prediction.

import { logMel, resampleTo16k } from './features.js';

const MODEL_URL = '/model/model.json';
const MEL_URL = '/model/mel.json';
const META_URL = '/model/model_meta.json';
const LOAD_TIMEOUT_MS = 20000;

function withTimeout(promise, ms, what) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${what} timed out`)), ms)),
  ]);
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json();
}

// audioContext: the AudioContext the app already uses
// sourceNode:   the MediaStreamAudioSourceNode of the existing mic stream
// onPrediction: called every 0.5 s with { label, confidence, crashProb, probs, modelVersion, synthetic }
// Rejects if the model cannot be loaded (the caller then falls back to the loudness rule).
export async function createCrashDetector({ audioContext, sourceNode, onPrediction }) {
  // TensorFlow.js is loaded only here, so the rest of the app does not pay for it.
  const tf = await withTimeout(import('@tensorflow/tfjs'), LOAD_TIMEOUT_MS, 'TensorFlow.js load');
  await tf.ready();

  const [cfg, meta, model] = await withTimeout(
    Promise.all([fetchJson(MEL_URL), fetchJson(META_URL), tf.loadLayersModel(MODEL_URL)]),
    LOAD_TIMEOUT_MS,
    'Model load',
  );
  const classes = meta.classes || cfg.classes;
  const crashIndex = classes.indexOf('crash');
  const melMatrix = tf.tensor2d(cfg.melMatrix, cfg.melMatrixShape, 'float32');

  const predictWindow = (samples16k) => {
    const probs = tf.tidy(() => model.predict(logMel(tf, samples16k, melMatrix, cfg)).dataSync());
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    const byClass = {};
    classes.forEach((name, i) => { byClass[name] = probs[i]; });
    return {
      label: classes[best],
      confidence: probs[best],
      crashProb: crashIndex >= 0 ? probs[crashIndex] : 0,
      probs: byClass,
      modelVersion: meta.modelVersion,
      synthetic: !!meta.synthetic,
    };
  };

  // one warm up run so the first real prediction is not slow
  predictWindow(new Float32Array(cfg.windowSamples));

  // rolling buffer holding the last 1 second at the microphone sample rate
  const srcRate = audioContext.sampleRate;
  const ringSize = Math.round(srcRate);
  const ring = new Float32Array(ringSize);
  let writePos = 0;
  let filled = 0;
  let sinceLast = 0;
  let stopped = false;
  const hopNative = Math.round(srcRate * (cfg.hopSamples / cfg.sampleRate));

  const runPrediction = () => {
    const native = new Float32Array(ringSize);
    native.set(ring.subarray(writePos));
    native.set(ring.subarray(0, writePos), ringSize - writePos);
    const resampled = resampleTo16k(native, srcRate, cfg.sampleRate);
    const window16k = new Float32Array(cfg.windowSamples);
    window16k.set(resampled.subarray(0, Math.min(resampled.length, cfg.windowSamples)));
    onPrediction(predictWindow(window16k));
  };

  // ScriptProcessorNode is old but works in every mobile browser without an extra worklet file.
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const silent = audioContext.createGain();
  silent.gain.value = 0; // the mic is never played back through the speaker
  processor.onaudioprocess = (event) => {
    if (stopped) return;
    const input = event.inputBuffer.getChannelData(0);
    for (let i = 0; i < input.length; i++) {
      ring[writePos] = input[i];
      writePos = (writePos + 1) % ringSize;
    }
    filled = Math.min(filled + input.length, ringSize);
    sinceLast += input.length;
    if (filled >= ringSize && sinceLast >= hopNative) {
      sinceLast = 0;
      try {
        runPrediction();
      } catch (err) {
        console.error('Crash detector prediction failed:', err);
      }
    }
  };
  sourceNode.connect(processor);
  processor.connect(silent);
  silent.connect(audioContext.destination);

  return {
    classes,
    modelVersion: meta.modelVersion,
    synthetic: !!meta.synthetic,
    numTensors: () => tf.memory().numTensors, // should stay flat while running
    stop() {
      if (stopped) return;
      stopped = true;
      processor.onaudioprocess = null;
      try { sourceNode.disconnect(processor); } catch { /* already disconnected */ }
      processor.disconnect();
      silent.disconnect();
      melMatrix.dispose();
      model.dispose();
    },
  };
}
