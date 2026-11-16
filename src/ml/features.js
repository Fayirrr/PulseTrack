// Audio features for the team trained crash sound CNN.
//
// These steps must stay the same as ml/features.py (the training side):
//   1 second of 16 kHz audio -> tf.signal.stft (frame 1024, step 256, hann)
//   -> power -> mel matrix from mel.json -> log(x + eps) -> per sample normalisation.
//
// No browser APIs are used here, so ml/check_export.mjs can run the same code in Node
// and compare it with Keras.

// Low pass filter + linear interpolation from the microphone rate (often 44.1 or 48 kHz) to 16 kHz.
export function resampleTo16k(input, srcRate, targetRate = 16000) {
  if (srcRate === targetRate) return Float32Array.from(input);
  let src = input;
  if (srcRate > targetRate) {
    // windowed sinc low pass a little under the new Nyquist frequency
    const taps = 63;
    const half = (taps - 1) / 2;
    const fc = (0.45 * targetRate) / srcRate; // cycles per sample
    const h = new Float32Array(taps);
    let sum = 0;
    for (let i = 0; i < taps; i++) {
      const n = i - half;
      const sinc = n === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
      const hamming = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (taps - 1));
      h[i] = sinc * hamming;
      sum += h[i];
    }
    for (let i = 0; i < taps; i++) h[i] /= sum;
    const filtered = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      let acc = 0;
      for (let k = 0; k < taps; k++) {
        const j = i + k - half;
        if (j >= 0 && j < input.length) acc += input[j] * h[k];
      }
      filtered[i] = acc;
    }
    src = filtered;
  }
  const outLen = Math.floor((input.length * targetRate) / srcRate);
  const out = new Float32Array(outLen);
  const step = srcRate / targetRate;
  for (let i = 0; i < outLen; i++) {
    const pos = i * step;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, src.length - 1);
    const frac = pos - i0;
    out[i] = src[i0] * (1 - frac) + src[i1] * frac;
  }
  return out;
}

// waveform: Float32Array or 1D tensor with cfg.windowSamples values in [-1, 1].
// melMatrix: 2D tensor [513, 64] built from mel.json.
// Returns a tensor [1, 59, 64, 1]. Call inside tf.tidy.
export function logMel(tf, waveform, melMatrix, cfg) {
  const signal = waveform instanceof tf.Tensor ? waveform : tf.tensor1d(waveform, 'float32');
  const stft = tf.signal.stft(signal, cfg.frameLength, cfg.frameStep, cfg.fftLength, tf.signal.hannWindow);
  const power = tf.square(tf.abs(stft));
  const mel = tf.matMul(power, melMatrix);
  const logMelSpec = tf.log(tf.add(mel, cfg.logEpsilon));
  const { mean, variance } = tf.moments(logMelSpec);
  const norm = tf.div(tf.sub(logMelSpec, mean), tf.add(tf.sqrt(variance), cfg.normEpsilon));
  return tf.reshape(norm, [1, cfg.numFrames, cfg.numMelBins, 1]);
}
