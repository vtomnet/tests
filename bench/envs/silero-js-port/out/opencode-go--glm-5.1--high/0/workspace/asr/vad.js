/**
 * Silero VAD (Voice Activity Detection) for Node.js using ONNX Runtime.
 *
 * Provides a streaming voice activity detector that processes audio in
 * fixed-size windows and returns speech probabilities, plus a higher-level
 * iterator that emits speech start / end events.
 *
 * Usage:
 *   import { VoiceActivityDetector, VADIterator } from "./asr/vad.js";
 *
 *   const vad = new VoiceActivityDetector("/path/to/silero_vad.onnx");
 *   await vad.init();
 *
 *   // Per-chapter probability
 *   const prob = await vad.process(float32Chunk);  // 512 samples @ 16kHz
 *
 *   // Or use the iterator for speech segments
 *   const iter = new VADIterator(vad);
 *   const event = await iter.process(float32Chunk);   // {start:n} | {end:n} | null
 */

import { InferenceSession, Tensor } from "onnxruntime-node";

// ── constants ──────────────────────────────────────────────────────────────

const WINDOW_16K = 512;   // samples per chunk @ 16 kHz
const WINDOW_8K  = 256;   // samples per chunk @ 8 kHz
const CONTEXT_16K = 64;
const CONTEXT_8K  = 32;
const STATE_SIZE  = 128;  // hidden-state width (always 128)

// ── helpers ────────────────────────────────────────────────────────────────

function windowSizeFor(sr) {
  if (sr === 16000) return WINDOW_16K;
  if (sr === 8000)  return WINDOW_8K;
  throw new Error(`Silero VAD only supports 8000 or 16000 Hz sample rates (got ${sr})`);
}

function contextSizeFor(sr) {
  return sr === 16000 ? CONTEXT_16K : CONTEXT_8K;
}

/**
 * Accept a Float32Array or a plain number[] and return a Float32Array
 * of exactly `len` elements, right-padding with zeros when the input
 * is shorter than the required window.
 */
function ensureWindow(audio, len) {
  if (audio instanceof Float32Array) {
    if (audio.length === len) return audio;
    const out = new Float32Array(len);
    out.set(audio.subarray(0, Math.min(audio.length, len)));
    return out;
  }
  const out = new Float32Array(len);
  for (let i = 0; i < Math.min(audio.length, len); i++) out[i] = audio[i];
  return out;
}

// ── VoiceActivityDetector ──────────────────────────────────────────────────

/**
 * Low-level wrapper around the Silero VAD ONNX model.
 *
 * Maintains the recurrent `state` and `context` buffers so that
 * consecutive calls to `process()` operate as a streaming detector.
 */
export class VoiceActivityDetector {
  /**
   * @param {string} modelPath  – path to the silero_vad.onnx file
   * @param {object} [opts]
   * @param {number} [opts.sampleRate=16000]
   * @param {number} [opts.batchSize=1]
   */
  constructor(modelPath, opts = {}) {
    this._modelPath  = modelPath;
    this._sampleRate = opts.sampleRate ?? 16000;
    this._batchSize  = opts.batchSize  ?? 1;

    /** @type {InferenceSession|null} */
    this._session = null;

    // recurrent state (kept between calls)
    this._state    = null;   // Float32Array, shape [2 * batchSize * STATE_SIZE]
    this._context  = null;   // Float32Array, shape [contextSize]
  }

  // ── lifecycle ─────────────────────────────────────────────────────────

  /** Load the ONNX model. Must be called before `process`. */
  async init() {
    this._session = await InferenceSession.create(this._modelPath, {
      executionProviders: ["cpu"],
      intraOpNumThreads: 1,
      interOpNumThreads: 1,
    });
    this.reset();
    return this;
  }

  /** Release the ONNX session and free resources. */
  async release() {
    if (this._session) {
      await this._session.release();
      this._session = null;
    }
  }

  /** Re-initialise the recurrent state and context. */
  reset() {
    const bs = this._batchSize;
    this._state   = new Float32Array(2 * bs * STATE_SIZE); // zeros
    this._context = new Float32Array(contextSizeFor(this._sampleRate)); // zeros
  }

  // ── properties ───────────────────────────────────────────────────────

  get sampleRate()   { return this._sampleRate; }
  get windowSize()   { return windowSizeFor(this._sampleRate); }
  get contextSize()  { return contextSizeFor(this._sampleRate); }

  // ── core inference ───────────────────────────────────────────────────

  /**
   * Async version of `process`.  Runs a single window of audio through the
   * model and returns the speech probability (0–1).
   *
   * @param {Float32Array|number[]} audio – PCM float samples;
   *   length should equal `this.windowSize` (shorter chunks are zero-padded).
   * @returns {Promise<number>} speech probability for this window
   */
  async processAsync(audio) {
    if (!this._session) throw new Error("Call init() before processAsync()");

    const sr   = this._sampleRate;
    const ws   = this.windowSize;
    const cs   = this.contextSize;
    const bs   = this._batchSize;

    const windowAudio = ensureWindow(audio, ws);

    // Concatenate [context, windowAudio] → input tensor [1, cs + ws]
    const inputLen = cs + ws;
    const inputBuf = new Float32Array(inputLen);
    inputBuf.set(this._context, 0);
    inputBuf.set(windowAudio, cs);

    const inputTensor  = new Tensor("float32", inputBuf, [bs, inputLen]);
    const stateTensor  = new Tensor("float32", this._state, [2, bs, STATE_SIZE]);
    const srTensor     = new Tensor("int64", BigInt64Array.from([BigInt(sr)]), []);

    const outputs = await this._session.run({
      input: inputTensor,
      state: stateTensor,
      sr:    srTensor,
    });

    // The model returns output names like "output" and "stateN" or similar.
    // Silero VAD outputs are: prob (shape [1,1]) and new state (shape [2,1,128]).
    // The output names vary by model version, so we use positional access.
    const outputNames = this._session.outputNames;

    // output is first, state is second
    const probTensor  = outputs[outputNames[0]];
    const stateTensor2 = outputs[outputNames[1]];

    // Update recurrent state
    this._state = stateTensor2.data instanceof Float32Array
      ? stateTensor2.data
      : new Float32Array(stateTensor2.data);

    // Save context = last `cs` samples of the input we just fed
    this._context = inputBuf.slice(inputLen - cs);

    // Return scalar probability
    return probTensor.data[0];
  }

  /**
   * Process one window and return speech probability (0–1).
   * Alias for `processAsync`.
   *
   * @param {Float32Array|number[]} audio
   * @returns {Promise<number>}
   */
  async process(audio) {
    return this.processAsync(audio);
  }

  /**
   * Run the model on a full audio buffer and return an array of
   * per-window speech probabilities.
   *
   * Resets state before processing.
   *
   * @param {Float32Array} audio – full audio waveform (16 kHz mono float32)
   * @returns {Promise<number[]>} per-window speech probabilities
   */
  async processFull(audio) {
    this.reset();
    const ws = this.windowSize;
    const probs = [];
    for (let offset = 0; offset < audio.length; offset += ws) {
      const chunk = audio.subarray(offset, Math.min(offset + ws, audio.length));
      const prob = await this.processAsync(chunk);
      probs.push(prob);
    }
    return probs;
  }
}

// ── VADIterator ─────────────────────────────────────────────────────────────

/**
 * Streaming voice-activity iterator that wraps a `VoiceActivityDetector`
 * and emits `{ start: sampleIndex }` / `{ end: sampleIndex }` events as
 * speech crosses the configured thresholds.
 *
 * Modeled after `silero_vad.VADIterator` in the Python API.
 */
export class VADIterator {
  /**
   * @param {VoiceActivityDetector} vad – an initialised VAD instance
   * @param {object} [opts]
   * @param {number} [opts.threshold=0.5]       – speech probability threshold
   * @param {number} [opts.negThreshold]       – non-speech threshold (default: threshold − 0.15)
   * @param {number} [opts.minSilenceMs=100]    – silence duration (ms) before emitting speech-end
   * @param {number} [opts.speechPadMs=30]     – padding (ms) added to each speech segment
   * @param {number} [opts.minSpeechMs=250]     – minimum speech segment duration (ms)
   */
  constructor(vad, opts = {}) {
    this._vad = vad;

    this._threshold     = opts.threshold     ?? 0.5;
    this._negThreshold  = opts.negThreshold  ?? Math.max(this._threshold - 0.15, 0.01);
    this._minSilenceMs  = opts.minSilenceMs  ?? 100;
    this._speechPadMs   = opts.speechPadMs   ?? 30;
    this._minSpeechMs   = opts.minSpeechMs   ?? 250;

    // derived sample counts
    this._minSilenceSamples = Math.floor(vad.sampleRate * this._minSilenceMs / 1000);
    this._speechPadSamples  = Math.floor(vad.sampleRate * this._speechPadMs  / 1000);
    this._minSpeechSamples  = Math.floor(vad.sampleRate * this._minSpeechMs  / 1000);

    this.reset();
  }

  reset() {
    this._vad.reset();
    this._triggered = false;
    this._tempEnd    = 0;   // sample index of tentative speech end
    this._currentSample = 0;
  }

  /**
   * Feed one window of audio and return an event (or `null`).
   *
   * Returns one of:
   *   - `{ start: number }`  – speech began at this sample
   *   - `{ end: number }`    – speech ended at this sample
   *   - `null`               – no state change
   *
   * @param {Float32Array|number[]} audio – one window of PCM float samples
   * @returns {Promise<{start:number}|{end:number}|null>}
   */
  async process(audio) {
    const ws = this._vad.windowSize;
    const audioWindow = audio instanceof Float32Array ? audio : new Float32Array(audio);
    this._currentSample += audioWindow.length;

    const prob = await this._vad.processAsync(audioWindow);

    // Speech resumed after a tentative end – cancel the pending end
    if (prob >= this._threshold && this._tempEnd) {
      this._tempEnd = 0;
    }

    // Speech starts
    if (prob >= this._threshold && !this._triggered) {
      this._triggered = true;
      const speechStart = Math.max(0, this._currentSample - this._speechPadSamples - ws);
      return { start: speechStart };
    }

    // Speech ends (below negative threshold while currently triggered)
    if (prob < this._negThreshold && this._triggered) {
      if (!this._tempEnd) {
        this._tempEnd = this._currentSample;
      }
      if (this._currentSample - this._tempEnd < this._minSilenceSamples) {
        return null; // still in silence, but not long enough yet
      }
      const speechEnd = this._tempEnd + this._speechPadSamples - ws;
      this._tempEnd   = 0;
      this._triggered = false;
      return { end: speechEnd };
    }

    return null;
  }

  // Accessors
  get threshold()     { return this._threshold; }
  get negThreshold() { return this._negThreshold; }
  get sampleRate()   { return this._vad.sampleRate; }
  get windowSize()   { return this._vad.windowSize; }
}

// ── Utility functions ──────────────────────────────────────────────────────

/**
 * Given speech probabilities and a threshold, compute speech segments
 * from a full audio buffer. This is the JS equivalent of Silero's
 * `get_speech_timestamps()`.
 *
 * @param {Float32Array} audio   – full audio waveform
 * @param {VoiceActivityDetector} vad – initialised VAD (will be reset)
 * @param {object} [opts]
 * @param {number} [opts.threshold=0.5]
 * @param {number} [opts.negThreshold]
 * @param {number} [opts.minSilenceMs=100]
 * @param {number} [opts.speechPadMs=30]
 * @param {number} [opts.minSpeechMs=250]
 * @returns {Promise<Array<{start:number,end:number}>>} speech segments (sample indices)
 */
export async function getSpeechTimestamps(audio, vad, opts = {}) {
  const threshold     = opts.threshold     ?? 0.5;
  const negThreshold  = opts.negThreshold  ?? Math.max(threshold - 0.15, 0.01);
  const minSilenceMs  = opts.minSilenceMs  ?? 100;
  const speechPadMs   = opts.speechPadMs   ?? 30;
  const minSpeechMs   = opts.minSpeechMs   ?? 250;

  const sr = vad.sampleRate;
  const ws = vad.windowSize;
  const minSilenceSamples = Math.floor(sr * minSilenceMs / 1000);
  const speechPadSamples  = Math.floor(sr * speechPadMs  / 1000);
  const minSpeechSamples  = Math.floor(sr * minSpeechMs   / 1000);

  vad.reset();

  // Collect per-window probabilities
  const probs = [];
  for (let offset = 0; offset < audio.length; offset += ws) {
    const end = Math.min(offset + ws, audio.length);
    const chunk = audio.subarray(offset, end);
    probs.push(await vad.processAsync(chunk));
  }

  // Convert probabilities into speech segments
  let triggered = false;
  let tempEnd   = 0;
  const speeches = [];
  let current = {};

  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    const curSample = ws * i;

    if (p >= threshold && tempEnd) {
      tempEnd = 0;
    }

    if (p >= threshold && !triggered) {
      triggered = true;
      current = { start: curSample };
      continue;
    }

    if (p < negThreshold && triggered) {
      if (!tempEnd) {
        tempEnd = curSample;
      }
      if (curSample - tempEnd < minSilenceSamples) {
        continue;
      }
      current.end = tempEnd;
      if (current.end - current.start > minSpeechSamples) {
        speeches.push(current);
      }
      current = {};
      triggered = false;
      tempEnd   = 0;
    }
  }

  // Handle trailing speech that never ended
  if (triggered && current.start !== undefined) {
    current.end = audio.length;
    if (current.end - current.start > minSpeechSamples) {
      speeches.push(current);
    }
  }

  // Pad segments and resolve overlaps
  for (let i = 0; i < speeches.length; i++) {
    if (i === 0) {
      speeches[i].start = Math.max(0, speeches[i].start - speechPadSamples);
    } else {
      const silence = speeches[i].start - speeches[i - 1].end;
      if (silence < 2 * speechPadSamples) {
        speeches[i - 1].end += Math.floor(silence / 2);
        speeches[i].start = Math.max(0, speeches[i].start - Math.floor(silence / 2));
      } else {
        speeches[i - 1].end = Math.min(audio.length, speeches[i - 1].end + speechPadSamples);
        speeches[i].start = Math.max(0, speeches[i].start - speechPadSamples);
      }
    }
    if (i === speeches.length - 1) {
      speeches[i].end = Math.min(audio.length, speeches[i].end + speechPadSamples);
    }
  }

  return speeches;
}

/**
 * Convert sample-based speech timestamps to seconds.
 *
 * @param {Array<{start:number,end:number}>} timestamps
 * @param {number} sampleRate
 * @returns {Array<{start:number,end:number}>}
 */
export function timestampsToSeconds(timestamps, sampleRate) {
  return timestamps.map(t => ({
    start: Number((t.start / sampleRate).toFixed(3)),
    end:   Number((t.end   / sampleRate).toFixed(3)),
  }));
}