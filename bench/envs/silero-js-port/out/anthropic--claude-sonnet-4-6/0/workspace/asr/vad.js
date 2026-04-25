/**
 * asr/vad.js — Silero VAD interface via onnxruntime-node
 *
 * Mirrors the three-part Python API from silero-vad/src/silero_vad/utils_vad.py:
 *
 *   SileroVad        — low-level ONNX wrapper (OnnxWrapper equivalent)
 *   VADIterator      — streaming chunk-at-a-time iterator
 *   getSpeechTimestamps — offline full-buffer segmentation
 *
 * Quick start:
 *
 *   import { createVad } from './asr/vad.js';
 *
 *   const { getSpeechTimestamps, release } = await createVad();
 *   const segments = await getSpeechTimestamps(float32Audio, { samplingRate: 16000 });
 *   // segments = [{ start: 512, end: 8192 }, ...]
 *   await release();
 */

import { InferenceSession, Tensor } from 'onnxruntime-node';
import { fileURLToPath } from 'node:url';
import { join, dirname } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

/** Default model path — the full-precision opset-16 ONNX shipped with silero-vad. */
export const DEFAULT_MODEL_PATH = join(
  __dirname,
  '../vendor/silero-vad/src/silero_vad/data/silero_vad.onnx',
);

const SUPPORTED_SAMPLE_RATES = [8000, 16000];

// ─────────────────────────────────────────────────────────────────────────────
// SileroVad — low-level ONNX wrapper
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Low-level wrapper around the silero_vad ONNX model.
 *
 * The model signature is:
 *   inputs  : input  [batch, context + window]  float32
 *             state  [2, batch, 128]             float32
 *             sr     [1]                         int64
 *   outputs : output [batch]                     float32   (speech probability)
 *             stateN [2, batch, 128]             float32   (updated RNN state)
 *
 * Context sizes:  64 samples @ 16 kHz,  32 samples @ 8 kHz
 * Window sizes:  512 samples @ 16 kHz, 256 samples @ 8 kHz
 */
export class SileroVad {
  /** @type {InferenceSession} */
  #session;

  // Internal RNN + context state ─────────────────────────────────────────────
  #state       = null;   // Float32Array [2 * batchSize * 128]
  #context     = null;   // Float32Array [batchSize * contextSize], null ⇒ first call
  #lastSr      = 0;
  #lastBatchSize = 0;

  /** @private — use SileroVad.create() */
  constructor(session) {
    this.#session = session;
    this.resetStates();
  }

  /**
   * Load the ONNX model and return a ready SileroVad instance.
   *
   * @param {string} [modelPath]       Path to silero_vad.onnx
   * @param {object} [sessionOptions]  onnxruntime-node SessionOptions overrides
   * @returns {Promise<SileroVad>}
   */
  static async create(modelPath = DEFAULT_MODEL_PATH, sessionOptions = {}) {
    const opts = {
      interOpNumThreads: 1,
      intraOpNumThreads: 1,
      graphOptimizationLevel: 'all',
      ...sessionOptions,
    };
    const session = await InferenceSession.create(modelPath, opts);
    return new SileroVad(session);
  }

  /**
   * Reset the model's RNN state and context buffer.
   * Call this before processing a new, unrelated audio stream.
   *
   * @param {number} [batchSize=1]
   */
  resetStates(batchSize = 1) {
    this.#state         = new Float32Array(2 * batchSize * 128);
    this.#context       = null;   // allocated lazily on first call
    this.#lastSr        = 0;
    this.#lastBatchSize = 0;
  }

  /**
   * Run inference on a single audio chunk.
   *
   * @param {Float32Array} chunk  Audio window (512 samples @16 kHz, 256 @8 kHz)
   * @param {number}       [sr=16000]
   * @returns {Promise<number>}   Speech probability in [0, 1]
   */
  async call(chunk, sr = 16000) {
    const [prob] = await this.callBatch([chunk], sr);
    return prob;
  }

  /**
   * Run inference on a batch of same-length audio chunks.
   *
   * @param {Float32Array[]} batch  One Float32Array per batch item
   * @param {number}         [sr=16000]
   * @returns {Promise<number[]>}   Speech probability per batch item
   */
  async callBatch(batch, sr = 16000) {
    // ── Normalise sample rate (multiples of 16 kHz → 16 kHz by decimation) ──
    sr = this.#normaliseSr(batch, sr);

    const numSamples  = sr === 16000 ? 512 : 256;
    const contextSize = sr === 16000 ?  64 :  32;
    const batchSize   = batch.length;

    if (batch[0].length !== numSamples) {
      throw new Error(
        `Expected ${numSamples} samples for ${sr} Hz, got ${batch[0].length}. ` +
        `Provide exactly 512 samples @16 kHz or 256 samples @8 kHz.`,
      );
    }

    // ── Reset states when batch size or sample rate changes ──────────────────
    if (!this.#lastBatchSize ||
        this.#lastSr        !== sr ||
        this.#lastBatchSize !== batchSize) {
      this.resetStates(batchSize);
    }

    // ── Initialise zero-padded context on first call ──────────────────────────
    if (!this.#context) {
      this.#context = new Float32Array(batchSize * contextSize);
    }

    // ── Build input: [batchSize, contextSize + numSamples] ───────────────────
    const windowLen = contextSize + numSamples;
    const inputData = new Float32Array(batchSize * windowLen);
    for (let b = 0; b < batchSize; b++) {
      inputData.set(
        this.#context.subarray(b * contextSize, (b + 1) * contextSize),
        b * windowLen,
      );
      inputData.set(batch[b], b * windowLen + contextSize);
    }

    // ── ONNX tensors ─────────────────────────────────────────────────────────
    const inputTensor = new Tensor('float32', inputData,                     [batchSize, windowLen]);
    const stateTensor = new Tensor('float32', this.#state,                   [2, batchSize, 128]);
    const srTensor    = new Tensor('int64',   BigInt64Array.of(BigInt(sr)),  [1]);

    const outputs = await this.#session.run({
      input: inputTensor,
      state: stateTensor,
      sr:    srTensor,
    });

    // ── Update context (last contextSize samples of each input window) ────────
    for (let b = 0; b < batchSize; b++) {
      this.#context.set(
        inputData.subarray((b + 1) * windowLen - contextSize, (b + 1) * windowLen),
        b * contextSize,
      );
    }

    this.#state         = Float32Array.from(outputs['stateN'].data);
    this.#lastSr        = sr;
    this.#lastBatchSize = batchSize;

    return Array.from(outputs['output'].data);
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  /**
   * Downsample batch in-place for multiples of 16 kHz; validate sample rate.
   * Mutates the batch array items.
   * @returns {number} Normalised sample rate
   */
  #normaliseSr(batch, sr) {
    if (sr !== 16000 && sr % 16000 === 0) {
      const step = sr / 16000;
      for (let b = 0; b < batch.length; b++) {
        const orig = batch[b];
        const ds   = new Float32Array(Math.ceil(orig.length / step));
        for (let i = 0, j = 0; i < orig.length; i += step, j++) ds[j] = orig[i];
        batch[b] = ds;
      }
      return 16000;
    }
    if (!SUPPORTED_SAMPLE_RATES.includes(sr)) {
      throw new Error(
        `Unsupported sample rate ${sr}. ` +
        `Supported: ${SUPPORTED_SAMPLE_RATES.join(', ')} (or multiples of 16000).`,
      );
    }
    return sr;
  }

  /** Release the underlying ONNX InferenceSession. */
  async release() {
    await this.#session.release();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// VADIterator — streaming chunk-at-a-time interface
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Streaming VAD iterator. Feed fixed-size audio chunks one at a time; receive
 * `{ start }` when speech is detected and `{ end }` when it stops.
 *
 * Mirrors the Python VADIterator class.
 *
 * @example
 * const iter = new VADIterator(vad, { samplingRate: 16000 });
 * for (const chunk of audioChunks) {
 *   const ev = await iter.call(chunk);
 *   if (ev?.start !== undefined) console.log('speech started at', ev.start);
 *   if (ev?.end   !== undefined) console.log('speech ended at',   ev.end);
 * }
 */
export class VADIterator {
  #vad;
  #threshold;
  #sr;
  #minSilenceSamples;
  #speechPadSamples;

  // Streaming state
  #triggered     = false;
  #tempEnd       = 0;
  #currentSample = 0;

  /**
   * @param {SileroVad} vad
   * @param {object}  [opts]
   * @param {number}  [opts.threshold=0.5]             Speech probability threshold
   * @param {number}  [opts.samplingRate=16000]         Must be 8000 or 16000
   * @param {number}  [opts.minSilenceDurationMs=100]  Silence to confirm end of speech
   * @param {number}  [opts.speechPadMs=30]            Padding added around each segment
   */
  constructor(vad, {
    threshold            = 0.5,
    samplingRate         = 16000,
    minSilenceDurationMs = 100,
    speechPadMs          = 30,
  } = {}) {
    if (!SUPPORTED_SAMPLE_RATES.includes(samplingRate)) {
      throw new Error(
        `VADIterator only supports sample rates: ${SUPPORTED_SAMPLE_RATES.join(', ')}.`,
      );
    }
    this.#vad               = vad;
    this.#threshold         = threshold;
    this.#sr                = samplingRate;
    this.#minSilenceSamples = samplingRate * minSilenceDurationMs / 1000;
    this.#speechPadSamples  = samplingRate * speechPadMs           / 1000;
  }

  /**
   * Reset all streaming state (and the underlying model's RNN state).
   * Call this before starting a new audio stream.
   */
  resetStates() {
    this.#vad.resetStates();
    this.#triggered     = false;
    this.#tempEnd       = 0;
    this.#currentSample = 0;
  }

  /**
   * Process one audio chunk.
   *
   * @param {Float32Array} chunk         Window of audio (512 @16 kHz, 256 @8 kHz)
   * @param {boolean}      [returnSeconds=false]  Return timestamps in seconds
   * @returns {Promise<{start: number}|{end: number}|null>}
   *   `{ start }` on speech onset, `{ end }` on speech end, `null` otherwise.
   */
  async call(chunk, returnSeconds = false) {
    const windowSize = chunk.length;
    this.#currentSample += windowSize;

    const prob         = await this.#vad.call(chunk, this.#sr);
    const negThreshold = this.#threshold - 0.15;

    // Clear temp_end if speech has resumed above the main threshold
    if (prob >= this.#threshold && this.#tempEnd) {
      this.#tempEnd = 0;
    }

    // Speech onset
    if (prob >= this.#threshold && !this.#triggered) {
      this.#triggered = true;
      const speechStart = Math.max(
        0, this.#currentSample - this.#speechPadSamples - windowSize,
      );
      return { start: toTimestamp(speechStart, returnSeconds, this.#sr) };
    }

    // Silence while triggered → start / confirm end
    if (prob < negThreshold && this.#triggered) {
      if (!this.#tempEnd) this.#tempEnd = this.#currentSample;

      if (this.#currentSample - this.#tempEnd < this.#minSilenceSamples) {
        return null;  // silence not long enough yet
      }

      const speechEnd = this.#tempEnd + this.#speechPadSamples - windowSize;
      this.#tempEnd   = 0;
      this.#triggered = false;
      return { end: toTimestamp(speechEnd, returnSeconds, this.#sr) };
    }

    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// getSpeechTimestamps — offline full-buffer segmentation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Process a complete audio buffer and return all detected speech segments.
 *
 * Mirrors the Python `get_speech_timestamps()` function, including:
 * - max-speech-duration splitting with best-silence selection
 * - speech padding and adjacent segment merging
 *
 * @param {Float32Array} audio           Mono float32 audio
 * @param {SileroVad}    vad             A SileroVad instance (state will be reset)
 * @param {object}       [opts]
 * @param {number}  [opts.threshold=0.5]             Speech probability threshold
 * @param {number}  [opts.samplingRate=16000]
 * @param {number}  [opts.minSpeechDurationMs=250]   Discard segments shorter than this
 * @param {number}  [opts.maxSpeechDurationS=Infinity] Split segments longer than this
 * @param {number}  [opts.minSilenceDurationMs=100]  Silence required to end a segment
 * @param {number}  [opts.speechPadMs=30]            Padding applied around each segment
 * @param {number}  [opts.negThreshold]              Default: threshold - 0.15 (min 0.01)
 * @param {boolean} [opts.returnSeconds=false]       Return timestamps in seconds
 * @param {number}  [opts.timeResolution=1]          Decimal places when returnSeconds=true
 * @param {function}[opts.onProgress]               Called with progress [0–100] each chunk
 * @returns {Promise<Array<{start: number, end: number}>>}
 */
export async function getSpeechTimestamps(audio, vad, {
  threshold            = 0.5,
  samplingRate         = 16000,
  minSpeechDurationMs  = 250,
  maxSpeechDurationS   = Infinity,
  minSilenceDurationMs = 100,
  speechPadMs          = 30,
  negThreshold         = null,
  returnSeconds        = false,
  timeResolution       = 1,
  onProgress           = null,
} = {}) {

  // Downsample if sample rate is a multiple of 16 kHz
  if (samplingRate > 16000 && samplingRate % 16000 === 0) {
    const step  = samplingRate / 16000;
    const ds    = new Float32Array(Math.ceil(audio.length / step));
    for (let i = 0, j = 0; i < audio.length; i += step, j++) ds[j] = audio[i];
    audio        = ds;
    samplingRate = 16000;
  }

  if (!SUPPORTED_SAMPLE_RATES.includes(samplingRate)) {
    throw new Error(`Unsupported sample rate: ${samplingRate}.`);
  }

  const windowSize = samplingRate === 16000 ? 512 : 256;
  const neg        = negThreshold ?? Math.max(threshold - 0.15, 0.01);

  const minSpeechSamples  = samplingRate * minSpeechDurationMs  / 1000;
  const speechPadSamples  = samplingRate * speechPadMs          / 1000;
  const minSilenceSamples = samplingRate * minSilenceDurationMs / 1000;
  // 98 ms hard-coded minimum silence at max-speech boundary (matches reference implementations)
  const minSilAtMaxSpeech = samplingRate * 98 / 1000;
  const maxSpeechSamples  = isFinite(maxSpeechDurationS)
    ? samplingRate * maxSpeechDurationS - windowSize - 2 * speechPadSamples
    : Infinity;

  const audioLength = audio.length;
  vad.resetStates();

  // ── Step 1: Collect per-chunk speech probabilities ────────────────────────
  const speechProbs = [];
  for (let start = 0; start < audioLength; start += windowSize) {
    let chunk = audio.subarray(start, start + windowSize);
    if (chunk.length < windowSize) {
      const padded = new Float32Array(windowSize);
      padded.set(chunk);
      chunk = padded;
    }
    speechProbs.push(await vad.call(chunk, samplingRate));

    if (onProgress) {
      const done = Math.min(start + windowSize, audioLength);
      onProgress(done / audioLength * 100);
    }
  }

  // ── Step 2: State-machine segmentation ───────────────────────────────────
  //
  // Faithful port of the Python get_speech_timestamps() with
  // use_max_poss_sil_at_max_speech=true (the default).
  //
  let triggered     = false;
  const speeches    = [];
  let currentSpeech = {};

  let tempEnd = 0, prevEnd = 0, nextStart = 0;
  // Candidate silence endpoints collected while in a long speech segment.
  // Each entry: { end: <sample>, dur: <silence_duration_samples> }
  const possibleEnds = [];

  for (let i = 0; i < speechProbs.length; i++) {
    const prob      = speechProbs[i];
    const curSample = windowSize * i;

    // ── Speech resumes after a temporary silence ──────────────────────────
    if (prob >= threshold && tempEnd) {
      const silDur = curSample - tempEnd;
      // Record this as a candidate cut-point if the silence was substantial
      if (silDur > minSilAtMaxSpeech) {
        possibleEnds.push({ end: tempEnd, dur: silDur });
      }
      tempEnd = 0;
      if (nextStart < prevEnd) nextStart = curSample;
    }

    // ── Speech onset ──────────────────────────────────────────────────────
    if (prob >= threshold && !triggered) {
      triggered             = true;
      currentSpeech.start   = curSample;
      continue;
    }

    // ── Max speech duration reached ───────────────────────────────────────
    if (triggered && curSample - currentSpeech.start > maxSpeechSamples) {
      if (possibleEnds.length) {
        // Use the longest silence we've seen — least abrupt cut
        const best       = possibleEnds.reduce((a, b) => b.dur > a.dur ? b : a);
        currentSpeech.end = best.end;
        speeches.push({ ...currentSpeech });
        currentSpeech     = {};
        nextStart         = best.end + best.dur;  // sample where speech resumed

        // If we're still before that resumption point, start a new segment there
        if (nextStart < curSample) {
          currentSpeech.start = nextStart;
        } else {
          triggered = false;
        }
        prevEnd = nextStart = tempEnd = 0;
        possibleEnds.length = 0;
      } else if (prevEnd) {
        // Fall back to the last significant silence boundary
        currentSpeech.end = prevEnd;
        speeches.push({ ...currentSpeech });
        currentSpeech     = {};
        if (nextStart < prevEnd) {
          triggered = false;
        } else {
          currentSpeech.start = nextStart;
        }
        prevEnd = nextStart = tempEnd = 0;
        possibleEnds.length = 0;
      } else {
        // No usable silence → cut here
        currentSpeech.end = curSample;
        speeches.push({ ...currentSpeech });
        currentSpeech     = {};
        prevEnd = nextStart = tempEnd = 0;
        triggered           = false;
        possibleEnds.length = 0;
        continue;
      }
    }

    // ── Silence while triggered ───────────────────────────────────────────
    if (prob < neg && triggered) {
      if (!tempEnd) tempEnd = curSample;
      const silDurNow = curSample - tempEnd;

      // Track a prev_end candidate for max-speech fallback
      if (silDurNow > minSilAtMaxSpeech) prevEnd = tempEnd;

      if (silDurNow < minSilenceSamples) continue;  // silence not long enough yet

      // Confirm end of speech segment
      currentSpeech.end = tempEnd;
      if (currentSpeech.end - currentSpeech.start > minSpeechSamples) {
        speeches.push({ ...currentSpeech });
      }
      currentSpeech       = {};
      prevEnd = nextStart = tempEnd = 0;
      triggered           = false;
      possibleEnds.length = 0;
      continue;
    }
  }

  // ── Flush any still-open speech at the end of audio ──────────────────────
  if (currentSpeech.start !== undefined &&
      audioLength - currentSpeech.start > minSpeechSamples) {
    currentSpeech.end = audioLength;
    speeches.push({ ...currentSpeech });
  }

  // ── Step 3: Apply speech padding, merge close segments ───────────────────
  for (let i = 0; i < speeches.length; i++) {
    const s = speeches[i];

    if (i === 0) {
      s.start = Math.max(0, s.start - speechPadSamples);
    }

    if (i < speeches.length - 1) {
      const silDur = speeches[i + 1].start - s.end;
      if (silDur < 2 * speechPadSamples) {
        // Gap is smaller than two pads → split evenly rather than overlapping
        s.end                 += Math.floor(silDur / 2);
        speeches[i + 1].start  = Math.max(0, speeches[i + 1].start - Math.floor(silDur / 2));
      } else {
        s.end                  = Math.min(audioLength, s.end + speechPadSamples);
        speeches[i + 1].start  = Math.max(0, speeches[i + 1].start - speechPadSamples);
      }
    } else {
      s.end = Math.min(audioLength, s.end + speechPadSamples);
    }
  }

  // ── Step 4: Optionally convert sample offsets → seconds ──────────────────
  if (returnSeconds) {
    const audioLengthSec = audioLength / samplingRate;
    for (const s of speeches) {
      s.start = Math.max(roundTo(s.start / samplingRate, timeResolution), 0);
      s.end   = Math.min(roundTo(s.end   / samplingRate, timeResolution), audioLengthSec);
    }
  }

  return speeches;
}

// ─────────────────────────────────────────────────────────────────────────────
// createVad — convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Load the Silero VAD model and return a ready-to-use object with all helpers
 * pre-bound.
 *
 * @param {string} [modelPath]       Defaults to the bundled silero_vad.onnx
 * @param {object} [sessionOptions]  onnxruntime-node SessionOptions overrides
 * @returns {Promise<{
 *   vad: SileroVad,
 *   call: (chunk: Float32Array, sr?: number) => Promise<number>,
 *   resetStates: (batchSize?: number) => void,
 *   getSpeechTimestamps: (audio: Float32Array, opts?: object) => Promise<Array>,
 *   createIterator: (opts?: object) => VADIterator,
 *   release: () => Promise<void>,
 * }>}
 *
 * @example
 * const vad = await createVad();
 * const segments = await vad.getSpeechTimestamps(audio, { samplingRate: 16000 });
 * await vad.release();
 */
export async function createVad(modelPath = DEFAULT_MODEL_PATH, sessionOptions = {}) {
  const vad = await SileroVad.create(modelPath, sessionOptions);

  return {
    /** The underlying SileroVad instance, in case you need direct access. */
    vad,

    /**
     * Run inference on one fixed-size audio chunk.
     * @param {Float32Array} chunk
     * @param {number}       [sr=16000]
     * @returns {Promise<number>} Speech probability in [0, 1]
     */
    call: (chunk, sr = 16000) => vad.call(chunk, sr),

    /**
     * Reset the model's internal RNN state.
     * @param {number} [batchSize=1]
     */
    resetStates: (batchSize = 1) => vad.resetStates(batchSize),

    /**
     * Offline segmentation of a complete audio buffer.
     * @param {Float32Array} audio
     * @param {object}       [opts]  See getSpeechTimestamps() for all options
     * @returns {Promise<Array<{start: number, end: number}>>}
     */
    getSpeechTimestamps: (audio, opts = {}) =>
      getSpeechTimestamps(audio, vad, opts),

    /**
     * Create a VADIterator bound to this model instance for streaming use.
     * @param {object} [opts]  See VADIterator constructor options
     * @returns {VADIterator}
     */
    createIterator: (opts = {}) => new VADIterator(vad, opts),

    /** Release the underlying ONNX InferenceSession. */
    release: () => vad.release(),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Round `value` to `decimals` decimal places.
 * @param {number} value
 * @param {number} decimals
 */
function roundTo(value, decimals) {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

/**
 * Convert a sample offset to the requested timestamp format.
 * @param {number}  sample
 * @param {boolean} inSeconds
 * @param {number}  sr
 */
function toTimestamp(sample, inSeconds, sr) {
  if (!inSeconds) return Math.round(sample);
  return Math.round(sample / sr * 1000) / 1000;  // 3 decimal places
}
