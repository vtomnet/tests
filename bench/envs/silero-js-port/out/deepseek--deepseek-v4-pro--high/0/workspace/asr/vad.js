/**
 * Silero VAD (Voice Activity Detection) using ONNX Runtime for JavaScript.
 *
 * Usage:
 *
 *   import { VAD, getSpeechTimestamps, VADIterator } from "./vad.js";
 *
 *   // One-shot: process an entire Float32Array of 16kHz mono audio
 *   const model = await VAD.load("silero_vad.onnx");
 *   const segments = await getSpeechTimestamps(audio, model, { returnSeconds: true });
 *   // segments = [{ start: 0.0, end: 1.2 }, ...]
 *
 *   // Streaming: feed chunks as they arrive
 *   const vad = new VADIterator(model, { threshold: 0.5, samplingRate: 16000 });
 *   for (const chunk of audioChunks) {
 *     const event = await vad.process(chunk);
 *     if (event) console.log(event); // { start: N } or { end: N }
 *   }
 */

// ---------------------------------------------------------------------------
// ONNX model wrapper (equivalent to OnnxWrapper in silero_vad/utils_vad.py)
// ---------------------------------------------------------------------------

const SUPPORTED_SAMPLE_RATES = [8000, 16000];
const WINDOW_SIZE_SAMPLES = { 8000: 256, 16000: 512 };
const CONTEXT_SIZE_SAMPLES = { 8000: 32, 16000: 64 };

export class VAD {
  /**
   * Load a Silero VAD ONNX model.
   * @param {string|ArrayBuffer|Uint8Array} modelPath - path or buffer
   * @returns {Promise<VAD>}
   */
  static async load(modelPath) {
    // Try Node.js backend first, fall back to common
    let ort;
    try {
      ort = await import("onnxruntime-node");
    } catch {
      ort = await import("onnxruntime-common");
    }

    const session = await ort.InferenceSession.create(modelPath, {
      graphOptimizationLevel: "all",
      intraOpNumThreads: 1,
      interOpNumThreads: 1,
    });

    return new VAD(session, ort);
  }

  constructor(session, ort) {
    this._session = session;
    this._ort = ort;
    this.reset();
  }

  /**
   * Reset internal state and context. Must be called before processing
   * a new audio stream.
   */
  reset() {
    // State: shape [2, 1, 128] of floats, initialized to zeros
    this._state = new Float32Array(2 * 1 * 128);
    // Context: shape [contextSize] of floats, initialized to zeros
    this._context = new Float32Array(0);
    this._lastSampleRate = 0;
    this._lastBatchSize = 0;
  }

  /**
   * Run inference on a single audio chunk.
   *
   * @param {Float32Array} x - audio samples (must be windowSize long)
   * @param {number} sampleRate - 8000 or 16000 (or multiple of 16000)
   * @returns {number} speech probability (0..1)
   */
  async call(x, sampleRate) {
    const { sr, data } = this._validateInput(x, sampleRate);
    x = data;
    const windowSize = WINDOW_SIZE_SAMPLES[sr];
    const contextSize = CONTEXT_SIZE_SAMPLES[sr];

    if (x.length !== windowSize) {
      throw new Error(
        `Expected ${windowSize} samples for ${sr} Hz, got ${x.length}`,
      );
    }

    // Reset state if sample rate or batch size changed
    if (!this._lastBatchSize) this.reset();
    if (this._lastSampleRate && this._lastSampleRate !== sr) this.reset();

    if (this._context.length === 0) {
      this._context = new Float32Array(contextSize);
    }

    // Prepend context to the input
    const effectiveSize = contextSize + windowSize;
    const input = new Float32Array(effectiveSize);
    input.set(this._context, 0);
    input.set(x, contextSize);

    // Build input tensors
    const inputTensor = new this._ort.Tensor("float32", input, [
      1,
      effectiveSize,
    ]);
    const stateTensor = new this._ort.Tensor("float32", this._state, [
      2,
      1,
      128,
    ]);
    const srTensor = new this._ort.Tensor(
      "int64",
      BigInt64Array.from([BigInt(sr)]),
      [1],
    );

    const feeds = {
      input: inputTensor,
      state: stateTensor,
      sr: srTensor,
    };

    const results = await this._session.run(feeds);

    const output = results.output.data[0]; // float32 scalar
    const stateN = results.stateN.data; // Float32Array [2*1*128]

    // Update state and context
    this._state.set(stateN);
    this._context.set(input.subarray(effectiveSize - contextSize));
    this._lastSampleRate = sr;
    this._lastBatchSize = 1;

    return output;
  }

  // -----------------------------------------------------------------------
  // Private helpers
  // -----------------------------------------------------------------------

  _validateInput(x, sampleRate) {
    let sr = sampleRate;

    // Support multiples of 16000 by decimating
    if (sr !== 16000 && sr % 16000 === 0) {
      const step = sr / 16000;
      sr = 16000;
      const reducedSize = Math.ceil(x.length / step);
      const reduced = new Float32Array(reducedSize);
      for (let i = 0, j = 0; i < x.length; i += step, j++) {
        reduced[j] = x[i];
      }
      x = reduced;
    }

    if (!SUPPORTED_SAMPLE_RATES.includes(sr)) {
      throw new Error(
        `Sample rate ${sampleRate} not supported. ` +
          `Use 8000, 16000, or multiples of 16000.`,
      );
    }

    // Check audio isn't too short (more than 31.25 windows per second)
    if (sr / x.length > 31.25) {
      throw new Error("Input audio chunk is too short");
    }

    return { sr, data: x };
  }
}

// ---------------------------------------------------------------------------
// getSpeechTimestamps (one-shot processing, equivalent to Python version)
// ---------------------------------------------------------------------------

/**
 * Detect speech segments in an audio buffer.
 *
 * @param {Float32Array} audio - mono 16kHz (or 8kHz) float samples
 * @param {VAD} model - a loaded VAD instance
 * @param {object} [options]
 * @param {number} [options.threshold=0.5] - speech probability threshold
 * @param {number} [options.samplingRate=16000] - sample rate (8000 or 16000)
 * @param {number} [options.minSpeechDurationMs=250]
 * @param {number} [options.maxSpeechDurationS=Infinity]
 * @param {number} [options.minSilenceDurationMs=100]
 * @param {number} [options.speechPadMs=30]
 * @param {number} [options.minSilenceAtMaxSpeechMs=98]
 * @param {boolean} [options.returnSeconds=false] - return seconds instead of samples
 * @param {number} [options.timeResolution=1] - decimal places for seconds
 * @returns {Promise<Array<{start: number, end: number}>>} speech segments
 */
export async function getSpeechTimestamps(audio, model, options = {}) {
  const {
    threshold = 0.5,
    samplingRate = 16000,
    minSpeechDurationMs = 250,
    maxSpeechDurationS = Infinity,
    minSilenceDurationMs = 100,
    speechPadMs = 30,
    minSilenceAtMaxSpeechMs = 98,
    returnSeconds = false,
    timeResolution = 1,
  } = options;

  const negThreshold = Math.max(threshold - 0.15, 0.01);

  // Validate sample rate
  let sr = samplingRate;
  let step = 1;
  if (sr > 16000 && sr % 16000 === 0) {
    step = sr / 16000;
    sr = 16000;
  }
  if (!SUPPORTED_SAMPLE_RATES.includes(sr)) {
    throw new Error(
      `Sample rate ${samplingRate} not supported. ` +
        `Use 8000, 16000, or multiples of 16000.`,
    );
  }

  const windowSize = WINDOW_SIZE_SAMPLES[sr];

  model.reset();

  const minSpeechSamples = (samplingRate * minSpeechDurationMs) / 1000;
  const speechPadSamples = (samplingRate * speechPadMs) / 1000;
  const maxSpeechSamples =
    samplingRate * maxSpeechDurationS - windowSize - 2 * speechPadSamples;
  const minSilenceSamples = (samplingRate * minSilenceDurationMs) / 1000;
  const minSilenceSamplesAtMaxSpeech =
    (samplingRate * minSilenceAtMaxSpeechMs) / 1000;
  const audioLengthSamples = audio.length;

  // Run inference across all windows
  const speechProbs = [];
  for (let i = 0; i < audioLengthSamples; i += windowSize) {
    let chunk = audio.subarray(i, i + windowSize);
    if (chunk.length < windowSize) {
      const padded = new Float32Array(windowSize);
      padded.set(chunk);
      chunk = padded;
    }
    const prob = await model.call(chunk, sr);
    speechProbs.push(prob);
  }

  return calculateSegments(speechProbs, {
    threshold,
    negThreshold,
    windowSize,
    audioLengthSamples,
    minSpeechSamples,
    maxSpeechSamples,
    minSilenceSamples,
    minSilenceSamplesAtMaxSpeech,
    speechPadSamples,
    returnSeconds,
    timeResolution,
    sampleRate: samplingRate,
    step,
  });
}

// ---------------------------------------------------------------------------
// VADIterator (streaming, equivalent to Python VADIterator)
// ---------------------------------------------------------------------------

export class VADIterator {
  /**
   * Create a streaming VAD iterator.
   *
   * @param {VAD} model - a loaded VAD instance
   * @param {object} [options]
   * @param {number} [options.threshold=0.5]
   * @param {number} [options.samplingRate=16000]
   * @param {number} [options.minSilenceDurationMs=100]
   * @param {number} [options.speechPadMs=30]
   */
  constructor(model, options = {}) {
    const {
      threshold = 0.5,
      samplingRate = 16000,
      minSilenceDurationMs = 100,
      speechPadMs = 30,
    } = options;

    if (!SUPPORTED_SAMPLE_RATES.includes(samplingRate)) {
      throw new Error(
        `VADIterator only supports sample rates ${SUPPORTED_SAMPLE_RATES}`,
      );
    }

    this._model = model;
    this._threshold = threshold;
    this._samplingRate = samplingRate;
    this._minSilenceSamples = (samplingRate * minSilenceDurationMs) / 1000;
    this._speechPadSamples = (samplingRate * speechPadMs) / 1000;
    this.reset();
  }

  /** Reset internal state. Call before processing a new stream. */
  reset() {
    this._model.reset();
    this._triggered = false;
    this._tempEnd = 0;
    this._currentSample = 0;
  }

  /**
   * Process an audio chunk. Returns a speech event when a segment
   * starts or ends, or null otherwise.
   *
   * @param {Float32Array} chunk - audio samples (must be windowSize long)
   * @param {object} [options]
   * @param {boolean} [options.returnSeconds=false]
   * @param {number} [options.timeResolution=1]
   * @returns {Promise<{start?: number}|{end?: number}|null>}
   */
  async process(chunk, options = {}) {
    const { returnSeconds = false, timeResolution = 1 } = options;

    const windowSize = WINDOW_SIZE_SAMPLES[this._samplingRate];
    if (chunk.length > windowSize) {
      // Split into exact window-size chunks if caller gives more
      let result = null;
      for (let i = 0; i < chunk.length; i += windowSize) {
        const sub = chunk.subarray(i, Math.min(i + windowSize, chunk.length));
        if (sub.length < windowSize) {
          const padded = new Float32Array(windowSize);
          padded.set(sub);
          // Don't process short tail — wait for more data
          break;
        }
        result = await this._processWindow(sub, returnSeconds, timeResolution);
        if (result) return result;
      }
      return null;
    }

    if (chunk.length < windowSize) {
      // Not enough data yet
      return null;
    }

    return this._processWindow(chunk, returnSeconds, timeResolution);
  }

  // Private: process exactly one window
  async _processWindow(chunk, returnSeconds, timeResolution) {
    const windowSize = WINDOW_SIZE_SAMPLES[this._samplingRate];
    this._currentSample += windowSize;

    const speechProb = await this._model.call(
      chunk,
      this._samplingRate,
    );
    const threshold = this._threshold;
    const negThreshold = threshold - 0.15;

    // Reset temp end if speech resumes after a dip
    if (speechProb >= threshold && this._tempEnd) {
      this._tempEnd = 0;
    }

    // Speech start
    if (speechProb >= threshold && !this._triggered) {
      this._triggered = true;
      const start = Math.max(
        0,
        this._currentSample - this._speechPadSamples - windowSize,
      );
      return {
        start: returnSeconds
          ? roundTo(start / this._samplingRate, timeResolution)
          : start,
      };
    }

    // Silence while triggered → potential end
    if (speechProb < negThreshold && this._triggered) {
      if (!this._tempEnd) {
        this._tempEnd = this._currentSample;
      }
      if (this._currentSample - this._tempEnd < this._minSilenceSamples) {
        return null;
      }
      // End detected
      const end =
        this._tempEnd + this._speechPadSamples - windowSize;
      this._tempEnd = 0;
      this._triggered = false;
      return {
        end: returnSeconds
          ? roundTo(end / this._samplingRate, timeResolution)
          : Math.round(end),
      };
    }

    return null;
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function calculateSegments(speechProbs, opts) {
  const {
    threshold,
    negThreshold,
    windowSize,
    audioLengthSamples,
    minSpeechSamples,
    maxSpeechSamples,
    minSilenceSamples,
    minSilenceSamplesAtMaxSpeech,
    speechPadSamples,
    returnSeconds,
    timeResolution,
    sampleRate,
    step,
  } = opts;

  let triggered = false;
  const speeches = [];
  let currentSpeech = {};
  let tempEnd = 0;
  let prevEnd = 0;
  let nextStart = 0;
  const possibleEnds = []; // [{end, duration}] for max-speech cut selection

  for (let i = 0; i < speechProbs.length; i++) {
    const speechProb = speechProbs[i];
    const curSample = windowSize * i;

    // Speech resumed after a temp_end: track candidate silence
    if (speechProb >= threshold && tempEnd) {
      const silDuration = curSample - tempEnd;
      if (silDuration > minSilenceSamplesAtMaxSpeech) {
        possibleEnds.push({ end: tempEnd, duration: silDuration });
      }
      tempEnd = 0;
      if (nextStart < prevEnd) {
        nextStart = curSample;
      }
    }

    // Start of speech
    if (speechProb >= threshold && !triggered) {
      triggered = true;
      currentSpeech = { start: curSample };
      continue;
    }

    // Max speech duration exceeded
    if (
      triggered &&
      curSample - currentSpeech.start > maxSpeechSamples
    ) {
      if (possibleEnds.length > 0) {
        // Use the longest silence within this segment
        let best = possibleEnds[0];
        for (const pe of possibleEnds) {
          if (pe.duration > best.duration) best = pe;
        }
        prevEnd = best.end;
        const dur = best.duration;

        currentSpeech.end = prevEnd;
        speeches.push({ ...currentSpeech });
        currentSpeech = {};
        nextStart = prevEnd + dur;

        if (nextStart < prevEnd + curSample) {
          currentSpeech = { start: nextStart };
        } else {
          triggered = false;
        }
        prevEnd = 0;
        nextStart = 0;
        tempEnd = 0;
        possibleEnds.length = 0;
      } else if (prevEnd) {
        // Legacy: use the last valid silence
        currentSpeech.end = prevEnd;
        speeches.push({ ...currentSpeech });
        currentSpeech = {};
        if (nextStart < prevEnd) {
          triggered = false;
        } else {
          currentSpeech = { start: nextStart };
        }
        prevEnd = 0;
        nextStart = 0;
        tempEnd = 0;
        possibleEnds.length = 0;
      } else {
        // Hard cut at current sample
        currentSpeech.end = curSample;
        speeches.push({ ...currentSpeech });
        currentSpeech = {};
        prevEnd = 0;
        nextStart = 0;
        tempEnd = 0;
        triggered = false;
        possibleEnds.length = 0;
        continue;
      }
    }

    // Silence while triggered
    if (speechProb < negThreshold && triggered) {
      if (!tempEnd) {
        tempEnd = curSample;
      }
      const silDurationNow = curSample - tempEnd;

      if (silDurationNow > minSilenceSamplesAtMaxSpeech) {
        prevEnd = tempEnd;
      }
      if (silDurationNow < minSilenceSamples) {
        continue;
      }
      // Silence long enough → end the segment
      currentSpeech.end = tempEnd;
      if (currentSpeech.end - currentSpeech.start > minSpeechSamples) {
        speeches.push({ ...currentSpeech });
      }
      currentSpeech = {};
      prevEnd = 0;
      nextStart = 0;
      tempEnd = 0;
      triggered = false;
      possibleEnds.length = 0;
      continue;
    }
  }

  // Handle trailing speech
  if (
    currentSpeech.start !== undefined &&
    audioLengthSamples - currentSpeech.start > minSpeechSamples
  ) {
    currentSpeech.end = audioLengthSamples;
    speeches.push({ ...currentSpeech });
  }

  // Apply padding
  for (let i = 0; i < speeches.length; i++) {
    const seg = speeches[i];
    if (i === 0) {
      seg.start = Math.max(0, Math.round(seg.start - speechPadSamples));
    }
    if (i !== speeches.length - 1) {
      const silenceDuration = speeches[i + 1].start - seg.end;
      if (silenceDuration < 2 * speechPadSamples) {
        seg.end += Math.floor(silenceDuration / 2);
        speeches[i + 1].start = Math.max(
          0,
          Math.round(speeches[i + 1].start - silenceDuration / 2),
        );
      } else {
        seg.end = Math.min(
          audioLengthSamples,
          Math.round(seg.end + speechPadSamples),
        );
        speeches[i + 1].start = Math.max(
          0,
          Math.round(speeches[i + 1].start - speechPadSamples),
        );
      }
    } else {
      seg.end = Math.min(
        audioLengthSamples,
        Math.round(seg.end + speechPadSamples),
      );
    }
  }

  // Apply step (for subsampled inputs) or convert to seconds
  if (returnSeconds) {
    const audioLenSec = audioLengthSamples / sampleRate;
    for (const seg of speeches) {
      seg.start = Math.max(
        0,
        roundTo(seg.start / sampleRate, timeResolution),
      );
      seg.end = Math.min(
        audioLenSec,
        roundTo(seg.end / sampleRate, timeResolution),
      );
    }
  } else if (step > 1) {
    for (const seg of speeches) {
      seg.start *= step;
      seg.end *= step;
    }
  }

  return speeches;
}

function roundTo(value, decimals) {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}
