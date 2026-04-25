// Silero VAD ONNX wrapper for JavaScript.
//
// This module mirrors the public pieces normally used from the Python
// silero-vad package: load_silero_vad(), get_speech_timestamps(), and
// VADIterator. It uses ONNX Runtime JS (usually `onnxruntime-node`) and the
// vendored Silero ONNX model.

export const DEFAULT_MODEL_PATH = new URL(
  "../vendor/silero-vad/src/silero_vad/data/silero_vad.onnx",
  import.meta.url,
);

const SUPPORTED_SAMPLE_RATES = [8000, 16000];
const THRESHOLD_GAP = 0.15;

async function loadOrt(ort) {
  if (ort) return ort.default ?? ort;

  try {
    const mod = await import("onnxruntime-node");
    return mod.default ?? mod;
  } catch (nodeError) {
    try {
      const mod = await import("onnxruntime-web");
      return mod.default ?? mod;
    } catch {
      throw new Error(
        "Silero VAD needs ONNX Runtime JS. Install `onnxruntime-node` " +
          "(or pass an ort module via { ort }). Original error: " +
          nodeError.message,
      );
    }
  }
}

function isTypedArray(value) {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

function isArrayLikeAudio(value) {
  return Array.isArray(value) || isTypedArray(value);
}

function isNestedAudio(value) {
  return Array.isArray(value) && value.length > 0 && isArrayLikeAudio(value[0]);
}

function toFloat32Samples(samples) {
  if (samples instanceof Float32Array) return samples;

  if (samples instanceof Int16Array) {
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) out[i] = Math.max(-1, samples[i] / 32768);
    return out;
  }

  if (samples instanceof Int32Array) {
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) out[i] = Math.max(-1, samples[i] / 2147483648);
    return out;
  }

  if (samples instanceof Uint8Array || samples instanceof Uint8ClampedArray) {
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) out[i] = (samples[i] - 128) / 128;
    return out;
  }

  if (Array.isArray(samples) || isTypedArray(samples)) {
    return Float32Array.from(samples);
  }

  throw new TypeError("Audio must be an Array, TypedArray, or a batch of those values");
}

function toBatch(samples) {
  if (isNestedAudio(samples)) return samples.map(toFloat32Samples);
  return [toFloat32Samples(samples)];
}

function downsampleByStep(samples, step) {
  if (step <= 1) return samples;
  const out = new Float32Array(Math.ceil(samples.length / step));
  for (let i = 0, j = 0; i < samples.length; i += step, j++) out[j] = samples[i];
  return out;
}

function validateAndPrepareBatch(samples, samplingRate, sampleRates = SUPPORTED_SAMPLE_RATES) {
  let sr = samplingRate;
  let batch = toBatch(samples);

  if (sr !== 16000 && sr % 16000 === 0) {
    const step = sr / 16000;
    batch = batch.map((row) => downsampleByStep(row, step));
    sr = 16000;
  }

  if (!sampleRates.includes(sr)) {
    throw new Error(`Supported sampling rates: ${sampleRates.join(", ")} (or multiples of 16000)`);
  }

  if (batch.length === 0 || batch[0].length === 0) {
    throw new Error("Input audio chunk is empty");
  }

  const width = batch[0].length;
  for (const row of batch) {
    if (row.length !== width) throw new Error("All batched audio chunks must have the same length");
  }

  if (sr / width > 31.25) {
    throw new Error("Input audio chunk is too short");
  }

  return { batch, samplingRate: sr };
}

function flattenBatch(batch) {
  const rows = batch.length;
  const cols = batch[0].length;
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) out.set(batch[r], r * cols);
  return { data: out, rows, cols };
}

function resolveOutput(results, preferredName, index) {
  if (results[preferredName]) return results[preferredName];
  const values = Object.values(results);
  if (!values[index]) throw new Error(`ONNX output ${preferredName} was not returned`);
  return values[index];
}

function resolveModelPath(modelPath) {
  if (modelPath instanceof URL) {
    if (modelPath.protocol !== "file:") return modelPath.href;

    // Minimal fileURLToPath equivalent without importing Node built-ins, so this
    // file still parses in browser-like runtimes.
    let pathname = decodeURIComponent(modelPath.pathname);
    if (typeof process !== "undefined" && process.platform === "win32" && /^\/[A-Za-z]:/.test(pathname)) {
      pathname = pathname.slice(1);
    }
    return pathname;
  }
  return String(modelPath);
}

function resetModel(model, batchSize) {
  if (typeof model.resetStates === "function") model.resetStates(batchSize);
  else if (typeof model.reset_states === "function") model.reset_states(batchSize);
}

function roundTo(value, digits = 1) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function cloneSegment(segment) {
  return { start: segment.start, end: segment.end };
}

/**
 * Stateful wrapper around the Silero ONNX model.
 *
 * Input chunks must be 512 samples at 16 kHz or 256 samples at 8 kHz. Audio
 * should be mono Float32 samples in [-1, 1]. Int PCM TypedArrays are accepted
 * and normalized for convenience.
 */
export class SileroVadOnnxModel {
  constructor(session, ort, options = {}) {
    this.session = session;
    this.ort = ort;
    this.sampleRates = options.sampleRates ?? SUPPORTED_SAMPLE_RATES;
    this.resetStates();
  }

  static async create(options = {}) {
    const ort = await loadOrt(options.ort);
    const modelPath = options.modelPath ?? DEFAULT_MODEL_PATH;
    const pathOrUrl = resolveModelPath(modelPath);

    const sessionOptions = {
      executionProviders: options.executionProviders ?? ["cpu"],
      interOpNumThreads: 1,
      intraOpNumThreads: 1,
      ...options.sessionOptions,
    };

    const session = await ort.InferenceSession.create(pathOrUrl, sessionOptions);
    return new SileroVadOnnxModel(session, ort, options);
  }

  resetStates(batchSize = 1) {
    this.state = new Float32Array(2 * batchSize * 128);
    this.context = new Float32Array(0);
    this.lastSamplingRate = 0;
    this.lastBatchSize = 0;
  }

  /** Alias matching the Python wrapper name. */
  reset_states(batchSize = 1) {
    this.resetStates(batchSize);
  }

  /**
   * Run one Silero frame. Returns one probability per batch item.
   */
  async call(samples, samplingRate) {
    let { batch, samplingRate: sr } = validateAndPrepareBatch(samples, samplingRate, this.sampleRates);
    const numberSamples = sr === 16000 ? 512 : 256;

    if (batch[0].length !== numberSamples) {
      throw new Error(
        `Provided number of samples is ${batch[0].length} ` +
          `(supported values: 256 for 8000 sample rate, 512 for 16000)`,
      );
    }

    const batchSize = batch.length;
    const contextSize = sr === 16000 ? 64 : 32;

    if (!this.lastBatchSize || (this.lastSamplingRate && this.lastSamplingRate !== sr) || this.lastBatchSize !== batchSize) {
      this.resetStates(batchSize);
    }

    if (this.context.length === 0) {
      this.context = new Float32Array(batchSize * contextSize);
    }

    const inputCols = contextSize + numberSamples;
    const input = new Float32Array(batchSize * inputCols);
    for (let r = 0; r < batchSize; r++) {
      input.set(this.context.subarray(r * contextSize, (r + 1) * contextSize), r * inputCols);
      input.set(batch[r], r * inputCols + contextSize);
    }

    const Tensor = this.ort.Tensor;
    const feeds = {
      input: new Tensor("float32", input, [batchSize, inputCols]),
      state: new Tensor("float32", this.state, [2, batchSize, 128]),
      sr: new Tensor("int64", new BigInt64Array([BigInt(sr)]), [1]),
    };

    const results = await this.session.run(feeds);
    const outputTensor = resolveOutput(results, "output", 0);
    const stateTensor = resolveOutput(results, "stateN", 1);

    this.state = new Float32Array(stateTensor.data);
    this.context = new Float32Array(batchSize * contextSize);
    for (let r = 0; r < batchSize; r++) {
      const rowStart = r * inputCols;
      this.context.set(input.subarray(rowStart + inputCols - contextSize, rowStart + inputCols), r * contextSize);
    }

    this.lastSamplingRate = sr;
    this.lastBatchSize = batchSize;

    return Array.from(outputTensor.data, Number);
  }

  /** Convenience for the common non-batched case. */
  async predict(samples, samplingRate) {
    const probs = await this.call(samples, samplingRate);
    return probs[0];
  }

  /**
   * Run VAD probabilities over a full clip, padding the final frame as Silero's
   * Python `audio_forward()` does.
   */
  async audioForward(audio, samplingRate) {
    let { batch, samplingRate: sr } = validateAndPrepareBatch(audio, samplingRate, this.sampleRates);
    const numSamples = sr === 16000 ? 512 : 256;
    const width = batch[0].length;
    const pad = width % numSamples ? numSamples - (width % numSamples) : 0;

    if (pad) {
      batch = batch.map((row) => {
        const padded = new Float32Array(width + pad);
        padded.set(row);
        return padded;
      });
    }

    this.resetStates(batch.length);
    const outs = [];
    for (let i = 0; i < batch[0].length; i += numSamples) {
      const frame = batch.map((row) => row.subarray(i, i + numSamples));
      outs.push(await this.call(frame, sr));
    }
    return outs;
  }
}

/** Python-style loader. */
export async function loadSileroVad(options = {}) {
  return SileroVadOnnxModel.create(options);
}

/** Python-style alias. */
export const load_silero_vad = loadSileroVad;

/**
 * Split a mono clip into speech timestamps, ported from silero_vad.utils_vad.
 *
 * Returns [{ start, end }, ...] in sample offsets by default, or seconds when
 * `returnSeconds` is true.
 */
export async function getSpeechTimestamps(audio, model, options = {}) {
  if (!model) model = await loadSileroVad(options);

  let samplingRate = options.samplingRate ?? options.sampling_rate ?? 16000;
  const threshold = options.threshold ?? 0.5;
  const minSpeechDurationMs = options.minSpeechDurationMs ?? options.min_speech_duration_ms ?? 250;
  const maxSpeechDurationS = options.maxSpeechDurationS ?? options.max_speech_duration_s ?? Infinity;
  const minSilenceDurationMs = options.minSilenceDurationMs ?? options.min_silence_duration_ms ?? 100;
  const speechPadMs = options.speechPadMs ?? options.speech_pad_ms ?? 30;
  const returnSeconds = options.returnSeconds ?? options.return_seconds ?? false;
  const timeResolution = options.timeResolution ?? options.time_resolution ?? 1;
  const progressTrackingCallback = options.progressTrackingCallback ?? options.progress_tracking_callback;
  const negThreshold = options.negThreshold ?? options.neg_threshold ?? Math.max(threshold - THRESHOLD_GAP, 0.01);
  const minSilenceAtMaxSpeech = options.minSilenceAtMaxSpeech ?? options.min_silence_at_max_speech ?? 98;
  const useMaxPossibleSilenceAtMaxSpeech =
    options.useMaxPossibleSilenceAtMaxSpeech ?? options.use_max_poss_sil_at_max_speech ?? true;

  let wav = toFloat32Samples(audio);
  let step = 1;
  if (samplingRate > 16000 && samplingRate % 16000 === 0) {
    step = samplingRate / 16000;
    samplingRate = 16000;
    wav = downsampleByStep(wav, step);
  }

  if (!SUPPORTED_SAMPLE_RATES.includes(samplingRate)) {
    throw new Error("Silero VAD supports only 8000 and 16000 Hz (or multiples of 16000)");
  }

  if (wav.length === 0) throw new Error("Input audio is empty");

  const windowSizeSamples = samplingRate === 16000 ? 512 : 256;
  resetModel(model);

  const minSpeechSamples = (samplingRate * minSpeechDurationMs) / 1000;
  const speechPadSamples = (samplingRate * speechPadMs) / 1000;
  const maxSpeechSamples = samplingRate * maxSpeechDurationS - windowSizeSamples - 2 * speechPadSamples;
  const minSilenceSamples = (samplingRate * minSilenceDurationMs) / 1000;
  const minSilenceSamplesAtMaxSpeech = (samplingRate * minSilenceAtMaxSpeech) / 1000;
  const audioLengthSamples = wav.length;

  const speechProbs = [];
  for (let currentStartSample = 0; currentStartSample < audioLengthSamples; currentStartSample += windowSizeSamples) {
    const end = currentStartSample + windowSizeSamples;
    let chunk = wav.subarray(currentStartSample, Math.min(end, audioLengthSamples));
    if (chunk.length < windowSizeSamples) {
      const padded = new Float32Array(windowSizeSamples);
      padded.set(chunk);
      chunk = padded;
    }

    speechProbs.push(await model.predict(chunk, samplingRate));

    if (progressTrackingCallback) {
      progressTrackingCallback((Math.min(end, audioLengthSamples) / audioLengthSamples) * 100);
    }
  }

  let triggered = false;
  const speeches = [];
  let currentSpeech = null;
  let tempEnd = 0;
  let prevEnd = 0;
  let nextStart = 0;
  let possibleEnds = [];

  for (let i = 0; i < speechProbs.length; i++) {
    const speechProb = speechProbs[i];
    const curSample = windowSizeSamples * i;

    if (speechProb >= threshold && tempEnd) {
      const silenceDuration = curSample - tempEnd;
      if (silenceDuration > minSilenceSamplesAtMaxSpeech) {
        possibleEnds.push([tempEnd, silenceDuration]);
      }
      tempEnd = 0;
      if (nextStart < prevEnd) nextStart = curSample;
    }

    if (speechProb >= threshold && !triggered) {
      triggered = true;
      currentSpeech = { start: curSample };
      continue;
    }

    if (triggered && curSample - currentSpeech.start > maxSpeechSamples) {
      if (useMaxPossibleSilenceAtMaxSpeech && possibleEnds.length) {
        let best = possibleEnds[0];
        for (const candidate of possibleEnds) if (candidate[1] > best[1]) best = candidate;

        [prevEnd] = best;
        const duration = best[1];
        currentSpeech.end = prevEnd;
        speeches.push(currentSpeech);
        currentSpeech = {};
        nextStart = prevEnd + duration;

        if (nextStart < prevEnd + curSample) currentSpeech.start = nextStart;
        else triggered = false;

        prevEnd = nextStart = tempEnd = 0;
        possibleEnds = [];
      } else if (prevEnd) {
        currentSpeech.end = prevEnd;
        speeches.push(currentSpeech);
        currentSpeech = {};
        if (nextStart < prevEnd) triggered = false;
        else currentSpeech.start = nextStart;
        prevEnd = nextStart = tempEnd = 0;
        possibleEnds = [];
      } else {
        currentSpeech.end = curSample;
        speeches.push(currentSpeech);
        currentSpeech = null;
        prevEnd = nextStart = tempEnd = 0;
        triggered = false;
        possibleEnds = [];
        continue;
      }
    }

    if (speechProb < negThreshold && triggered) {
      if (!tempEnd) tempEnd = curSample;
      const silenceDurationNow = curSample - tempEnd;

      if (!useMaxPossibleSilenceAtMaxSpeech && silenceDurationNow > minSilenceSamplesAtMaxSpeech) {
        prevEnd = tempEnd;
      }

      if (silenceDurationNow < minSilenceSamples) continue;

      currentSpeech.end = tempEnd;
      if (currentSpeech.end - currentSpeech.start > minSpeechSamples) speeches.push(currentSpeech);
      currentSpeech = null;
      prevEnd = nextStart = tempEnd = 0;
      triggered = false;
      possibleEnds = [];
    }
  }

  if (currentSpeech && audioLengthSamples - currentSpeech.start > minSpeechSamples) {
    currentSpeech.end = audioLengthSamples;
    speeches.push(currentSpeech);
  }

  for (let i = 0; i < speeches.length; i++) {
    const speech = speeches[i];
    if (i === 0) speech.start = Math.trunc(Math.max(0, speech.start - speechPadSamples));

    if (i !== speeches.length - 1) {
      const nextSpeech = speeches[i + 1];
      const silenceDuration = nextSpeech.start - speech.end;
      if (silenceDuration < 2 * speechPadSamples) {
        speech.end += Math.trunc(silenceDuration / 2);
        nextSpeech.start = Math.trunc(Math.max(0, nextSpeech.start - silenceDuration / 2));
      } else {
        speech.end = Math.trunc(Math.min(audioLengthSamples, speech.end + speechPadSamples));
        nextSpeech.start = Math.trunc(Math.max(0, nextSpeech.start - speechPadSamples));
      }
    } else {
      speech.end = Math.trunc(Math.min(audioLengthSamples, speech.end + speechPadSamples));
    }
  }

  if (returnSeconds) {
    const audioLengthSeconds = audioLengthSamples / samplingRate;
    return speeches.map((speech) => ({
      start: Math.max(roundTo(speech.start / samplingRate, timeResolution), 0),
      end: Math.min(roundTo(speech.end / samplingRate, timeResolution), audioLengthSeconds),
    }));
  }

  if (step > 1) {
    return speeches.map((speech) => ({ start: speech.start * step, end: speech.end * step }));
  }

  return speeches.map(cloneSegment);
}

export const get_speech_timestamps = getSpeechTimestamps;

/** Streaming Silero VAD iterator. Feed one frame at a time. */
export class VADIterator {
  constructor(model, options = {}) {
    this.model = model;
    this.threshold = options.threshold ?? 0.5;
    this.samplingRate = options.samplingRate ?? options.sampling_rate ?? 16000;

    if (!SUPPORTED_SAMPLE_RATES.includes(this.samplingRate)) {
      throw new Error("VADIterator supports only 8000 and 16000 Hz");
    }

    const minSilenceDurationMs = options.minSilenceDurationMs ?? options.min_silence_duration_ms ?? 100;
    const speechPadMs = options.speechPadMs ?? options.speech_pad_ms ?? 30;

    this.minSilenceSamples = (this.samplingRate * minSilenceDurationMs) / 1000;
    this.speechPadSamples = (this.samplingRate * speechPadMs) / 1000;
    this.resetStates();
  }

  resetStates() {
    resetModel(this.model);
    this.triggered = false;
    this.tempEnd = 0;
    this.currentSample = 0;
  }

  reset_states() {
    this.resetStates();
  }

  async process(chunk, options = {}) {
    const returnSeconds = options.returnSeconds ?? options.return_seconds ?? false;
    const timeResolution = options.timeResolution ?? options.time_resolution ?? 1;
    const samples = toFloat32Samples(chunk);
    const windowSizeSamples = samples.length;

    this.currentSample += windowSizeSamples;
    const speechProb = await this.model.predict(samples, this.samplingRate);

    if (speechProb >= this.threshold && this.tempEnd) this.tempEnd = 0;

    if (speechProb >= this.threshold && !this.triggered) {
      this.triggered = true;
      const speechStart = Math.max(0, this.currentSample - this.speechPadSamples - windowSizeSamples);
      return {
        start: returnSeconds ? roundTo(speechStart / this.samplingRate, timeResolution) : Math.trunc(speechStart),
      };
    }

    if (speechProb < this.threshold - THRESHOLD_GAP && this.triggered) {
      if (!this.tempEnd) this.tempEnd = this.currentSample;
      if (this.currentSample - this.tempEnd < this.minSilenceSamples) return null;

      const speechEnd = this.tempEnd + this.speechPadSamples - windowSizeSamples;
      this.tempEnd = 0;
      this.triggered = false;
      return {
        end: returnSeconds ? roundTo(speechEnd / this.samplingRate, timeResolution) : Math.trunc(speechEnd),
      };
    }

    return null;
  }

  /** Alias for Python-like call sites. */
  async call(chunk, options = {}) {
    return this.process(chunk, options);
  }
}

export const VadIterator = VADIterator;

export function collectChunks(timestamps, wav, options = {}) {
  const seconds = options.seconds ?? false;
  const samplingRate = options.samplingRate ?? options.sampling_rate;
  if (seconds && !samplingRate) throw new Error("samplingRate is required when seconds is true");

  const audio = toFloat32Samples(wav);
  const chunks = [];
  let total = 0;
  for (const ts of timestamps) {
    const start = seconds ? Math.round(ts.start * samplingRate) : ts.start;
    const end = seconds ? Math.round(ts.end * samplingRate) : ts.end;
    const chunk = audio.subarray(start, end);
    chunks.push(chunk);
    total += chunk.length;
  }

  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

export const collect_chunks = collectChunks;

export function dropChunks(timestamps, wav, options = {}) {
  const seconds = options.seconds ?? false;
  const samplingRate = options.samplingRate ?? options.sampling_rate;
  if (seconds && !samplingRate) throw new Error("samplingRate is required when seconds is true");

  const audio = toFloat32Samples(wav);
  const chunks = [];
  let total = 0;
  let currentStart = 0;

  for (const ts of timestamps) {
    const start = seconds ? Math.round(ts.start * samplingRate) : ts.start;
    const end = seconds ? Math.round(ts.end * samplingRate) : ts.end;
    const chunk = audio.subarray(currentStart, start);
    chunks.push(chunk);
    total += chunk.length;
    currentStart = end;
  }

  const tail = audio.subarray(currentStart);
  chunks.push(tail);
  total += tail.length;

  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

export const drop_chunks = dropChunks;

/**
 * Small facade for consumers that want a single `vad` interface.
 */
export async function createVad(options = {}) {
  const model = options.model ?? (await loadSileroVad(options));
  return {
    model,
    async getSpeechTimestamps(audio, opts = {}) {
      return getSpeechTimestamps(audio, model, { ...options, ...opts });
    },
    async process(chunk, opts = {}) {
      if (!this.iterator) this.iterator = new VADIterator(model, { ...options, ...opts });
      return this.iterator.process(chunk, opts);
    },
    reset() {
      resetModel(model);
      if (this.iterator) this.iterator.resetStates();
    },
  };
}

export const utils = {
  getSpeechTimestamps,
  get_speech_timestamps,
  VADIterator,
  VadIterator,
  collectChunks,
  collect_chunks,
  dropChunks,
  drop_chunks,
};

export const vad = {
  create: createVad,
  load: loadSileroVad,
  loadSileroVad,
  load_silero_vad,
  getSpeechTimestamps,
  get_speech_timestamps,
  VADIterator,
  VadIterator,
  SileroVadOnnxModel,
  collectChunks,
  collect_chunks,
  dropChunks,
  drop_chunks,
  utils,
  DEFAULT_MODEL_PATH,
};

export default vad;
