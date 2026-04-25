import { InferenceSession, Tensor } from "onnxruntime-node";

const VALID_SAMPLE_RATES = [8000, 16000];

/**
 * Silero VAD iterator. Loads an ONNX model and detects speech segments
 * in mono audio using the same algorithm as the official C++/C# examples.
 */
export class VadIterator {
  /**
   * @param {InferenceSession} session - Loaded ONNX Runtime session.
   * @param {object} [options]
   * @param {number} [options.sampleRate=16000] - 8000 or 16000.
   * @param {number} [options.threshold=0.5] - Speech probability threshold.
   * @param {number} [options.minSilenceDurationMs=100] - Min silence to split speech.
   * @param {number} [options.speechPadMs=30] - Padding applied to segments.
   * @param {number} [options.minSpeechDurationMs=250] - Min speech segment length.
   * @param {number} [options.maxSpeechDurationS=Infinity] - Max speech segment length in seconds.
   * @param {number} [options.frameSizeMs=32] - Frame/window size in milliseconds.
   */
  constructor(session, options = {}) {
    this.session = session;
    this.sampleRate = options.sampleRate ?? 16000;
    this.threshold = options.threshold ?? 0.5;
    this.negThreshold = this.threshold - 0.15;

    if (!VALID_SAMPLE_RATES.includes(this.sampleRate)) {
      throw new Error(
        `Unsupported sample rate ${this.sampleRate}. Only 8000 and 16000 are supported.`
      );
    }

    const srPerMs = this.sampleRate / 1000;
    const frameSizeMs = options.frameSizeMs ?? 32;
    this.windowSizeSamples = frameSizeMs * srPerMs;
    this.contextSize = this.sampleRate === 16000 ? 64 : 32;
    this.effectiveWindowSize = this.windowSizeSamples + this.contextSize;

    const minSpeechDurationMs = options.minSpeechDurationMs ?? 250;
    const maxSpeechDurationS = options.maxSpeechDurationS ?? Infinity;
    const minSilenceDurationMs = options.minSilenceDurationMs ?? 100;
    const speechPadMs = options.speechPadMs ?? 30;

    this.minSpeechSamples = srPerMs * minSpeechDurationMs;
    this.speechPadSamples = srPerMs * speechPadMs;
    this.maxSpeechSamples =
      this.sampleRate * maxSpeechDurationS -
      this.windowSizeSamples -
      2 * this.speechPadSamples;
    this.minSilenceSamples = srPerMs * minSilenceDurationMs;
    this.minSilenceSamplesAtMaxSpeech = srPerMs * 98;

    this._resetStates();
  }

  /**
   * Async factory. Creates the ONNX session and returns a VadIterator.
   * @param {object} [options]
   * @param {string} [options.modelPath="./vendor/silero-vad/src/silero_vad/data/silero_vad.onnx"]
   * @param {number} [options.sampleRate=16000]
   * @param {number} [options.threshold=0.5]
   * @param {number} [options.minSilenceDurationMs=100]
   * @param {number} [options.speechPadMs=30]
   * @param {number} [options.minSpeechDurationMs=250]
   * @param {number} [options.maxSpeechDurationS=Infinity]
   * @param {number} [options.frameSizeMs=32]
   * @returns {Promise<VadIterator>}
   */
  static async create(options = {}) {
    const modelPath =
      options.modelPath ??
      "./vendor/silero-vad/src/silero_vad/data/silero_vad.onnx";
    const session = await InferenceSession.create(modelPath, {
      interOpNumThreads: 1,
      intraOpNumThreads: 1,
      graphOptimizationLevel: "all",
    });
    return new VadIterator(session, options);
  }

  _resetStates() {
    this.state = new Float32Array(2 * 1 * 128);
    this.context = new Float32Array(this.contextSize);
    this.triggered = false;
    this.tempEnd = 0;
    this.prevEnd = 0;
    this.nextStart = 0;
    this.currentSample = 0;
    this.currentSpeech = { start: -1, end: -1 };
    this.speeches = [];
    this.audioLengthSamples = 0;
  }

  /**
   * Reset the internal VAD state (context, ONNX state, segment buffers).
   */
  reset() {
    this._resetStates();
  }

  /**
   * Run inference on a single chunk of exactly `windowSizeSamples` frames.
   * Updates internal state and returns the speech probability.
   * @param {Float32Array} chunk
   * @returns {Promise<number>} speech probability [0..1]
   */
  async call(chunk) {
    if (chunk.length !== this.windowSizeSamples) {
      throw new Error(
        `Chunk size ${chunk.length} does not match expected window size ${this.windowSizeSamples}`
      );
    }

    const inputData = new Float32Array(this.effectiveWindowSize);
    inputData.set(this.context, 0);
    inputData.set(chunk, this.contextSize);

    const inputTensor = new Tensor("float32", inputData, [
      1,
      this.effectiveWindowSize,
    ]);
    const stateTensor = new Tensor("float32", this.state, [2, 1, 128]);
    const srTensor = new Tensor(
      "int64",
      new BigInt64Array([BigInt(this.sampleRate)]),
      [1]
    );

    const results = await this.session.run({
      input: inputTensor,
      state: stateTensor,
      sr: srTensor,
    });

    const speechProb = results.output.data[0];
    this.state.set(results.stateN.data);

    // Update context with the last contextSize samples
    this.context.set(
      inputData.subarray(this.effectiveWindowSize - this.contextSize)
    );

    this.currentSample += this.windowSizeSamples;

    // ---- VAD state machine (mirrors official C++ reference) ----
    if (speechProb >= this.threshold) {
      if (this.tempEnd !== 0) {
        this.tempEnd = 0;
        if (this.nextStart < this.prevEnd) {
          this.nextStart = this.currentSample - this.windowSizeSamples;
        }
      }
      if (!this.triggered) {
        this.triggered = true;
        this.currentSpeech.start = this.currentSample - this.windowSizeSamples;
      }
      return speechProb;
    }

    if (
      this.triggered &&
      this.currentSample - this.currentSpeech.start > this.maxSpeechSamples
    ) {
      if (this.prevEnd > 0) {
        this.currentSpeech.end = this.prevEnd;
        this.speeches.push({ ...this.currentSpeech });
        this.currentSpeech = { start: -1, end: -1 };
        if (this.nextStart < this.prevEnd) {
          this.triggered = false;
        } else {
          this.currentSpeech.start = this.nextStart;
        }
        this.prevEnd = 0;
        this.nextStart = 0;
        this.tempEnd = 0;
      } else {
        this.currentSpeech.end = this.currentSample;
        this.speeches.push({ ...this.currentSpeech });
        this.currentSpeech = { start: -1, end: -1 };
        this.prevEnd = 0;
        this.nextStart = 0;
        this.tempEnd = 0;
        this.triggered = false;
      }
      return speechProb;
    }

    if (speechProb >= this.negThreshold && speechProb < this.threshold) {
      return speechProb;
    }

    if (speechProb < this.negThreshold) {
      if (this.triggered) {
        if (this.tempEnd === 0) {
          this.tempEnd = this.currentSample;
        }
        if (
          this.currentSample - this.tempEnd >
          this.minSilenceSamplesAtMaxSpeech
        ) {
          this.prevEnd = this.tempEnd;
        }
        if (this.currentSample - this.tempEnd >= this.minSilenceSamples) {
          this.currentSpeech.end = this.tempEnd;
          if (
            this.currentSpeech.end - this.currentSpeech.start >
            this.minSpeechSamples
          ) {
            this.speeches.push({ ...this.currentSpeech });
            this.currentSpeech = { start: -1, end: -1 };
            this.prevEnd = 0;
            this.nextStart = 0;
            this.tempEnd = 0;
            this.triggered = false;
          }
        }
      }
      return speechProb;
    }

    return speechProb;
  }

  /**
   * Process an entire audio buffer and return detected speech segments.
   * @param {Float32Array|number[]} audio - Mono audio samples, ideally normalized to [-1, 1].
   * @returns {Promise<Array<{start:number,end:number,startSec:number,endSec:number}>>}
   */
  async process(audio) {
    this.reset();
    const samples =
      audio instanceof Float32Array ? audio : new Float32Array(audio);
    this.audioLengthSamples = samples.length;

    for (
      let i = 0;
      i <= samples.length - this.windowSizeSamples;
      i += this.windowSizeSamples
    ) {
      const chunk = samples.subarray(i, i + this.windowSizeSamples);
      await this.call(chunk);
    }

    if (this.currentSpeech.start >= 0) {
      this.currentSpeech.end = this.audioLengthSamples;
      this.speeches.push({ ...this.currentSpeech });
      this.currentSpeech = { start: -1, end: -1 };
      this.prevEnd = 0;
      this.nextStart = 0;
      this.tempEnd = 0;
      this.triggered = false;
    }

    return this._applyPadding(this.speeches);
  }

  _applyPadding(segments) {
    if (segments.length === 0) return [];

    const result = segments.map((s) => ({
      start: s.start,
      end: s.end,
      startSec: s.start / this.sampleRate,
      endSec: s.end / this.sampleRate,
    }));

    for (let i = 0; i < result.length; i++) {
      const item = result[i];
      if (i === 0) {
        item.start = Math.max(0, item.start - this.speechPadSamples);
        item.startSec = item.start / this.sampleRate;
      }

      if (i !== result.length - 1) {
        const nextItem = result[i + 1];
        const silenceDuration = nextItem.start - item.end;
        if (silenceDuration < 2 * this.speechPadSamples) {
          const pad = Math.floor(silenceDuration / 2);
          item.end += pad;
          nextItem.start = Math.max(0, nextItem.start - pad);
        } else {
          item.end = Math.min(
            this.audioLengthSamples,
            item.end + this.speechPadSamples
          );
          nextItem.start = Math.max(
            0,
            nextItem.start - this.speechPadSamples
          );
        }
        item.endSec = item.end / this.sampleRate;
        nextItem.startSec = nextItem.start / this.sampleRate;
      } else {
        item.end = Math.min(
          this.audioLengthSamples,
          item.end + this.speechPadSamples
        );
        item.endSec = item.end / this.sampleRate;
      }
    }

    // Merge overlapping segments
    const merged = [];
    let left = result[0].start;
    let right = result[0].end;
    for (let i = 1; i < result.length; i++) {
      if (result[i].start > right) {
        merged.push({
          start: left,
          end: right,
          startSec: left / this.sampleRate,
          endSec: right / this.sampleRate,
        });
        left = result[i].start;
        right = result[i].end;
      } else {
        right = Math.max(right, result[i].end);
      }
    }
    merged.push({
      start: left,
      end: right,
      startSec: left / this.sampleRate,
      endSec: right / this.sampleRate,
    });

    return merged;
  }
}

/**
 * Convenience function mirroring the Python `get_speech_timestamps` API.
 * Creates a temporary VadIterator, runs detection, and returns segments.
 *
 * @param {Float32Array|number[]} audio - Mono audio samples.
 * @param {object} [options] - Same options as VadIterator.create().
 * @returns {Promise<Array<{start:number,end:number,startSec:number,endSec:number}>>}
 */
export async function getSpeechTimestamps(audio, options = {}) {
  const vad = await VadIterator.create(options);
  return vad.process(audio);
}
