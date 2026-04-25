import { InferenceSession, Tensor } from "onnxruntime-common";

// Silero VAD model configuration
const CONTEXT_SAMPLES = {
  16000: 64,
  8000: 32,
};

const STATE_SHAPE = [2, 1, 128];
const STATE_SIZE = 2 * 1 * 128; // 256

/**
 * Creates a Silero VAD instance.
 *
 * @param {Object} options
 * @param {string|ArrayBuffer|Uint8Array} options.model - Path to ONNX model, or model bytes
 * @param {number} [options.sampleRate=16000] - Sample rate: 16000 or 8000
 * @param {number} [options.threshold=0.5] - Speech probability threshold
 * @param {number} [options.windowSizeMs=32] - Window size in milliseconds
 * @param {number} [options.minSilenceDurationMs=100] - Minimum silence duration before ending speech
 * @param {number} [options.speechPadMs=30] - Padding around speech segments
 * @param {number} [options.minSpeechDurationMs=250] - Minimum speech duration to keep
 * @param {number} [options.maxSpeechDurationS=Infinity] - Maximum speech duration before forced split
 * @param {import("onnxruntime-common").InferenceSession.SessionOptions} [options.sessionOptions] - ORT session options
 * @returns {Promise<VadIterator>}
 */
export async function createVad({
  model,
  sampleRate = 16000,
  threshold = 0.5,
  windowSizeMs = 32,
  minSilenceDurationMs = 100,
  speechPadMs = 30,
  minSpeechDurationMs = 250,
  maxSpeechDurationS = Infinity,
  sessionOptions,
} = {}) {
  if (sampleRate !== 16000 && sampleRate !== 8000) {
    throw new Error("sampleRate must be 16000 or 8000");
  }

  const session = await InferenceSession.create(model, {
    interOpNumThreads: 1,
    intraOpNumThreads: 1,
    graphOptimizationLevel: "all",
    ...sessionOptions,
  });

  const srPerMs = sampleRate / 1000;
  const windowSizeSamples = windowSizeMs * srPerMs;
  const contextSize = CONTEXT_SAMPLES[sampleRate];
  const effectiveWindowSize = windowSizeSamples + contextSize;

  const minSilenceSamples = srPerMs * minSilenceDurationMs;
  const minSpeechSamples = srPerMs * minSpeechDurationMs;
  const speechPadSamples = srPerMs * speechPadMs;
  const maxSpeechSamples =
    sampleRate * maxSpeechDurationS - windowSizeSamples - 2 * speechPadSamples;
  const minSilenceSamplesAtMaxSpeech = srPerMs * 98;

  return new VadIterator({
    session,
    sampleRate,
    threshold,
    windowSizeSamples,
    contextSize,
    effectiveWindowSize,
    minSilenceSamples,
    minSpeechSamples,
    speechPadSamples,
    maxSpeechSamples,
    minSilenceSamplesAtMaxSpeech,
  });
}

class VadIterator {
  constructor(params) {
    this.params = params;
    this.reset();
  }

  reset() {
    this._state = new Float32Array(STATE_SIZE);
    this._context = new Float32Array(this.params.contextSize);
    this._triggered = false;
    this._tempEnd = 0;
    this._currentSample = 0;
    this._prevEnd = 0;
    this._nextStart = 0;
    this._currentSpeech = null;
    this._speeches = [];
  }

  /**
   * Run a single inference step on one audio chunk.
   * @param {Float32Array} chunk - Audio samples (length = windowSizeSamples)
   * @returns {Promise<number>} Speech probability
   */
  async _predict(chunk) {
    const { session, effectiveWindowSize, contextSize } = this.params;

    // Build input: [context | chunk]
    const input = new Float32Array(effectiveWindowSize);
    input.set(this._context, 0);
    input.set(chunk, contextSize);

    // Create tensors
    const inputTensor = new Tensor("float32", input, [1, effectiveWindowSize]);
    const stateTensor = new Tensor("float32", this._state, STATE_SHAPE);
    const srTensor = new Tensor(
      "int64",
      new BigInt64Array([BigInt(this.params.sampleRate)]),
      [1]
    );

    const feeds = {
      input: inputTensor,
      state: stateTensor,
      sr: srTensor,
    };

    const results = await session.run(feeds);

    const speechProb = results.output.data[0];
    this._state = new Float32Array(results.stateN.data);

    // Update context with last contextSize samples from input
    this._context = input.slice(input.length - contextSize);

    return speechProb;
  }

  /**
   * Process audio samples and return speech timestamps.
   * @param {Float32Array} samples - Audio samples (16-bit float normalized to [-1, 1])
   * @returns {Promise<Array<{start: number, end: number}>>} Speech segments in samples
   */
  async process(samples) {
    this.reset();

    const { windowSizeSamples, threshold } = this.params;
    const numSamples = samples.length;
    const speeches = [];

    let triggered = false;
    let tempEnd = 0;
    let currentSample = 0;
    let prevEnd = 0;
    let nextStart = 0;
    let currentSpeech = null;

    for (let offset = 0; offset + windowSizeSamples <= numSamples; offset += windowSizeSamples) {
      const chunk = samples.slice(offset, offset + windowSizeSamples);
      const speechProb = await this._predict(chunk);
      currentSample += windowSizeSamples;

      if (speechProb >= threshold) {
        if (tempEnd !== 0) {
          tempEnd = 0;
          if (nextStart < prevEnd) {
            nextStart = currentSample - windowSizeSamples;
          }
        }
        if (!triggered) {
          triggered = true;
          currentSpeech = { start: currentSample - windowSizeSamples, end: 0 };
        }
        continue;
      }

      // Max speech duration exceeded
      if (
        triggered &&
        currentSample - currentSpeech.start > this.params.maxSpeechSamples
      ) {
        if (prevEnd > 0) {
          currentSpeech.end = prevEnd;
          speeches.push(currentSpeech);
          currentSpeech = null;
          if (nextStart < prevEnd) {
            triggered = false;
          } else {
            currentSpeech = { start: nextStart, end: 0 };
          }
          prevEnd = 0;
          nextStart = 0;
          tempEnd = 0;
        } else {
          currentSpeech.end = currentSample;
          speeches.push(currentSpeech);
          currentSpeech = null;
          prevEnd = 0;
          nextStart = 0;
          tempEnd = 0;
          triggered = false;
        }
        continue;
      }

      // Hysteresis zone — keep going without triggering end
      if (
        speechProb >= threshold - 0.15 &&
        speechProb < threshold
      ) {
        continue;
      }

      // Below silence threshold
      if (speechProb < threshold - 0.15) {
        if (triggered) {
          if (tempEnd === 0) {
            tempEnd = currentSample;
          }
          if (currentSample - tempEnd > this.params.minSilenceSamplesAtMaxSpeech) {
            prevEnd = tempEnd;
          }
          if (currentSample - tempEnd >= this.params.minSilenceSamples) {
            currentSpeech.end = tempEnd;
            if (
              currentSpeech.end - currentSpeech.start >
              this.params.minSpeechSamples
            ) {
              speeches.push(currentSpeech);
              currentSpeech = null;
              prevEnd = 0;
              nextStart = 0;
              tempEnd = 0;
              triggered = false;
            }
          }
        }
        continue;
      }
    }

    // Finalize any open speech segment
    if (triggered && currentSpeech) {
      currentSpeech.end = numSamples;
      speeches.push(currentSpeech);
    }

    // Apply padding and filter short segments
    const { speechPadSamples, minSpeechSamples } = this.params;
    const result = [];
    for (const seg of speeches) {
      const start = Math.max(0, seg.start - speechPadSamples);
      const end = Math.min(numSamples, seg.end + speechPadSamples);
      if (end - start >= minSpeechSamples) {
        result.push({ start, end });
      }
    }

    return result;
  }

  /**
   * Process a single chunk for streaming usage.
   * Returns the speech probability for this chunk.
   * Use getSpeechTimestamps() after streaming to retrieve segments.
   * @param {Float32Array} chunk - Audio chunk (length = windowSizeSamples)
   * @returns {Promise<number>} Speech probability
   */
  async processChunk(chunk) {
    const { threshold, windowSizeSamples } = this.params;

    if (chunk.length !== windowSizeSamples) {
      throw new Error(
        `Chunk length must be ${windowSizeSamples}, got ${chunk.length}`
      );
    }

    const speechProb = await this._predict(chunk);
    this._currentSample += windowSizeSamples;

    if (speechProb >= threshold) {
      if (this._tempEnd !== 0) {
        this._tempEnd = 0;
        if (this._nextStart < this._prevEnd) {
          this._nextStart = this._currentSample - windowSizeSamples;
        }
      }
      if (!this._triggered) {
        this._triggered = true;
        this._currentSpeech = {
          start: this._currentSample - windowSizeSamples,
          end: 0,
        };
      }
      return speechProb;
    }

    if (
      this._triggered &&
      this._currentSample - this._currentSpeech.start >
        this.params.maxSpeechSamples
    ) {
      if (this._prevEnd > 0) {
        this._currentSpeech.end = this._prevEnd;
        this._speeches.push(this._currentSpeech);
        this._currentSpeech = null;
        if (this._nextStart < this._prevEnd) {
          this._triggered = false;
        } else {
          this._currentSpeech = { start: this._nextStart, end: 0 };
        }
        this._prevEnd = 0;
        this._nextStart = 0;
        this._tempEnd = 0;
      } else {
        this._currentSpeech.end = this._currentSample;
        this._speeches.push(this._currentSpeech);
        this._currentSpeech = null;
        this._prevEnd = 0;
        this._nextStart = 0;
        this._tempEnd = 0;
        this._triggered = false;
      }
      return speechProb;
    }

    if (speechProb >= threshold - 0.15 && speechProb < threshold) {
      return speechProb;
    }

    if (speechProb < threshold - 0.15) {
      if (this._triggered) {
        if (this._tempEnd === 0) {
          this._tempEnd = this._currentSample;
        }
        if (
          this._currentSample - this._tempEnd >
          this.params.minSilenceSamplesAtMaxSpeech
        ) {
          this._prevEnd = this._tempEnd;
        }
        if (
          this._currentSample - this._tempEnd >=
          this.params.minSilenceSamples
        ) {
          this._currentSpeech.end = this._tempEnd;
          if (
            this._currentSpeech.end - this._currentSpeech.start >
            this.params.minSpeechSamples
          ) {
            this._speeches.push(this._currentSpeech);
            this._currentSpeech = null;
            this._prevEnd = 0;
            this._nextStart = 0;
            this._tempEnd = 0;
            this._triggered = false;
          }
        }
      }
    }

    return speechProb;
  }

  /**
   * Get detected speech timestamps after streaming.
   * Call after processChunk() for each chunk, then call this to finalize.
   * @returns {Array<{start: number, end: number}>} Speech segments in samples
   */
  getSpeechTimestamps() {
    const speeches = [...this._speeches];

    // Finalize any open segment
    if (this._triggered && this._currentSpeech) {
      speeches.push({
        ...this._currentSpeech,
        end: this._currentSample,
      });
    }

    // Apply padding and filter
    const { speechPadSamples, minSpeechSamples } = this.params;
    const result = [];
    for (const seg of speeches) {
      const start = Math.max(0, seg.start - speechPadSamples);
      const end = Math.min(this._currentSample, seg.end + speechPadSamples);
      if (end - start >= minSpeechSamples) {
        result.push({ start, end });
      }
    }

    return result;
  }

  /**
   * Convert sample-based timestamps to seconds.
   * @param {Array<{start: number, end: number}>} timestamps
   * @returns {Array<{start: number, end: number}>} Timestamps in seconds
   */
  timestampsToSeconds(timestamps) {
    const sr = this.params.sampleRate;
    return timestamps.map((t) => ({
      start: t.start / sr,
      end: t.end / sr,
    }));
  }
}
