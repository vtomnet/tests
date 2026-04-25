/**
 * Silero VAD (Voice Activity Detection) interface using ONNX Runtime.
 *
 * Based on the C++ and Python implementations from silero-vad vendor.
 * Uses ONNX Runtime for Node.js to perform inference.
 */

import { InferenceSession, Tensor } from 'onnxruntime-node';
import * as fs from 'fs';
import * as path from 'path';

export interface SpeechSegment {
  start: number;  // in seconds
  end: number;    // in seconds
}

export interface VADOptions {
  /** Path to the ONNX model file */
  modelPath?: string;
  /** Sample rate (default: 16000) */
  sampleRate?: 8000 | 16000;
  /** Window frame size in ms (default: 32, which gives 512 samples at 16kHz) */
  windowFrameSize?: number;
  /** Detection threshold (default: 0.5) */
  threshold?: number;
  /** Minimum silence duration in ms to split speech chunks (default: 100) */
  minSilenceDurationMs?: number;
  /** Speech padding duration in ms (default: 30) */
  speechPadMs?: number;
  /** Minimum speech duration in ms (default: 250) */
  minSpeechDurationMs?: number;
  /** Maximum speech duration in seconds (default: Infinity) */
  maxSpeechDurationS?: number;
  /** Minimum speech samples (overrides minSpeechDurationMs if set) */
  minSpeechSamples?: number;
  /** Minimum silence samples (overrides minSilenceDurationMs if set) */
  minSilenceSamples?: number;
}

/**
 * Default ONNX model filename
 */
const DEFAULT_MODEL_FILENAME = 'silero_vad.onnx';

/**
 * Default model search paths
 */
const DEFAULT_MODEL_PATHS = [
  path.join(process.cwd(), 'models', DEFAULT_MODEL_FILENAME),
  path.join(process.cwd(), 'data', DEFAULT_MODEL_FILENAME),
  path.join(__dirname, '..', 'models', DEFAULT_MODEL_FILENAME),
  path.join(__dirname, '..', '..', 'models', DEFAULT_MODEL_FILENAME),
];

/**
 * Internal state for VAD iterator
 */
interface VADState {
  triggered: boolean;
  tempEnd: number;
  currentSample: number;
  prevEnd: number;
  nextStart: number;
  speeches: SpeechSegment[];
  currentSpeech: { start: number; end: number } | null;
  context: Float32Array;
  state: Float32Array;
}

/**
 * Silero VAD implementation using ONNX Runtime.
 */
export class VadIterator {
  private session: InferenceSession | null = null;
  private sampleRate: number;
  private threshold: number;
  private windowSizeSamples: number;
  private contextSamples: number;
  private effectiveWindowSize: number;
  private srPerMs: number;
  private speechPadSamples: number;
  private minSpeechSamples: number;
  private maxSpeechSamples: number;
  private minSilenceSamples: number;
  private minSilenceSamplesAtMaxSpeech: number;

  private state: VADState;
  private inputNodeNames: string[];
  private outputNodeNames: string[];

  /**
   * Create a new VAD iterator.
   * @param options VAD configuration options
   */
  constructor(options: VADOptions = {}) {
    const {
      sampleRate = 16000,
      windowFrameSize = 32,
      threshold = 0.5,
      minSilenceDurationMs = 100,
      speechPadMs = 30,
      minSpeechDurationMs = 250,
      maxSpeechDurationS = Infinity,
      minSpeechSamples,
      minSilenceSamples,
    } = options;

    this.sampleRate = sampleRate;
    this.threshold = threshold;
    this.windowSizeSamples = windowFrameSize * (sampleRate / 1000);
    this.contextSamples = 64;  // Always 64 samples for context (4ms at 16kHz)
    this.effectiveWindowSize = this.windowSizeSamples + this.contextSamples;
    this.srPerMs = sampleRate / 1000;
    this.speechPadSamples = this.speechPadMs * this.srPerMs;

    // Calculate min speech samples
    const calculatedMinSpeechSamples = this.srPerMs * minSpeechDurationMs;
    this.minSpeechSamples = minSpeechSamples ?? calculatedMinSpeechSamples;

    // Calculate max speech samples (subtract window and padding)
    this.maxSpeechSamples = (sampleRate * maxSpeechDurationS - this.windowSizeSamples - 2 * this.speechPadSamples);

    // Calculate min silence samples
    const calculatedMinSilenceSamples = this.srPerMs * minSilenceDurationMs;
    this.minSilenceSamples = minSilenceSamples ?? calculatedMinSilenceSamples;

    this.minSilenceSamplesAtMaxSpeech = this.srPerMs * 98;

    this.inputNodeNames = ['input', 'state', 'sr'];
    this.outputNodeNames = ['output', 'stateN'];

    // Initialize state
    this.state = this.createInitialState();
  }

  /**
   * Create initial state for the VAD
   */
  private createInitialState(): VADState {
    return {
      triggered: false,
      tempEnd: 0,
      currentSample: 0,
      prevEnd: 0,
      nextStart: 0,
      speeches: [],
      currentSpeech: null,
      context: new Float32Array(this.contextSamples),
      state: new Float32Array(2 * 1 * 128),  // [2, 1, 128] state size
    };
  }

  /**
   * Load the ONNX model.
   * @param modelPath Path to the ONNX model file
   */
  async loadModel(modelPath?: string): Promise<void> {
    let resolvedPath = modelPath;

    if (!resolvedPath) {
      // Try to find the model in default locations
      for (const tryPath of DEFAULT_MODEL_PATHS) {
        if (fs.existsSync(tryPath)) {
          resolvedPath = tryPath;
          break;
        }
      }
    }

    if (!resolvedPath) {
      throw new Error(
        `Model not found. Please specify modelPath or place '${DEFAULT_MODEL_FILENAME}' ` +
        `in one of: ${DEFAULT_MODEL_PATHS.join(', ')}`
      );
    }

    this.session = await InferenceSession.create(resolvedPath, {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'all',
    });
  }

  /**
   * Reset the VAD state
   */
  reset(): void {
    this.state = this.createInitialState();
  }

  /**
   * Process a chunk of audio and return speech probability.
   * @param dataChunk Audio data as Float32Array (windowSizeSamples samples)
   * @returns Speech probability (0-1)
   */
  private async predict(dataChunk: Float32Array): Promise<number> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    // Build input with context: [context | current_chunk]
    const newData = new Float32Array(this.effectiveWindowSize);
    newData.set(this.state.context, 0);
    newData.set(dataChunk, this.contextSamples);

    // Create input tensor [1, effectiveWindowSize]
    const inputTensor = new Tensor('float32', newData, [1, this.effectiveWindowSize]);

    // Create state tensor [2, 1, 128]
    const stateTensor = new Tensor('float32', this.state.state, [2, 1, 128]);

    // Create sample rate tensor [1]
    const srTensor = new Tensor('int64', BigInt64Array.from([BigInt(this.sampleRate)]), [1]);

    // Run inference
    const outputs = await this.session.run({
      input: inputTensor,
      state: stateTensor,
      sr: srTensor,
    });

    // Get speech probability from output
    const outputData = outputs['output'] as Tensor;
    const speechProb = (outputData.data as Float32Array)[0];

    // Get updated state
    const stateNData = outputs['stateN'] as Tensor;
    const newState = stateNData.data as Float32Array;
    this.state.state.set(newState);

    // Update context with last contextSamples from newData
    this.state.context.set(newData.subarray(newData.length - this.contextSamples));

    return speechProb;
  }

  /**
   * Process a chunk of audio and update speech detection state.
   * @param dataChunk Audio data as Float32Array
   * @returns Speech probability
   */
  private async processChunk(dataChunk: Float32Array): Promise<number> {
    const speechProb = await this.predict(dataChunk);
    this.state.currentSample += this.windowSizeSamples;

    // Speech detected above threshold
    if (speechProb >= this.threshold) {
      if (this.state.tempEnd !== 0) {
        this.state.tempEnd = 0;
        if (this.state.nextStart < this.state.prevEnd) {
          this.state.nextStart = this.state.currentSample - this.windowSizeSamples;
        }
      }
      if (!this.state.triggered) {
        this.state.triggered = true;
        this.state.currentSpeech = { start: this.state.currentSample - this.windowSizeSamples, end: 0 };
      }
      return speechProb;
    }

    // Speech segment too long
    if (this.state.triggered && this.state.currentSpeech) {
      const speechDuration = this.state.currentSample - this.state.currentSpeech.start;
      if (speechDuration > this.maxSpeechSamples) {
        if (this.state.prevEnd > 0) {
          this.state.currentSpeech.end = this.state.prevEnd;
          this.state.speeches.push(this.state.currentSpeech);
          this.state.currentSpeech = null;
          if (this.state.nextStart < this.state.prevEnd) {
            this.state.triggered = false;
          } else if (this.state.currentSpeech) {
            this.state.currentSpeech.start = this.state.nextStart;
          }
          this.state.prevEnd = 0;
          this.state.nextStart = 0;
          this.state.tempEnd = 0;
        } else {
          this.state.currentSpeech.end = this.state.currentSample;
          this.state.speeches.push(this.state.currentSpeech);
          this.state.currentSpeech = null;
          this.state.prevEnd = 0;
          this.state.nextStart = 0;
          this.state.tempEnd = 0;
          this.state.triggered = false;
        }
      }
    }

    // Near threshold but not quite speech
    if (speechProb >= (this.threshold - 0.15) && speechProb < this.threshold) {
      return speechProb;
    }

    // Below threshold - silence
    if (speechProb < (this.threshold - 0.15)) {
      if (this.state.triggered) {
        if (this.state.tempEnd === 0) {
          this.state.tempEnd = this.state.currentSample;
        }
        if (this.state.currentSample - this.state.tempEnd > this.minSilenceSamplesAtMaxSpeech) {
          this.state.prevEnd = this.state.tempEnd;
        }
        if (this.state.currentSample - this.state.tempEnd >= this.minSilenceSamples) {
          if (this.state.currentSpeech) {
            this.state.currentSpeech.end = this.state.tempEnd;
            if (this.state.currentSpeech.end - this.state.currentSpeech.start > this.minSpeechSamples) {
              this.state.speeches.push(this.state.currentSpeech);
              this.state.currentSpeech = null;
              this.state.prevEnd = 0;
              this.state.nextStart = 0;
              this.state.tempEnd = 0;
              this.state.triggered = false;
            }
          }
        }
      }
      return speechProb;
    }

    return speechProb;
  }

  /**
   * Get speech timestamps from audio buffer.
   * @param audio Float32Array of audio samples (assumes 16-bit PCM normalized to [-1, 1])
   * @returns Array of speech segments with start/end times in seconds
   */
  async getSpeechTimestamps(audio: Float32Array): Promise<SpeechSegment[]> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    // Reset state for new processing
    this.reset();

    const numSamples = audio.length;

    // Process audio in chunks
    for (let j = 0; j < numSamples; j += this.windowSizeSamples) {
      if (j + this.windowSizeSamples > numSamples) {
        break;
      }

      // Extract chunk (convert to Float32 if needed)
      const chunk = audio.slice(j, j + this.windowSizeSamples);
      await this.processChunk(chunk);
    }

    // Handle remaining speech segment
    if (this.state.currentSpeech && this.state.currentSpeech.start >= 0) {
      this.state.currentSpeech.end = numSamples;
      this.state.speeches.push(this.state.currentSpeech);
      this.state.currentSpeech = null;
    }

    // Convert sample indices to seconds
    const result: SpeechSegment[] = this.state.speeches.map((speech) => ({
      start: Math.round((speech.start / this.sampleRate) * 10) / 10,
      end: Math.round((speech.end / this.sampleRate) * 10) / 10,
    }));

    // Reset for next use
    this.reset();

    return result;
  }

  /**
   * Get speech probabilities for audio chunks.
   * Useful for real-time applications and visualization.
   * @param audio Float32Array of audio samples
   * @param onProg Optional callback for each chunk's probability
   * @returns Array of {start, end, probability} objects
   */
  async getSpeechProbs(
    audio: Float32Array,
    onProg?: (prob: number, chunkStart: number) => void
  ): Promise<Array<{ start: number; end: number; prob: number }>> {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    this.reset();

    const numSamples = audio.length;
    const results: Array<{ start: number; end: number; prob: number }> = [];

    for (let j = 0; j < numSamples; j += this.windowSizeSamples) {
      if (j + this.windowSizeSamples > numSamples) {
        break;
      }

      const chunk = audio.slice(j, j + this.windowSizeSamples);
      const prob = await this.predict(chunk);

      if (onProg) {
        onProg(prob, j);
      }

      results.push({
        start: j / this.sampleRate,
        end: (j + this.windowSizeSamples) / this.sampleRate,
        prob,
      });

      this.state.currentSample += this.windowSizeSamples;
    }

    this.reset();

    return results;
  }
}

/**
 * Simple VAD function that creates a VadIterator, loads model and returns speech timestamps.
 * This is the typical silero-vad interface pattern.
 * 
 * @param audio Float32Array of audio samples
 * @param modelPath Optional path to ONNX model
 * @param options Optional VAD configuration
 * @returns Promise resolving to array of speech segments
 */
export async function get_speech_timestamps(
  audio: Float32Array,
  modelPath?: string,
  options: VADOptions = {}
): Promise<SpeechSegment[]> {
  const vad = new VadIterator(options);
  await vad.loadModel(modelPath);
  return vad.getSpeechTimestamps(audio);
}

/**
 * Get speech probabilities for audio chunks.
 * 
 * @param audio Float32Array of audio samples
 * @param modelPath Optional path to ONNX model
 * @param options Optional VAD configuration
 * @param onProg Optional callback for each chunk's probability
 * @returns Promise resolving to array of probability results
 */
export async function get_speech_probs(
  audio: Float32Array,
  modelPath?: string,
  options: VADOptions = {},
  onProg?: (prob: number, chunkStart: number) => void
): Promise<Array<{ start: number; end: number; prob: number }>> {
  const vad = new VadIterator(options);
  await vad.loadModel(modelPath);
  return vad.getSpeechProbs(audio, onProg);
}

/**
 * Load the silero VAD model and utilities.
 * This mimics the torch.hub.load pattern from Python.
 * 
 * @param modelPath Optional path to ONNX model
 * @returns Object with utils including get_speech_timestamps, get_speech_probs
 */
export async function load_silero_vad(modelPath?: string) {
  const vad = new VadIterator();
  if (modelPath) {
    await vad.loadModel(modelPath);
  }

  return {
    /** Get speech timestamps from audio */
    get_speech_timestamps: (audio: Float32Array) => vad.getSpeechTimestamps(audio),
    /** Get speech probabilities from audio */
    get_speech_probs: (audio: Float32Array, onProg?: (prob: number, chunkStart: number) => void) =>
      vad.getSpeechProbs(audio, onProg),
    /** Load model from path */
    load_model: (path: string) => vad.loadModel(path),
    /** Reset VAD state */
    reset: () => vad.reset(),
    /** The VadIterator instance */
    vad,
  };
}

/**
 * Convert 16-bit PCM audio to Float32 normalized to [-1, 1]
 * @param audioData Int16Array or Buffer of 16-bit PCM audio
 * @returns Float32Array normalized to [-1, 1]
 */
export function int16ToFloat32(audioData: Int16Array | Buffer): Float32Array {
  const float32 = new Float32Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    float32[i] = audioData[i] / 32768.0;
  }
  return float32;
}

/**
 * Convert Float32 audio to 16-bit PCM
 * @param audioData Float32Array normalized to [-1, 1]
 * @returns Int16Array
 */
export function float32ToInt16(audioData: Float32Array): Int16Array {
  const int16 = new Int16Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    const val = Math.max(-1, Math.min(1, audioData[i])) * 32767;
    int16[i] = Math.round(val);
  }
  return int16;
}

export default {
  VadIterator,
  get_speech_timestamps,
  get_speech_probs,
  load_silero_vad,
  int16ToFloat32,
  float32ToInt16,
};