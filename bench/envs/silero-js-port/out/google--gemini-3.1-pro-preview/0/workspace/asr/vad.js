const ort = require('onnxruntime-node');

class SileroVad {
    constructor(modelPath, options = {}) {
        this.modelPath = modelPath;
        this.sampleRate = options.sampleRate || 16000;
        this.windowSizeMs = options.windowSizeMs || 32;
        this.threshold = options.threshold || 0.5;
        this.minSilenceDurationMs = options.minSilenceDurationMs || 100;
        this.speechPadMs = options.speechPadMs || 30;
        this.minSpeechDurationMs = options.minSpeechDurationMs || 250;
        this.maxSpeechDurationS = options.maxSpeechDurationS || Infinity;

        this.srPerMs = this.sampleRate / 1000;
        this.windowSizeSamples = this.windowSizeMs * this.srPerMs;
        this.contextSamples = this.sampleRate === 8000 ? 32 : 64;
        
        this.effectiveWindowSize = this.windowSizeSamples + this.contextSamples;

        this.minSpeechSamples = this.srPerMs * this.minSpeechDurationMs;
        this.maxSpeechSamples = (this.sampleRate * this.maxSpeechDurationS - this.windowSizeSamples - 2 * this.speechPadMs * this.srPerMs);
        this.minSilenceSamples = this.srPerMs * this.minSilenceDurationMs;
        this.minSilenceSamplesAtMaxSpeech = this.srPerMs * 98;
        this.speechPadSamples = this.speechPadMs * this.srPerMs;

        this.session = null;
        this._state = new Float32Array(2 * 1 * 128);
        this._context = new Float32Array(this.contextSamples);
        this.srTensor = new ort.Tensor('int64', new BigInt64Array([BigInt(this.sampleRate)]), [1]);
        
        this.resetStates();
    }

    async init() {
        this.session = await ort.InferenceSession.create(this.modelPath);
    }

    resetStates() {
        this._state.fill(0);
        this._context.fill(0);
        this.triggered = false;
        this.tempEnd = 0;
        this.currentSample = 0;
        this.prevEnd = 0;
        this.nextStart = 0;
        this.speeches = [];
        this.currentSpeech = { start: -1, end: -1 };
    }

    async predict(dataChunk) {
        if (!this.session) {
            throw new Error('Model not initialized. Call init() first.');
        }

        const newData = new Float32Array(this.effectiveWindowSize);
        newData.set(this._context, 0);
        newData.set(dataChunk, this.contextSamples);

        const inputTensor = new ort.Tensor('float32', newData, [1, this.effectiveWindowSize]);
        const stateTensor = new ort.Tensor('float32', this._state, [2, 1, 128]);

        const feeds = {
            input: inputTensor,
            state: stateTensor,
            sr: this.srTensor
        };

        const results = await this.session.run(feeds);
        
        const speechProb = results.output.data[0];
        this._state.set(results.stateN.data);
        this.currentSample += this.windowSizeSamples;

        this._context.set(newData.subarray(newData.length - this.contextSamples));

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

        if (this.triggered && ((this.currentSample - this.currentSpeech.start) > this.maxSpeechSamples)) {
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

        if ((speechProb >= (this.threshold - 0.15)) && (speechProb < this.threshold)) {
            return speechProb;
        }

        if (speechProb < (this.threshold - 0.15)) {
            if (this.triggered) {
                if (this.tempEnd === 0) {
                    this.tempEnd = this.currentSample;
                }
                if (this.currentSample - this.tempEnd > this.minSilenceSamplesAtMaxSpeech) {
                    this.prevEnd = this.tempEnd;
                }
                if ((this.currentSample - this.tempEnd) >= this.minSilenceSamples) {
                    this.currentSpeech.end = this.tempEnd;
                    if (this.currentSpeech.end - this.currentSpeech.start > this.minSpeechSamples) {
                        this.speeches.push({ ...this.currentSpeech });
                    }
                    this.currentSpeech = { start: -1, end: -1 };
                    this.prevEnd = 0;
                    this.nextStart = 0;
                    this.tempEnd = 0;
                    this.triggered = false;
                }
            }
            return speechProb;
        }
        
        return speechProb;
    }

    async process(inputWav) {
        this.resetStates();
        const audioLengthSamples = inputWav.length;

        for (let j = 0; j < audioLengthSamples; j += this.windowSizeSamples) {
            if (j + this.windowSizeSamples > audioLengthSamples) break;
            const chunk = inputWav.subarray(j, j + this.windowSizeSamples);
            await this.predict(chunk);
        }

        if (this.currentSpeech.start >= 0) {
            this.currentSpeech.end = audioLengthSamples;
            this.speeches.push({ ...this.currentSpeech });
            this.currentSpeech = { start: -1, end: -1 };
            this.prevEnd = 0;
            this.nextStart = 0;
            this.tempEnd = 0;
            this.triggered = false;
        }
        
        return this.speeches.map(s => ({
            start: s.start / this.sampleRate,
            end: s.end / this.sampleRate
        }));
    }
}

module.exports = { SileroVad };
