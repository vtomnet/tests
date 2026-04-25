import {
  gptTranscribe, gptMiniTranscribe,
  geminiPro, geminiFlash, gemma,
  voxtralSmall, voxtralMini, voxtralMiniRealtime,
  cohereTranscribe,
} from "providers.js";

const MODELS = [
  gptTranscribe,
  gptMiniTranscribe,

  geminiPro,
  geminiFlash,
  gemma,

  voxtralSmall,
  voxtralMini,
  voxtralMiniRealtime,

  // cohereTranscribe,
];

console.log(MODELS);
