import path from "node:path";
import { fileURLToPath } from "node:url";

export const THINKING_LEVELS = new Set([
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
]);

export function projectRoot() {
  return path.dirname(fileURLToPath(import.meta.url));
}

export function sanitizeModelPath(model) {
  return model.replaceAll("/", "--").replaceAll(":", "--");
}

export function parseModelArg(modelArg) {
  let head = modelArg;
  let thinkingLevel;
  const colon = modelArg.lastIndexOf(":");
  if (colon !== -1) {
    const suffix = modelArg.slice(colon + 1);
    if (!THINKING_LEVELS.has(suffix)) {
      throw new Error(
        `invalid thinking level in model '${modelArg}': ${suffix}`,
      );
    }
    head = modelArg.slice(0, colon);
    thinkingLevel = suffix;
  }

  const slash = head.indexOf("/");
  if (slash === -1) {
    throw new Error(`model must be provider/model[:thinking]: ${modelArg}`);
  }
  const provider = head.slice(0, slash);
  const modelId = head.slice(slash + 1);
  if (!provider || !modelId) {
    throw new Error(`model must be provider/model[:thinking]: ${modelArg}`);
  }
  return { provider, modelId, thinkingLevel };
}

export function resolveModel(modelArg, modelRegistry) {
  const { provider, modelId, thinkingLevel } = parseModelArg(modelArg);
  const model = modelRegistry.find(provider, modelId);
  if (!model) throw new Error(`model not found: ${provider}/${modelId}`);
  return { model, thinkingLevel };
}
