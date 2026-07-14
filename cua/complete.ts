import {
  type Api,
  type AssistantMessage,
  type Context,
  getModel,
  getModels,
  getProviders,
  type ImageContent,
  type Model,
  stream as piStream,
  type TextContent,
  type Tool,
  type ToolCall,
  type Usage,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import type { CompleteCallRecord, RunStats } from "./types.ts";

export { Type };
export type { Tool };

export interface CompleteArgs {
  label: string;
  prompt: string;
  systemPrompt?: string;
  imagePath?: string;
  tools: Tool[];
  requiredTool?: string;
  model?: string;
  apiKey?: string;
  maxTokens?: number;
  temperature?: number;
  meta?: Record<string, unknown>;
}

export interface CompleteResult {
  message: AssistantMessage;
  toolCalls: ToolCall[];
}

const DEFAULT_MODEL = "openai/gpt-5.5";
let modelSpec = Deno.env.get("CUA_MODEL") || Deno.env.get("MODEL") ||
  DEFAULT_MODEL;
let records: CompleteCallRecord[] | null = null;

export function setDefaultModel(nextModelSpec: string): void {
  const clean = nextModelSpec.trim();
  if (!clean) throw new Error("model must not be empty");
  modelSpec = clean;
}

export function getDefaultModel(): string {
  return modelSpec;
}

export function beginCompleteRecording(): void {
  if (records) {
    throw new Error("a benchmark run is already recording completions");
  }
  records = [];
}

export function takeCompleteRecording(): CompleteCallRecord[] {
  if (!records) throw new Error("no benchmark run is recording completions");
  const out = records;
  records = null;
  return out;
}

function usageSummary(usage: Usage | undefined): CompleteCallRecord["usage"] {
  if (!usage) return null;
  return {
    input: usage.input,
    output: usage.output,
    cacheRead: usage.cacheRead,
    cacheWrite: usage.cacheWrite,
    totalTokens: usage.totalTokens,
  };
}

function toolCallsFrom(message: AssistantMessage): ToolCall[] {
  return message.content.filter((item): item is ToolCall =>
    item.type === "toolCall"
  );
}

function recordCall(record: CompleteCallRecord): void {
  records?.push(record);
}

function average(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function summarizeCompleteCalls(
  calls: CompleteCallRecord[],
  durationMs: number,
): RunStats {
  const errors =
    calls.filter((call) =>
      call.error || call.stopReason === "error" || call.stopReason === "aborted"
    ).length;
  const ttfts = calls.map((call) => call.ttftMs).filter((
    value,
  ): value is number => value !== null);
  const tps = calls.map((call) => call.tps).filter((value): value is number =>
    value !== null
  );
  return {
    calls: calls.length,
    errors,
    errorRate: calls.length ? errors / calls.length : 0,
    durationMs,
    cost: calls.reduce((sum, call) => sum + call.cost, 0),
    avgTtftMs: average(ttfts),
    avgTps: average(tps),
  };
}

function providerAndId(spec: string): { provider: string; id: string } | null {
  const colon = spec.indexOf(":");
  if (colon > 0) {
    return { provider: spec.slice(0, colon), id: spec.slice(colon + 1) };
  }

  const slash = spec.indexOf("/");
  if (slash <= 0) return null;

  const provider = spec.slice(0, slash);
  const providers = getProviders() as string[];
  if (!providers.includes(provider)) return null;
  return { provider, id: spec.slice(slash + 1) };
}

function builtinModel(spec: string): Model<Api> {
  const explicit = providerAndId(spec);
  if (explicit) {
    const model = getModel(explicit.provider as never, explicit.id as never) as
      | Model<Api>
      | undefined;
    if (!model) {
      throw new Error(`unknown model ${explicit.provider}/${explicit.id}`);
    }
    return model;
  }

  const matches: Model<Api>[] = [];
  for (const provider of getProviders() as string[]) {
    for (const model of getModels(provider as never) as Model<Api>[]) {
      if (model.id === spec) matches.push(model);
    }
  }

  if (matches.length === 1) return matches[0];
  if (matches.length === 0) throw new Error(`unknown model ${spec}`);
  throw new Error(
    `ambiguous model ${spec}; use provider/model or provider:model`,
  );
}

function customModel(spec: string): Model<Api> {
  const baseUrl = Deno.env.get("CUA_MODEL_BASE_URL");
  if (!baseUrl) throw new Error("CUA_MODEL_BASE_URL is not set");
  const id = Deno.env.get("CUA_MODEL_ID") || spec;
  const provider = Deno.env.get("CUA_MODEL_PROVIDER") || "custom";
  const api = (Deno.env.get("CUA_MODEL_API") || "openai-completions") as Api;
  return {
    id,
    name: id,
    api,
    provider,
    baseUrl,
    reasoning: false,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128_000,
    maxTokens: 4096,
  };
}

function resolveModel(spec: string): Model<Api> {
  if (Deno.env.get("CUA_MODEL_BASE_URL")) return customModel(spec);
  return builtinModel(spec);
}

function customApiKey(
  model: Model<Api>,
  explicit: string | undefined,
): string | undefined {
  if (explicit?.trim()) return explicit;
  if (!Deno.env.get("CUA_MODEL_BASE_URL")) return undefined;
  const direct = Deno.env.get("CUA_API_KEY");
  if (direct) return direct;
  const envName = `${
    model.provider.toUpperCase().replaceAll(/[^A-Z0-9]/g, "_")
  }_API_KEY`;
  return Deno.env.get(envName);
}

function mimeType(path: string): string {
  const lower = path.toLowerCase();
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".webp")) return "image/webp";
  throw new Error(`unsupported image type for ${path}`);
}

async function userContent(
  args: CompleteArgs,
): Promise<string | (TextContent | ImageContent)[]> {
  if (!args.imagePath) return args.prompt;
  const image = await Deno.readFile(args.imagePath);
  return [
    { type: "text", text: args.prompt },
    {
      type: "image",
      data: bytesToBase64(image),
      mimeType: mimeType(args.imagePath),
    },
  ];
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function modelName(model: Model<Api>): string {
  return `${model.provider}/${model.id}`;
}

function generationTps(
  usage: Usage | undefined,
  durationMs: number,
  ttftMs: number | null,
): number | null {
  if (!usage || ttftMs === null || usage.output <= 0) return null;
  const generationSeconds = (durationMs - ttftMs) / 1000;
  if (generationSeconds <= 0) return null;
  return usage.output / generationSeconds;
}

function finishRecord(
  base: Omit<
    CompleteCallRecord,
    | "finishedAt"
    | "durationMs"
    | "ttftMs"
    | "tps"
    | "cost"
    | "usage"
    | "stopReason"
    | "error"
    | "toolCalls"
  >,
  started: number,
  ttftMs: number | null,
  message: AssistantMessage | null,
  error: string | null,
): CompleteCallRecord {
  const durationMs = performance.now() - started;
  const usage = message?.usage;
  return {
    ...base,
    finishedAt: new Date().toISOString(),
    durationMs,
    ttftMs,
    tps: generationTps(usage, durationMs, ttftMs),
    cost: usage?.cost.total ?? 0,
    usage: usageSummary(usage),
    stopReason: message?.stopReason ?? null,
    error,
    toolCalls: message
      ? toolCallsFrom(message).map((call) => ({
        id: call.id,
        name: call.name,
        arguments: call.arguments,
      }))
      : [],
  };
}

export async function complete(args: CompleteArgs): Promise<CompleteResult> {
  const model = resolveModel(args.model || modelSpec);
  const started = performance.now();
  const startedAt = new Date().toISOString();
  let ttftMs: number | null = null;
  let message: AssistantMessage | null = null;
  let error: string | null = null;

  const base = {
    id: crypto.randomUUID(),
    label: args.label,
    model: modelName(model),
    startedAt,
    meta: args.meta || {},
  };

  try {
    const context: Context = {
      systemPrompt: args.systemPrompt,
      messages: [
        {
          role: "user",
          content: await userContent(args),
          timestamp: Date.now(),
        },
      ],
      tools: args.tools,
    };

    const options: Record<string, unknown> = {
      toolChoice: "required",
      maxTokens: args.maxTokens ?? 512,
    };
    if (args.temperature !== undefined) options.temperature = args.temperature;
    const apiKey = customApiKey(model, args.apiKey);
    if (apiKey) options.apiKey = apiKey;

    const stream = piStream(model, context, options);
    for await (const event of stream) {
      if (ttftMs === null && event.type !== "start") {
        ttftMs = performance.now() - started;
      }
    }
    message = await stream.result();

    if (message.stopReason === "error" || message.stopReason === "aborted") {
      throw new Error(message.errorMessage || message.stopReason);
    }

    const toolCalls = toolCallsFrom(message);
    const requiredTool = args.requiredTool || args.tools[0]?.name;
    if (requiredTool && !toolCalls.some((call) => call.name === requiredTool)) {
      throw new Error(`model did not call ${requiredTool}`);
    }

    return { message, toolCalls };
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
    throw err;
  } finally {
    recordCall(finishRecord(base, started, ttftMs, message, error));
  }
}

export function toolArguments(
  result: CompleteResult,
  name: string,
): Record<string, unknown> {
  const call = result.toolCalls.find((item) => item.name === name);
  if (!call) throw new Error(`missing tool call ${name}`);
  return call.arguments;
}

export function numericToolPoint(
  result: CompleteResult,
  name: string,
): { x: number; y: number } {
  const args = toolArguments(result, name);
  const x = Number(args.x);
  const y = Number(args.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new Error(`${name} returned non-numeric coordinates`);
  }
  return { x, y };
}
