export type Json = null | boolean | number | string | Json[] | {
  [key: string]: Json;
};

export interface Point {
  x: number;
  y: number;
}

export interface Box extends Point {
  width: number;
  height: number;
}

export interface CompleteCallRecord {
  id: string;
  label: string;
  model: string;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  ttftMs: number | null;
  tps: number | null;
  cost: number;
  usage: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    totalTokens: number;
  } | null;
  stopReason: string | null;
  error: string | null;
  toolCalls: Array<{
    id: string;
    name: string;
    arguments: Record<string, unknown>;
  }>;
  meta: Record<string, unknown>;
}

export interface RunStats {
  calls: number;
  errors: number;
  errorRate: number;
  durationMs: number;
  cost: number;
  avgTtftMs: number | null;
  avgTps: number | null;
}

export interface SavedRun {
  test: string;
  run: number;
  model: string;
  score: number | null;
  error: string | null;
  startedAt: string;
  finishedAt: string;
  durationMs: number;
  stats: RunStats;
  calls: CompleteCallRecord[];
}

export interface BenchmarkModule {
  title?: string;
  description?: string;
  build(): Promise<void> | void;
  run(): Promise<number> | number;
  view(run?: SavedRun): Promise<string> | string;
}
