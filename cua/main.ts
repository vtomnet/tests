import {
  dirname,
  extname,
  fromFileUrl,
  join,
  normalize,
  SEPARATOR,
  toFileUrl,
} from "@std/path";
import {
  beginCompleteRecording,
  getDefaultModel,
  setDefaultModel,
  summarizeCompleteCalls,
  takeCompleteRecording,
} from "./complete.ts";
import { escapeHtml } from "./html.ts";
import type { BenchmarkModule, SavedRun } from "./types.ts";

const ROOT = dirname(fromFileUrl(import.meta.url));
const TESTS_DIR = join(ROOT, "tests");
const OUT_DIR = join(ROOT, "out");
const UI_DIST = join(ROOT, "dist", "ui");
const HOST = Deno.env.get("HOST") || "127.0.0.1";
const PORT = Number(Deno.env.get("PORT") || 8000);

interface LoadedTest {
  id: string;
  dir: string;
  module: BenchmarkModule;
}

let tests = new Map<string, LoadedTest>();
let activeJob: string | null = null;

function assertPort(port: number): void {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("PORT must be an integer from 1 to 65535");
  }
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

async function requestJson(req: Request): Promise<Record<string, unknown>> {
  const text = await req.text();
  if (!text.trim()) return {};
  const value = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("request body must be a JSON object");
  }
  return value as Record<string, unknown>;
}

function httpError(
  status: number,
  message: string,
): Error & { status: number } {
  const err = new Error(message) as Error & { status: number };
  err.status = status;
  return err;
}

async function withJob<T>(name: string, fn: () => Promise<T>): Promise<T> {
  if (activeJob) throw httpError(409, `${activeJob} is already running`);
  activeJob = name;
  try {
    return await fn();
  } finally {
    activeJob = null;
  }
}

async function ensureUiBundle(): Promise<void> {
  try {
    await Deno.stat(join(UI_DIST, "index.html"));
    return;
  } catch {
    // build below
  }

  const command = new Deno.Command(Deno.execPath(), {
    args: ["task", "build-ui"],
    cwd: ROOT,
    stdout: "inherit",
    stderr: "inherit",
  });
  const status = await command.output();
  if (!status.success) throw new Error("UI build failed");
}

async function loadTests(): Promise<Map<string, LoadedTest>> {
  const loaded = new Map<string, LoadedTest>();
  const entries = [];
  for await (const entry of Deno.readDir(TESTS_DIR)) {
    if (entry.isDirectory) entries.push(entry.name);
  }
  entries.sort();

  for (const id of entries) {
    const path = join(TESTS_DIR, id, "main.ts");
    const stat = await Deno.stat(path).catch(() => null);
    if (!stat?.isFile) continue;
    const module = await import(toFileUrl(path).href) as BenchmarkModule;
    for (const name of ["build", "run", "view"] as const) {
      if (typeof module[name] !== "function") {
        throw new Error(`${id}/main.ts does not export ${name}()`);
      }
    }
    loaded.set(id, { id, dir: join(TESTS_DIR, id), module });
  }
  return loaded;
}

function testOrThrow(id: string): LoadedTest {
  const test = tests.get(id);
  if (!test) throw httpError(404, `unknown test ${id}`);
  return test;
}

async function ensureOutDir(testId: string): Promise<string> {
  const dir = join(OUT_DIR, testId);
  await Deno.mkdir(dir, { recursive: true });
  return dir;
}

async function readRunFile(path: string): Promise<SavedRun> {
  return JSON.parse(await Deno.readTextFile(path)) as SavedRun;
}

async function runsFor(testId: string): Promise<SavedRun[]> {
  const dir = join(OUT_DIR, testId);
  const entries: Array<{ n: number; path: string }> = [];
  try {
    for await (const entry of Deno.readDir(dir)) {
      const match = entry.isFile ? entry.name.match(/^run_(\d+)\.json$/) : null;
      if (match) {
        entries.push({ n: Number(match[1]), path: join(dir, entry.name) });
      }
    }
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return [];
    throw err;
  }

  entries.sort((a, b) => a.n - b.n);
  return await Promise.all(entries.map((entry) => readRunFile(entry.path)));
}

async function nextRunNumber(testId: string): Promise<number> {
  const runs = await runsFor(testId);
  return runs.reduce((max, run) => Math.max(max, run.run), 0) + 1;
}

async function writeRun(run: SavedRun): Promise<void> {
  const dir = await ensureOutDir(run.test);
  await Deno.writeTextFile(
    join(dir, `run_${run.run}.json`),
    JSON.stringify(run, null, 2),
  );
}

async function viewHtml(test: LoadedTest, run?: SavedRun): Promise<string> {
  try {
    return await test.module.view(run);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return `<p class="bench-error">${escapeHtml(message)}</p>`;
  }
}

function publicRun(run: SavedRun): SavedRun {
  return run;
}

async function testSummary(test: LoadedTest): Promise<Record<string, unknown>> {
  const runs = await runsFor(test.id);
  const latest = runs.at(-1);
  return {
    id: test.id,
    title: test.module.title || test.id,
    description: test.module.description || "",
    latestRun: latest ? publicRun(latest) : null,
    latestView: await viewHtml(test, latest),
  };
}

async function allTestsResponse(): Promise<Response> {
  const summaries = [];
  for (const test of tests.values()) summaries.push(await testSummary(test));
  return json({ model: getDefaultModel(), activeJob, tests: summaries });
}

async function oneTestResponse(id: string): Promise<Response> {
  const test = testOrThrow(id);
  const runs = await runsFor(id);
  const fullRuns = [];
  for (const run of runs) {
    fullRuns.push({ ...publicRun(run), view: await viewHtml(test, run) });
  }
  return json({
    model: getDefaultModel(),
    activeJob,
    test: {
      id: test.id,
      title: test.module.title || test.id,
      description: test.module.description || "",
      runs: fullRuns,
      emptyView: runs.length ? null : await viewHtml(test),
    },
  });
}

async function buildTest(id: string): Promise<Response> {
  const test = testOrThrow(id);
  await withJob(`build ${id}`, async () => {
    await test.module.build();
  });
  return oneTestResponse(id);
}

async function runTest(id: string, req: Request): Promise<Response> {
  const test = testOrThrow(id);
  const body = await requestJson(req);
  const requestedModel = typeof body.model === "string"
    ? body.model.trim()
    : "";
  if (requestedModel) setDefaultModel(requestedModel);

  await withJob(`run ${id}`, async () => {
    const runNumber = await nextRunNumber(id);
    const startedAt = new Date().toISOString();
    const started = performance.now();
    let score: number | null = null;
    let error: string | null = null;

    beginCompleteRecording();
    try {
      const value = await test.module.run();
      if (!Number.isFinite(value)) {
        throw new Error(`${id} returned a non-finite score`);
      }
      score = value;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
    const calls = takeCompleteRecording();
    const durationMs = performance.now() - started;
    const finishedAt = new Date().toISOString();
    await writeRun({
      test: id,
      run: runNumber,
      model: getDefaultModel(),
      score,
      error,
      startedAt,
      finishedAt,
      durationMs,
      stats: summarizeCompleteCalls(calls, durationMs),
      calls,
    });
  });

  return oneTestResponse(id);
}

function safeJoin(root: string, rel: string): string {
  const path = normalize(join(root, rel));
  if (path !== root && !path.startsWith(root + SEPARATOR)) {
    throw httpError(403, "path escapes static root");
  }
  return path;
}

function contentType(path: string): string {
  const ext = extname(path).toLowerCase();
  if (ext === ".html") return "text/html; charset=utf-8";
  if (ext === ".js") return "text/javascript; charset=utf-8";
  if (ext === ".css") return "text/css; charset=utf-8";
  if (ext === ".json") return "application/json; charset=utf-8";
  if (ext === ".png") return "image/png";
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  if (ext === ".svg") return "image/svg+xml";
  return "application/octet-stream";
}

async function fileResponse(path: string): Promise<Response> {
  try {
    return new Response(await Deno.readFile(path), {
      headers: {
        "Content-Type": contentType(path),
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) {
      throw httpError(404, "file not found");
    }
    throw err;
  }
}

async function staticResponse(pathname: string): Promise<Response | null> {
  if (pathname === "/" || pathname === "/index.html") {
    return fileResponse(join(UI_DIST, "index.html"));
  }
  if (pathname.startsWith("/assets/")) {
    return fileResponse(
      safeJoin(UI_DIST, decodeURIComponent(pathname.slice(1))),
    );
  }
  if (pathname.startsWith("/files/tests/")) {
    return fileResponse(
      safeJoin(
        TESTS_DIR,
        decodeURIComponent(pathname.slice("/files/tests/".length)),
      ),
    );
  }
  if (pathname.startsWith("/files/out/")) {
    return fileResponse(
      safeJoin(
        OUT_DIR,
        decodeURIComponent(pathname.slice("/files/out/".length)),
      ),
    );
  }
  return null;
}

async function route(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const staticFile = await staticResponse(url.pathname);
  if (staticFile) return staticFile;

  if (req.method === "GET" && url.pathname === "/api/tests") {
    return allTestsResponse();
  }

  const testMatch = url.pathname.match(/^\/api\/tests\/([^/]+)$/);
  if (testMatch) {
    const id = decodeURIComponent(testMatch[1]);
    if (req.method === "GET") return oneTestResponse(id);
  }

  const actionMatch = url.pathname.match(
    /^\/api\/tests\/([^/]+)\/(build|run)$/,
  );
  if (actionMatch && req.method === "POST") {
    const id = decodeURIComponent(actionMatch[1]);
    const action = actionMatch[2];
    if (action === "build") return buildTest(id);
    if (action === "run") return runTest(id, req);
  }

  throw httpError(404, "not found");
}

async function main(): Promise<void> {
  assertPort(PORT);
  await ensureUiBundle();
  tests = await loadTests();
  Deno.serve({ hostname: HOST, port: PORT }, (req) => {
    return route(req).catch((err) => {
      const status = Number((err as { status?: number }).status || 500);
      const message = err instanceof Error ? err.message : String(err);
      return json({ error: { message } }, status);
    });
  });
  console.log(`Serving http://${HOST}:${PORT}`);
}

if (import.meta.main) await main();
