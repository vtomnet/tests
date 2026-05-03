// Runs all benchmark envs under Gondolin and Pi.
//
// Directory shape:
//   pi.d/                  Pi auth, model config, sessions
//   envs/<env>/manifest.toml  prompt = "...", enabled = true
//   envs/<env>/init.sh        setup script run in the VM before the agent
//   envs/<env>/test.sh        validation script run in the VM after the agent
//   envs/<env>/workspace/     fixture copied into ephemeral guest /workspace
//   envs/<env>/out/<model>/0/ changed/new files copied back from guest /workspace
//   envs/<env>/results.json   binary test results for generated outputs

import { existsSync } from "node:fs";
import { lstat, readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { parse as parseToml } from "@std/toml";
import { projectRoot as getProjectRoot, sanitizeModelPath } from "./lib.js";
import { runSavedTest, runTest, type TestResult } from "./vm.ts";

const MODELS = [
  "openai-codex/gpt-5.5:high",
  // "google/gemini-3.1-pro-preview:high",
  // "anthropic/claude-opus-4-7:high",
  // "xai/grok-4.20-0309-reasoning:high",
  "opencode-go/kimi-k2.6:high",
  "opencode-go/glm-5.1:high",
  "opencode-go/mimo-v2.5-pro:high",
  "opencode-go/minimax-m2.7:high",
  "deepseek/deepseek-v4-pro:high",
];

type Manifest = { enabled: false } | { enabled: true; prompt: string };

type ModelResult = TestResult & {
  model: string;
  outPath: string;
};

type ResultsFile = {
  version: 3;
  results: Record<string, ModelResult>;
};

function log(message: string) {
  console.log(`[bench] ${message}`);
}

function assertString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await lstat(filePath);
    return true;
  } catch (err) {
    if (err instanceof Deno.errors.NotFound) return false;
    if (
      err && typeof err === "object" && "code" in err &&
      (err as { code?: string }).code === "ENOENT"
    ) return false;
    throw err;
  }
}

async function readText(filePath: string): Promise<string> {
  return new TextDecoder().decode(await readFile(filePath));
}

async function readManifest(testDir: string): Promise<Manifest> {
  const manifestPath = path.join(testDir, "manifest.toml");
  const raw = await readText(manifestPath);
  let toml: unknown;
  try {
    toml = parseToml(raw);
  } catch (err) {
    throw new Error(
      `${manifestPath}: invalid TOML: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
  }
  if (!isRecord(toml)) {
    throw new Error(`${manifestPath}: manifest must be an object`);
  }

  if (toml.enabled !== undefined && typeof toml.enabled !== "boolean") {
    throw new Error(`${manifestPath}: enabled must be a boolean`);
  }
  if (toml.enabled === false) return { enabled: false };

  return {
    enabled: true,
    prompt: assertString(toml.prompt, `${manifestPath}: prompt`),
  };
}

async function readResultsFile(
  resultsPath: string,
): Promise<ResultsFile | undefined> {
  if (!(await pathExists(resultsPath))) return undefined;

  const parsed = JSON.parse(await readText(resultsPath)) as unknown;
  if (!isRecord(parsed) || parsed.version !== 3 || !isRecord(parsed.results)) {
    return undefined;
  }

  const results: Record<string, ModelResult> = {};
  for (const [model, value] of Object.entries(parsed.results)) {
    if (!isRecord(value)) continue;
    if (typeof value.success !== "boolean") continue;
    if (typeof value.exitCode !== "number") continue;
    if (typeof value.stdout !== "string") continue;
    if (typeof value.stderr !== "string") continue;
    if (typeof value.outPath !== "string") continue;
    results[model] = {
      model,
      outPath: value.outPath,
      success: value.success,
      exitCode: value.exitCode,
      stdout: value.stdout,
      stderr: value.stderr,
    };
  }

  return { version: 3, results };
}

function modelResult(
  testDir: string,
  model: string,
  outPath: string,
  result: TestResult,
): ModelResult {
  return {
    model,
    outPath: path.relative(testDir, outPath),
    ...result,
  };
}

async function runEnv(
  projectRoot: string,
  testDir: string,
  envName: string,
  manifest: Extract<Manifest, { enabled: true }>,
): Promise<string[]> {
  const initCode = await readText(path.join(testDir, "init.sh"));
  const testCode = await readText(path.join(testDir, "test.sh"));
  const workspacePath = path.join(testDir, "workspace");
  if (!(await pathExists(workspacePath))) {
    throw new Error(`${workspacePath} does not exist`);
  }

  const resultsPath = path.join(testDir, "results.json");
  const existing = await readResultsFile(resultsPath);
  const results: Record<string, ModelResult> = {};
  const generatedPaths: string[] = [];
  let resultsChanged = !existing;

  for (const model of MODELS) {
    const outPath = path.join(testDir, "out", sanitizeModelPath(model), "0");
    const existingResult = existing?.results[model];
    if (existsSync(outPath)) {
      log(`${envName}: skipping ${model}; ${outPath} exists`);
      if (existingResult) {
        results[model] = existingResult;
        continue;
      }

      const testResult = await runSavedTest(
        model,
        workspacePath,
        outPath,
        initCode,
        testCode,
        { projectRoot, envName },
      );
      results[model] = modelResult(testDir, model, outPath, testResult);
      resultsChanged = true;
      continue;
    }

    await rm(outPath, { recursive: true, force: true });
    const testResult = await runTest(
      model,
      manifest.prompt,
      workspacePath,
      outPath,
      initCode,
      testCode,
      { projectRoot, envName },
    );
    generatedPaths.push(outPath);
    results[model] = modelResult(testDir, model, outPath, testResult);
    resultsChanged = true;
  }

  if (resultsChanged) {
    await writeFile(
      resultsPath,
      `${
        JSON.stringify({ version: 3, results } satisfies ResultsFile, null, 2)
      }\n`,
    );
    generatedPaths.push(resultsPath);
  }

  return generatedPaths;
}

async function gitCommitGenerated(projectRoot: string, paths: string[]) {
  if (paths.length === 0) return;

  const relativePaths = paths.map((p) => path.relative(projectRoot, p)).filter((
    p,
  ) => p && !p.startsWith(".."));
  if (relativePaths.length === 0) return;

  const add = await new Deno.Command("git", {
    args: ["add", ...relativePaths],
    cwd: projectRoot,
    stdout: "inherit",
    stderr: "inherit",
  }).output();
  if (!add.success) {
    throw new Error(`git add failed with exit code ${add.code}`);
  }

  const status = await new Deno.Command("git", {
    args: ["diff", "--cached", "--quiet"],
    cwd: projectRoot,
  }).output();
  if (status.success) return;

  const commit = await new Deno.Command("git", {
    args: ["commit", "-m", "bench: add generated results"],
    cwd: projectRoot,
    stdout: "inherit",
    stderr: "inherit",
  }).output();
  if (!commit.success) {
    throw new Error(`git commit failed with exit code ${commit.code}`);
  }
}

async function main() {
  const projectRoot = getProjectRoot();
  const envsDir = path.join(projectRoot, "envs");
  const generatedPaths: string[] = [];

  const envEntries = await readdir(envsDir, { withFileTypes: true });
  envEntries.sort((a, b) => a.name.localeCompare(b.name));

  for (const entry of envEntries) {
    if (!entry.isDirectory()) continue;

    const testDir = path.join(envsDir, entry.name);
    const manifest = await readManifest(testDir);
    if (!manifest.enabled) {
      log(`${entry.name}: skipped; manifest enabled=false`);
      continue;
    }

    generatedPaths.push(
      ...await runEnv(projectRoot, testDir, entry.name, manifest),
    );
  }

  await gitCommitGenerated(projectRoot, generatedPaths);
}

if (import.meta.main) await main();
