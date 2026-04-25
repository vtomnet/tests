// Runs all benchmark envs under Gondolin and Pi.
//
// Directory shape:
//   pi.d/                  Pi auth, model config, sessions
//   envs/<env>/manifest.toml  prompt = "...", judgePrompt = "...", enabled = true
//   envs/<env>/init.sh        setup script run in the VM before the agent
//   envs/<env>/workspace/     fixture copied into ephemeral guest /workspace
//   envs/<env>/out/<model>/0/ changed/new files copied back from guest /workspace
//   envs/<env>/results.json   model ranking and match judgements over generated outputs

import { existsSync } from "node:fs";
import { lstat, readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { parse as parseToml } from "@std/toml";
import { projectRoot as getProjectRoot, sanitizeModelPath } from "./lib.js";
import {
  type AgentResult,
  type FileMap,
  judge,
  type JudgePrompt,
  type JudgeResult,
} from "./judge.ts";

import { runTest } from "./vm.ts";

const MODELS = [
  "openai-codex/gpt-5.5:high",
  "opencode-go/kimi-k2.6:high",
  "opencode-go/glm-5.1:high",
  "deepseek/deepseek-v4-pro:high",
  "opencode-go/mimo-v2.5-pro:high",
  "opencode-go/minimax-m2.7:high",
  "anthropic/claude-sonnet-4-6",
  "google/gemini-3.1-pro-preview",
];

type Manifest = { enabled: false } | {
  enabled: true;
  prompt: string;
  judgePrompt: string;
};

type PlayedMatch = {
  phase: string;
  a: string;
  b: string;
  winner: string;
  result: JudgeResult;
};

type ByeMatch = { phase: string; bye: string; winner: string };
type MatchRecord = PlayedMatch | ByeMatch;

type ResultsFile = {
  version: 2;
  strategy: "double-elimination" | "double-elimination+binary-insertion";
  ranking: string[];
  matches: MatchRecord[];
};

function log(message: string) {
  console.log(`[bench] ${message}`);
}

function assertString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  return value;
}

function assertRelativeInside(rel: string, label: string) {
  if (
    !rel || path.isAbsolute(rel) || rel === ".." ||
    rel.startsWith(`..${path.sep}`)
  ) {
    throw new Error(`${label} escapes its base directory: ${rel}`);
  }
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

async function readManifest(testDir: string): Promise<Manifest> {
  const manifestPath = path.join(testDir, "manifest.toml");
  const raw = await readFile(manifestPath, "utf8");
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
  if (!toml || typeof toml !== "object" || Array.isArray(toml)) {
    throw new Error(`${manifestPath}: manifest must be an object`);
  }
  const manifest = toml as {
    prompt?: unknown;
    judgePrompt?: unknown;
    enabled?: unknown;
  };
  if (manifest.enabled !== undefined && typeof manifest.enabled !== "boolean") {
    throw new Error(`${manifestPath}: enabled must be a boolean`);
  }
  if (manifest.enabled === false) return { enabled: false };

  const prompt = assertString(manifest.prompt, `${manifestPath}: prompt`);
  const judgePrompt = assertString(
    manifest.judgePrompt,
    `${manifestPath}: judgePrompt`,
  );
  return { enabled: true, prompt, judgePrompt };
}

async function readText(filePath: string): Promise<string> {
  return new TextDecoder().decode(await readFile(filePath));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

async function readResultsFile(
  resultsPath: string,
): Promise<ResultsFile | undefined> {
  if (!(await pathExists(resultsPath))) return undefined;
  const parsed = JSON.parse(await readText(resultsPath)) as unknown;
  if (!isRecord(parsed)) return undefined;
  if (!Array.isArray(parsed.ranking)) return undefined;
  if (!parsed.ranking.every((value) => typeof value === "string")) {
    return undefined;
  }
  if (!Array.isArray(parsed.matches)) return undefined;
  return parsed as ResultsFile;
}

async function readDirectFiles(dir: string): Promise<FileMap> {
  const files: FileMap = {};
  const entries = await readdir(dir, { withFileTypes: true });
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    files[entry.name] = await readText(path.join(dir, entry.name));
  }
  return files;
}

async function readRecursiveFiles(root: string): Promise<FileMap> {
  const files: FileMap = {};
  if (!(await pathExists(root))) return files;

  async function walk(dir: string) {
    const entries = await readdir(dir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(fullPath);
      } else if (entry.isFile()) {
        const relative = path.relative(root, fullPath);
        assertRelativeInside(relative, "output file path");
        files[relative.split(path.sep).join("/")] = await readText(fullPath);
      }
    }
  }

  await walk(root);
  return files;
}

async function readAgentResults(testDir: string): Promise<AgentResult[]> {
  const outDir = path.join(testDir, "out");
  if (!(await pathExists(outDir))) return [];

  const knownModels = new Map(
    MODELS.map((model) => [sanitizeModelPath(model), model]),
  );
  const entries = await readdir(outDir, { withFileTypes: true });
  entries.sort((a, b) => a.name.localeCompare(b.name));

  const modelOrder = new Map(MODELS.map((model, index) => [model, index]));
  const results: AgentResult[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.endsWith(".tmp")) continue;
    const runDir = path.join(outDir, entry.name, "0");
    if (!(await pathExists(runDir))) continue;
    results.push({
      name: knownModels.get(entry.name) ?? entry.name,
      files: await readRecursiveFiles(path.join(runDir, "workspace")),
    });
  }
  results.sort((a, b) =>
    (modelOrder.get(a.name) ?? Number.MAX_SAFE_INTEGER) -
      (modelOrder.get(b.name) ?? Number.MAX_SAFE_INTEGER) ||
    a.name.localeCompare(b.name)
  );
  return results;
}

async function judgeMatch(
  envName: string,
  prompt: JudgePrompt,
  a: AgentResult,
  b: AgentResult,
  phase: string,
  matches: MatchRecord[],
): Promise<AgentResult> {
  log(`${envName}: ${phase}: judging ${a.name} vs ${b.name}`);
  const result = await judge(prompt, a, b);
  const winner = result.judgement === "a" ? a : b;
  matches.push({
    phase,
    a: a.name,
    b: b.name,
    winner: winner.name,
    result,
  });
  return winner;
}

type Entrant = { result: AgentResult; seed: number; losses: number };

async function doubleEliminationRanking(
  envName: string,
  prompt: JudgePrompt,
  results: AgentResult[],
): Promise<{ ranking: AgentResult[]; matches: MatchRecord[] }> {
  let active: Entrant[] = results.map((result, seed) => ({
    result,
    seed,
    losses: 0,
  }));
  const eliminated: Entrant[] = [];
  const matches: MatchRecord[] = [];
  let round = 1;

  while (active.length > 1) {
    active.sort((a, b) => a.losses - b.losses || a.seed - b.seed);
    const next: Entrant[] = [];
    const phase = `double-elimination round ${round}`;

    for (let i = 0; i < active.length; i += 2) {
      const a = active[i];
      const b = active[i + 1];
      if (!b) {
        matches.push({ phase, bye: a.result.name, winner: a.result.name });
        next.push(a);
        continue;
      }

      const winnerResult = await judgeMatch(
        envName,
        prompt,
        a.result,
        b.result,
        phase,
        matches,
      );
      const winner = winnerResult.name === a.result.name ? a : b;
      const loser = winner === a ? b : a;
      loser.losses++;
      if (loser.losses >= 2) eliminated.push(loser);
      else next.push(loser);
      next.push(winner);
    }

    active = next;
    round++;
  }

  return {
    ranking: [...active, ...eliminated.reverse()].map((entrant) =>
      entrant.result
    ),
    matches,
  };
}

async function insertIntoRanking(
  envName: string,
  prompt: JudgePrompt,
  ranking: AgentResult[],
  candidate: AgentResult,
  matches: MatchRecord[],
) {
  let low = 0;
  let high = ranking.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    const winner = await judgeMatch(
      envName,
      prompt,
      candidate,
      ranking[mid],
      `binary insertion for ${candidate.name}`,
      matches,
    );
    if (winner.name === candidate.name) high = mid;
    else low = mid + 1;
  }
  ranking.splice(low, 0, candidate);
}

async function resultsNeedJudge(testDir: string): Promise<boolean> {
  const resultsPath = path.join(testDir, "results.json");
  const existing = await readResultsFile(resultsPath);
  if (!existing) return true;

  const ranked = new Set(existing.ranking);
  const outputs = await readAgentResults(testDir);
  return outputs.some((result) => !ranked.has(result.name));
}

async function runJudge(
  testDir: string,
  manifest: Extract<Manifest, { enabled: true }>,
) {
  const envName = path.basename(testDir);
  log(`${envName}: running judge`);
  const prompt: JudgePrompt = {
    judgePrompt: manifest.judgePrompt,
    originalPrompt: manifest.prompt,
    referenceFiles: await readDirectFiles(path.join(testDir, "workspace")),
  };

  const results = await readAgentResults(testDir);
  const byName = new Map(results.map((result) => [result.name, result]));
  const resultsPath = path.join(testDir, "results.json");
  const existing = await readResultsFile(resultsPath);

  if (!existing) {
    const ranked = await doubleEliminationRanking(envName, prompt, results);
    await writeFile(
      resultsPath,
      `${
        JSON.stringify(
          {
            version: 2,
            strategy: "double-elimination",
            ranking: ranked.ranking.map((result) => result.name),
            matches: ranked.matches,
          } satisfies ResultsFile,
          null,
          2,
        )
      }\n`,
    );
    return;
  }

  const ranking = existing.ranking
    .map((name) => byName.get(name))
    .filter((result): result is AgentResult => result !== undefined);
  if (ranking.length === 0 && results.length > 1) {
    const ranked = await doubleEliminationRanking(envName, prompt, results);
    await writeFile(
      resultsPath,
      `${
        JSON.stringify(
          {
            version: 2,
            strategy: "double-elimination",
            ranking: ranked.ranking.map((result) => result.name),
            matches: ranked.matches,
          } satisfies ResultsFile,
          null,
          2,
        )
      }\n`,
    );
    return;
  }
  const ranked = new Set(ranking.map((result) => result.name));
  const newResults = results.filter((result) => !ranked.has(result.name));
  const matches = existing.matches.slice();

  for (const result of newResults) {
    await insertIntoRanking(envName, prompt, ranking, result, matches);
  }

  await writeFile(
    resultsPath,
    `${
      JSON.stringify(
        {
          version: 2,
          strategy: "double-elimination+binary-insertion",
          ranking: ranking.map((result) => result.name),
          matches,
        } satisfies ResultsFile,
        null,
        2,
      )
    }\n`,
  );
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

    const initCode = await readFile(path.join(testDir, "init.sh"), "utf8");
    const workspacePath = path.join(testDir, "workspace");
    if (!(await pathExists(workspacePath))) {
      throw new Error(`${workspacePath} does not exist`);
    }

    let addedAny = false;
    for (const model of MODELS) {
      const outPath = path.join(testDir, "out", sanitizeModelPath(model), "0");
      if (existsSync(outPath)) {
        log(`${entry.name}: skipping ${model}; ${outPath} exists`);
        continue;
      }
      await rm(outPath, { recursive: true, force: true });
      await runTest(model, manifest.prompt, workspacePath, outPath, initCode, {
        projectRoot,
        envName: entry.name,
      });
      generatedPaths.push(outPath);
      addedAny = true;
    }

    const resultsPath = path.join(testDir, "results.json");
    if (addedAny || await resultsNeedJudge(testDir)) {
      await runJudge(testDir, manifest);
      generatedPaths.push(resultsPath);
    }
  }

  await gitCommitGenerated(projectRoot, generatedPaths);
}

if (import.meta.main) await main();
