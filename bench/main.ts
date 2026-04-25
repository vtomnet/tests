// Runs all benchmark envs under Gondolin and Pi.
//
// Directory shape:
//   pi.d/                  Pi auth, model config, sessions
//   envs/<env>/manifest.json  { "prompt": "...", "enabled": true }
//   envs/<env>/init.sh        setup script run in the VM before the agent
//   envs/<env>/workspace/     fixture copied into ephemeral guest /workspace
//   envs/<env>/out/<model>/0/ changed/new files copied back from guest /workspace
//   envs/<env>/judge.ts       run after any new model outputs are produced

import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  readdir,
  readFile,
  readlink,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createHttpHooks, MemoryProvider, VM } from "@earendil-works/gondolin";
import {
  AuthStorage,
  type BashOperations,
  createAgentSession,
  createBashToolDefinition,
  createEditToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
  DefaultResourceLoader,
  type EditOperations,
  ModelRegistry,
  type ReadOperations,
  SessionManager,
  SettingsManager,
  type ToolDefinition,
  type WriteOperations,
} from "@mariozechner/pi-coding-agent";

const MODELS = [
  "openai-codex/gpt-5.5:high",
  "opencode-go/kimi-k2.6:high",
  "opencode-go/glm-5.1:high",
  "opencode-go/deepseek-v4-pro:high",
  "opencode-go/mimo-v2.5-pro:high",
  "opencode-go/minimax-m2.7:high",
];

const GUEST_WORKSPACE = "/workspace";
const GUEST_PATH =
  "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

const THINKING_LEVELS = new Set<ThinkingLevel>([
  "off",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
]);

type SnapshotEntry = { hash: string; mode: number; size: number };
type Snapshot = Map<string, SnapshotEntry>;
type Manifest = { enabled: false } | { enabled: true; prompt: string };

type RunTestOptions = {
  projectRoot: string;
  envName: string;
};

function log(message: string) {
  console.log(`[bench] ${message}`);
}

function shQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
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

function guestEnv(extra?: Record<string, string>): Record<string, string> {
  return {
    HOME: "/root",
    LANG: "C.UTF-8",
    LC_ALL: "C.UTF-8",
    PATH: GUEST_PATH,
    TERM: "xterm-256color",
    ...extra,
  };
}

function sanitizeModelPath(model: string): string {
  return model.replaceAll("/", "--").replaceAll(":", "--");
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
  const manifestPath = path.join(testDir, "manifest.json");
  const raw = await readFile(manifestPath, "utf8");
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `${manifestPath}: invalid JSON: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
  }
  if (!json || typeof json !== "object" || Array.isArray(json)) {
    throw new Error(`${manifestPath}: manifest must be an object`);
  }
  const manifest = json as { prompt?: unknown; enabled?: unknown };
  if (manifest.enabled !== undefined && typeof manifest.enabled !== "boolean") {
    throw new Error(`${manifestPath}: enabled must be a boolean`);
  }
  if (manifest.enabled === false) return { enabled: false };

  const prompt = assertString(manifest.prompt, `${manifestPath}: prompt`);
  return { enabled: true, prompt };
}

function hashBuffer(buffer: Buffer): string {
  return createHash("sha256").update(buffer).digest("hex");
}

type WorkspaceProvider = {
  mkdir(
    path: string,
    options?: { recursive?: boolean; mode?: number },
  ): Promise<unknown>;
  writeFile?(
    path: string,
    data: Buffer,
    options?: { mode?: number },
  ): Promise<void>;
  symlink?(target: string, path: string, type?: string): Promise<void>;
};

async function copyHostTreeToProvider(
  provider: WorkspaceProvider,
  hostPath: string,
  providerPath: string,
) {
  const stat = await lstat(hostPath);

  if (stat.isDirectory()) {
    if (providerPath !== "/") {
      await provider.mkdir(providerPath, {
        recursive: true,
        mode: stat.mode & 0o7777,
      });
    }
    const entries = await readdir(hostPath, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      await copyHostTreeToProvider(
        provider,
        path.join(hostPath, entry.name),
        path.posix.join(providerPath, entry.name),
      );
    }
    return;
  }

  await provider.mkdir(path.posix.dirname(providerPath), { recursive: true });

  if (stat.isSymbolicLink()) {
    if (!provider.symlink) {
      throw new Error("workspace provider does not support symlinks");
    }
    await provider.symlink(await readlink(hostPath), providerPath);
    return;
  }

  if (!stat.isFile()) return;
  if (!provider.writeFile) {
    throw new Error("workspace provider does not support writeFile");
  }
  await provider.writeFile(providerPath, await readFile(hostPath), {
    mode: stat.mode & 0o7777,
  });
}

async function snapshotWorkspace(vm: VM): Promise<Snapshot> {
  const result = await vm.exec(
    ["/bin/sh", "-lc", `find ${shQuote(GUEST_WORKSPACE)} -type f -print0`],
    { cwd: GUEST_WORKSPACE, env: guestEnv() },
  );
  if (!result.ok) {
    throw new Error(
      `workspace snapshot failed (${result.exitCode}): ${result.stderr}`,
    );
  }

  const snapshot: Snapshot = new Map();
  const paths = result.stdoutBuffer.toString("utf8").split("\0").filter(
    Boolean,
  );
  paths.sort((a, b) => a.localeCompare(b));
  for (const guestPath of paths) {
    const [stat, bytes] = await Promise.all([
      vm.fs.stat(guestPath),
      vm.fs.readFile(guestPath),
    ]);
    snapshot.set(guestPath, {
      hash: hashBuffer(bytes),
      mode: stat.mode & 0o7777,
      size: stat.size,
    });
  }
  return snapshot;
}

async function saveChangedFiles(
  vm: VM,
  before: Snapshot,
  after: Snapshot,
  outPath: string,
): Promise<number> {
  let count = 0;
  for (const [guestPath, entry] of after) {
    const old = before.get(guestPath);
    if (
      old && old.hash === entry.hash && old.mode === entry.mode &&
      old.size === entry.size
    ) continue;

    if (!guestPath.startsWith("/")) {
      throw new Error(`guest path is not absolute: ${guestPath}`);
    }
    const relative = guestPath.slice(1);
    assertRelativeInside(
      relative.split("/").join(path.sep),
      "guest output path",
    );
    const hostPath = path.join(outPath, ...relative.split("/"));
    await mkdir(path.dirname(hostPath), { recursive: true });
    await writeFile(hostPath, await vm.fs.readFile(guestPath));
    await chmod(hostPath, entry.mode || 0o644);
    count++;
  }

  if (count === 0) await mkdir(outPath, { recursive: true });
  return count;
}

function parseModelArg(
  modelArg: string,
): { provider: string; modelId: string; thinkingLevel?: ThinkingLevel } {
  let head = modelArg;
  let thinkingLevel: ThinkingLevel | undefined;
  const colon = modelArg.lastIndexOf(":");
  if (colon !== -1) {
    const suffix = modelArg.slice(colon + 1);
    if (!THINKING_LEVELS.has(suffix as ThinkingLevel)) {
      throw new Error(
        `invalid thinking level in model '${modelArg}': ${suffix}`,
      );
    }
    head = modelArg.slice(0, colon);
    thinkingLevel = suffix as ThinkingLevel;
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

function resolveModel(modelArg: string, modelRegistry: ModelRegistry) {
  const { provider, modelId, thinkingLevel } = parseModelArg(modelArg);
  const model = modelRegistry.find(provider, modelId);
  if (!model) throw new Error(`model not found: ${provider}/${modelId}`);
  return { model, thinkingLevel };
}

function createGondolinReadOps(vm: VM): ReadOperations {
  return {
    readFile: (guestPath) => vm.fs.readFile(guestPath),
    access: (guestPath) => vm.fs.access(guestPath),
    detectImageMimeType: async (guestPath) => {
      const result = await vm.exec([
        "/bin/sh",
        "-lc",
        `file --mime-type -b ${shQuote(guestPath)}`,
      ], {
        cwd: GUEST_WORKSPACE,
        env: guestEnv(),
      });
      if (!result.ok) return null;
      const mime = result.stdout.trim();
      return ["image/jpeg", "image/png", "image/gif", "image/webp"].includes(
          mime,
        )
        ? mime
        : null;
    },
  };
}

function createGondolinWriteOps(vm: VM): WriteOperations {
  return {
    mkdir: (guestPath) => vm.fs.mkdir(guestPath, { recursive: true }),
    writeFile: async (guestPath, content) => {
      await vm.fs.mkdir(path.posix.dirname(guestPath), { recursive: true });
      await vm.fs.writeFile(guestPath, content);
    },
  };
}

function createGondolinEditOps(vm: VM): EditOperations {
  const readOps = createGondolinReadOps(vm);
  const writeOps = createGondolinWriteOps(vm);
  return {
    access: readOps.access,
    readFile: readOps.readFile,
    writeFile: writeOps.writeFile,
  };
}

function createGondolinBashOps(vm: VM): BashOperations {
  return {
    exec: async (command, cwd, { onData, signal, timeout }) => {
      const controller = new AbortController();
      const onAbort = () => controller.abort();
      signal?.addEventListener("abort", onAbort, { once: true });

      let timedOut = false;
      const timer = timeout && timeout > 0
        ? setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, timeout * 1000)
        : undefined;

      try {
        const proc = vm.exec(["/bin/bash", "-lc", command], {
          cwd,
          env: guestEnv(),
          signal: controller.signal,
          stdout: "pipe",
          stderr: "pipe",
        });
        for await (const chunk of proc.output()) onData(chunk.data);
        const result = await proc;
        return { exitCode: result.exitCode };
      } catch (err) {
        if (signal?.aborted) throw new Error("aborted");
        if (timedOut) throw new Error(`timeout:${timeout}`);
        throw err;
      } finally {
        if (timer) clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
      }
    },
  };
}

function customTools(vm: VM): ToolDefinition[] {
  return [
    createReadToolDefinition(GUEST_WORKSPACE, {
      operations: createGondolinReadOps(vm),
    }) as unknown as ToolDefinition,
    createWriteToolDefinition(GUEST_WORKSPACE, {
      operations: createGondolinWriteOps(vm),
    }) as unknown as ToolDefinition,
    createEditToolDefinition(GUEST_WORKSPACE, {
      operations: createGondolinEditOps(vm),
    }) as unknown as ToolDefinition,
    createBashToolDefinition(GUEST_WORKSPACE, {
      operations: createGondolinBashOps(vm),
    }) as unknown as ToolDefinition,
  ];
}

async function runAgent(
  vm: VM,
  modelArg: string,
  prompt: string,
  projectRoot: string,
  envName: string,
) {
  const agentDir = path.join(projectRoot, "pi.d");
  await mkdir(agentDir, { recursive: true });

  const authStorage = AuthStorage.create(path.join(agentDir, "auth.json"));
  const modelRegistry = ModelRegistry.create(
    authStorage,
    path.join(agentDir, "models.json"),
  );
  const { model, thinkingLevel } = resolveModel(modelArg, modelRegistry);
  const settingsManager = SettingsManager.create(GUEST_WORKSPACE, agentDir);
  const sessionManager = SessionManager.create(
    GUEST_WORKSPACE,
    path.join(agentDir, "sessions"),
  );
  const resourceLoader = new DefaultResourceLoader({
    cwd: GUEST_WORKSPACE,
    agentDir,
    settingsManager,
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
  });
  await resourceLoader.reload();

  const { session, modelFallbackMessage } = await createAgentSession({
    cwd: GUEST_WORKSPACE,
    agentDir,
    authStorage,
    modelRegistry,
    model,
    thinkingLevel,
    sessionManager,
    settingsManager,
    resourceLoader,
    tools: ["read", "bash", "edit", "write"],
    customTools: customTools(vm),
  });

  if (modelFallbackMessage) log(modelFallbackMessage);
  log(
    `${envName}: pi session ${session.sessionId} ${session.sessionFile ?? ""}`,
  );

  const unsubscribe = session.subscribe((event) => {
    if (
      event.type === "message_update" &&
      event.assistantMessageEvent.type === "text_delta"
    ) {
      Deno.stdout.writeSync(
        new TextEncoder().encode(event.assistantMessageEvent.delta),
      );
    } else if (event.type === "tool_execution_start") {
      console.log(
        `\n[tool:start] ${event.toolName} ${JSON.stringify(event.args)}`,
      );
    } else if (event.type === "tool_execution_end") {
      console.log(
        `[tool:end] ${event.toolName}${event.isError ? " error" : ""}`,
      );
    } else if (event.type === "auto_retry_start") {
      console.log(
        `[retry] ${event.attempt}/${event.maxAttempts}: ${event.errorMessage}`,
      );
    }
  });

  try {
    await session.prompt(prompt, { expandPromptTemplates: false });
  } finally {
    unsubscribe();
    session.dispose();
  }

  const last = session.state.messages.findLast((message) =>
    message.role === "assistant"
  ) as { stopReason?: string; errorMessage?: string } | undefined;
  if (last?.stopReason === "error" || last?.stopReason === "aborted") {
    throw new Error(
      last.errorMessage ?? `assistant stopped: ${last.stopReason}`,
    );
  }
}

async function runInit(vm: VM, initCode: string, envName: string) {
  log(`${envName}: running init.sh`);
  const proc = vm.exec(["/bin/bash", "-lc", initCode], {
    cwd: GUEST_WORKSPACE,
    env: guestEnv(),
    stdout: "pipe",
    stderr: "pipe",
  });
  for await (const chunk of proc.output()) Deno.stdout.writeSync(chunk.data);
  const result = await proc;
  if (!result.ok) {
    throw new Error(`${envName}: init.sh failed (${result.exitCode})`);
  }
}

async function runTest(
  model: string,
  prompt: string,
  workspacePath: string,
  outPath: string,
  initCode: string,
  options: RunTestOptions,
) {
  let networkEnabled = true;
  const { httpHooks, env } = createHttpHooks({
    allowedHosts: ["*"],
    isRequestAllowed: () => networkEnabled,
  });

  const workspaceProvider = new MemoryProvider();
  const vm = await VM.create({
    httpHooks,
    env,
    vfs: {
      mounts: {
        [GUEST_WORKSPACE]: workspaceProvider,
      },
    },
  });

  try {
    log(`${options.envName}: copying workspace`);
    await copyHostTreeToProvider(workspaceProvider, workspacePath, "/");
    await runInit(vm, initCode, options.envName);
    const before = await snapshotWorkspace(vm);

    networkEnabled = false;
    log(`${options.envName}: running ${model}`);
    await runAgent(vm, model, prompt, options.projectRoot, options.envName);

    const after = await snapshotWorkspace(vm);
    const tempOutPath = `${outPath}.tmp`;
    await rm(tempOutPath, { recursive: true, force: true });
    try {
      const count = await saveChangedFiles(vm, before, after, tempOutPath);
      await rename(tempOutPath, outPath);
      log(`${options.envName}: saved ${count} changed file(s) to ${outPath}`);
    } finally {
      await rm(tempOutPath, { recursive: true, force: true });
    }
  } finally {
    await vm.close();
  }
}

async function runJudge(testDir: string) {
  const judgePath = path.join(testDir, "judge.ts");
  if (!existsSync(judgePath)) throw new Error(`${judgePath} does not exist`);

  log(`${path.basename(testDir)}: running judge.ts`);
  const command = new Deno.Command(Deno.execPath(), {
    args: ["run", "--allow-all", judgePath],
    cwd: testDir,
    stdout: "piped",
    stderr: "piped",
  });
  const result = await command.output();
  const stdout = new TextDecoder().decode(result.stdout);
  const stderr = new TextDecoder().decode(result.stderr);
  if (!result.success) {
    throw new Error(`${judgePath} failed (${result.code})\n${stdout}${stderr}`);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout);
  } catch (err) {
    throw new Error(
      `${judgePath} did not print JSON: ${
        err instanceof Error ? err.message : String(err)
      }\n${stdout}${stderr}`,
    );
  }
  await writeFile(
    path.join(testDir, "results.json"),
    `${JSON.stringify(parsed, null, 2)}\n`,
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
  const projectRoot = path.dirname(fileURLToPath(import.meta.url));
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

    if (addedAny) {
      await runJudge(testDir);
      generatedPaths.push(path.join(testDir, "results.json"));
    }
  }

  await gitCommitGenerated(projectRoot, generatedPaths);
}

if (import.meta.main) await main();
