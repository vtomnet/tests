import process from "node:process";
import { createInterface } from "node:readline/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { AuthStorage, ModelRegistry } from "@mariozechner/pi-coding-agent";

type AuthInfo = { url: string; instructions?: string };
type PromptInfo = {
  message: string;
  placeholder?: string;
  allowEmpty?: boolean;
};

function usage(providers: { id: string; name: string }[]): string {
  const list = providers.map((provider) =>
    `  ${provider.id.padEnd(24)} ${provider.name}`
  ).join("\n");
  return `Usage: deno task login <provider>\n\nOAuth providers:\n${list}\n`;
}

async function main() {
  const providerId = Deno.args[0];
  const projectRoot = path.dirname(fileURLToPath(import.meta.url));
  const agentDir = path.join(projectRoot, "pi.d");
  const authStorage = AuthStorage.create(path.join(agentDir, "auth.json"));
  const modelRegistry = ModelRegistry.create(
    authStorage,
    path.join(agentDir, "models.json"),
  );
  const modelError = modelRegistry.getError();
  if (modelError) throw new Error(modelError);

  const providers = authStorage.getOAuthProviders();
  if (!providerId || providerId === "-h" || providerId === "--help") {
    console.log(usage(providers));
    Deno.exit(providerId ? 0 : 1);
  }

  const provider = providers.find((candidate) => candidate.id === providerId);
  if (!provider) {
    console.error(`Unknown OAuth provider: ${providerId}\n`);
    console.error(usage(providers));
    Deno.exit(1);
  }

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const manualInputAbort = new AbortController();

  async function ask(info: PromptInfo): Promise<string> {
    const suffix = info.placeholder ? ` (${info.placeholder})` : "";
    const answer = await rl.question(`${info.message}${suffix}: `);
    if (!answer && !info.allowEmpty) throw new Error("empty input");
    return answer;
  }

  try {
    console.log(`Logging in to ${provider.name} (${provider.id})`);
    await authStorage.login(provider.id, {
      onAuth: (info: AuthInfo) => {
        console.log(`\nOpen this URL in your browser:\n${info.url}\n`);
        if (info.instructions) console.log(`${info.instructions}\n`);
      },
      onPrompt: ask,
      onProgress: (message: string) => console.log(message),
      onManualCodeInput: () =>
        rl.question(
          "Paste redirect URL/code here if browser callback does not complete: ",
          {
            signal: manualInputAbort.signal,
          },
        ),
    });
    console.log(
      `Logged in to ${provider.name}. Credentials saved to ${
        path.join(agentDir, "auth.json")
      }`,
    );
  } finally {
    manualInputAbort.abort();
    rl.close();
  }
}

if (import.meta.main) await main();
