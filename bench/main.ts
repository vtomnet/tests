// MODELS = [
//   "openai-codex/gpt-5.5:high",
//   "opencode-go/kimi-k2.6:high",
//   "opencode-go/glm-5.1:high",
// ];

// unsure if any of these params should have richer types
// model name follows syntax `provider_name/model_name:think_level`
async function runTest(model: string, prompt: string, workspacePath: string, outPath: string, initCode: string) {
  // init gondolin vm
  // mount workspacePath as /workspace (ephemeral, e.g. copied into a MemoryProvider)
  // eval initCode. should probably allow all network access for the init code, then disallow
  // network reqs for the agent.

  // init pi agent
  //
  // give it 4 tools: read, write, edit, bash. modify the tools to execute inside the vm, similar
  // to <https://github.com/earendil-works/gondolin/blob/main/host/examples/pi-gondolin.ts>. Note,
  // we don't need everything from that; e.g., we don't need to route '!' commands to the vm, since
  // we're not running pi interactively.
  //
  // use a custom resourceloader to avoid accidentally reading skills and such we don't mean to.
  // should use at least ${agentDir}/sessions/, ${agentDir}/auth.json
  //
  // run agent with prompt

  // track all modified/created files in the VM (either via gondolin hooks if it gives them, or
  // diff at the end) and write modified/created files into outPath. Preserve directory structure,
  // e.g. if /workspace/vendor/ghostty/src/main.zig is modified, then it should be saved to
  // ${outPath}/workspace/vendor/ghostty/src/main.zig. Only track modified/created files in
  // /workspace.
}

// for (testDir in ${scriptDir}/envs/*) {
//   read ${testDir}/manifest.json. fields: "prompt"
//   initCode = readFile(`${testDir}/init.sh`);
//   for (model in MODELS) {
//     // the '0' is included b/c we may want to do best-of-n or similar later
//     const outPath = `${testDir}/out/${model.replace("/", "--")}/0/`;
//     if outPath does not exist {
//       runTest(model, prompt, `${testDir}/workspace`, outPath, initCode);
//     }
//   }
//   if any outPaths added {
//     run ${testDir}/judge.ts
//     write returned results to ${testDir}/results.json
//   }
// }
// make an automated git commit adding all newly created/updated outPath's and results.json's

// will later add a ${scriptDir}/viz.html or something to browse results. In the results viewer,
// should show (with pretty graphs?) info such as token usage, wall clock time, cost details, tool
// calls, precise HTTP requests/responses (with keys redacted), etc. Can go a step further and do
// pareto frontier, most attractive quandrant (a la ArtificialAnalysis), etc. Also, put this
// visualizer on a website eventually. Maybe automate with Github Actions eventually.

// still need to brainstorm structure for results.json, and approach for judge.ts.
//
// The "LLM council" will be one-shot GPT-5.5, Opus-4.7, and Gemini-3.1-Pro. Tournament style: LLMs
// are paired together, judged two at a time. GPT and Claude are each asked to rank the two: A
// better than B, B better than A, both good, or both bad. If they agree, the winner is decided. If
// they disagree, Gemini-3.1-Pro is called with those two options, to break the tie. This keeps
// judges' context length minimal, gives nice "buckets" of performance, and makes it cheap to
// evaluate single new LLMs, by only needing to compare them to ~≤50% of models previously judged.
// Can use binary search, starting with comparing the new model to a mid-tier model, and going from
// there to determine its ranking.
//
// ISSUE: If framing as "pick which one to commit," the one that is more complete and complex will
// always win over the one that is simpler but less complete. Not sure this is a good reward
// function. Prompt to judges is open-ended; can specify that the solution is binary (pass/fail,
// should bias toward reject both or accept both), or that it is nuanced and the result can be
// scalar. Should give the judges all of the edited/created files for context, plus at least the
// most notable project files that were given to the agents. E.g., the main.ts file in the
// silero-js-port env. Maybe run this file as well to see if the written code even works, and
// compare to expected result for correctness, and include the result of this test in the judges'
// context.
//
// when relevant, prompt to agent should specify a way to evaluate the code it writes. e.g., in the
// silero js port env, it might be provided a main.js file that shows what the interface should
// look like (imports VadIterator or whatever and calls that), and the LLM can use this to test
// that its code is correct. Prompt should encourage this with e.g. "Test your code to make sure it
// works" or similar.
//
// Consider single-elimination tournament, double-elimination tournament, or [swiss-system
// tournament](https://en.wikipedia.org/wiki/Swiss-system_tournament) as an alternative to
// pass@N/best-of-N; it's not really clear to me how traditional best-of-N would work with a
// tournament system. The alternative is not to compare the agents directly in a tournament style
// at all, but rather to just assign them a score in isolation, leaderboard-style. I don't see how
// this would work with scalar problems such as the silero-js-port env and LLM as judge, since the
// LLM cannot reliably judge consistently across runs.

// manifest.json should later include permission scoping for web access, etc.

// To be explicit about file structure, here is an example:
// /          # project root, i.e. ~/tests/bench. is git root
//   main.ts  # runs the vm, pi, handles git snapshots of results, prints results, etc.
//   pi.d/    # where pi agent data is stored
//     sessions/
//     auth.json
//   envs/    # environments for tests. each subdir is one test
//     example-env/
//       manifest.json      # for now only specifies prompt. will include more settings in future
//       init.sh            # sets up the environment. installs dependencies and whatnot.
//       judge.ts           #
//       workspace/...      # files to be included in the VM
//       out/(model)/0/...  # files that the agent edited/wrote in the VM
