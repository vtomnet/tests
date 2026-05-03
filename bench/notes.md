Some notes about the tournament approach, like using gemini for tiebreaking, got lost. Recover
those.

TODO be conscious of prompt caching. maybe be clever about context building for judging

---

Junk ideas below. Doesn't really work. Main problem is recreating the environment is difficult or impossible. And producing a reliably correct north star for each problem is also probably difficult. But could maybe take inspiration from prompts sourced from HF at least, and go from there? Idk.

CrowdBench - pull from public agent sessions on huggingface, continually update a la LiveBench. make the whole thing as open as possible.
Also, you can create your own benchmark based on your own pi/opencode/codex/etc sessions.
Continually pull new data from new/updated huggingface datasets.
When a new model is released, deprecate questions that were publicly available at the time of its release.
Deprecate questions that were publicly available at the time a new model is released. Can be smart about this: track the model/coding agent used. If it's Claude, and the new model released is Claude, deprecate all prompts whose timestamp is 1 day prior to the Claude release. Likewise for GPT (depending on OpenAI's training policy?). Otherwise, deprecate prompts if they were *publicly shared* ≥1 day prior to a new model's release.
If we find this to be too aggressive, then make it e.g. 1 week instead of 1 day, or only deprecate the prompt if any models that may have been trained on it actually beat it, where their predecessor could not.
Probably also deprecate prompts when they get saturated (most/all models tested on them succeed, at pass@N). Are there any issues with doing this? Is this what LiveBench does?
//
> CrowdBench is a rolling release LLM benchmark derived from publicly available agent traces.
//
Aim for, idk, 200 prompts? Maybe 100 or 50 to start, in the interest of cost.
//
Need to detect prompts that look to be mostly pasted from something else. Examples: log messages, pasted data from a webpage, etc. Alternatively, classify and filter for messages that are *specifically* feature requests. Also need to filter out messages which depend on data we cannot access, such as if the prompt says to reference particular files... oh... that could be a problem. hm. we don't actually have reliable access to *any* files. Uhm. We can make educated guesses for some of them based on timestamps and pathnames, like assuming the obvious ~/Downloads/$REPO git clones and rewinding to the latest commit as of message timestamp, in my case. Otherwise, I think we just need a very good classifier process to detect whether the prompt relies on data we don't have access to. E.g., public repos, we could assume the prompt is starting from the latest commit in that repo, but that's not guaranteed.
We could consider just modifying the prompts a bit if they're mostly usable.
I suppose we could get smart and use read/grep outputs to estimate whether the repo matches the expected commit. If it does, then keep.
Must also throw out any prompts that would be expensive to run due to other APIs, such as those that involve using Vast API, ASR or LLM API, etc.
Should also decide whether questions are internally coherent. Throw out if self-contradictory or nonsensical.
Also need to throw out many prompts that require web search, since web data will have changed. (Unless we can potentially limit search queries to only older data?)
How many prompts does this leave us with?
And need to figure out if LLM can even be a reliable judge in the first place, without a human setting the intended outcome. If a problem is difficult for all the LLMs, then it's probably impossible for them to reliably judge the results.
//
Looking forward, this could mostly be solved with a pi extension that essentially just does `git diff` on every invocation and stores this somewhere. Plus maybe stores full file reads. Basically do everything we can to make the environment fully reproduceable.


