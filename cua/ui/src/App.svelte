<script lang="ts">
  import { onMount } from "svelte";
  import Button from "./lib/components/ui/Button.svelte";
  import Card from "./lib/components/ui/Card.svelte";
  import Input from "./lib/components/ui/Input.svelte";
  import Badge from "./lib/components/ui/Badge.svelte";

  type RunStats = {
    calls: number;
    errors: number;
    errorRate: number;
    durationMs: number;
    cost: number;
    avgTtftMs: number | null;
    avgTps: number | null;
  };

  type SavedRun = {
    test: string;
    run: number;
    model: string;
    score: number | null;
    error: string | null;
    startedAt: string;
    finishedAt: string;
    durationMs: number;
    stats: RunStats;
    view?: string;
  };

  type TestSummary = {
    id: string;
    title: string;
    description: string;
    latestRun: SavedRun | null;
    latestView: string;
  };

  type TestDetail = {
    id: string;
    title: string;
    description: string;
    runs: SavedRun[];
    emptyView: string | null;
  };

  let model = "";
  let tests: TestSummary[] = [];
  let selected = "";
  let detail: TestDetail | null = null;
  let busy = "";
  let status = "";
  let error = "";

  $: detailRuns = detail ? [...detail.runs].reverse() : [];

  function formatMs(value: number | null | undefined): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    if (value < 1000) return `${value.toFixed(0)}ms`;
    return `${(value / 1000).toFixed(2)}s`;
  }

  function formatNumber(value: number | null | undefined, places = 2): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return value.toFixed(places);
  }

  function formatCost(value: number | null | undefined): string {
    if (value === null || value === undefined || !Number.isFinite(value)) return "—";
    return `$${value.toFixed(5)}`;
  }

  function statsText(run: SavedRun | null): string {
    if (!run) return "No runs yet.";
    return `${run.stats.calls} calls · ${run.stats.errors} errors · ${formatMs(run.durationMs)} · ${formatCost(run.stats.cost)} · TTFT ${formatMs(run.stats.avgTtftMs)} · TPS ${formatNumber(run.stats.avgTps, 1)}`;
  }

  async function api(path: string, init?: RequestInit) {
    const response = await fetch(path, init);
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload?.error?.message || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async function loadTests() {
    const payload = await api("/api/tests");
    tests = payload.tests;
    if (!model) model = localStorage.getItem("cua-bench-model") || payload.model || "";
  }

  async function openTest(id: string) {
    selected = id;
    detail = null;
    const payload = await api(`/api/tests/${encodeURIComponent(id)}`);
    detail = payload.test;
  }

  async function refreshDetail() {
    if (!selected) return;
    const payload = await api(`/api/tests/${encodeURIComponent(selected)}`);
    detail = payload.test;
  }

  async function rebuild(id: string) {
    busy = `building ${id}`;
    status = busy;
    error = "";
    try {
      await api(`/api/tests/${encodeURIComponent(id)}/build`, { method: "POST" });
      status = `built ${id}`;
      await loadTests();
      if (selected === id) await refreshDetail();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      busy = "";
    }
  }

  async function newRun(id: string) {
    busy = `running ${id}`;
    status = busy;
    error = "";
    localStorage.setItem("cua-bench-model", model);
    try {
      await api(`/api/tests/${encodeURIComponent(id)}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });
      status = `ran ${id}`;
      await loadTests();
      if (selected === id) await refreshDetail();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      busy = "";
    }
  }

  onMount(() => {
    model = localStorage.getItem("cua-bench-model") || "";
    loadTests().catch((err) => {
      error = err instanceof Error ? err.message : String(err);
    });
  });
</script>

<header class="topbar">
  <div>
    <h1>CUA Benchmark Suite</h1>
    <p>Lower scores are better.</p>
  </div>
  <div class="model-row">
    <label for="model">model</label>
    <Input id="model" bind:value={model} placeholder="openai/gpt-5.5" />
    <Button variant="secondary" disabled={Boolean(busy)} on:click={loadTests}>Refresh</Button>
  </div>
</header>

{#if status || error}
  <div class:error={Boolean(error)} class="status">{error || status}</div>
{/if}

{#if selected}
  <main class="page">
    <div class="detail-head">
      <Button variant="ghost" on:click={() => { selected = ""; detail = null; }}>← All tests</Button>
      {#if detail}
        <div class="detail-title">
          <h2>{detail.title}</h2>
          <p>{detail.description}</p>
        </div>
        <div class="actions">
          <Button variant="secondary" disabled={Boolean(busy)} on:click={() => rebuild(detail!.id)}>Rebuild</Button>
          <Button disabled={Boolean(busy)} on:click={() => newRun(detail!.id)}>New run</Button>
        </div>
      {/if}
    </div>

    {#if !detail}
      <p>Loading…</p>
    {:else if detailRuns.length === 0}
      <Card>
        <h3>No runs yet</h3>
        {#if detail.emptyView}<div class="view-html">{@html detail.emptyView}</div>{/if}
      </Card>
    {:else}
      {#each detailRuns as run}
        <Card class="run-card">
          <div class="run-head">
            <div>
              <h3>run_{run.run}</h3>
              <p>{new Date(run.startedAt).toLocaleString()} · {run.model}</p>
            </div>
            <div class="score">
              <Badge tone={run.error ? "bad" : "default"}>score {formatNumber(run.score, 3)}</Badge>
            </div>
          </div>
          {#if run.error}<p class="run-error">{run.error}</p>{/if}
          <p class="muted">{statsText(run)}</p>
          {#if run.view}<div class="view-html">{@html run.view}</div>{/if}
        </Card>
      {/each}
    {/if}
  </main>
{:else}
  <main class="page grid">
    {#each tests as test}
      <Card>
        <div class="card-head">
          <button class="title-button" on:click={() => openTest(test.id)}>{test.title}</button>
          <Badge tone={test.latestRun?.error ? "bad" : "muted"}>{test.id}</Badge>
        </div>
        <p>{test.description}</p>
        <div class="actions">
          <Button variant="secondary" disabled={Boolean(busy)} on:click={() => rebuild(test.id)}>Rebuild</Button>
          <Button disabled={Boolean(busy)} on:click={() => newRun(test.id)}>New run</Button>
        </div>
        <div class="latest">
          <p><strong>Latest:</strong> {test.latestRun ? `run_${test.latestRun.run} · score ${formatNumber(test.latestRun.score, 3)}` : "none"}</p>
          <p class="muted">{statsText(test.latestRun)}</p>
          {#if test.latestRun?.error}<p class="run-error">{test.latestRun.error}</p>{/if}
        </div>
        <div class="view-html preview">{@html test.latestView}</div>
      </Card>
    {/each}
  </main>
{/if}
