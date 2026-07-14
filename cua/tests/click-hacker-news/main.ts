import puppeteer from "puppeteer";
import { dirname, fromFileUrl, join } from "@std/path";
import { complete, numericToolPoint, type Tool, Type } from "../../complete.ts";
import { distanceToBox, rounded } from "../../geometry.ts";
import { assetPath, escapeHtml, pct } from "../../html.ts";
import type { Box, CompleteCallRecord, Point, SavedRun } from "../../types.ts";

export const title = "Click Hacker News";
export const description =
  "Click HN titles, domains, and comments from a screenshot.";

const DIR = dirname(fromFileUrl(import.meta.url));
const SCREENSHOT = "screenshot.png";
const SCREENSHOT_PATH = join(DIR, SCREENSHOT);
const META_PATH = join(DIR, "articles.json");
const HN_URL = "https://news.ycombinator.com";
const CHROME = Deno.env.get("PUPPETEER_EXECUTABLE_PATH") ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const CLICK_TOOL = "click";

interface ArticleBoxes {
  title: Box;
  domain: Box | null;
  comments: Box | null;
}

interface Article {
  id: string;
  title: string;
  domain: string | null;
  boxes: ArticleBoxes;
}

interface HnMetadata {
  url: string;
  fetchedAt: string;
  screenshot: string;
  width: number;
  height: number;
  articles: Article[];
}

const clickTool: Tool = {
  name: CLICK_TOOL,
  description: "Click a point in the screenshot.",
  parameters: Type.Object({
    x: Type.Number({
      description: "x coordinate in screenshot pixels from the left edge",
    }),
    y: Type.Number({
      description: "y coordinate in screenshot pixels from the top edge",
    }),
  }),
};

async function readMetadata(): Promise<HnMetadata> {
  return JSON.parse(await Deno.readTextFile(META_PATH)) as HnMetadata;
}

function pngDimensions(bytes: Uint8Array): { width: number; height: number } {
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < signature.length; i += 1) {
    if (bytes[i] !== signature[i]) throw new Error("screenshot is not a PNG");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { width: view.getUint32(16), height: view.getUint32(20) };
}

async function ensureChrome(): Promise<void> {
  await Deno.stat(CHROME).catch(() => {
    throw new Error(
      `Chrome not found at ${CHROME}; set PUPPETEER_EXECUTABLE_PATH`,
    );
  });
}

export async function build(): Promise<void> {
  await ensureChrome();
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
    defaultViewport: { width: 1280, height: 900, deviceScaleFactor: 1 },
  });

  try {
    const page = await browser.newPage();
    await page.goto(HN_URL, { waitUntil: "networkidle2", timeout: 45_000 });
    await page.screenshot({ path: SCREENSHOT_PATH, fullPage: true });

    const dimensions = pngDimensions(await Deno.readFile(SCREENSHOT_PATH));
    const articles = await page.evaluate(() => {
      const doc = (globalThis as unknown as { document: any }).document;
      function box(
        element: any,
      ): { x: number; y: number; width: number; height: number } | null {
        if (!element) return null;
        const rect = element.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        return {
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height,
        };
      }

      return Array.from(doc.querySelectorAll("tr.athing")).map((row: any) => {
        const titleEl = row.querySelector(".titleline > a");
        const domainEl = row.querySelector(".sitestr");
        const subtext = row.nextElementSibling?.querySelector(".subtext") ||
          null;
        const commentsEl =
          Array.from(subtext?.querySelectorAll("a") || []).find((link: any) => {
            const href = link.getAttribute("href") || "";
            return href.startsWith("item?id=");
          }) || null;
        const titleBox = box(titleEl);
        if (!titleEl || !titleBox) return null;
        return {
          id: row.getAttribute("id") || "",
          title: titleEl.textContent?.trim() || "",
          domain: domainEl?.textContent?.trim() || null,
          boxes: {
            title: titleBox,
            domain: box(domainEl),
            comments: box(commentsEl),
          },
        };
      }).filter((article: any): article is {
        id: string;
        title: string;
        domain: string | null;
        boxes: {
          title: { x: number; y: number; width: number; height: number };
          domain:
            | { x: number; y: number; width: number; height: number }
            | null;
          comments:
            | { x: number; y: number; width: number; height: number }
            | null;
        };
      } => Boolean(article && article.title));
    });

    await Deno.writeTextFile(
      META_PATH,
      JSON.stringify(
        {
          url: HN_URL,
          fetchedAt: new Date().toISOString(),
          screenshot: SCREENSHOT,
          ...dimensions,
          articles,
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close();
  }
}

function randomInt(min: number, max: number): number {
  return min + Math.floor(Math.random() * (max - min + 1));
}

function shuffled<T>(values: T[]): T[] {
  const out = [...values];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = randomInt(0, i);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function pickTwo(articles: Article[], kind: keyof ArticleBoxes): Article[] {
  const candidates = articles.filter((article) => article.boxes[kind]);
  if (candidates.length < 2) {
    throw new Error(`not enough Hacker News articles with ${kind} boxes`);
  }
  return shuffled(candidates).slice(0, 2);
}

function promptFor(kind: keyof ArticleBoxes, article: Article): string {
  if (kind === "title") return `Click the article title: ${article.title}`;
  if (kind === "domain") {
    return `Click the domain shown next to the article title: ${article.title}`;
  }
  return `Click the comments link for the article title: ${article.title}`;
}

async function runClick(
  metadata: HnMetadata,
  kind: keyof ArticleBoxes,
  article: Article,
): Promise<number> {
  const targetBox = article.boxes[kind];
  if (!targetBox) throw new Error(`${article.title} has no ${kind} box`);

  const result = await complete({
    label: `hn ${kind} ${article.id}`,
    imagePath: SCREENSHOT_PATH,
    tools: [clickTool],
    requiredTool: CLICK_TOOL,
    systemPrompt:
      `You are controlling a browser from a ${metadata.width}x${metadata.height} screenshot. Coordinates are screenshot pixels with origin at the top-left corner. Always call ${CLICK_TOOL}; do not answer in text.`,
    prompt: promptFor(kind, article),
    meta: {
      kind: "click-hacker-news",
      targetKind: kind,
      articleId: article.id,
      title: article.title,
      domain: article.domain,
      targetBox,
      image: "tests/click-hacker-news/screenshot.png",
      width: metadata.width,
      height: metadata.height,
    },
  });

  return distanceToBox(numericToolPoint(result, CLICK_TOOL), targetBox);
}

export async function run(): Promise<number> {
  const metadata = await readMetadata();
  const tasks: Array<{ kind: keyof ArticleBoxes; article: Article }> = [];
  for (const kind of ["title", "domain", "comments"] as const) {
    for (const article of pickTwo(metadata.articles, kind)) {
      tasks.push({ kind, article });
    }
  }

  const missPenalty = Math.hypot(metadata.width, metadata.height);
  let score = 0;
  for (const task of tasks) {
    try {
      score += await runClick(metadata, task.kind, task.article);
    } catch {
      score += missPenalty;
    }
  }
  return rounded(score, 3);
}

function pointFromCall(call: CompleteCallRecord): Point | null {
  const toolCall = call.toolCalls.find((item) => item.name === CLICK_TOOL);
  if (!toolCall) return null;
  const x = Number(toolCall.arguments.x);
  const y = Number(toolCall.arguments.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

function boxOverlay(call: CompleteCallRecord, metadata: HnMetadata): string {
  const box = call.meta.targetBox as Box | undefined;
  const kind = String(call.meta.targetKind || "target");
  if (!box) return "";
  return `<span class="bench-box bench-box-${escapeHtml(kind)}" title="${
    escapeHtml(kind)
  } ${escapeHtml(call.meta.title || "")}" style="left:${
    pct(box.x, metadata.width)
  };top:${pct(box.y, metadata.height)};width:${
    pct(box.width, metadata.width)
  };height:${pct(box.height, metadata.height)}"></span>`;
}

function clickOverlay(call: CompleteCallRecord, metadata: HnMetadata): string {
  const point = pointFromCall(call);
  if (!point) return "";
  const kind = String(call.meta.targetKind || "click");
  return `<span class="bench-click bench-click-${escapeHtml(kind)}" title="${
    escapeHtml(kind)
  } click ${Math.round(point.x)},${Math.round(point.y)}" style="left:${
    pct(point.x, metadata.width)
  };top:${pct(point.y, metadata.height)}"></span>`;
}

function legend(run: SavedRun | undefined): string {
  const rows = (run?.calls || [])
    .filter((call) => call.meta.kind === "click-hacker-news")
    .map((call) => {
      const point = pointFromCall(call);
      const targetBox = call.meta.targetBox as Box | undefined;
      const score = point && targetBox ? distanceToBox(point, targetBox) : null;
      return `<li>${escapeHtml(call.meta.targetKind || "target")} · ${
        escapeHtml(call.meta.title || "")
      } · ${score === null ? "miss" : rounded(score, 2)}</li>`;
    });
  return rows.length ? `<ol class="bench-legend">${rows.join("")}</ol>` : "";
}

export async function view(run?: SavedRun): Promise<string> {
  const metadata = await readMetadata();
  const calls = (run?.calls || []).filter((call) =>
    call.meta.kind === "click-hacker-news"
  );
  return `
    <div class="bench-view bench-hn">
      <div class="bench-frame" style="width:${metadata.width}px">
        <img src="${
    assetPath("tests/click-hacker-news/screenshot.png")
  }" width="${metadata.width}" height="${metadata.height}" alt="Hacker News screenshot">
        <div class="bench-overlay">
          ${calls.map((call) => boxOverlay(call, metadata)).join("")}
          ${calls.map((call) => clickOverlay(call, metadata)).join("")}
        </div>
      </div>
      ${legend(run)}
    </div>`;
}
