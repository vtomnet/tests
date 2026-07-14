// @deno-types="npm:@types/pngjs@6.0.5"
import { PNG } from "pngjs";
import { dirname, fromFileUrl, join } from "@std/path";
import { complete, numericToolPoint, type Tool, Type } from "../../complete.ts";
import { distance, rounded } from "../../geometry.ts";
import { assetPath, escapeHtml, pct } from "../../html.ts";
import type { CompleteCallRecord, Point, SavedRun } from "../../types.ts";

export const title = "Pinpoint dot simple";
export const description =
  "Find a requested 3px dot by color in synthetic images.";

const DIR = dirname(fromFileUrl(import.meta.url));
const META_PATH = join(DIR, "dots.json");
const WIDTH = 1024;
const HEIGHT = 768;
const IMAGE_COUNT = 10;
const RADIUS = 3;
const MIN_DOT_DISTANCE = 10;
const COORD_TOOL = "return_coords";

interface Dot extends Point {
  id: number;
  color: string;
  radius: number;
}

interface DotImage {
  id: number;
  file: string;
  background: string;
  dots: Dot[];
}

interface DotMetadata {
  width: number;
  height: number;
  radius: number;
  images: DotImage[];
}

const coordTool: Tool = {
  name: COORD_TOOL,
  description:
    "Return the center coordinates of the requested dot in image pixels.",
  parameters: Type.Object({
    x: Type.Number({
      description: "x coordinate in pixels from the left edge",
    }),
    y: Type.Number({ description: "y coordinate in pixels from the top edge" }),
  }),
};

function randomInt(min: number, max: number): number {
  return min + Math.floor(Math.random() * (max - min + 1));
}

function randomHexByte(): string {
  return randomInt(0, 255).toString(16).padStart(2, "0");
}

function randomColor(): string {
  return `#${randomHexByte()}${randomHexByte()}${randomHexByte()}`;
}

function rgb(color: string): [number, number, number] {
  return [1, 3, 5].map((offset) =>
    parseInt(color.slice(offset, offset + 2), 16)
  ) as [number, number, number];
}

function colorDistance(a: string, b: string): number {
  const aa = rgb(a);
  const bb = rgb(b);
  return Math.hypot(aa[0] - bb[0], aa[1] - bb[1], aa[2] - bb[2]);
}

function uniqueDotColor(used: Set<string>, background: string): string {
  while (true) {
    const color = randomColor();
    if (!used.has(color) && colorDistance(color, background) > 80) return color;
  }
}

function pointFarEnough(point: Point, dots: Dot[]): boolean {
  return dots.every((dot) => distance(point, dot) >= MIN_DOT_DISTANCE);
}

function randomDot(
  id: number,
  dots: Dot[],
  usedColors: Set<string>,
  background: string,
): Dot {
  while (true) {
    const point = {
      x: randomInt(RADIUS + 1, WIDTH - RADIUS - 2),
      y: randomInt(RADIUS + 1, HEIGHT - RADIUS - 2),
    };
    if (!pointFarEnough(point, dots)) continue;
    const color = uniqueDotColor(usedColors, background);
    usedColors.add(color);
    return { id, ...point, color, radius: RADIUS };
  }
}

function setPixel(png: PNG, x: number, y: number, color: string): void {
  const [r, g, b] = rgb(color);
  const offset = (y * png.width + x) * 4;
  png.data[offset] = r;
  png.data[offset + 1] = g;
  png.data[offset + 2] = b;
  png.data[offset + 3] = 255;
}

function fill(png: PNG, color: string): void {
  for (let y = 0; y < png.height; y += 1) {
    for (let x = 0; x < png.width; x += 1) setPixel(png, x, y, color);
  }
}

function drawDot(png: PNG, dot: Dot): void {
  for (let y = dot.y - dot.radius; y <= dot.y + dot.radius; y += 1) {
    for (let x = dot.x - dot.radius; x <= dot.x + dot.radius; x += 1) {
      if (x < 0 || x >= png.width || y < 0 || y >= png.height) continue;
      if (Math.hypot(x - dot.x, y - dot.y) <= dot.radius) {
        setPixel(png, x, y, dot.color);
      }
    }
  }
}

async function writeImage(image: DotImage): Promise<void> {
  const png = new PNG({ width: WIDTH, height: HEIGHT });
  fill(png, image.background);
  for (const dot of image.dots) drawDot(png, dot);
  await Deno.writeFile(join(DIR, image.file), PNG.sync.write(png));
}

async function readMetadata(): Promise<DotMetadata> {
  return JSON.parse(await Deno.readTextFile(META_PATH)) as DotMetadata;
}

function choose<T>(values: T[]): T {
  return values[randomInt(0, values.length - 1)];
}

export async function build(): Promise<void> {
  const images: DotImage[] = [];
  for (let i = 0; i < IMAGE_COUNT; i += 1) {
    const background = randomColor();
    const usedColors = new Set<string>();
    const dots: Dot[] = [];
    const dotCount = randomInt(7, 18);
    for (let dotId = 0; dotId < dotCount; dotId += 1) {
      dots.push(randomDot(dotId, dots, usedColors, background));
    }

    const image = {
      id: i,
      file: `image_${String(i).padStart(2, "0")}.png`,
      background,
      dots,
    };
    await writeImage(image);
    images.push(image);
  }

  await Deno.writeTextFile(
    META_PATH,
    JSON.stringify(
      { width: WIDTH, height: HEIGHT, radius: RADIUS, images },
      null,
      2,
    ),
  );
}

async function runOne(image: DotImage, dot: Dot): Promise<number> {
  const result = await complete({
    label: `pinpoint ${image.file} ${dot.color}`,
    imagePath: join(DIR, image.file),
    tools: [coordTool],
    requiredTool: COORD_TOOL,
    systemPrompt:
      `You are measuring a ${WIDTH}x${HEIGHT} image. Coordinates are pixel coordinates with origin at the top-left corner. Always call ${COORD_TOOL}; do not answer in text.`,
    prompt:
      `Find the center of the dot whose color is exactly ${dot.color}. Call ${COORD_TOOL} with its x and y coordinates.`,
    meta: {
      kind: "pinpoint-dot-simple",
      imageIndex: image.id,
      image: `tests/pinpoint-dot-simple/${image.file}`,
      width: WIDTH,
      height: HEIGHT,
      target: dot,
    },
  });
  return distance(numericToolPoint(result, COORD_TOOL), dot);
}

export async function run(): Promise<number> {
  const metadata = await readMetadata();
  let score = 0;
  const missPenalty = Math.hypot(metadata.width, metadata.height);

  for (const image of metadata.images) {
    const dot = choose(image.dots);
    try {
      score += await runOne(image, dot);
    } catch {
      score += missPenalty;
    }
  }

  return rounded(score, 3);
}

function pointFromCall(call: CompleteCallRecord): Point | null {
  const toolCall = call.toolCalls.find((item) => item.name === COORD_TOOL);
  if (!toolCall) return null;
  const x = Number(toolCall.arguments.x);
  const y = Number(toolCall.arguments.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

function callsByImage(
  run: SavedRun | undefined,
): Map<number, CompleteCallRecord> {
  const out = new Map<number, CompleteCallRecord>();
  for (const call of run?.calls || []) {
    if (call.meta.kind !== "pinpoint-dot-simple") continue;
    const imageIndex = Number(call.meta.imageIndex);
    if (Number.isInteger(imageIndex)) out.set(imageIndex, call);
  }
  return out;
}

function dotOverlay(dot: Dot, width: number, height: number): string {
  return `<span class="bench-dot" title="${
    escapeHtml(dot.color)
  } ${dot.x},${dot.y}" style="left:${pct(dot.x, width)};top:${
    pct(dot.y, height)
  };border-color:${escapeHtml(dot.color)}"></span>`;
}

function predictionOverlay(
  point: Point,
  width: number,
  height: number,
): string {
  return `<span class="bench-click" title="model ${Math.round(point.x)},${
    Math.round(point.y)
  }" style="left:${pct(point.x, width)};top:${pct(point.y, height)}"></span>`;
}

function targetOverlay(dot: Dot, width: number, height: number): string {
  return `<span class="bench-target" title="target ${dot.color} ${dot.x},${dot.y}" style="left:${
    pct(dot.x, width)
  };top:${pct(dot.y, height)}"></span>`;
}

export async function view(run?: SavedRun): Promise<string> {
  const metadata = await readMetadata();
  const calls = callsByImage(run);
  const cards = metadata.images.map((image) => {
    const call = calls.get(image.id);
    const target = call?.meta.target as Dot | undefined;
    const point = call ? pointFromCall(call) : null;
    const dist = point && target ? distance(point, target) : null;
    return `
      <figure class="bench-figure">
        <div class="bench-frame" style="width:${metadata.width}px">
          <img src="${
      assetPath(`tests/pinpoint-dot-simple/${image.file}`)
    }" width="${metadata.width}" height="${metadata.height}" alt="${
      escapeHtml(image.file)
    }">
          <div class="bench-overlay">
            ${
      image.dots.map((dot) => dotOverlay(dot, metadata.width, metadata.height))
        .join("")
    }
            ${
      target ? targetOverlay(target, metadata.width, metadata.height) : ""
    }
            ${
      point ? predictionOverlay(point, metadata.width, metadata.height) : ""
    }
          </div>
        </div>
        <figcaption>${escapeHtml(image.file)}${
      target
        ? ` · target ${escapeHtml(target.color)} · dist ${
          rounded(dist || 0, 2)
        }`
        : ""
    }</figcaption>
      </figure>`;
  });

  return `<div class="bench-view bench-pinpoint">${cards.join("")}</div>`;
}
