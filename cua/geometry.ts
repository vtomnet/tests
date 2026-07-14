import type { Box, Point } from "./types.ts";

export function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function boxCenter(box: Box): Point {
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

export function pointInBox(point: Point, box: Box): boolean {
  return point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y &&
    point.y <= box.y + box.height;
}

export function distanceToBox(point: Point, box: Box): number {
  if (pointInBox(point, box)) return 0;
  const x = Math.max(box.x, Math.min(point.x, box.x + box.width));
  const y = Math.max(box.y, Math.min(point.y, box.y + box.height));
  return distance(point, { x, y });
}

export function rounded(value: number, places = 3): number {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}
