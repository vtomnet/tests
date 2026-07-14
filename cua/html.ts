export function escapeHtml(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function pct(value: number, total: number): string {
  return `${(value / total) * 100}%`;
}

export function assetPath(path: string): string {
  return `/files/${path.split("/").map(encodeURIComponent).join("/")}`;
}

export function formatNumber(
  value: number | null | undefined,
  places = 2,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(places);
}
