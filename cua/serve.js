import { JSON5 } from "bun";

const port = Number(process.env.PORT || 3000);
const apiKey = process.env.OPENROUTER_API_KEY || "";
const htmlPath = new URL("./index.html", import.meta.url);
const imagePath = new URL("./image.png", import.meta.url);
const modelsPath = new URL("./models.json5", import.meta.url);

function injectApiKey(html, key) {
  const payload = `<script>window.__OPENROUTER_API_KEY__ = ${JSON.stringify(key)};</script>`;
  return html.replace("<script>", `${payload}\n  <script>`);
}

const html = injectApiKey(await Bun.file(htmlPath).text(), apiKey);
const image = await Bun.file(imagePath).bytes();
const models = JSON5.parse(await Bun.file(modelsPath).text());
const modelsJson = JSON.stringify(models, null, 2);

const server = Bun.serve({
  port,
  fetch(req) {
    const { pathname } = new URL(req.url);

    if (pathname === "/") {
      return new Response(html, {
        headers: { "Content-Type": "text/html; charset=utf-8" }
      });
    }

    if (pathname === "/image.png") {
      return new Response(image, {
        headers: { "Content-Type": "image/png" }
      });
    }

    if (pathname === "/models.json") {
      return new Response(modelsJson, {
        headers: { "Content-Type": "application/json; charset=utf-8" }
      });
    }

    return new Response("not found\n", {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" }
    });
  }
});

console.log(`serving ${server.url.href}`);
