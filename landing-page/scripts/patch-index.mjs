/**
 * Patch index.mjs after build: inject the SSR service (ssr.mjs) into the
 * h3App request pipeline so TanStack Start SSR actually renders pages.
 *
 * Without this patch, index.mjs routes ALL requests to the renderer template
 * (renderIndexHTML), which returns raw index.html with %head% placeholders.
 * With this patch, the h3App fetcher first checks if a request should be
 * handled by the SSR service, and if so, invokes it directly.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const indexPath = resolve(__dirname, "../.vercel/output/functions/__server.func/index.mjs");

let indexContent;
try {
  indexContent = readFileSync(indexPath, "utf8");
} catch {
  // File doesn't exist (dev mode, etc.) — skip patch
  console.log("[patch-index] index.mjs not found, skipping");
  process.exit(0);
}

// Only patch if not already patched (check for our injection marker)
if (indexContent.includes("__nitro_ssr_handler__")) {
  console.log("[patch-index] Already patched, skipping");
  process.exit(0);
}

// Inject: add ssr service reference and modify appHandler to call it
const patch = `
// --- SIP patch: invoke SSR service before renderer ---
var __nitro_ssr_lazy = { fetch(req) { return import("./_ssr/ssr.mjs").then(m => m.default.fetch(req)); } };
async function __nitro_ssr_handler(req) {
  var url = req.url ? new URL(req.url) : null;
  if (url && req.method === "GET" && url.pathname === "/api/cron/wake") {
    return await __nitro_ssr_lazy.fetch(req);
  }
  return null; // fall through to h3App
}
// --- end SIP patch ---
`;

const oldHandler = `	let appHandler = (req) => {
		req.context ||= {};
		req.context.nitro = req.context.nitro || { errors: [] };
		return h3App.fetch(req);
	};`;

const newHandler = `	let appHandler = async (req) => {
		req.context ||= {};
		req.context.nitro = req.context.nitro || { errors: [] };
		// SIP patch: check SSR service first for specific routes
		var ssrResult = await __nitro_ssr_handler(req);
		if (ssrResult) return ssrResult;
		return h3App.fetch(req);
	};`;

if (!indexContent.includes(oldHandler)) {
  console.log("[patch-index] Could not find appHandler pattern, skipping");
  process.exit(0);
}

const patched = patch + indexContent.replace(oldHandler, newHandler);
writeFileSync(indexPath, patched, "utf8");
console.log("[patch-index] Patched index.mjs to invoke SSR service");