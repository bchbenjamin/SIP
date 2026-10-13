import "./lib/error-capture";

import { consumeLastCapturedError } from "./lib/error-capture";
import { renderErrorPage } from "./lib/error-page";

type ServerEntry = {
  fetch: (request: Request, env: unknown, ctx: unknown) => Promise<Response> | Response;
};

let serverEntryPromise: Promise<ServerEntry> | undefined;

async function getServerEntry(): Promise<ServerEntry> {
  if (!serverEntryPromise) {
    serverEntryPromise = import("@tanstack/react-start/server-entry").then(
      (m) => (m.default ?? m) as ServerEntry,
    );
  }
  return serverEntryPromise;
}

// h3 swallows in-handler throws into a normal 500 Response with body
// {"unhandled":true,"message":"HTTPError"} — try/catch alone never fires for those.
async function normalizeCatastrophicSsrResponse(response: Response): Promise<Response> {
  if (response.status < 500) return response;
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) return response;

  const body = await response.clone().text();
  if (!body.includes('"unhandled":true') || !body.includes('"message":"HTTPError"')) {
    return response;
  }

  console.error(consumeLastCapturedError() ?? new Error(`h3 swallowed SSR error: ${body}`));
  return new Response(renderErrorPage(), {
    status: 500,
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}

const BACKEND_URL = "https://api.sip.bchbenjamin.in/actuator/health";

async function handleCronWake(): Promise<Response> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

    const res = await fetch(BACKEND_URL, {
      signal: controller.signal,
      headers: { "User-Agent": "SIP-Cron/1.0" },
    });

    clearTimeout(timeout);

    if (!res.ok) {
      console.error(`[cron/wake] Backend returned ${res.status}`);
      return new Response(
        JSON.stringify({ ok: false, backendStatus: res.status }),
        { status: 502, headers: { "Content-Type": "application/json" } },
      );
    }

    console.log(`[cron/wake] Backend healthy — ${new Date().toISOString()}`);
    return new Response(
      JSON.stringify({ ok: true, at: new Date().toISOString() }),
      { headers: { "Content-Type": "application/json" } },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[cron/wake] Failed: ${msg}`);
    return new Response(
      JSON.stringify({ ok: false, error: msg }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }
}

export default {
  async fetch(request: Request, env: unknown, ctx: unknown) {
    // Vercel Cron: /api/cron/wake every 10 minutes keeps Render free-tier backend awake.
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/api/cron/wake") {
      return handleCronWake();
    }

    try {
      const handler = await getServerEntry();
      const response = await handler.fetch(request, env, ctx);
      return await normalizeCatastrophicSsrResponse(response);
    } catch (error) {
      console.error(error);
      return new Response(renderErrorPage(), {
        status: 500,
        headers: { "content-type": "text/html; charset=utf-8" },
      });
    }
  },
};
