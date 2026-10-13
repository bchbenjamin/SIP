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

// Cron handler: keep the Render free-tier backend awake.
// Retries once after 5s if the first attempt times out.
async function handleCronWake(): Promise<Response> {
  const BACKEND_URL = "https://api.sip.bchbenjamin.in/actuator/health";
  const TIMEOUT_MS = 15000;
  async function ping(): Promise<{ ok: boolean; status?: number; error?: string }> {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
      const res = await fetch(BACKEND_URL, {
        signal: controller.signal,
        headers: { "User-Agent": "SIP-Cron/1.0" },
      });
      clearTimeout(timer);
      return { ok: res.ok, status: res.status };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { ok: false, error: msg };
    }
  }
  let result = await ping();
  if (!result.ok && result.error?.includes("aborted")) {
    // Backend may have been asleep; retry once after a short delay
    await new Promise((r) => setTimeout(r, 5000));
    result = await ping();
  }
  if (!result.ok) {
    return new Response(
      JSON.stringify({ ok: false, backendStatus: result.status, error: result.error }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }
  return new Response(
    JSON.stringify({ ok: true, at: new Date().toISOString() }),
    { headers: { "Content-Type": "application/json" } },
  );
}

export default {
  async fetch(request: Request, env: unknown, ctx: unknown) {
    const url = new URL(request.url);

    // Cron wake — handled at entry level before SSR routing
    if (request.method === "GET" && url.pathname === "/api/cron/wake") {
      return handleCronWake();
    }

    // Health check — plain text so we know the function is running
    if (request.method === "GET" && url.pathname === "/api/health") {
      return new Response(
        JSON.stringify({
          function: "ok",
          time: new Date().toISOString(),
          url: request.url,
          pathname: url.pathname,
        }),
        { headers: { "Content-Type": "application/json" } },
      );
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