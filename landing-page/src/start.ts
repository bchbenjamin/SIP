import { createStart, createMiddleware } from "@tanstack/react-start";

import { renderErrorPage } from "./lib/error-page";

const BACKEND_URL = "https://api.sip.bchbenjamin.in/actuator/health";
const CRON_SECRET = process.env.CRON_SECRET ?? "sip-cron-secret-change-me";

const cronWakeMiddleware = createMiddleware().server(async ({ next, request }) => {
  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/api/cron/wake") {
    const authHeader = request.headers.get("authorization") ?? "";
    if (authHeader !== `Bearer ${CRON_SECRET}`) {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      const wakeRes = await fetch(BACKEND_URL, {
        signal: controller.signal,
        headers: { "User-Agent": "SIP-Cron/1.0" },
      });
      clearTimeout(timeout);
      if (!wakeRes.ok) {
        return new Response(JSON.stringify({ ok: false, backendStatus: wakeRes.status }), {
          status: 502,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify({ ok: true, at: new Date().toISOString() }), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return new Response(JSON.stringify({ ok: false, error: msg }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }
  }
  return next();
});

const errorMiddleware = createMiddleware().server(async ({ next }) => {
  try {
    return await next();
  } catch (error) {
    if (error != null && typeof error === "object" && "statusCode" in error) {
      throw error;
    }
    console.error(error);
    return new Response(renderErrorPage(), {
      status: 500,
      headers: { "Content-Type": "text/html; charset=utf-8" },
    });
  }
});

export const startInstance = createStart(() => ({
  requestMiddleware: [cronWakeMiddleware, errorMiddleware],
}));
