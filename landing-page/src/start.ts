import { createStart } from "@tanstack/react-start";

const BACKEND_URL = "https://api.sip.bchbenjamin.in/actuator/health";

export const startInstance = createStart(() => ({
  requestMiddleware: [
    {
      type: "request",
      server: async ({ next, request }) => {
        const url = new URL(request.url);
        if (request.method === "GET" && url.pathname === "/api/cron/wake") {
          try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 8000);
            const res = await fetch(BACKEND_URL, {
              signal: controller.signal,
              headers: { "User-Agent": "SIP-Cron/1.0" },
            });
            clearTimeout(timeout);
            if (!res.ok) {
              return new Response(
                JSON.stringify({ ok: false, backendStatus: res.status }),
                { status: 502, headers: { "Content-Type": "application/json" } },
              );
            }
            return new Response(
              JSON.stringify({ ok: true, at: new Date().toISOString() }),
              { headers: { "Content-Type": "application/json" } },
            );
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            return new Response(
              JSON.stringify({ ok: false, error: msg }),
              { status: 503, headers: { "Content-Type": "application/json" } },
            );
          }
        }
        return next();
      },
    },
  ],
}));