// Vercel Serverless Function: keeps the backend awake on free Render tier.
// Vercel Cron triggers this every 10 minutes via vercel.json.
// Runs OUTSIDE the TanStack Start SSR layer — no external deps required.

export const dynamic = "force-dynamic";

const BACKEND_URL = "https://api.sip.bchbenjamin.in/actuator/health";

export async function GET() {
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