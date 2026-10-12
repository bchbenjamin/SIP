// Vercel Serverless Function: runs at /api/cron/wake
// Triggered by Vercel Cron every 10 minutes (defined in vercel.json).
// Vercel injects headers and context — no extra deps needed.

export const dynamic = "force-dynamic";

const BACKEND_URL = "https://api.sip.bchbenjamin.in/actuator/health";

export async function GET(request: Request) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);

    const wakeRes = await fetch(BACKEND_URL, {
      signal: controller.signal,
      headers: { "User-Agent": "SIP-Cron/1.0" },
    });

    clearTimeout(timeout);

    if (!wakeRes.ok) {
      console.error(`[cron/wake] Backend returned ${wakeRes.status}`);
      return new Response(
        JSON.stringify({ ok: false, backendStatus: wakeRes.status }),
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
    console.error(`[cron/wake] Failed to reach backend: ${msg}`);
    return new Response(
      JSON.stringify({ ok: false, error: msg }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );
  }
}