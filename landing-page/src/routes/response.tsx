import { createFileRoute } from "@tanstack/react-router";
import {
  Siren,
  Volume2,
  AlertTriangle,
  ShieldAlert,
  Radio,
  PhoneCall,
  Activity,
  Zap,
} from "lucide-react";
import { PageTransition, Reveal, TerminalLabel } from "../components/PageTransition";

export const Route = createFileRoute("/response")({
  head: () => ({
    meta: [
      { title: "Threat Response System — AI Street Safety Device Network" },
      {
        name: "description",
        content:
          "Two-tier autonomous response: localized acoustic/visual deterrence for low-severity threats, and sub-500ms API escalation to police and EMS for the rest.",
      },
      { property: "og:title", content: "Two-Tier Threat Response" },
      {
        property: "og:description",
        content:
          "Deterrable vs non-deterrable threats — handled at the edge in under half a second.",
      },
    ],
  }),
  component: ResponsePage,
});

function ResponsePage() {
  return (
    <PageTransition>
      {/* HEADER */}
      <section className="border-b border-iron py-20">
        <div className="mx-auto max-w-7xl px-6">
          <Reveal>
            <TerminalLabel>threat_response.md</TerminalLabel>
            <h1 className="max-w-4xl text-4xl text-paper md:text-6xl">
              Two tiers.
              <br />
              One proportional response.
            </h1>
            <p className="mt-6 max-w-3xl text-fog">
              Every detection runs through a two-stage classifier. The second
              stage decides which tier handles the threat — minimizing false
              dispatch while never under-reacting to severe events. Ambiguous
              cases default to escalation. Safety-first.
            </p>
          </Reveal>
        </div>
      </section>

      {/* TIER 1 */}
      <section className="border-b border-iron py-24">
        <div className="mx-auto grid max-w-7xl gap-12 px-6 lg:grid-cols-[1.1fr_1fr] lg:items-start">
          <Reveal>
            <article className="border-l-2 border-ember bg-graphite p-8">
              <div className="flex items-center gap-3">
                <span className="border border-ember px-2 py-0.5 text-xs uppercase tracking-widest text-ember">
                  tier_1
                </span>
                <span className="text-xs uppercase tracking-widest text-ash">
                  deterrable
                </span>
              </div>
              <h2 className="mt-4 text-3xl text-paper">
                Active On-Site Deterrence
              </h2>
              <p className="mt-4 text-fog">
                Wildlife intrusion, loitering, trespass. A localized ML
                sub-classifier verifies the entity, then triggers
                ML-generated acoustic and visual deterrents directly from the
                node — no human, no cloud, no dispatch.
              </p>

              <ul className="mt-6 space-y-2 text-sm">
                <SpecRow k="trigger" v="wildlife / loitering / trespass" />
                <SpecRow k="model" v="YOLOv8n + species classifier" />
                <SpecRow k="action" v="targeted-frequency speaker + IR strobe" />
                <SpecRow k="latency" v="~ 200 ms" />
                <SpecRow k="fallback" v="auto-escalate to tier_2 on timeout" />
              </ul>
            </article>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="space-y-4">
              <FlowStep icon={Activity} title="DETECT" body="Fused sensor input crosses confidence threshold." />
              <FlowStep icon={AlertTriangle} title="VERIFY" body="Sub-classifier confirms entity type, rejects false positives." />
              <FlowStep icon={Volume2} title="DETER" body="ML-generated acoustic burst at species-appropriate frequency." />
              <FlowStep icon={Siren} title="VISUAL" body="LED strobe + buzzer in parallel for redundancy." />
              <FlowStep icon={Zap} title="ESCALATE?" body="If target persists past timeout → tier_2 dispatch." last />
            </div>
          </Reveal>
        </div>
      </section>

      {/* TIER 2 */}
      <section className="border-b border-iron py-24">
        <div className="mx-auto grid max-w-7xl gap-12 px-6 lg:grid-cols-[1fr_1.1fr] lg:items-start">
          <Reveal>
            <div className="space-y-4">
              <FlowStep icon={Activity} title="DETECT" body="Severity classifier flags non-deterrable threat." />
              <FlowStep icon={ShieldAlert} title="PACKAGE" body="GPS, timestamped clip, classification, confidence." />
              <FlowStep icon={PhoneCall} title="DISPATCH" body="Structured API call to local police station." />
              <FlowStep icon={Radio} title="BROADCAST" body="Parallel notification to EMS + nearby residents (SMS / app)." />
              <FlowStep icon={Siren} title="PERSIST" body="Local deterrents remain active; full event logged." last />
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <article className="border-l-2 border-ember bg-graphite p-8 lg:order-first lg:hidden" />
            <article className="border-l-2 border-ember bg-graphite p-8">
              <div className="flex items-center gap-3">
                <span className="border border-ember px-2 py-0.5 text-xs uppercase tracking-widest text-ember">
                  tier_2
                </span>
                <span className="text-xs uppercase tracking-widest text-ash">
                  non-deterrable
                </span>
              </div>
              <h2 className="mt-4 text-3xl text-paper">
                Emergency Escalation
              </h2>
              <p className="mt-4 text-fog">
                Armed intrusion, robbery, assault, road accidents, medical
                emergencies. The node packages evidence and dispatches
                automated, structured alerts to multiple authorities in
                parallel — closing the 15–30 minute response gap.
              </p>

              <ul className="mt-6 space-y-2 text-sm">
                <SpecRow k="trigger" v="weapons / assault / accident / fire" />
                <SpecRow k="model" v="YOLOv8s + audio LSTM (screams, gunshots)" />
                <SpecRow k="action" v="MQTT + REST → police, EMS, locals" />
                <SpecRow k="latency" v="< 500 ms end-to-end" />
                <SpecRow k="payload" v="class, gps, snapshot, confidence" />
              </ul>
            </article>
          </Reveal>
        </div>
      </section>

      {/* COMPARISON TABLE */}
      <section className="py-24">
        <div className="mx-auto max-w-7xl px-6">
          <Reveal>
            <TerminalLabel>compare.tbl</TerminalLabel>
            <h2 className="text-3xl text-paper md:text-4xl">
              Tier 1 vs Tier 2
            </h2>
          </Reveal>
          <Reveal delay={0.05}>
            <div className="mt-10 overflow-x-auto border border-iron">
              <table className="w-full text-left text-sm">
                <thead className="bg-carbon text-xs uppercase tracking-widest text-steel">
                  <tr>
                    <th className="border-b border-iron p-4">attribute</th>
                    <th className="border-b border-l border-iron p-4 text-paper">
                      tier_1 · deterrable
                    </th>
                    <th className="border-b border-l border-iron p-4 text-paper">
                      tier_2 · non-deterrable
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ["trigger", "wildlife, loitering, trespass", "weapons, assault, accident, fire"],
                    ["model", "YOLOv8n + species/sub-class", "YOLOv8s + audio LSTM"],
                    ["action", "acoustic + visual deterrent", "API dispatch to authorities"],
                    ["network", "local only (in-node)", "MQTT mesh + LTE uplink"],
                    ["latency", "~ 200 ms", "< 500 ms"],
                    ["escalation", "auto → tier_2 on timeout", "logged + persisted"],
                  ].map(([a, b, c]) => (
                    <tr key={a} className="border-b border-iron last:border-b-0">
                      <td className="p-4 text-ember">{a}</td>
                      <td className="border-l border-iron p-4 text-fog">{b}</td>
                      <td className="border-l border-iron p-4 text-fog">{c}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>
        </div>
      </section>
    </PageTransition>
  );
}

function SpecRow({ k, v }: { k: string; v: string }) {
  return (
    <li className="flex flex-col gap-1 border-b border-iron pb-2 sm:flex-row sm:items-baseline sm:gap-4">
      <span className="w-24 shrink-0 text-xs uppercase tracking-widest text-steel">
        {k}
      </span>
      <span className="text-paper">{v}</span>
    </li>
  );
}

function FlowStep({
  icon: Icon,
  title,
  body,
  last,
}: {
  icon: typeof Activity;
  title: string;
  body: string;
  last?: boolean;
}) {
  return (
    <div className="relative">
      <div className="flex gap-4 border border-iron bg-carbon p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center border border-iron bg-graphite">
          <Icon className="h-4 w-4 text-ember" />
        </div>
        <div>
          <p className="text-xs uppercase tracking-widest text-ember">
            {title}
          </p>
          <p className="mt-1 text-sm text-fog">{body}</p>
        </div>
      </div>
      {!last && (
        <div className="ml-9 h-4 w-px bg-iron" aria-hidden />
      )}
    </div>
  );
}
