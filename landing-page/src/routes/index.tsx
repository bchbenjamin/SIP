import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpRight, Cpu, Radio, ShieldAlert } from "lucide-react";
import { PageTransition, Reveal, TerminalLabel } from "../components/PageTransition";

const DRIVE_URL =
  "https://drive.google.com/drive/folders/1-srQ-2YbDc3Ptqc_WZ5A1padu-PmNhae";

const HERO_IMG =
  "https://images.unsplash.com/photo-1573164574511-73c773193279?auto=format&fit=crop&w=1920&q=80";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Proactive Edge-AI Street Safety — AI Street Safety Device Network" },
      {
        name: "description",
        content:
          "A vandalism-resilient mesh of edge-AI nodes that detects, deters, and dispatches — replacing passive CCTV with active street safety.",
      },
      {
        property: "og:title",
        content: "Proactive Edge-AI Street Safety",
      },
      {
        property: "og:description",
        content:
          "Multi-modal sensing + two-tier autonomous response, deployed as a fault-tolerant mesh.",
      },
      { property: "og:image", content: HERO_IMG },
    ],
  }),
  component: HomePage,
});

const stats = [
  { k: "25%", v: "chain-snatcher apprehension rate · India 2025" },
  { k: "15–30 min", v: "emergency response in isolated zones" },
  { k: "< 500 ms", v: "sensor→response latency target (edge)" },
];

const sdgs = [
  {
    code: "SDG 12",
    title: "Law Enforcement & Governance",
    desc: "Automated, evidence-backed alerts to local police; monitors public infrastructure.",
  },
  {
    code: "SDG 06",
    title: "Disaster Management & Emergency",
    desc: "Real-time early warning for fires, accidents, mass-casualty events.",
  },
  {
    code: "SDG 08",
    title: "Urban Structure & Transport",
    desc: "Detects road accidents early; protects petrol stations, transit corridors.",
  },
];

function HomePage() {
  return (
    <PageTransition>
      {/* HERO */}
      <section className="relative min-h-[92vh] overflow-hidden border-b border-iron">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${HERO_IMG})` }}
          aria-hidden
        />
        <div className="absolute inset-0 bg-void/85" aria-hidden />
        <div className="absolute inset-0 grid-bg opacity-60" aria-hidden />
        <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-void to-transparent" aria-hidden />

        <div className="relative mx-auto flex min-h-[92vh] max-w-7xl flex-col justify-center px-6 py-24">
          <Reveal>
            <p className="text-xs uppercase tracking-[0.3em] text-ember">
              &gt; ASIP_127 // street-safety-device-network
            </p>
          </Reveal>
          <Reveal delay={0.05}>
            <h1 className="mt-6 text-4xl leading-[1.05] text-paper sm:text-6xl md:text-7xl">
              PROACTIVE
              <br />
              EDGE-AI
              <br />
              <span className="text-ember">STREET_SAFETY</span>
              <span className="ml-2 inline-block h-[0.85em] w-[0.5em] translate-y-[0.05em] bg-ember align-baseline animate-pulse" />
            </h1>
          </Reveal>
          <Reveal delay={0.15}>
            <p className="mt-8 max-w-2xl text-base text-fog md:text-lg">
              CCTV records. Ours <span className="text-paper">acts</span>. A
              fault-tolerant mesh of Raspberry Pi 5 nodes fuses video, thermal
              and motion to detect threats, deter them on-site, and escalate
              the rest to police and emergency services — in under half a
              second, at the edge.
            </p>
          </Reveal>
          <Reveal delay={0.25}>
            <div className="mt-10 flex flex-col gap-3 sm:flex-row">
              <a
                href={DRIVE_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center justify-between gap-4 border border-ember bg-ember px-6 py-3 text-sm font-bold text-void transition-all hover:bg-paper hover:border-paper"
              >
                <span>[ ACCESS PROJECT RESOURCES ]</span>
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </a>
              <Link
                to="/response"
                className="inline-flex items-center justify-between gap-4 border border-iron bg-transparent px-6 py-3 text-sm text-paper transition-colors hover:border-paper"
              >
                <span>[ SEE THREAT RESPONSE ]</span>
                <span className="text-ember">→</span>
              </Link>
            </div>
          </Reveal>
          <Reveal delay={0.4}>
            <div className="mt-16 flex flex-wrap items-center gap-x-8 gap-y-3 text-xs text-steel">
              <span>
                <span className="text-ember">●</span> system_status: prototype_phase_2
              </span>
              <span>build: 24UTAI13</span>
              <span>institution: atria.edu</span>
            </div>
          </Reveal>
        </div>
      </section>

      {/* PROBLEM STRIP */}
      <section className="border-b border-iron py-24">
        <div className="mx-auto max-w-7xl px-6">
          <Reveal>
            <TerminalLabel>problem.md</TerminalLabel>
            <h2 className="max-w-3xl text-3xl text-paper md:text-4xl">
              Existing surveillance is{" "}
              <span className="text-ash line-through">passive</span> — recording
              after the fact, dependent on humans, blind when vandalized.
            </h2>
          </Reveal>
          <div className="mt-12 grid gap-px bg-iron sm:grid-cols-3">
            {stats.map((s, i) => (
              <Reveal key={s.k} delay={i * 0.05}>
                <div className="h-full bg-void p-6">
                  <p className="text-3xl text-ember md:text-4xl">{s.k}</p>
                  <p className="mt-3 text-sm text-fog">{s.v}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* SOLUTION OVERVIEW */}
      <section className="border-b border-iron py-24">
        <div className="mx-auto grid max-w-7xl gap-12 px-6 lg:grid-cols-2">
          <Reveal>
            <TerminalLabel>solution.md</TerminalLabel>
            <h2 className="text-3xl text-paper md:text-4xl">
              Detect. Deter. Dispatch.
            </h2>
            <p className="mt-6 text-fog">
              Each node runs a two-stage classifier on a Raspberry Pi 5,
              fusing HD/4K night-vision video, MLX90640 thermal data, and PIR
              motion. The first stage finds anomalies. The second decides:
              <span className="text-paper"> deterrable</span> threats trigger
              ML-generated acoustic and visual deterrents on-site;
              <span className="text-paper"> non-deterrable</span> threats fire
              a structured API alert to police and emergency services in
              under 500 ms — with geo-coordinates, snapshot, and confidence.
            </p>
            <p className="mt-4 text-fog">
              Every node is also a peer in an MQTT mesh. Vandalize one — its
              neighbours expand sensitivity automatically. There are no blind
              spots.
            </p>
            <Link
              to="/hardware"
              className="mt-8 inline-flex items-center gap-2 border-b border-ember pb-1 text-sm text-paper hover:text-ember"
            >
              inspect_hardware <span className="text-ember">→</span>
            </Link>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="border border-iron bg-carbon">
              <div className="flex items-center justify-between border-b border-iron px-4 py-2 text-xs text-steel">
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-ember" />
                  architecture.txt
                </span>
                <span>read_only</span>
              </div>
              <pre className="overflow-x-auto p-6 text-xs leading-relaxed text-fog md:text-sm">
                {`[ LAYER 1: SENSOR ARRAY ]
  ├── HD/4K Night-Vision Cam (CSI)
  ├── MLX90640 Thermal (I2C, 110° FoV)
  └── PIR Motion (low-power wake)
                │
                ▼  multi-modal fusion
[ LAYER 2: EDGE CORE  — Raspberry Pi 5 ]
  stage_1 → general threat detector
  stage_2 → severity classifier
                │
        ┌───────┴────────┐
   deterrable      non-deterrable
        │                │
[ TIER 1 ]         [ TIER 2 ]
 acoustic +         police  + EMS
 visual             + locals (SMS)
 deterrence         < 500 ms dispatch
                │
                ▼
[ LAYER 4: MQTT MESH ]
  node ←→ node ←→ node
  on_vandalize → neighbours
                  expand FoV`}
              </pre>
            </div>
          </Reveal>
        </div>
      </section>

      {/* SDG BAND */}
      <section className="py-24">
        <div className="mx-auto max-w-7xl px-6">
          <Reveal>
            <TerminalLabel>sdg_alignment.md</TerminalLabel>
            <h2 className="text-3xl text-paper md:text-4xl">
              Aligned to three UN SDGs
            </h2>
          </Reveal>
          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {sdgs.map((s, i) => {
              const Icon = i === 0 ? ShieldAlert : i === 1 ? Radio : Cpu;
              return (
                <Reveal key={s.code} delay={i * 0.05}>
                  <article className="h-full border-l-2 border-ember bg-graphite p-6">
                    <div className="flex items-center justify-between">
                      <span className="text-xs uppercase tracking-widest text-ember">
                        {s.code}
                      </span>
                      <Icon className="h-4 w-4 text-steel" />
                    </div>
                    <h3 className="mt-4 text-lg text-paper">{s.title}</h3>
                    <p className="mt-3 text-sm text-fog">{s.desc}</p>
                  </article>
                </Reveal>
              );
            })}
          </div>
        </div>
      </section>
    </PageTransition>
  );
}
