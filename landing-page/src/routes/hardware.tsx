import { createFileRoute } from "@tanstack/react-router";
import {
  Cpu,
  Camera,
  Thermometer,
  Radar,
  Mic,
  Radio,
  Battery,
  Network,
  ArrowUpRight,
} from "lucide-react";
import { PageTransition, Reveal, TerminalLabel } from "../components/PageTransition";

export const Route = createFileRoute("/hardware")({
  head: () => ({
    meta: [
      { title: "Hardware Infrastructure — AI Street Safety Device Network" },
      {
        name: "description",
        content:
          "Raspberry Pi 5 edge core, HD/4K night-vision cameras, MLX90640 thermal, PIR motion, 4G uplink — wired into a fault-tolerant MQTT mesh.",
      },
      { property: "og:title", content: "Hardware Infrastructure" },
      {
        property: "og:description",
        content:
          "Spec sheets for every component in the street safety node, plus the vandalism-resilient mesh.",
      },
    ],
  }),
  component: HardwarePage,
});

type Part = {
  id: string;
  name: string;
  role: string;
  icon: typeof Cpu;
  span: string;
  specs: { k: string; v: string }[];
  note?: string;
};

const parts: Part[] = [
  {
    id: "rpi5",
    name: "Raspberry Pi 5 (4GB)",
    role: "edge_core",
    icon: Cpu,
    span: "md:col-span-2 md:row-span-2",
    specs: [
      { k: "cpu", v: "Quad Cortex-A76 @ 2.4 GHz" },
      { k: "ram", v: "4 GB LPDDR4X" },
      { k: "io", v: "PCIe 2.0 · USB-C PD 27 W" },
      { k: "inference", v: "YOLOv8n/s · TFLite · PyTorch" },
      { k: "status", v: "SETTLED — non-negotiable" },
    ],
    note: "Runs the full two-stage classifier locally. Upgrade path: AI HAT+ (Hailo-10H, 40 TOPS) for production.",
  },
  {
    id: "cam",
    name: "RPi Camera Module 3 Wide (NoIR)",
    role: "visual_input",
    icon: Camera,
    span: "md:col-span-2",
    specs: [
      { k: "sensor", v: "Sony IMX708 · 12 MP" },
      { k: "fov", v: "120° wide" },
      { k: "interface", v: "CSI · 30 fps @ 1080p" },
      { k: "night", v: "+ external IR illuminator" },
    ],
  },
  {
    id: "thermal",
    name: "MLX90640 Thermal Sensor",
    role: "heat_signature",
    icon: Thermometer,
    span: "",
    specs: [
      { k: "res", v: "32 × 24 pixels" },
      { k: "fov", v: "110°" },
      { k: "range", v: "0–80 °C · ±1.5 °C" },
      { k: "bus", v: "I2C · 4 Hz" },
    ],
  },
  {
    id: "pir",
    name: "PIR Motion Sensor",
    role: "wake_trigger",
    icon: Radar,
    span: "",
    specs: [
      { k: "type", v: "passive infrared" },
      { k: "draw", v: "< 50 μA idle" },
      { k: "role", v: "keeps core in low-power state" },
      { k: "wake", v: "gpio interrupt → fusion" },
    ],
  },
  {
    id: "mic",
    name: "ReSpeaker 2-Mic Pi HAT",
    role: "audio_pipeline",
    icon: Mic,
    span: "md:col-span-2",
    specs: [
      { k: "type", v: "dual MEMS · omnidirectional" },
      { k: "snr", v: "65 dB" },
      { k: "interface", v: "I2S (HAT stack)" },
      { k: "model", v: "LSTM + CNN on MFCC" },
      { k: "detects", v: "screams · distress · gunshots" },
    ],
  },
  {
    id: "lte",
    name: "SIM7600E-H 4G HAT",
    role: "uplink",
    icon: Radio,
    span: "",
    specs: [
      { k: "standard", v: "LTE Cat-4 + 3G/2G fallback" },
      { k: "bands", v: "B1/B3/B5/B8/B40/B41 (IN)" },
      { k: "extra", v: "GPS bundled" },
      { k: "protocol", v: "MQTT + SMS" },
    ],
  },
  {
    id: "deter",
    name: "Deterrent Module",
    role: "tier_1_actuator",
    icon: Radio,
    span: "",
    specs: [
      { k: "audio", v: "wide-band external speaker" },
      { k: "visual", v: "high-lumen LED strobe" },
      { k: "drive", v: "PWM via GPIO" },
      { k: "trigger", v: "ML-generated waveform" },
    ],
  },
  {
    id: "power",
    name: "Battery + Solar Pack",
    role: "field_power",
    icon: Battery,
    span: "md:col-span-2",
    specs: [
      { k: "battery", v: "Li-ion · field-swappable" },
      { k: "solar", v: "20 W panel · MPPT controller" },
      { k: "duty", v: "PIR-gated wake reduces avg draw" },
      { k: "uptime", v: "off-grid roadside deployment" },
    ],
  },
];

function HardwarePage() {
  return (
    <PageTransition>
      {/* HEADER */}
      <section className="border-b border-iron py-20">
        <div className="mx-auto max-w-7xl px-6">
          <Reveal>
            <TerminalLabel>hardware_bom.md</TerminalLabel>
            <h1 className="max-w-4xl text-4xl text-paper md:text-6xl">
              A street-deployable
              <br />
              <span className="text-ember">node</span>, fully specced.
            </h1>
            <p className="mt-6 max-w-3xl text-fog">
              Off-the-shelf, India-sourceable components — chosen for inference
              throughput, low-light performance, low idle draw, and field
              repairability. Every part below is evaluated against documented
              alternatives in CONTEXT §9.
            </p>
            <div className="mt-8">
              <a
                href="https://sip-video.vercel.app/"
                target="_blank"
                rel="noopener noreferrer"
                className="group inline-flex items-center justify-between gap-4 border border-ember bg-ember px-6 py-3 text-sm font-bold text-void transition-all hover:bg-paper hover:border-paper"
              >
                <span>[ VIEW HARDWARE VISUALIZATION ]</span>
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </a>
            </div>
          </Reveal>
        </div>
      </section>

      {/* BENTO GRID */}
      <section className="border-b border-iron py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="grid auto-rows-[minmax(180px,_auto)] gap-px bg-iron md:grid-cols-4">
            {parts.map((p, i) => (
              <Reveal key={p.id} delay={i * 0.04} className={p.span}>
                <article className="flex h-full flex-col bg-graphite p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex h-10 w-10 items-center justify-center border border-iron bg-carbon">
                      <p.icon className="h-5 w-5 text-ember" />
                    </div>
                    <span className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-steel">
                      <span className="h-1.5 w-1.5 rounded-full bg-ember" />
                      {p.role}
                    </span>
                  </div>
                  <h3 className="mt-5 text-lg text-paper">{p.name}</h3>
                  <ul className="mt-4 space-y-1.5 text-xs">
                    {p.specs.map((s) => (
                      <li
                        key={s.k}
                        className="flex items-baseline gap-3 border-b border-iron pb-1"
                      >
                        <span className="w-20 shrink-0 text-steel">{s.k}</span>
                        <span className="text-fog">{s.v}</span>
                      </li>
                    ))}
                  </ul>
                  {p.note && (
                    <p className="mt-4 border-l-2 border-ember pl-3 text-xs text-ash">
                      {p.note}
                    </p>
                  )}
                </article>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* MESH RESILIENCE */}
      <section className="py-24">
        <div className="mx-auto grid max-w-7xl gap-12 px-6 lg:grid-cols-[1fr_1.1fr] lg:items-center">
          <Reveal>
            <TerminalLabel>mesh_resilience.md</TerminalLabel>
            <h2 className="text-3xl text-paper md:text-5xl">
              Vandalize one.
              <br />
              The mesh adapts.
            </h2>
            <p className="mt-6 text-fog">
              Every node is a peer in an MQTT-synced network. Each one
              maintains awareness of its neighbours' status. The moment a node
              drops offline — power loss, vandalism, hardware fault — adjacent
              nodes detect it and{" "}
              <span className="text-paper">
                autonomously expand their detection sensitivity and inference
                frequency
              </span>{" "}
              to cover the resulting blind spot.
            </p>
            <p className="mt-4 text-fog">
              This property is absent from every existing reviewed system. It
              is the network's primary differentiator and the reason single
              points of failure no longer exist in the deployment.
            </p>
            <ul className="mt-8 space-y-2 text-sm">
              <li className="flex items-baseline gap-3 border-b border-iron pb-2">
                <span className="w-32 shrink-0 text-xs uppercase tracking-widest text-steel">
                  protocol
                </span>
                <span className="text-paper">MQTT over local wireless</span>
              </li>
              <li className="flex items-baseline gap-3 border-b border-iron pb-2">
                <span className="w-32 shrink-0 text-xs uppercase tracking-widest text-steel">
                  topology
                </span>
                <span className="text-paper">peer-to-peer mesh</span>
              </li>
              <li className="flex items-baseline gap-3 border-b border-iron pb-2">
                <span className="w-32 shrink-0 text-xs uppercase tracking-widest text-steel">
                  heartbeat
                </span>
                <span className="text-paper">state-sync, second-resolution</span>
              </li>
              <li className="flex items-baseline gap-3">
                <span className="w-32 shrink-0 text-xs uppercase tracking-widest text-steel">
                  uplink
                </span>
                <span className="text-paper">LTE Cat-4 via SIM7600E-H</span>
              </li>
            </ul>
          </Reveal>

          <Reveal delay={0.1}>
            <MeshDiagram />
          </Reveal>
        </div>
      </section>
    </PageTransition>
  );
}

function MeshDiagram() {
  // 6 nodes positioned in a hex-ish layout. Node #4 is vandalized.
  const nodes = [
    { id: 1, x: 18, y: 22, dead: false },
    { id: 2, x: 50, y: 12, dead: false },
    { id: 3, x: 82, y: 22, dead: false },
    { id: 4, x: 50, y: 50, dead: true },
    { id: 5, x: 22, y: 78, dead: false },
    { id: 6, x: 78, y: 78, dead: false },
  ];
  const edges: [number, number, boolean][] = [
    [1, 2, false],
    [2, 3, false],
    [1, 4, true],
    [3, 4, true],
    [2, 4, true],
    [4, 5, true],
    [4, 6, true],
    [1, 5, false],
    [3, 6, false],
    [5, 6, false],
  ];

  return (
    <div className="border border-iron bg-carbon">
      <div className="flex items-center justify-between border-b border-iron px-4 py-2 text-xs text-steel">
        <span className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-ember" />
          mesh_topology.svg
        </span>
        <span>node_4 = OFFLINE · rerouting</span>
      </div>
      <div className="relative aspect-square w-full grid-bg">
        <svg
          viewBox="0 0 100 100"
          className="absolute inset-0 h-full w-full"
          preserveAspectRatio="none"
        >
          {edges.map(([a, b, dashed], idx) => {
            const na = nodes.find((n) => n.id === a)!;
            const nb = nodes.find((n) => n.id === b)!;
            const isDead = na.dead || nb.dead;
            return (
              <line
                key={idx}
                x1={na.x}
                y1={na.y}
                x2={nb.x}
                y2={nb.y}
                stroke={isDead ? "#3a3a3a" : "#da5c2c"}
                strokeWidth={isDead ? 0.2 : 0.35}
                strokeDasharray={dashed || isDead ? "1 1.5" : "0"}
                opacity={isDead ? 0.4 : 0.9}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
        {nodes.map((n) => (
          <div
            key={n.id}
            className="absolute flex -translate-x-1/2 -translate-y-1/2 flex-col items-center"
            style={{ left: `${n.x}%`, top: `${n.y}%` }}
          >
            <div
              className={`flex h-10 w-10 items-center justify-center border ${
                n.dead
                  ? "border-slate bg-void text-steel"
                  : "border-ember bg-graphite text-ember"
              }`}
            >
              {n.dead ? (
                <Network className="h-4 w-4" />
              ) : (
                <Cpu className="h-4 w-4" />
              )}
            </div>
            <span
              className={`mt-1 text-[10px] uppercase tracking-widest ${
                n.dead ? "text-steel line-through" : "text-fog"
              }`}
            >
              node_{n.id}
            </span>
          </div>
        ))}
      </div>
      <div className="border-t border-iron px-4 py-2 text-xs text-ash">
        <span className="text-ember">●</span> ember links = active sync ·
        dashed = traffic re-routed around offline node
      </div>
    </div>
  );
}
