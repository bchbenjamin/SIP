import { Link } from "@tanstack/react-router";

export function Footer() {
  return (
    <footer className="mt-32 border-t border-iron bg-void">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="flex flex-col md:flex-row justify-between gap-10">
          <div>
            <p className="text-xs uppercase tracking-widest text-steel">
              &gt; project
            </p>
            <p className="mt-3 text-paper">AI-Powered Street Safety Device Network</p>
            <p className="mt-1 text-xs text-ash">
              ASIP_127 / 24UTAI13
            </p>
          </div>
          <div className="flex items-center">
            <Link
              to="/team"
              className="inline-flex items-center justify-between gap-4 border border-iron bg-carbon px-6 py-3 text-sm text-paper transition-colors hover:border-paper"
            >
              <span>[ VIEW TEAM & INSTITUTE DETAILS ]</span>
              <span className="text-ember">→</span>
            </Link>
          </div>
        </div>
        <div className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-iron pt-6 text-xs text-steel sm:flex-row">
          <span>© 2026 · AI-Powered Street Safety Device Network · All rights reserved.</span>
          <span>
            <span className="text-ember">●</span> system_status: prototype_phase_2
          </span>
        </div>
      </div>
    </footer>
  );
}
