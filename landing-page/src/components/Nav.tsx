import { Link } from "@tanstack/react-router";
import { ArrowUpRight } from "lucide-react";

const links = [
  { to: "/", label: "HOME" },
  { to: "/response", label: "RESPONSE" },
  { to: "/hardware", label: "HARDWARE" },
  { to: "/team", label: "TEAM" },
] as const;

export function Nav() {
  return (
    <header className="sticky top-0 z-50 border-b border-iron bg-void/90 backdrop-blur-sm">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-2 text-paper">
          <span className="text-ember">●</span>
          <span className="text-sm tracking-widest">ASIP_127</span>
          <span className="hidden text-xs text-ash sm:inline">
            / street-safety-net
          </span>
        </Link>
        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-1 text-xs sm:gap-2 sm:text-sm">
            {links.map((l) => (
              <Link
                key={l.to}
                to={l.to}
                activeOptions={{ exact: l.to === "/" }}
                className="group px-2 py-1 text-fog transition-colors hover:text-paper"
                activeProps={{ className: "text-paper" }}
              >
                {({ isActive }) => (
                  <span className="flex items-center gap-1">
                    <span className={isActive ? "text-ember" : "text-steel"}>
                      [
                    </span>
                    <span>{l.label}</span>
                    <span className={isActive ? "text-ember" : "text-steel"}>
                      ]
                    </span>
                  </span>
                )}
              </Link>
            ))}
          </nav>
          <a
            href="https://drive.google.com/drive/folders/1-srQ-2YbDc3Ptqc_WZ5A1padu-PmNhae"
            target="_blank"
            rel="noopener noreferrer"
            className="group hidden sm:inline-flex items-center gap-2 border border-ember bg-ember px-4 py-1.5 text-xs font-bold text-void transition-all hover:bg-paper hover:border-paper"
          >
            <span>[ ACCESS PROJECT RESOURCES ]</span>
            <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </a>
        </div>
      </div>
    </header>
  );
}
