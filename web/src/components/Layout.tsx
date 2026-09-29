import { useQuery } from "@tanstack/react-query";
import {
  Activity, BookOpen, Box, BrainCircuit, Cpu, Database, Dna, FlaskConical, HeartPulse, LayoutDashboard, Menu,
  Microscope, Moon, Orbit, Sun, TestTubes, Timer, X,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api, type Health } from "@/lib/api";
import { useTheme } from "@/lib/theme";
import { cx } from "./ui";

type Item = { to: string; label: string; icon: ReactNode };
const NAV: { group: string | null; items: Item[] }[] = [
  { group: null, items: [{ to: "/", label: "Overview", icon: <LayoutDashboard /> }] },
  {
    group: "Physiology",
    items: [
      { to: "/physiology/meal", label: "Meal simulation", icon: <Activity /> },
      { to: "/physiology/beta-cell", label: "β-cell dynamics", icon: <Timer /> },
      { to: "/physiology/minimal-model", label: "Insulin sensitivity", icon: <FlaskConical /> },
      { to: "/physiology/clinical", label: "Clinical indices", icon: <HeartPulse /> },
    ],
  },
  {
    group: "Cell & molecular",
    items: [
      { to: "/cells", label: "Cell theatre", icon: <Microscope /> },
      { to: "/dna", label: "DNA lab", icon: <Dna /> },
    ],
  },
  {
    group: "RNA structure",
    items: [
      { to: "/rna/sequences", label: "Sequences", icon: <TestTubes /> },
      { to: "/rna/predict", label: "3D prediction", icon: <Cpu /> },
      { to: "/rna/structures", label: "Structure viewer", icon: <Box /> },
    ],
  },
  {
    group: "Data & models",
    items: [
      { to: "/data", label: "Dataset explorer", icon: <Database /> },
      { to: "/risk", label: "Diabetes risk model", icon: <Orbit /> },
    ],
  },
  { group: "AI", items: [{ to: "/assistant", label: "Research assistant", icon: <BrainCircuit /> }] },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="size-8 rounded-lg bg-brand grid place-items-center">
        <svg viewBox="0 0 32 32" className="size-5" aria-hidden="true">
          <path d="M11 6c0 8 10 8 10 10s-10 2-10 10M21 6c0 8-10 8-10 10s10 2 10 10" stroke="#F6F5F1" strokeWidth="2.4" fill="none" strokeLinecap="round" />
        </svg>
      </div>
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-[-0.01em] text-ink">GlucoLab</div>
        <div className="text-[11px] text-muted">Computational biology</div>
      </div>
    </div>
  );
}

function StatusRow({ label, ok, detail }: { label: string; ok: boolean | undefined; detail?: string }) {
  return (
    <div className="flex items-center justify-between text-[12px]">
      <span className="text-muted">{label}</span>
      <span className="flex items-center gap-1.5 text-ink-2">
        <span className={cx("size-1.5 rounded-full", ok === undefined ? "bg-faint" : ok ? "bg-ok" : "bg-warn")} />
        {detail ?? (ok === undefined ? "…" : ok ? "Ready" : "Off")}
      </span>
    </div>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { mode, toggle } = useTheme();
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get<Health>("/api/health"), refetchInterval: 15000 });
  const kaggle = useQuery({ queryKey: ["kaggle-status"], queryFn: () => api.get<{ configured: boolean }>("/api/datasets/kaggle/status") });
  return (
    <div className="flex h-full flex-col">
      <div className="px-4 pt-5 pb-4"><Logo /></div>
      <nav className="flex-1 overflow-y-auto px-3 pb-4" aria-label="Main">
        {NAV.map((g, gi) => (
          <div key={gi} className={cx(gi > 0 && "mt-5")}>
            {g.group && <div className="px-2 mb-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-faint">{g.group}</div>}
            <ul className="flex flex-col gap-0.5">
              {g.items.map((it) => (
                <li key={it.to}>
                  <NavLink
                    to={it.to} end={it.to === "/"} onClick={onNavigate}
                    className={({ isActive }) => cx(
                      "group flex items-center gap-2.5 rounded-lg px-2 h-8 text-[13.5px] transition-colors [&_svg]:size-4 [&_svg]:shrink-0",
                      isActive ? "bg-surface text-ink font-medium shadow-card border border-line" : "text-ink-2 hover:bg-surface-3 border border-transparent",
                    )}
                  >
                    {({ isActive }) => (
                      <>
                        <span className={isActive ? "text-brand-text" : "text-faint group-hover:text-muted"}>{it.icon}</span>
                        {it.label}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div className="border-t border-line px-4 py-3.5 flex flex-col gap-1.5">
        <StatusRow label="API" ok={health.isError ? false : health.data ? true : undefined} detail={health.isError ? "Offline" : undefined} />
        <StatusRow label="AI assistant" ok={health.data?.ai_configured} detail={health.data ? (health.data.ai_configured ? health.data.ai_model ?? "Ready" : "No key") : undefined} />
        <StatusRow label="Kaggle" ok={kaggle.data?.configured} detail={kaggle.data ? (kaggle.data.configured ? "Connected" : "No token") : undefined} />
        <div className="flex items-center justify-between mt-2">
          <NavLink to="/references" onClick={onNavigate} className="flex items-center gap-1.5 text-[12px] text-muted hover:text-ink">
            <BookOpen className="size-3.5" /> References
          </NavLink>
          <button onClick={toggle} className="size-7 grid place-items-center rounded-md text-muted hover:bg-surface-3 hover:text-ink" aria-label="Toggle colour theme">
            {mode === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}

export function Layout() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [loc.pathname]);
  return (
    <div className="min-h-screen">
      <aside className="hidden lg:block fixed inset-y-0 left-0 w-[248px] border-r border-line bg-surface-2">
        <Sidebar />
      </aside>
      {/* mobile top bar */}
      <div className="lg:hidden sticky top-0 z-30 flex items-center justify-between h-14 px-4 border-b border-line bg-surface/90 backdrop-blur">
        <Logo />
        <button onClick={() => setOpen(true)} className="size-9 grid place-items-center rounded-lg border border-line" aria-label="Open menu"><Menu className="size-5" /></button>
      </div>
      {open && (
        <div className="lg:hidden fixed inset-0 z-40">
          <div className="absolute inset-0 bg-black/30" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-[272px] bg-surface-2 border-r border-line shadow-pop">
            <button onClick={() => setOpen(false)} className="absolute right-3 top-5 size-8 grid place-items-center rounded-md text-muted" aria-label="Close menu"><X className="size-5" /></button>
            <Sidebar onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
      <main className="lg:pl-[248px]">
        <div className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-10 py-8 lg:py-10">
          <Outlet />
          <footer className="mt-16 pt-6 border-t border-line text-[12px] text-faint flex flex-wrap gap-x-4 gap-y-1 justify-between">
            <span>GlucoLab · research and education software, not a medical device.</span>
            <span>Models validated against their primary publications · see References</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
