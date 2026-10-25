import {
  Activity, BookOpen, Languages, MessageSquareText, ChartSpline, Box, BrainCircuit, Cpu, Database, Dna, FlaskConical, HeartPulse, LayoutDashboard, Menu,
  Microscope, Moon, Orbit, Pill, Scissors, ScanSearch, Sigma, Sun, TestTubes, Timer, X, Zap,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { TRANSLATED_ROUTES, useI18n, type Key } from "@/lib/i18n";
import { useTheme } from "@/lib/theme";
import { InstallApp } from "./InstallApp";
import { cx } from "./ui";

type Item = { to: string; label: Key; icon: ReactNode };
const NAV: { group: Key | null; items: Item[] }[] = [
  { group: null, items: [{ to: "/", label: "nav.overview", icon: <LayoutDashboard /> }, { to: "/methods", label: "nav.math", icon: <Sigma /> }] },
  {
    group: "nav.group.physiology",
    items: [
      { to: "/physiology/meal", label: "nav.meal", icon: <Activity /> },
      { to: "/physiology/beta-cell", label: "nav.betacell", icon: <Timer /> },
      { to: "/physiology/minimal-model", label: "nav.minimal", icon: <FlaskConical /> },
      { to: "/physiology/clinical", label: "nav.clinical", icon: <HeartPulse /> },
      { to: "/physiology/cgm", label: "nav.cgm", icon: <ChartSpline /> },
      { to: "/physiology/biophysics", label: "nav.biophysics", icon: <Zap /> },
    ],
  },
  {
    group: "nav.group.cell",
    items: [
      { to: "/cells/explorer", label: "nav.explorer", icon: <ScanSearch /> },
      { to: "/cells", label: "nav.theatre", icon: <Microscope /> },
      { to: "/dna", label: "nav.dna", icon: <Dna /> },
      { to: "/genetics/monogenic", label: "nav.monogenic", icon: <Pill /> },
      { to: "/genetics/crispr", label: "nav.crispr", icon: <Scissors /> },
    ],
  },
  {
    group: "nav.group.rna",
    items: [
      { to: "/rna/sequences", label: "nav.sequences", icon: <TestTubes /> },
      { to: "/rna/predict", label: "nav.predict", icon: <Cpu /> },
      { to: "/rna/structures", label: "nav.structures", icon: <Box /> },
    ],
  },
  {
    group: "nav.group.data",
    items: [
      { to: "/data", label: "nav.datasets", icon: <Database /> },
      { to: "/risk", label: "nav.risk", icon: <Orbit /> },
    ],
  },
  { group: "nav.group.ai", items: [{ to: "/assistant", label: "nav.assistant", icon: <BrainCircuit /> }] },
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

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { mode, toggle } = useTheme();
  const { lang, setLang, t } = useI18n();
  return (
    <div className="flex h-full flex-col">
      <div className="px-4 pt-5 pb-4"><Logo /></div>
      <nav className="flex-1 overflow-y-auto px-3 pb-4" aria-label="Main">
        {NAV.map((g, gi) => (
          <div key={gi} className={cx(gi > 0 && "mt-5")}>
            {g.group && <div className="px-2 mb-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-faint">{t(g.group)}</div>}
            <ul className="flex flex-col gap-0.5">
              {g.items.map((it) => (
                <li key={it.to}>
                  <NavLink
                    to={it.to} end={it.to === "/" || it.to === "/cells"} onClick={onNavigate}
                    className={({ isActive }) => cx(
                      "group flex items-center gap-2.5 rounded-lg px-2 h-8 text-[13.5px] transition-colors [&_svg]:size-4 [&_svg]:shrink-0",
                      isActive ? "bg-surface text-ink font-medium shadow-card border border-line" : "text-ink-2 hover:bg-surface-3 border border-transparent",
                    )}
                  >
                    {({ isActive }) => (
                      <>
                        <span className={isActive ? "text-brand-text" : "text-faint group-hover:text-muted"}>{it.icon}</span>
                        {t(it.label)}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div className="border-t border-line px-4 py-3.5 flex flex-col gap-2.5">
        <InstallApp />
        <NavLink to="/feedback" onClick={onNavigate} className="flex items-center gap-1.5 text-[12.5px] font-medium text-brand-text hover:underline">
          <MessageSquareText className="size-3.5" /> {t("layout.feedback")}
        </NavLink>
        <div className="flex items-center justify-between gap-2">
          <NavLink to="/references" onClick={onNavigate} className="flex items-center gap-1.5 text-[12px] text-muted hover:text-ink">
            <BookOpen className="size-3.5" /> {t("layout.references")}
          </NavLink>
          <div className="flex items-center gap-1">
            <label className="flex items-center gap-1 text-muted" title={t("layout.language")}>
              <Languages className="size-3.5" aria-hidden="true" />
              <select value={lang} onChange={(e) => setLang(e.target.value as "en" | "sw")} aria-label={t("layout.language")}
                className="bg-transparent text-[12px] text-ink-2 rounded-md py-0.5 pr-1 hover:bg-surface-3 cursor-pointer">
                <option value="en">English</option>
                <option value="sw">Kiswahili</option>
              </select>
            </label>
            <button onClick={toggle} className="size-7 grid place-items-center rounded-md text-muted hover:bg-surface-3 hover:text-ink" aria-label={t("layout.theme")}>
              {mode === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function Layout() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const { lang, t } = useI18n();
  useEffect(() => { window.scrollTo(0, 0); }, [loc.pathname]);
  const translated = TRANSLATED_ROUTES.includes(loc.pathname);
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
          {lang === "sw" && (
            <div className="mb-6 rounded-lg border border-line bg-surface-2 px-3.5 py-2.5 text-[12.5px] text-ink-2 leading-5">
              {t("layout.draft")}{!translated && <> <b>{t("layout.untranslated")}</b></>}
            </div>
          )}
          <Outlet />
          <footer className="mt-16 pt-6 border-t border-line text-[12px] text-faint flex flex-wrap gap-x-4 gap-y-1 justify-between">
            <span>{t("layout.footer1")}</span>
            <span>{t("layout.footer2")}</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
