import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

type Mode = "light" | "dark";
const Ctx = createContext<{ mode: Mode; toggle: () => void; colors: Record<string, string> }>({
  mode: "light", toggle: () => {}, colors: {},
});

const VARS = ["ink", "muted", "faint", "line", "surface", "brand", "viz-1", "viz-2", "viz-3", "viz-4", "viz-5", "viz-grid", "danger", "warn", "ok"];

function readColors(): Record<string, string> {
  const cs = getComputedStyle(document.documentElement);
  return Object.fromEntries(VARS.map((v) => [v, cs.getPropertyValue(`--${v}`).trim()]));
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>(() => {
    try {
      const saved = localStorage.getItem("glucolab-theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch { /* storage unavailable */ }
    return "light";
  });
  const [colors, setColors] = useState<Record<string, string>>({});
  useEffect(() => {
    document.documentElement.classList.toggle("dark", mode === "dark");
    try { localStorage.setItem("glucolab-theme", mode); } catch { /* ignore */ }
    setColors(readColors());
  }, [mode]);
  const value = useMemo(() => ({ mode, toggle: () => setMode((m) => (m === "dark" ? "light" : "dark")), colors }), [mode, colors]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useTheme = () => useContext(Ctx);
