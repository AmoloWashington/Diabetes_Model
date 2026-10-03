import { useEffect, useRef, useState } from "react";
import { useTheme } from "@/lib/theme";
import { Callout, Spinner } from "../ui";

export type ColorBy = "confidence" | "nucleotide" | "chain" | "spectrum";
export type StyleKind = "auto" | "trace" | "cartoon" | "stick" | "sphere" | "surface";

// Confidence bins follow the convention popularised by AlphaFold's pLDDT.
export const CONF_BINS = [
  { min: 90, label: "Very high (>90)", color: "#1b563e" },
  { min: 70, label: "Confident (70–90)", color: "#4fa37e" },
  { min: 50, label: "Low (50–70)", color: "#d29a3c" },
  { min: 0, label: "Very low (<50)", color: "#c4553f" },
];
const NUC: Record<string, string> = { A: "#3e9b74", U: "#c4553f", T: "#c4553f", G: "#d29a3c", C: "#5a7fb0" };
const CHAINS = ["#1b563e", "#c07a12", "#6a5acd", "#b42318", "#3f7f7a", "#8a6d3b"];

type Atom = { b: number; resn: string; chain: string; atom: string; resi: number; x: number; y: number; z: number };

export function MolViewer({ data, format = "pdb", colorBy = "confidence", style = "auto", height = 480, spin = false, pairs }: {
  data: string; format?: "pdb" | "cif"; colorBy?: ColorBy; style?: StyleKind; height?: number; spin?: boolean;
  /** 0-based base pairs; drawn as rungs between paired C1' atoms (C1'-only models). */
  pairs?: [number, number][];
}) {
  const el = useRef<HTMLDivElement>(null);
  const viewer = useRef<import("3dmol").GLViewer | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const { mode } = useTheme();

  useEffect(() => {
    let alive = true;
    import("3dmol").then(($3Dmol) => {
      if (!alive || !el.current) return;
      try {
        el.current.innerHTML = "";
        const v = $3Dmol.createViewer(el.current, { backgroundColor: mode === "dark" ? "#15181d" : "#ffffff", antialias: true });
        viewer.current = v;
        v.addModel(data, format);
        const atoms = v.selectedAtoms({}) as unknown as Atom[];
        const c1Only = atoms.length > 0 && atoms.every((a) => a.atom === "C1'" || a.atom === "C1*");
        const colorfunc = (a: Atom) => {
          if (colorBy === "confidence") return (CONF_BINS.find((b) => a.b >= b.min) ?? CONF_BINS[3]).color;
          if (colorBy === "nucleotide") return NUC[a.resn.trim().slice(-1)] ?? "#9a9ea9";
          if (colorBy === "chain") return CHAINS[(a.chain.charCodeAt(0) || 65) % CHAINS.length];
          return undefined;
        };
        const cs = colorBy === "spectrum" ? { colorscheme: "spectrum" } : { colorfunc };
        const kind = style === "auto" ? (c1Only ? "trace" : "cartoon") : style;
        if (kind === "trace") v.setStyle({}, { stick: { radius: 0.35, ...cs }, sphere: { radius: 1.25, ...cs } } as never);
        else if (kind === "cartoon") v.setStyle({}, { cartoon: { ...cs, thickness: 0.6 }, stick: { radius: 0.15, hidden: false, ...cs } } as never);
        else if (kind === "stick") v.setStyle({}, { stick: { radius: 0.25, ...cs } } as never);
        else if (kind === "surface") {
          v.setStyle({}, { cartoon: { ...cs, thickness: 0.6 } } as never);
          v.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.82, ...(colorBy === "spectrum" ? { colorscheme: "spectrum" } : { colorfunc }) } as never);
        }
        else v.setStyle({}, { sphere: { scale: 0.9, ...cs } } as never);
        if (c1Only && pairs?.length) {
          const byResi = new Map(atoms.map((a) => [a.resi, a]));
          for (const [i, j] of pairs) {
            const a = byResi.get(i + 1), b = byResi.get(j + 1);
            if (a && b) {
              v.addCylinder({
                start: { x: a.x, y: a.y, z: a.z }, end: { x: b.x, y: b.y, z: b.z },
                radius: 0.22, color: mode === "dark" ? "#6c717c" : "#b9b3a4", fromCap: 1, toCap: 1,
              } as never);
            }
          }
        }
        const labelStyle = {
          backgroundColor: mode === "dark" ? "#1a1e24" : "#ffffff", backgroundOpacity: 0.92, fontColor: mode === "dark" ? "#eceef1" : "#15171c",
          fontSize: 12, borderThickness: 1, borderColor: mode === "dark" ? "#363c46" : "#d3cfc3", inFront: true,
        };
        // Hover: residue name, number and chain
        v.setHoverable({}, true, (atom: Atom & { hoverLabel?: unknown }) => {
          if (!atom.hoverLabel) atom.hoverLabel = v.addLabel(`${atom.resn.trim()}${atom.resi} · chain ${atom.chain}`, { ...labelStyle, position: { x: atom.x, y: atom.y, z: atom.z } } as never);
        }, (atom: Atom & { hoverLabel?: unknown }) => {
          if (atom.hoverLabel) { v.removeLabel(atom.hoverLabel as never); delete atom.hoverLabel; }
        });
        // Structural labels for coarse-grained models: ends, helices and hairpin loops
        if (c1Only) {
          const byResi = new Map(atoms.map((a) => [a.resi, a]));
          const resis = [...byResi.keys()].sort((a, b) => a - b);
          const put = (text: string, a?: Atom) => a && v.addLabel(text, { ...labelStyle, fontSize: 11, position: { x: a.x, y: a.y, z: a.z } } as never);
          put("5′", byResi.get(resis[0]));
          put("3′", byResi.get(resis[resis.length - 1]));
          if (pairs?.length) {
            const sorted = [...pairs].sort((a, b) => a[0] - b[0]);
            const stems: [number, number][][] = [];
            for (const p of sorted) {
              const last = stems.at(-1)?.at(-1);
              if (last && p[0] === last[0] + 1 && p[1] === last[1] - 1) stems.at(-1)!.push(p); else stems.push([p]);
            }
            const paired = new Set(pairs.flat());
            stems.forEach((st, k) => {
              const [i, j] = st[Math.floor(st.length / 2)];
              const a = byResi.get(i + 1), b = byResi.get(j + 1);
              if (a && b) v.addLabel(`Helix ${k + 1} · ${st.length} bp`, { ...labelStyle, fontSize: 11, position: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 } } as never);
              const [ii, jj] = st[st.length - 1];
              let hairpin = jj - ii > 1;
              for (let q = ii + 1; q < jj && hairpin; q++) if (paired.has(q)) hairpin = false;
              if (hairpin) put(`Hairpin loop · ${jj - ii - 1} nt`, byResi.get(Math.round((ii + jj) / 2) + 1));
            });
          }
        }
        v.zoomTo();
        if (spin) v.spin("y", 0.6);
        v.render();
        setState("ready");
      } catch (e) {
        console.error(e);
        setState("error");
      }
    }).catch(() => setState("error"));
    return () => {
      alive = false;
      try { viewer.current?.clear(); } catch { /* ignore */ }
      viewer.current = null;
    };
  }, [data, format, colorBy, style, mode, spin, pairs]);

  useEffect(() => {
    const onResize = () => { viewer.current?.resize(); viewer.current?.render(); };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return (
    <div className="relative w-full" style={{ height }}>
      <div ref={el} className="absolute inset-0" />
      {state === "loading" && <div className="absolute inset-0 grid place-items-center"><Spinner label="Loading 3D viewer…" /></div>}
      {state === "error" && <div className="absolute inset-0 grid place-items-center p-6"><Callout tone="warn">The 3D viewer could not render this structure (WebGL unavailable or unsupported file).</Callout></div>}
    </div>
  );
}

export function ConfidenceLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-muted">
      {CONF_BINS.map((b) => (
        <span key={b.label} className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm" style={{ background: b.color }} />{b.label}</span>
      ))}
    </div>
  );
}
