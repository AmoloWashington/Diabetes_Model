import { Circle, Crosshair, Info, RotateCcw, Square, Tag, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { AskAI } from "@/components/AskAI";
import { Badge, Button, Callout, Card, PageHeader, Segmented, Slider, cx, useToast } from "@/components/ui";
import { useRecorder } from "@/lib/useRecorder";
import type { CellExplorer as Explorer, ImagingMode } from "@/lib/viz/cell3d";
import { PARTS } from "@/lib/viz/cellParts";

type PartKey = keyof typeof PARTS;
const JUMPS: PartKey[] = ["nucleus", "nucleolus", "rer", "golgi", "mito", "granule", "centrosome", "cilium"];

function zoomLevel(d: number) {
  if (d > 20) return { label: "Whole cell", level: 0 };
  if (d > 6) return { label: "Organelles", level: 1 };
  return { label: "Sub-structures", level: 2 };
}

export default function CellExplorerPage() {
  const toast = useToast();
  const host = useRef<HTMLDivElement>(null);
  const ex = useRef<Explorer | null>(null);
  const [mode, setMode] = useState<ImagingMode>("illustrated");
  const [cutaway, setCutaway] = useState(true);
  const [labels, setLabels] = useState(true);
  const [glucose, setGlucose] = useState(150);
  const [selected, setSelected] = useState<PartKey | null>(null);
  const [dist, setDist] = useState(20);
  const [error, setError] = useState<string | null>(null);
  const rec = useRecorder(() => ex.current?.canvas, "glucolab-cell-3d");

  useEffect(() => {
    let alive = true;
    import("@/lib/viz/cell3d").then(({ CellExplorer }) => {
      if (!alive || !host.current) return;
      try {
        const e = new CellExplorer(host.current);
        e.onSelect = (p) => setSelected(p as PartKey | null);
        let last = 0;
        e.onZoom = (d) => { const now = performance.now(); if (now - last > 150) { last = now; setDist(d); } };
        ex.current = e;
        e.setGlucose(150);
      } catch (err) {
        console.error(err);
        setError("3D rendering (WebGL) is not available in this browser.");
      }
    }).catch(() => setError("Could not load the 3D renderer."));
    return () => { alive = false; ex.current?.dispose(); ex.current = null; };
  }, []);

  useEffect(() => { ex.current?.setMode(mode); }, [mode]);
  useEffect(() => { ex.current?.setCutaway(cutaway); }, [cutaway]);
  useEffect(() => { if (ex.current) ex.current.showLabels = labels; }, [labels]);
  useEffect(() => { ex.current?.setGlucose(glucose); }, [glucose]);

  const z = zoomLevel(dist);
  const info = selected ? PARTS[selected] : null;
  const fold = Math.max(0, (1.5434 + 0.11 * (glucose - 91.76)) / 1.5434);

  return (
    <div>
      <PageHeader eyebrow="Cell & molecular" title="3D cell explorer"
        description="An interactive 3D model of a pancreatic β-cell. Zoom in to reveal finer structures, click any part to learn its role, and switch between illustration, fluorescence and electron-microscopy styles."
        actions={<>
          <AskAI page="3D cell explorer" summary={`Viewing a 3D β-cell model in ${mode} mode at plasma glucose ${glucose} mg/dl (secretion ${fold.toFixed(2)}× basal).${info ? ` Selected structure: ${info.name}.` : ""}`}
            question={info ? `Explain the ${info.name} of the β-cell and its role in insulin secretion.` : "Walk me through how this β-cell turns glucose into insulin secretion."} />
          <Button variant={rec.recording ? "danger" : "secondary"} icon={rec.recording ? <Square className="size-4" /> : <Circle className="size-4 fill-current text-danger" />}
            onClick={() => { const e = rec.toggle(); if (e) toast(e, "error"); }}>{rec.recording ? "Stop & save" : "Record video"}</Button>
        </>}
      />
      <Card className="mb-4">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-4">
          <div>
            <div className="text-[12.5px] font-medium text-ink-2 mb-1.5">Imaging style</div>
            <Segmented<ImagingMode> value={mode} onChange={setMode} options={[
              { value: "illustrated", label: "Illustrated" },
              { value: "fluorescence", label: "Fluorescence" },
              { value: "em", label: <><span className="sm:hidden">EM</span><span className="hidden sm:inline">Electron micrograph</span></> },
            ]} />
          </div>
          <div className="w-64"><Slider label="Plasma glucose (drives exocytosis)" value={glucose} min={40} max={400} step={5} onChange={setGlucose} format={(v) => `${v} mg/dl · ×${fold.toFixed(1)}`} /></div>
          <label className="flex items-center gap-2 text-[13px] text-ink-2 h-9"><input type="checkbox" checked={cutaway} disabled={mode === "em"} onChange={(e) => setCutaway(e.target.checked)} />Membrane cutaway</label>
          <label className="flex items-center gap-2 text-[13px] text-ink-2 h-9"><input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} />Labels</label>
          <Button size="sm" icon={<RotateCcw className="size-4" />} onClick={() => ex.current?.resetView()}>Reset view</Button>
        </div>
      </Card>

      <div className="grid xl:grid-cols-[1fr_320px] gap-4 items-start">
        <div className="relative rounded-xl overflow-hidden border border-line shadow-card" data-mode={mode}>
          {error ? <div className="p-8"><Callout tone="warn">{error}</Callout></div> : <div ref={host} className="relative h-[620px] w-full" aria-label="3D model of a pancreatic beta cell" />}
          <div className="absolute top-3 left-3 flex flex-col gap-1.5 pointer-events-none">
            <span className={cx("rounded-md px-2 py-1 text-[11.5px] font-medium backdrop-blur border", mode === "fluorescence" ? "bg-black/60 text-white/90 border-white/15" : "bg-white/85 text-ink border-line")}>
              Zoom level: {z.label}
            </span>
            <span className={cx("rounded-md px-2 py-1 text-[11px] backdrop-blur border", mode === "fluorescence" ? "bg-black/60 text-white/70 border-white/15" : "bg-white/85 text-muted border-line")}>
              Scroll to zoom · drag to rotate · click a structure
            </span>
          </div>
          {mode === "em" && <div className="absolute bottom-3 left-3 rounded-md bg-white/85 border border-line px-2 py-1 text-[11px] text-muted">Thin-section view, styled after transmission EM (rendered model, not a micrograph)</div>}
          {mode === "fluorescence" && <div className="absolute bottom-3 left-3 rounded-md bg-black/60 border border-white/15 px-2 py-1 text-[11px] text-white/70">Styled after multi-colour confocal imaging (rendered model)</div>}
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            {info ? (
              <div>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <Badge tone={info.level === 2 ? "accent" : "brand"}>{info.level === 2 ? "Sub-structure" : info.level === 1 ? "Organelle" : "Cell"}</Badge>
                    <h3 className="text-[16px] font-semibold mt-2">{info.name}</h3>
                  </div>
                  <button onClick={() => setSelected(null)} className="size-7 grid place-items-center rounded-md text-muted hover:bg-surface-3" aria-label="Close"><X className="size-4" /></button>
                </div>
                <p className="text-[13.5px] text-ink-2 mt-2 leading-6">{info.summary}</p>
                <div className="mt-3 rounded-lg bg-surface-2 border border-line p-3">
                  <div className="text-[12px] font-semibold text-muted mb-1">Role in insulin secretion</div>
                  <p className="text-[13px] text-ink-2 leading-5">{info.role}</p>
                </div>
                {info.source && <p className="text-[12px] text-faint mt-2">Source: {info.source}</p>}
                <Button size="sm" className="mt-3" icon={<Crosshair className="size-4" />} onClick={() => ex.current?.focus(selected!)}>Zoom to</Button>
              </div>
            ) : (
              <div className="flex gap-3">
                <Info className="size-5 text-brand-text shrink-0 mt-0.5" />
                <p className="text-[13px] text-muted leading-5">Click a structure in the cell to see what it is and how it contributes to insulin secretion. Labels for finer structures appear as you zoom closer to them.</p>
              </div>
            )}
          </Card>
          <Card>
            <div className="text-[12.5px] font-semibold text-ink-2 mb-2 flex items-center gap-1.5"><Tag className="size-3.5" />Go to</div>
            <div className="flex flex-wrap gap-1.5">
              {JUMPS.map((p) => (
                <button key={p} onClick={() => { setSelected(p); ex.current?.focus(p); }}
                  className={cx("rounded-md border px-2 py-1 text-[12.5px]", selected === p ? "bg-brand-soft border-brand text-brand-text" : "border-line-strong text-ink-2 hover:bg-surface-2")}>
                  {PARTS[p].name}
                </button>
              ))}
            </div>
          </Card>
          <Callout>
            Granules travel along microtubules to the membrane and fuse at a rate proportional to the meal model's static secretion,
            S = S<sub>b</sub> + β(G − G<sub>b</sub>), currently <b>{fold.toFixed(2)}× basal</b>. Geometry and counts are schematic: a real β-cell holds about 10,000 granules (Rorsman & Renström 2003), and a subset is drawn.
          </Callout>
        </div>
      </div>
    </div>
  );
}
