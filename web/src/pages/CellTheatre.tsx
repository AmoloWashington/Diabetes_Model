import { Circle, Pause, Play, Square } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Button, Callout, Card, PageHeader, Segmented, Slider, useToast } from "@/components/ui";
import { useLastMeal } from "@/lib/simStore";
import { useRecorder } from "@/lib/useRecorder";
import type { CellTheatre as Theatre } from "@/lib/viz/cells";

type Scene = "beta" | "muscle" | "dogma";

export default function CellTheatre() {
  const toast = useToast();
  const canvas = useRef<HTMLCanvasElement>(null);
  const theatre = useRef<Theatre | null>(null);
  const meal = useLastMeal();
  const [scene, setScene] = useState<Scene>("beta");
  const [legend, setLegend] = useState("");
  const [readout, setReadout] = useState<string[]>([]);
  const [useSim, setUseSim] = useState(!!meal);
  const [t, setT] = useState(60);
  const [playing, setPlaying] = useState(false);
  const [glucose, setGlucose] = useState(150);
  const [ir, setIr] = useState(0);
  const rec = useRecorder(() => canvas.current, "glucolab-cell");

  useEffect(() => {
    let alive = true;
    import("@/lib/viz/cells").then(({ CellTheatre }) => {
      if (!alive || !canvas.current) return;
      theatre.current = new CellTheatre(canvas.current, setReadout);
      setLegend(theatre.current.setScene("beta"));
    });
    return () => { alive = false; theatre.current?.dispose(); };
  }, []);

  useEffect(() => { if (theatre.current) setLegend(theatre.current.setScene(scene)); }, [scene]);

  const series = useMemo(() => {
    if (!meal) return null;
    const Sb = meal.basal.Sb;
    return { t: meal.series.t_min, g: meal.series.glucose_mg_dl, i: meal.series.insulin_pmol_l, fold: meal.series.secretion_pmol_kg_min.map((v) => v / Sb) };
  }, [meal]);

  useEffect(() => {
    const th = theatre.current;
    if (!th) return;
    if (useSim && series) {
      const idx = Math.min(series.t.length - 1, Math.max(0, series.t.findIndex((x) => x >= t)));
      th.setDrive({ glucose: series.g[idx], insulin: series.i[idx], secretion: Math.max(0, series.fold[idx]), ir: ir / 100 });
    } else {
      // Static secretion of the Dalla Man model with published normal values: S = S_b + β (G − G_b)
      const Gb = 91.76, beta = 0.11, Sb = 1.5434;
      const fold = Math.max(0, (Sb + beta * (glucose - Gb)) / Sb);
      th.setDrive({ glucose, secretion: fold, insulin: 25.49 * fold, ir: ir / 100 });
    }
  }, [useSim, series, t, glucose, ir, legend]);

  useEffect(() => {
    if (!playing || !series) return;
    const max = series.t[series.t.length - 1];
    const id = setInterval(() => setT((x) => (x + 1 > max ? 0 : x + 1)), 60);
    return () => clearInterval(id);
  }, [playing, series]);

  const maxT = series ? series.t[series.t.length - 1] : 480;

  return (
    <div>
      <PageHeader
        eyebrow="Cell & molecular"
        title="Cell theatre"
        description="Animations driven by the simulated plasma glucose, insulin and secretion rate. The sequence of molecular events follows the cited reviews; geometry is schematic."
        actions={
          <Button variant={rec.recording ? "danger" : "secondary"} icon={rec.recording ? <Square className="size-4" /> : <Circle className="size-4 fill-current text-danger" />}
            onClick={() => { const err = rec.toggle(); if (err) toast(err, "error"); }}>
            {rec.recording ? "Stop & save video" : "Record video"}
          </Button>
        }
      />
      <Card className="mb-4">
        <div className="flex flex-col gap-4">
          <Segmented<Scene> value={scene} onChange={setScene} options={[
            { value: "beta", label: "β-cell secretion" },
            { value: "muscle", label: "Muscle: insulin → GLUT4" },
            { value: "dogma", label: "INS gene → insulin" },
          ]} />
          <div className="grid md:grid-cols-[auto_1fr_1fr] gap-5 items-end">
            <label className="flex items-center gap-2 text-[13px] text-ink-2 h-9">
              <input type="checkbox" checked={useSim} disabled={!series} onChange={(e) => setUseSim(e.target.checked)} />
              Drive from meal simulation
            </label>
            {useSim && series ? (
              <div className="flex items-end gap-3">
                <div className="flex-1"><Slider label="Time after start" value={t} min={0} max={maxT} step={1} onChange={setT} format={(v) => `${v} min`} /></div>
                <Button size="sm" icon={playing ? <Pause className="size-4" /> : <Play className="size-4" />} onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</Button>
              </div>
            ) : (
              <Slider label="Plasma glucose" value={glucose} min={40} max={400} step={1} onChange={setGlucose} format={(v) => `${v} mg/dl`} />
            )}
            <Slider label="Insulin resistance (signal lost)" value={ir} min={0} max={95} step={5} onChange={setIr} format={(v) => `${v}%`} />
          </div>
          {!series && <p className="text-[12.5px] text-muted">Run a <Link to="/physiology/meal" className="text-brand-text font-medium hover:underline">meal simulation</Link> to drive the cells with the model's time course.</p>}
        </div>
      </Card>
      <div className="relative rounded-xl overflow-hidden border border-line shadow-card bg-[#050910]">
        <canvas ref={canvas} width={1200} height={680} className="block w-full h-auto aspect-[1200/680]" onClick={() => theatre.current?.click()} aria-label="Cell animation" />
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1 pointer-events-none">
          {readout.map((l) => <span key={l} className="rounded-md bg-black/60 text-white/90 font-mono text-[11.5px] px-2 py-0.5 backdrop-blur">{l}</span>)}
        </div>
      </div>
      <div className="mt-4">
        <Callout><div className="[&_ol]:list-decimal [&_ol]:ml-5 [&_ol]:mt-1.5 [&_li]:my-0.5 [&_b]:text-ink" dangerouslySetInnerHTML={{ __html: legend }} /></Callout>
      </div>
    </div>
  );
}
