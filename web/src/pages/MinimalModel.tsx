import { useMutation } from "@tanstack/react-query";
import { FlaskConical, Play, Sigma } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { TimeSeries, useChartColors } from "@/components/charts";
import { Button, Callout, Card, CardHeader, Field, Input, PageHeader, Stat, Textarea, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt, rowsOf } from "@/lib/format";

interface Sim { t_min: number[]; glucose_mg_dl: number[]; insulin_uU_ml: number[]; G0_mg_dl: number; Kg_pct_per_min: number | null }
interface Fit {
  SG_per_min: number; SI_per_min_per_uU_ml: number; p2_per_min: number; G0_mg_dl: number; cv_percent: Record<string, number | null>;
  fitted_t_min: number[]; fitted_glucose_mg_dl: number[]; converged: boolean; rmse_mg_dl: number; note: string;
}

export default function MinimalModel() {
  const toast = useToast();
  const c = useChartColors();
  const [p, setP] = useState({ SI: 5, SG: 0.025, p2: 0.025, dose_g_per_kg: 0.3, Gb: 90, Ib: 10 });
  const [sim, setSim] = useState<Sim | null>(null);
  const [text, setText] = useState("");
  const [fit, setFit] = useState<{ fit: Fit; rows: number[][] } | null>(null);

  const simulate = useMutation({
    mutationFn: () => api.post<Sim>("/api/physiology/ivgtt", { ...p, SI: p.SI * 1e-4 }),
    onSuccess: setSim, onError: (e: Error) => toast(e.message, "error"),
  });
  const runFit = useMutation({
    mutationFn: async () => {
      const rows = text.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#")).map((l) => l.split(/[,;\s]+/).map(Number));
      if (rows.some((r) => r.length < 3 || r.some((v) => !Number.isFinite(v)))) throw new Error("Each line needs three numbers: t, glucose, insulin.");
      const f = await api.post<Fit>("/api/physiology/ivgtt/fit", { t_min: rows.map((r) => r[0]), glucose_mg_dl: rows.map((r) => r[1]), insulin_uU_ml: rows.map((r) => r[2]) });
      return { fit: f, rows };
    },
    onSuccess: setFit, onError: (e: Error) => toast(e.message, "error"),
  });
  useEffect(() => { simulate.mutate(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const simData = useMemo(() => sim ? rowsOf({ t: sim.t_min, g: sim.glucose_mg_dl, i: sim.insulin_uU_ml }) : [], [sim]);
  const fitData = useMemo(() => {
    if (!fit) return [];
    const m = new Map<number, { t: number; obs?: number; fit?: number }>();
    fit.rows.filter((r) => r[0] >= 0).forEach((r) => m.set(r[0], { t: r[0], obs: r[1] }));
    fit.fit.fitted_t_min.forEach((t, i) => m.set(t, { ...(m.get(t) ?? { t }), fit: fit.fit.fitted_glucose_mg_dl[i] }));
    return [...m.values()].sort((a, b) => a.t - b.t) as unknown as Record<string, number>[];
  }, [fit]);

  const example = () => {
    if (!sim) return;
    const ts = [0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180];
    const at = (arr: number[], t: number) => arr[Math.max(0, sim.t_min.findIndex((x) => x >= t))];
    const gauss = () => Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());
    setText(`# SYNTHETIC data from the simulation above, 1.5% glucose noise (true S_I = ${p.SI}×10⁻⁴)\n# t_min, glucose_mg_dl, insulin_uU_ml\n` +
      ts.map((t) => t === 0 ? `0, ${p.Gb}, ${p.Ib}` : `${t}, ${(at(sim.glucose_mg_dl, t) * (1 + 0.015 * gauss())).toFixed(1)}, ${at(sim.insulin_uU_ml, t).toFixed(1)}`).join("\n"));
  };
  const cv = (k: string) => (fit?.fit.cv_percent[k] == null ? "–" : `CV ${fmt(fit.fit.cv_percent[k]!, 1)}%`);

  return (
    <div>
      <PageHeader eyebrow="Physiology" title="Insulin sensitivity: minimal model"
        description={<>Bergman et al. (Am J Physiol 1979): dG/dt = −(S<sub>G</sub> + X)G + S<sub>G</sub>G<sub>b</sub>, dX/dt = −p₂X + p₃(I − I<sub>b</sub>), S<sub>I</sub> = p₃/p₂.</>} />
      <div className="grid xl:grid-cols-2 gap-5">
        <Card>
          <CardHeader title="Forward IVGTT simulation" subtitle="Defaults are illustrative, typical-order values." />
          <div className="grid grid-cols-3 gap-3">
            <Field label="S_I (×10⁻⁴)"><Input type="number" step={0.1} value={p.SI} onChange={(e) => setP({ ...p, SI: +e.target.value })} /></Field>
            <Field label="S_G (min⁻¹)"><Input type="number" step={0.001} value={p.SG} onChange={(e) => setP({ ...p, SG: +e.target.value })} /></Field>
            <Field label="p₂ (min⁻¹)"><Input type="number" step={0.001} value={p.p2} onChange={(e) => setP({ ...p, p2: +e.target.value })} /></Field>
            <Field label="Dose (g/kg)"><Input type="number" step={0.05} value={p.dose_g_per_kg} onChange={(e) => setP({ ...p, dose_g_per_kg: +e.target.value })} /></Field>
            <Field label="G_b (mg/dl)"><Input type="number" value={p.Gb} onChange={(e) => setP({ ...p, Gb: +e.target.value })} /></Field>
            <Field label="I_b (µU/ml)"><Input type="number" value={p.Ib} onChange={(e) => setP({ ...p, Ib: +e.target.value })} /></Field>
          </div>
          <Button className="mt-4" variant="primary" icon={<Play className="size-4" />} loading={simulate.isPending} onClick={() => simulate.mutate()}>Simulate</Button>
          {sim && (
            <div className="mt-5">
              <TimeSeries data={simData} x="t" xLabel="min" height={250}
                series={[{ key: "g", label: "Glucose (mg/dl)", color: c.v2 }, { key: "i", label: "Insulin (µU/ml)", color: c.v1, axis: "right" }]} />
              <p className="text-[12.5px] text-muted mt-2">G₀ = {fmt(sim.G0_mg_dl, 0)} mg/dl · K<sub>G</sub> (10–40 min) = {fmt(sim.Kg_pct_per_min, 2)} %/min</p>
            </div>
          )}
        </Card>
        <Card>
          <CardHeader title="Estimate S_I from data" subtitle="Nonlinear least squares with measured insulin as the forcing function." />
          <Field label="FSIVGTT samples: t_min, glucose_mg_dl, insulin_uU_ml (8–60 rows)">
            <Textarea rows={9} className="font-mono text-[12px]" value={text} onChange={(e) => setText(e.target.value)} placeholder="0, 90, 10&#10;2, 245, 80&#10;…" />
          </Field>
          <div className="flex gap-2 mt-3">
            <Button variant="primary" icon={<Sigma className="size-4" />} loading={runFit.isPending} onClick={() => runFit.mutate()} disabled={!text.trim()}>Fit minimal model</Button>
            <Button icon={<FlaskConical className="size-4" />} onClick={example}>Synthetic example</Button>
          </div>
          {fit && (
            <div className="mt-5 flex flex-col gap-4">
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <Stat label="S_I (×10⁻⁴)" value={fmt(fit.fit.SI_per_min_per_uU_ml * 1e4, 3)} hint={cv("SI")} />
                <Stat label="S_G (min⁻¹)" value={fmt(fit.fit.SG_per_min, 4)} hint={cv("SG")} />
                <Stat label="p₂ (min⁻¹)" value={fmt(fit.fit.p2_per_min, 4)} hint={cv("p2")} />
                <Stat label="RMSE" value={fmt(fit.fit.rmse_mg_dl, 2)} unit="mg/dl" hint={fit.fit.converged ? "Converged" : "Not converged"} />
              </div>
              <TimeSeries data={fitData} x="t" xLabel="min" height={230}
                series={[{ key: "obs", label: "Measured", color: c.v2, points: true }, { key: "fit", label: "Minimal-model fit", color: c.v1 }]} />
              <Callout>{fit.fit.note}</Callout>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
