import { useMutation } from "@tanstack/react-query";
import { MethodsPanel } from "@/components/Methods";
import { Microscope, Play, Rows3 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { Badge, Button, Callout, Card, CardHeader, Field, Input, PageHeader, Select, Slider, Stat, useToast } from "@/components/ui";
import { api, type MealResult } from "@/lib/api";
import { fmt, rowsOf } from "@/lib/format";
import { setLastMeal } from "@/lib/simStore";
import { AskAI } from "@/components/AskAI";

type Pheno = "normal" | "insulin_resistant" | "type2";
const LABEL: Record<Pheno, string> = { normal: "Healthy adult", insulin_resistant: "Insulin resistant", type2: "Type 2 diabetes" };

export default function MealLab() {
  const toast = useToast();
  const nav = useNavigate();
  const c = useChartColors();
  const [f, setF] = useState({ phenotype: "normal" as Pheno, carbs1: 75, time1: 0, carbs2: 0, time2: 300, bw: 78, duration: 480, override: false, si: 1, bf: 1 });
  const [results, setResults] = useState<MealResult[]>([]);
  const color: Record<string, string> = { normal: c.v1, insulin_resistant: c.v2, type2: c.v3 };

  const run = useMutation({
    mutationFn: async (phenos: Pheno[]) => {
      const meals = [
        ...(f.carbs1 > 0 ? [{ time_min: f.time1, carbs_g: f.carbs1 }] : []),
        ...(f.carbs2 > 0 ? [{ time_min: f.time2, carbs_g: f.carbs2 }] : []),
      ];
      return Promise.all(phenos.map((p) => api.post<MealResult>("/api/physiology/meal", {
        meals, phenotype: p, body_weight_kg: f.bw, duration_min: f.duration,
        ...(f.override ? { insulin_sensitivity_scale: f.si, beta_cell_function_scale: f.bf } : {}),
      })));
    },
    onSuccess: (r) => { setResults(r); setLastMeal(r[0]); },
    onError: (e: Error) => toast(e.message, "error"),
  });

  useEffect(() => { run.mutate(["normal"]); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const primary = results[0];
  const s = primary?.summary;
  const glucoseData = useMemo(() => {
    if (!results.length) return [];
    const every = Math.max(1, Math.floor(results[0].series.t_min.length / 400));
    return rowsOf({ t: results[0].series.t_min, ...Object.fromEntries(results.map((r) => [r.phenotype.key, r.series.glucose_mg_dl])) }, every);
  }, [results]);
  const insulinData = useMemo(() => {
    if (!results.length) return [];
    const every = Math.max(1, Math.floor(results[0].series.t_min.length / 400));
    return rowsOf({ t: results[0].series.t_min, ...Object.fromEntries(results.map((r) => [r.phenotype.key, r.series.insulin_pmol_l])) }, every);
  }, [results]);
  const fluxData = useMemo(() => primary ? rowsOf({
    t: primary.series.t_min, ra: primary.series.ra_mg_kg_min, egp: primary.series.egp_mg_kg_min,
    uid: primary.series.uid_mg_kg_min, renal: primary.series.renal_mg_kg_min,
  }, Math.max(1, Math.floor(primary.series.t_min.length / 400))) : [], [primary]);

  const set = <K extends keyof typeof f>(k: K, v: (typeof f)[K]) => setF((x) => ({ ...x, [k]: v }));
  const g2 = s?.glucose_2h_mg_dl ?? null;
  const tone = (v: number | null, a: number, b: number) => (v === null ? undefined : v < a ? "ok" : v < b ? "warn" : "danger") as "ok" | "warn" | "danger" | undefined;

  return (
    <div>
      <PageHeader
        eyebrow="Physiology"
        title="Meal simulation"
        description={<>Post-prandial glucose and insulin from the Dalla Man–Rizza–Cobelli (2007) meal model. Only the healthy-adult phenotype uses the published parameter set; the others are labelled scalings of it.</>}
        actions={<>
          <AskAI page="Meal simulation" question="Interpret this meal simulation: what drives the glucose and insulin curves, and how do the phenotypes differ mechanistically?"
            summary={results.map((r) => `${r.phenotype.label}: peak glucose ${r.summary.peak_glucose_mg_dl.toFixed(0)} mg/dl at ${r.summary.time_to_peak_min.toFixed(0)} min, 2-h glucose ${r.summary.glucose_2h_mg_dl?.toFixed(0) ?? "n/a"} mg/dl, peak insulin ${r.summary.peak_insulin_pmol_l.toFixed(0)} pmol/l, time in range ${r.summary.time_in_range_70_180_pct.toFixed(0)}%, renal excretion ${r.summary.renal_excretion_mg_per_kg.toFixed(1)} mg/kg`).join("\n") + (results.length ? `\nMeals: carbs ${f.carbs1} g at ${f.time1} min${f.carbs2 ? `, ${f.carbs2} g at ${f.time2} min` : ""}; body weight ${f.bw} kg.` : "")} />
          <Button icon={<Microscope className="size-4" />} onClick={() => nav("/cells")} disabled={!primary}>Play in Cell theatre</Button>
        </>}
      />
      <div className="grid xl:grid-cols-[320px_1fr] gap-5 items-start">
        <Card className="xl:sticky xl:top-8">
          <CardHeader title="Scenario" />
          <form className="flex flex-col gap-4" onSubmit={(e) => { e.preventDefault(); run.mutate([f.phenotype]); }}>
            <Field label="Phenotype">
              <Select value={f.phenotype} onChange={(e) => set("phenotype", e.target.value as Pheno)}>
                <option value="normal">Healthy adult · published</option>
                <option value="insulin_resistant">Insulin resistant · illustrative</option>
                <option value="type2">Type 2 diabetes · illustrative</option>
              </Select>
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Meal 1 carbs (g)"><Input type="number" min={0} max={400} value={f.carbs1} onChange={(e) => set("carbs1", +e.target.value)} /></Field>
              <Field label="At (min)"><Input type="number" min={0} max={4000} step={5} value={f.time1} onChange={(e) => set("time1", +e.target.value)} /></Field>
              <Field label="Meal 2 carbs (g)"><Input type="number" min={0} max={400} value={f.carbs2} onChange={(e) => set("carbs2", +e.target.value)} /></Field>
              <Field label="At (min)"><Input type="number" min={0} max={4000} step={5} value={f.time2} onChange={(e) => set("time2", +e.target.value)} /></Field>
              <Field label="Body weight (kg)"><Input type="number" min={30} max={250} value={f.bw} onChange={(e) => set("bw", +e.target.value)} /></Field>
              <Field label="Duration (min)"><Input type="number" min={30} max={4320} step={30} value={f.duration} onChange={(e) => set("duration", +e.target.value)} /></Field>
            </div>
            <details className="group rounded-lg border border-line px-3 py-2.5">
              <summary className="cursor-pointer text-[13px] font-medium text-ink-2 list-none flex justify-between">Custom parameters <span className="text-faint group-open:rotate-180 transition">⌄</span></summary>
              <div className="flex flex-col gap-4 mt-3">
                <label className="flex items-center gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={f.override} onChange={(e) => set("override", e.target.checked)} /> Override phenotype scalings</label>
                <Slider label="Insulin sensitivity ×" value={f.si} min={0.05} max={2} step={0.05} onChange={(v) => setF((x) => ({ ...x, si: v, override: true }))} format={(v) => v.toFixed(2)} />
                <Slider label="β-cell function ×" value={f.bf} min={0.05} max={2} step={0.05} onChange={(v) => setF((x) => ({ ...x, bf: v, override: true }))} format={(v) => v.toFixed(2)} />
              </div>
            </details>
            <div className="flex gap-2">
              <Button type="submit" variant="primary" loading={run.isPending} icon={<Play className="size-4" />} className="flex-1">Simulate</Button>
              <Button type="button" onClick={() => run.mutate(["normal", "insulin_resistant", "type2"])} icon={<Rows3 className="size-4" />}>Compare</Button>
            </div>
            <p className="text-[12px] text-faint">Meals enter as impulses of glucose-equivalent carbohydrate, as in the original model.</p>
          </form>
        </Card>

        <div className="flex flex-col gap-5 min-w-0">
          {s && (
            <div className="grid grid-cols-2 md:grid-cols-3 2xl:grid-cols-6 gap-3">
              <Stat label="Peak glucose" value={fmt(s.peak_glucose_mg_dl, 0)} unit="mg/dl" tone={tone(s.peak_glucose_mg_dl, 180, 250)} />
              <Stat label="2-hour glucose" value={g2 === null ? "–" : fmt(g2, 0)} unit="mg/dl" tone={tone(g2, 140, 200)} hint="<140 normal · ≥200 diabetic" />
              <Stat label="Time to peak" value={fmt(s.time_to_peak_min, 0)} unit="min" />
              <Stat label="Peak insulin" value={fmt(s.peak_insulin_pmol_l, 0)} unit="pmol/l" />
              <Stat label="Time in range" value={fmt(s.time_in_range_70_180_pct, 0)} unit="%" tone={s.time_in_range_70_180_pct > 95 ? "ok" : s.time_in_range_70_180_pct > 70 ? "warn" : "danger"} hint="70–180 mg/dl" />
              <Stat label="Renal excretion" value={fmt(s.renal_excretion_mg_per_kg, 1)} unit="mg/kg" />
            </div>
          )}
          <ChartCard
            title="Plasma glucose"
            subtitle="mg/dl · shaded band 70–180 mg/dl · dashed lines: 2-h OGTT thresholds (ADA)"
            actions={<div className="flex flex-wrap justify-end gap-1.5">{results.map((r) => <Badge key={r.phenotype.key} tone={r.phenotype.published_parameter_set ? "brand" : "accent"}>{LABEL[r.phenotype.key as Pheno]}{!r.phenotype.published_parameter_set && " · illustrative"}</Badge>)}</div>}
          >
            <TimeSeries
              data={glucoseData} x="t" xLabel="min" height={300}
              series={results.map((r) => ({ key: r.phenotype.key, label: LABEL[r.phenotype.key as Pheno], color: color[r.phenotype.key] }))}
              refAreas={[{ y1: 70, y2: 180 }]}
              refLines={[
                { y: 140, label: "140 · impaired tolerance", color: c.v2 },
                { y: 200, label: "200 · diabetes", color: c.v3 },
                ...(f.carbs1 > 0 ? [{ x: f.time1 + 120, label: "2 h" }] : []),
              ]}
              yDomain={[40, "auto"]}
            />
          </ChartCard>
          <div className="grid lg:grid-cols-2 gap-5">
            <ChartCard title="Plasma insulin" subtitle="pmol/l">
              <TimeSeries data={insulinData} x="t" xLabel="min" height={240}
                series={results.map((r) => ({ key: r.phenotype.key, label: LABEL[r.phenotype.key as Pheno], color: color[r.phenotype.key] }))} yDomain={[0, "auto"]} />
            </ChartCard>
            <ChartCard title="Glucose fluxes" subtitle={`mg/kg/min · ${primary ? LABEL[primary.phenotype.key as Pheno] : ""}`}>
              <TimeSeries data={fluxData} x="t" xLabel="min" height={240} yDomain={[0, "auto"]} series={[
                { key: "ra", label: "Gut appearance", color: c.v2 },
                { key: "egp", label: "Liver production", color: c.v4 },
                { key: "uid", label: "Insulin-dependent use", color: c.v1 },
                { key: "renal", label: "Renal excretion", color: c.v3 },
              ]} />
            </ChartCard>
          </div>
          {primary && (
            <Callout title={primary.phenotype.label}>
              {primary.phenotype.description} Basal state: G = {fmt(primary.basal.Gb, 1)} mg/dl, I = {fmt(primary.basal.Ib, 1)} pmol/l;
              k<sub>p1</sub> = {fmt(primary.basal.kp1, 3)} mg/kg/min derived from the steady-state constraints.
            </Callout>
          )}
        </div>
      </div>
      <div className="mt-6"><MethodsPanel id="meal" /></div>
    </div>
  );
}
