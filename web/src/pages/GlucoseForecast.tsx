import { useMutation, useQuery } from "@tanstack/react-query";
import { FileUp, FlaskConical } from "lucide-react";
import { useRef, useState } from "react";
import { AskAI } from "@/components/AskAI";
import { ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { MethodsPanel } from "@/components/Methods";
import { Badge, Button, Callout, Card, CardHeader, PageHeader, Stat, Table, cx, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

interface Row { horizon_min: number; model: string; rmse: number; mae: number; n: number; skill_vs_persistence: number | null }
interface Example { horizon_min: number; t_hours: number[]; actual: number[]; predicted: number[] }
interface Result { table: Row[]; example: Example | null; n_subjects: number; days?: number; source?: string; unit_detected?: string }

const MODELS = ["persistence", "linear", "ridge"] as const;
const MODEL_LABEL: Record<string, string> = { persistence: "Persistence (baseline)", linear: "Linear slope", ridge: "Ridge regression" };
const HORIZONS = [30, 60];

export default function GlucoseForecast() {
  const toast = useToast();
  const c = useChartColors();
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploadName, setUploadName] = useState<string | null>(null);

  const bench = useQuery({ queryKey: ["forecast-benchmark"], queryFn: () => api.get<Result>("/api/forecast/benchmark") });
  const evalMut = useMutation({
    mutationFn: (csv: string) => api.post<Result>("/api/forecast/evaluate", { csv }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const r = (evalMut.isSuccess && uploadName ? evalMut.data : null) ?? bench.data;
  const onUpload = r === evalMut.data && !!uploadName;

  function cell(h: number, model: string): Row | undefined {
    return r?.table.find((x) => x.horizon_min === h && x.model === model);
  }
  const exampleRows = (r?.example?.t_hours ?? []).map((t, i) => ({ t, actual: r!.example!.actual[i], predicted: r!.example!.predicted[i] }));

  const summary = r ? `Glucose forecasting benchmark (${onUpload ? "uploaded CGM" : "synthetic in-silico cohort, " + r.n_subjects + " virtual subjects"}). Predicting glucose 30 and 60 min ahead from the last hour of CGM. RMSE (mg/dl): ${HORIZONS.map((h) => MODELS.map((m) => `${m} ${h}min ${fmt(cell(h, m)?.rmse, 1)}`).join(", ")).join("; ")}. Ridge skill vs persistence: 30 min ${fmt((cell(30, "ridge")?.skill_vs_persistence ?? 0) * 100, 0)}%, 60 min ${fmt((cell(60, "ridge")?.skill_vs_persistence ?? 0) * 100, 0)}%.` : "";

  async function handleFile(f: File | undefined) {
    if (!f) return;
    if (f.size > 15 * 1024 * 1024) { toast("File is larger than 15 MB.", "error"); return; }
    setUploadName(f.name);
    evalMut.mutate(await f.text());
  }

  return (
    <div>
      <PageHeader eyebrow="Physiology · Benchmark" title="Glucose forecasting"
        description="Can we predict where blood glucose is heading? This benchmark forecasts glucose 30 and 60 minutes ahead from the last hour of CGM, and scores several models honestly against the standard 'persistence' baseline, with no data leakage."
        actions={<AskAI page="Glucose forecasting benchmark" summary={summary} question="Explain this glucose forecasting benchmark: what do RMSE and skill-vs-persistence mean, and is beating persistence a real result?" />} />

      <Callout tone="warn" title="What these numbers are, and are not">
        The built-in cohort is <b>synthetic</b> — virtual subjects from the Dalla Man meal model, not real patients and not the OhioT1DM benchmark. The numbers compare the models against each other on reproducible in-silico data; they are not a clinical accuracy claim and are not comparable to published results on real CGM. Upload your own CGM file below to run the same benchmark on real data. This is research and education software, not a medical device.
      </Callout>

      <div className="mt-5 flex flex-wrap items-center gap-3">
        {onUpload ? <Badge tone="brand">{uploadName}</Badge> : <Badge tone="accent">Synthetic cohort · {r?.n_subjects ?? "…"} virtual subjects{r?.days ? ` · ${r.days} days each` : ""}</Badge>}
        <input ref={fileRef} type="file" accept=".csv,.txt,text/csv" className="hidden" onChange={(e) => { handleFile(e.target.files?.[0]); e.target.value = ""; }} />
        <Button size="sm" variant="secondary" icon={<FileUp className="size-4" />} loading={evalMut.isPending} onClick={() => fileRef.current?.click()}>Benchmark my CGM file</Button>
        {onUpload && <Button size="sm" variant="ghost" icon={<FlaskConical className="size-4" />} onClick={() => { setUploadName(null); evalMut.reset(); }}>Back to synthetic cohort</Button>}
        {r?.unit_detected && <span className="text-[12px] text-muted">values read as {r.unit_detected}</span>}
      </div>

      {r && (
        <>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-5">
            {HORIZONS.map((h) => (
              <Stat key={`p${h}`} label={`Persistence RMSE @ ${h} min`} value={fmt(cell(h, "persistence")?.rmse, 1)} unit="mg/dl" hint="the baseline to beat" />
            ))}
            {HORIZONS.map((h) => {
              const skill = cell(h, "ridge")?.skill_vs_persistence ?? 0;
              return <Stat key={`r${h}`} label={`Ridge skill @ ${h} min`} value={fmt(skill * 100, 0)} unit="%" tone={skill > 0 ? "ok" : "warn"} hint={`RMSE ${fmt(cell(h, "ridge")?.rmse, 1)} mg/dl`} />;
            })}
          </div>

          <Card padded={false} className="mt-5">
            <div className="px-5 pt-5"><CardHeader title="Accuracy by model and horizon" subtitle="RMSE and MAE in mg/dl (lower is better); skill is improvement over persistence" /></div>
            <Table head={["Model", "Horizon", "RMSE", "MAE", "Skill vs persistence", "Windows"]}>
              {HORIZONS.flatMap((h) => MODELS.map((m) => {
                const row = cell(h, m);
                if (!row) return null;
                const skill = row.skill_vs_persistence;
                return (
                  <tr key={`${h}-${m}`} className={cx(m === "ridge" && "bg-brand-soft/40")}>
                    <td className="px-3 py-2.5 font-medium">{MODEL_LABEL[m]}</td>
                    <td className="px-3 py-2.5 num text-ink-2">{h} min</td>
                    <td className="px-3 py-2.5 num">{fmt(row.rmse, 1)}</td>
                    <td className="px-3 py-2.5 num text-ink-2">{fmt(row.mae, 1)}</td>
                    <td className={cx("px-3 py-2.5 num", skill === null ? "text-faint" : skill > 0 ? "text-ok" : skill < 0 ? "text-warn" : "text-muted")}>
                      {skill === null ? "—" : `${skill > 0 ? "+" : ""}${fmt(skill * 100, 1)}%`}
                    </td>
                    <td className="px-3 py-2.5 num text-faint">{row.n.toLocaleString()}</td>
                  </tr>
                );
              }))}
            </Table>
          </Card>

          {r.example && exampleRows.length > 0 && (
            <ChartCard className="mt-5" title={`Example: ${r.example.horizon_min}-minute-ahead prediction`} subtitle="Ridge model on a held-out test window (mg/dl)">
              <TimeSeries data={exampleRows} x="t" xLabel="hours into the test window" height={240}
                refAreas={[{ y1: 70, y2: 180, color: c.v1 }]}
                series={[
                  { key: "actual", label: "Actual glucose", color: c.v1 },
                  { key: "predicted", label: `Predicted (${r.example.horizon_min} min ahead)`, color: c.v2, dashed: true },
                ]} />
            </ChartCard>
          )}

          <Callout className="mt-5">
            <b>Why persistence is the baseline.</b> Glucose changes slowly, so "it will stay about where it is now" is already a strong prediction — any model has to beat it to be worth anything. Here ridge regression beats persistence by about {fmt((cell(30, "ridge")?.skill_vs_persistence ?? 0) * 100, 0)}% at 30 minutes and {fmt((cell(60, "ridge")?.skill_vs_persistence ?? 0) * 100, 0)}% at 60 minutes on the synthetic cohort, while naive slope extrapolation does worse than persistence because it amplifies noise. The honest next step is to run this on real CGM (OhioT1DM) and compare against published deep-learning results.
          </Callout>
          <div className="mt-5"><MethodsPanel id="forecast" /></div>
        </>
      )}
      {bench.isLoading && !r && <p className="text-[13px] text-muted mt-6">Running the benchmark…</p>}
    </div>
  );
}
