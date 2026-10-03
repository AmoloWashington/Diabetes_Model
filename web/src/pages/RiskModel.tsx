import { useMutation, useQuery } from "@tanstack/react-query";
import { MethodsPanel } from "@/components/Methods";
import { Stethoscope } from "lucide-react";
import { useState } from "react";
import { Bars, ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { Button, Callout, Card, CardHeader, Field, Input, PageHeader, Select, Slider, Spinner, Stat, Table, cx, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt, pct } from "@/lib/format";

const SYMPTOMS: [string, string][] = [
  ["polyuria", "Polyuria"], ["polydipsia", "Polydipsia"], ["sudden_weight_loss", "Sudden weight loss"], ["weakness", "Weakness"],
  ["polyphagia", "Polyphagia"], ["genital_thrush", "Genital thrush"], ["visual_blurring", "Visual blurring"], ["itching", "Itching"],
  ["irritability", "Irritability"], ["delayed_healing", "Delayed healing"], ["partial_paresis", "Partial paresis"],
  ["muscle_stiffness", "Muscle stiffness"], ["alopecia", "Alopecia"], ["obesity", "Obesity"],
];
type M = Record<string, number>;
interface Card_ {
  dataset: { name: string; source: string; rows: number; unique_rows: number; duplicate_rows: number; positive: number; negative: number; age_range: number[] };
  evaluation: { scheme: string; models: Record<string, { pooled: M; ci95: Record<string, number[]> }>; naive_leaky_cv: M; leakage_note: string;
    roc_curve: { fpr: number[]; tpr: number[] }; calibration: { mean_pred: number; observed: number; n: number }[] };
  explanations: { odds_ratios: { label: string; odds_ratio: number; per: string }[]; permutation_importance: { label: string; auc_drop: number; sd_across_folds: number }[] };
  limitations: string[];
}
interface Pred {
  probability: number; probability_logistic: number; probability_random_forest: number; logistic_95ci: number[];
  explanation: { contributions: { label: string; log_odds: number }[] }; train_prevalence: number; warnings: string[]; disclaimer: string;
  prevalence_adjusted?: { target_prevalence: number; probability: number; assumption: string };
}
const NAMES: Record<string, string> = { logistic: "Logistic regression", random_forest: "Random forest", ensemble: "Ensemble (primary)" };

export default function RiskModel() {
  const toast = useToast();
  const c = useChartColors();
  const [age, setAge] = useState(45);
  const [male, setMale] = useState(false);
  const [on, setOn] = useState<Record<string, boolean>>({ polyuria: true, polydipsia: true });
  const [prev, setPrev] = useState(0);
  const card = useQuery({ queryKey: ["risk-card"], queryFn: () => api.get<Card_>("/api/risk/model-card"), staleTime: Infinity });
  const pred = useMutation({
    mutationFn: () => api.post<Pred>("/api/risk/predict", {
      patient: { age, male, ...Object.fromEntries(SYMPTOMS.map(([k]) => [k, !!on[k]])) }, ...(prev ? { target_prevalence: prev / 100 } : {}),
    }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const p = pred.data;
  const ev = card.data?.evaluation;

  return (
    <div>
      <PageHeader eyebrow="Data & models" title="Diabetes risk model"
        description="Symptom-based classifier trained on the UCI early-stage dataset (Sylhet Diabetes Hospital, Bangladesh), evaluated with duplicate-aware grouped cross-validation. It is not a population screening test." />
      <div className="grid xl:grid-cols-[340px_1fr] gap-5 items-start mb-8">
        <Card>
          <CardHeader title="Patient" />
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Age (years)"><Input type="number" min={1} max={120} value={age} onChange={(e) => setAge(+e.target.value)} /></Field>
              <Field label="Sex"><Select value={male ? "m" : "f"} onChange={(e) => setMale(e.target.value === "m")}><option value="f">Female</option><option value="m">Male</option></Select></Field>
            </div>
            <div>
              <div className="text-[12.5px] font-medium text-ink-2 mb-2">Symptoms present</div>
              <div className="flex flex-wrap gap-1.5">
                {SYMPTOMS.map(([k, l]) => (
                  <button key={k} type="button" aria-pressed={!!on[k]} onClick={() => setOn({ ...on, [k]: !on[k] })}
                    className={cx("rounded-md border px-2 py-1 text-[12.5px] transition-colors", on[k] ? "bg-brand-soft border-brand text-brand-text font-medium" : "border-line-strong text-ink-2 hover:bg-surface-2")}>{l}</button>
                ))}
              </div>
            </div>
            <Slider label="Re-express for prevalence" value={prev} min={0} max={50} step={1} onChange={setPrev} format={(v) => (v ? `${v}%` : "off")} />
            <Button variant="primary" icon={<Stethoscope className="size-4" />} loading={pred.isPending} onClick={() => pred.mutate()}>Estimate probability</Button>
          </div>
        </Card>
        <div className="flex flex-col gap-5 min-w-0">
          {p ? (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <Stat label="Ensemble probability" value={pct(p.probability, 1)} tone={p.probability < 0.2 ? "ok" : p.probability < 0.5 ? "warn" : "danger"} hint={`training prevalence ${pct(p.train_prevalence, 1)}`} />
                <Stat label="Logistic (95% interval)" value={pct(p.probability_logistic, 1)} hint={`${pct(p.logistic_95ci[0])} – ${pct(p.logistic_95ci[1])}`} />
                <Stat label="Random forest" value={pct(p.probability_random_forest, 1)} />
                <Stat label="At chosen prevalence" value={p.prevalence_adjusted ? pct(p.prevalence_adjusted.probability, 1) : "–"} hint={p.prevalence_adjusted ? `prevalence ${pct(p.prevalence_adjusted.target_prevalence, 0)}` : "set a prevalence"} />
              </div>
              {p.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
              <ChartCard title="Why: log-odds contributions" subtitle="Exact additive contributions of the logistic model relative to a reference patient (mean age, female, no symptoms). Positive values raise risk.">
                <Bars horizontal data={p.explanation.contributions.slice(0, 12).map((x) => ({ f: x.label, v: +x.log_odds.toFixed(3) }))} x="f" y="v"
                  height={Math.max(160, 34 * Math.min(12, p.explanation.contributions.length))} xLabel="log-odds" colorFn={(r) => (Number(r.v) > 0 ? c.v3 : c.v1)} />
              </ChartCard>
              <p className="text-[12px] text-faint">{p.disclaimer} {p.prevalence_adjusted?.assumption}</p>
            </>
          ) : <Card><p className="text-[13px] text-muted">Select symptoms and estimate to see the probability and its explanation.</p></Card>}
        </div>
      </div>

      <h2 className="text-[18px] font-semibold tracking-[-0.01em] mb-4">Model card</h2>
      {!card.data || !ev ? <Card><Spinner label="Training with repeated grouped cross-validation (first run only, about 30 s)…" /></Card> : (
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <Stat label="Records" value={card.data.dataset.rows} hint={`${card.data.dataset.unique_rows} unique`} />
            <Stat label="Exact duplicate rows" value={card.data.dataset.duplicate_rows} tone="warn" />
            <Stat label="Leak-free ROC AUC" value={fmt(ev.models.ensemble.pooled.auc, 3)} hint={`95% CI ${fmt(ev.models.ensemble.ci95.auc[0], 3)}–${fmt(ev.models.ensemble.ci95.auc[1], 3)}`} />
            <Stat label="Accuracy: naive → leak-free" value={`${fmt(100 * ev.naive_leaky_cv.accuracy, 1)} → ${fmt(100 * ev.models.ensemble.pooled.accuracy, 1)}%`} tone="danger" />
          </div>
          <Card>
            <CardHeader title="Validation" subtitle={ev.scheme} />
            <Table head={["Model", "ROC AUC", "Brier ↓", "Accuracy", "Sensitivity", "Specificity"]}>
              {Object.entries(ev.models).map(([k, m]) => (
                <tr key={k}><td className="px-3 py-2.5 font-medium">{NAMES[k] ?? k}</td>
                  {["auc", "brier", "accuracy", "sensitivity", "specificity"].map((x) => (
                    <td key={x} className="px-3 py-2.5"><div className="num font-mono text-[12.5px]">{fmt(m.pooled[x], 3)}</div><div className="num font-mono text-[11px] text-faint">[{fmt(m.ci95[x][0], 3)}, {fmt(m.ci95[x][1], 3)}]</div></td>
                  ))}</tr>
              ))}
              <tr className="bg-danger-soft/50"><td className="px-3 py-2.5 text-danger">Naive random K-fold (leaky)</td>
                {["auc", "brier", "accuracy", "sensitivity", "specificity"].map((x) => <td key={x} className="px-3 py-2.5 num font-mono text-[12.5px] text-danger">{fmt(ev.naive_leaky_cv[x], 3)}</td>)}</tr>
            </Table>
            <p className="text-[12px] text-faint mt-3">95% CIs from 500 bootstrap resamples of duplicate groups. {ev.leakage_note}</p>
          </Card>
          <div className="grid lg:grid-cols-2 gap-5">
            <ChartCard title="ROC curve" subtitle="Pooled out-of-fold predictions">
              <TimeSeries data={ev.roc_curve.fpr.map((f, i) => ({ fpr: f, tpr: ev.roc_curve.tpr[i], chance: f }))} x="fpr" xLabel="false positive rate" height={260} xDomain={[0, 1]} yDomain={[0, 1]}
                series={[{ key: "tpr", label: `Ensemble (AUC ${fmt(ev.models.ensemble.pooled.auc, 3)})`, color: c.v1 }, { key: "chance", label: "Chance", color: c.faint, dashed: true }]} />
            </ChartCard>
            <ChartCard title="Calibration" subtitle="Observed frequency vs mean predicted probability (deciles)">
              <TimeSeries data={ev.calibration.map((b) => ({ x: b.mean_pred, obs: b.observed, ideal: b.mean_pred }))} x="x" xLabel="predicted" height={260} xDomain={[0, 1]} yDomain={[0, 1]}
                series={[{ key: "obs", label: "Observed", color: c.v4 }, { key: "ideal", label: "Perfect calibration", color: c.faint, dashed: true }]} />
            </ChartCard>
          </div>
          <div className="grid lg:grid-cols-2 gap-5">
            <Card><CardHeader title="Adjusted odds ratios" subtitle="Logistic regression" />
              <Table head={["Feature", "OR", "Per"]}>{card.data.explanations.odds_ratios.map((o) => (
                <tr key={o.label}><td className="px-3 py-2">{o.label}</td><td className="px-3 py-2 num font-mono text-[12.5px]">{fmt(o.odds_ratio, 2)}</td><td className="px-3 py-2 text-muted text-[12px]">{o.per}</td></tr>))}</Table>
            </Card>
            <Card><CardHeader title="Held-out permutation importance" subtitle="Drop in AUC when a feature is shuffled" />
              <Table head={["Feature", "ΔAUC", "SD"]}>{card.data.explanations.permutation_importance.map((o) => (
                <tr key={o.label}><td className="px-3 py-2">{o.label}</td><td className="px-3 py-2 num font-mono text-[12.5px]">{fmt(o.auc_drop, 4)}</td><td className="px-3 py-2 num font-mono text-[12px] text-muted">{fmt(o.sd_across_folds, 4)}</td></tr>))}</Table>
            </Card>
          </div>
          <Card><CardHeader title="Data and limitations" subtitle={`${card.data.dataset.name}. ${card.data.dataset.source}`} />
            <ul className="list-disc ml-5 text-[13px] text-ink-2 space-y-1">{card.data.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
          </Card>
        </div>
      )}
      <div className="mt-6"><MethodsPanel id="risk" /></div>
    </div>
  );
}
