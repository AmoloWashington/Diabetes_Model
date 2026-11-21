import { useMutation, useQuery } from "@tanstack/react-query";
import { MethodsPanel } from "@/components/Methods";
import { Activity, Stethoscope } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { AskAI } from "@/components/AskAI";
import { Bars, ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { Badge, Button, Callout, Card, CardHeader, Field, Input, PageHeader, Select, Slider, Spinner, Stat, Table, cx, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt, pct } from "@/lib/format";

interface ScreenResult {
  probability_pct: number; population_rate_pct: number; percentile: number; fold_vs_population: number;
  band: "lower" | "average" | "higher"; recommend_test: boolean; test_threshold_pct: number; note: string;
}

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
        description="Two complementary models. First, a non-invasive pre-screening estimate from body measurements alone, built on real US population data (NHANES). Below it, a symptom-based classifier trained on a clinical dataset. Neither is a diagnosis." />

      <PreScreen />

      <div className="flex items-center gap-2 mt-10 mb-4">
        <Stethoscope className="size-4 text-brand-text" />
        <h2 className="text-[18px] font-semibold tracking-[-0.01em]">Symptom-based classifier</h2>
      </div>
      <Callout className="mb-4">
        A different question from the pre-screening tool above. This model is trained on patients who <b>already presented with symptoms</b> at a clinic (the UCI early-stage dataset, Sylhet Diabetes Hospital, Bangladesh), so it estimates how a symptom picture relates to diabetes in that setting — <b>not</b> the chance that a random person in the population has diabetes. It is not a population screening test.
      </Callout>
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

/** Non-invasive pre-screening from NHANES: body measurements only, no blood test. */
function PreScreen() {
  const toast = useToast();
  const c = useChartColors();
  const [age, setAge] = useState(50);
  const [sex, setSex] = useState<"male" | "female">("female");
  const [weight, setWeight] = useState(82);
  const [height, setHeight] = useState(170);
  const [waist, setWaist] = useState(96);
  const bmi = weight > 0 && height > 0 ? weight / (height / 100) ** 2 : 0;

  const run = useMutation({
    mutationFn: () => api.post<ScreenResult>("/api/nhanes/screen", { age, sex, bmi: +bmi.toFixed(1), waist_cm: waist }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const r = run.data;
  const summary = r
    ? `NHANES non-invasive pre-screening: a ${age}-year-old ${sex}, BMI ${bmi.toFixed(1)}, waist ${waist} cm. Estimated probability of undiagnosed diabetes ${r.probability_pct}% vs a ${r.population_rate_pct}% US adult average (${r.fold_vs_population}× population, ${r.percentile}th percentile). Blood test ${r.recommend_test ? "recommended" : "not indicated"} at this model's 80%-sensitivity threshold.`
    : "";

  const tone = r ? (r.band === "higher" ? "danger" : r.band === "average" ? "warn" : "ok") : "ok";

  return (
    <div className="grid xl:grid-cols-[340px_1fr] gap-5 items-start">
      <Card>
        <CardHeader title="Your measurements" subtitle="No blood test needed" />
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Age (years)"><Input type="number" min={20} max={100} value={age} onChange={(e) => setAge(+e.target.value)} /></Field>
            <Field label="Sex"><Select value={sex} onChange={(e) => setSex(e.target.value as "male" | "female")}><option value="female">Female</option><option value="male">Male</option></Select></Field>
            <Field label="Weight (kg)"><Input type="number" min={30} max={250} value={weight} onChange={(e) => setWeight(+e.target.value)} /></Field>
            <Field label="Height (cm)"><Input type="number" min={120} max={220} value={height} onChange={(e) => setHeight(+e.target.value)} /></Field>
            <Field label="Waist (cm)"><Input type="number" min={50} max={200} value={waist} onChange={(e) => setWaist(+e.target.value)} /></Field>
            <Field label="BMI (computed)"><Input type="number" value={+bmi.toFixed(1)} readOnly tabIndex={-1} className="bg-surface-2 text-muted" /></Field>
          </div>
          <Button variant="primary" icon={<Activity className="size-4" />} loading={run.isPending}
            disabled={bmi < 12 || bmi > 70} onClick={() => run.mutate()}>Estimate from measurements</Button>
          <p className="text-[11.5px] text-faint">Model fitted on 4,747 US adults in NHANES 2017–2018. <Link to="/data/nhanes" className="text-brand-text hover:underline">See how it was built and validated →</Link></p>
        </div>
      </Card>

      <div className="flex flex-col gap-5 min-w-0">
        {r ? (
          <>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Badge tone={r.recommend_test ? "accent" : "brand"}>{r.recommend_test ? "A blood test is worthwhile" : "Below the testing threshold"}</Badge>
              <AskAI page="Non-invasive diabetes pre-screening (NHANES)" summary={summary}
                question="Explain my non-invasive pre-screening estimate: what the probability and percentile mean, and whether I should get a blood test." />
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat label="Your estimate" value={fmt(r.probability_pct, 1)} unit="%" tone={tone} hint="chance of undiagnosed diabetes" />
              <Stat label="US adult average" value={fmt(r.population_rate_pct, 1)} unit="%" hint="NHANES prevalence" />
              <Stat label="vs population" value={`${fmt(r.fold_vs_population, 2)}×`} tone={tone} hint={`${fmt(r.percentile, 0)}th percentile`} />
              <Stat label="Testing threshold" value={fmt(r.test_threshold_pct, 1)} unit="%" hint="catches ~80% of cases" />
            </div>
            <Callout tone={r.recommend_test ? "warn" : "ok"} title={r.recommend_test ? "A blood test is worth doing" : "Lower than the model's testing threshold"}>
              {r.recommend_test
                ? `Your estimate (${fmt(r.probability_pct, 1)}%) is at or above the ${fmt(r.test_threshold_pct, 1)}% cut this model uses to catch about 80% of undiagnosed diabetes. That means a simple HbA1c or fasting-glucose test is worthwhile — most of the people this flags will not have diabetes, but it is the group where testing pays off.`
                : `Your estimate (${fmt(r.probability_pct, 1)}%) is below the ${fmt(r.test_threshold_pct, 1)}% cut this model uses to prioritise testing. That is reassuring but does not rule diabetes out — anyone with symptoms or risk factors should still be tested.`}
            </Callout>
            <ChartCard title="Where you sit" subtitle="Your estimated probability vs the US adult average (NHANES)">
              <Bars horizontal data={[{ f: "You", v: +r.probability_pct.toFixed(1) }, { f: "US adult average", v: +r.population_rate_pct.toFixed(1) }]}
                x="f" y="v" height={120} xLabel="probability of undiagnosed diabetes (%)"
                colorFn={(row) => (row.f === "You" ? (r.band === "higher" ? c.v3 : r.band === "average" ? c.v2 : c.v1) : c.faint)} />
            </ChartCard>
            <p className="text-[12px] text-faint">{r.note}</p>
          </>
        ) : (
          <Card>
            <p className="text-[13px] text-muted">Enter your measurements to estimate the chance of <b>undiagnosed</b> diabetes from body measurements alone — the same features the NHANES model was validated on (ROC AUC 0.75, no blood test). This is a triage aid to decide whether a blood test is worthwhile, not a diagnosis.</p>
          </Card>
        )}
      </div>
    </div>
  );
}
