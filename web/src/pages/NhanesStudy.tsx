import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { AskAI } from "@/components/AskAI";
import { Bars, ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { MethodsPanel } from "@/components/Methods";
import { Badge, Callout, Card, CardHeader, PageHeader, Stat, Table, cx } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

interface Prevalence {
  n_fasting: number; diabetes_pct: number; prediabetes_pct: number; diagnosed_self_pct: number; undiagnosed_pct: number;
}
interface ByBmi { group: string; median_homa_ir: number; n: number }
interface Homa { n: number; median: number; q1: number; q3: number; r_with_bmi: number; by_bmi: ByBmi[] }
interface Binned { hba1c: number; mean_fpg: number; n: number }
interface Hba1c { r_hba1c_fpg: number; mean_fpg: number; mean_eag_formula: number; binned: Binned[] }
interface Screening {
  n: number; positives: number; prevalence_pct: number; auc: number; auc_age_only: number; features: string;
  at_80_sensitivity: { sensitivity: number; specificity: number }; roc: { fpr: number; tpr: number }[];
}
interface Study {
  source: string; n_adults: number; prevalence: Prevalence;
  prevalence_by_age: { group: string; diabetes_pct: number; n: number }[];
  homa: Homa; hba1c: Hba1c; screening: Screening; caveats: string[];
}
interface Cycle { cycle: string; n_adults: number; n_fasting: number; diabetes_pct: number; prediabetes_pct: number; diagnosed_pct: number; undiagnosed_pct: number }
interface Calibration {
  bins: { pred: number; obs: number; n: number }[];
  brier: number; brier_baseline: number; mean_predicted: number; observed: number; slope: number; intercept: number; ece: number;
}
interface Validation {
  cycles: Cycle[];
  external: { train: string; test: string; features: string; auc_external: number; auc_apparent: number; n_test: number; positives_test: number };
  calibration: Calibration;
  note: string;
}
interface FairRow { group: string; n: number; positives: number; prevalence_pct: number | null; auc: number | null; predicted_pct: number | null; observed_pct: number | null; calib_gap_pct: number | null }
interface Fairness { overall_auc: number; n: number; features: string; by_sex: FairRow[]; by_age: FairRow[]; by_race: FairRow[]; findings: string[]; note: string }

export default function NhanesStudy() {
  const c = useChartColors();
  const [tab, setTab] = useState<"prevalence" | "homa" | "hba1c" | "screen" | "validate" | "fairness">("prevalence");
  const q = useQuery({ queryKey: ["nhanes-study"], queryFn: () => api.get<Study>("/api/nhanes/study") });
  const vq = useQuery({ queryKey: ["nhanes-validation"], queryFn: () => api.get<Validation>("/api/nhanes/validation") });
  const fq = useQuery({ queryKey: ["nhanes-fairness"], queryFn: () => api.get<Fairness>("/api/nhanes/fairness") });
  const r = q.data;
  const v = vq.data;
  const f = fq.data;

  const summary = r ? `Real-data study on NHANES 2017–2018 (${r.n_adults.toLocaleString()} US adults 20+). Weighted laboratory-defined diabetes prevalence ${r.prevalence.diabetes_pct}% and prediabetes ${r.prevalence.prediabetes_pct}%; ${r.prevalence.undiagnosed_pct}% of lab-defined diabetes is undiagnosed. HOMA-IR median ${r.homa.median} (IQR ${r.homa.q1}–${r.homa.q3}), Pearson r with BMI ${r.homa.r_with_bmi}. HbA1c vs fasting glucose r=${r.hba1c.r_hba1c_fpg}. A non-invasive pre-screening model (${r.screening.features}) reaches ROC AUC ${r.screening.auc} vs ${r.screening.auc_age_only} for age alone.` : "";

  const tabs = [
    { k: "prevalence", label: "Prevalence" },
    { k: "homa", label: "Insulin resistance" },
    { k: "hba1c", label: "HbA1c vs glucose" },
    { k: "screen", label: "Pre-screening model" },
    { k: "validate", label: "Does it hold up?" },
    { k: "fairness", label: "Who does it work for?" },
  ] as const;

  const rocData = r?.screening.roc.map((p) => ({ fpr: p.fpr, tpr: p.tpr, chance: p.fpr })) ?? [];

  return (
    <div>
      <PageHeader eyebrow="Data & models · Real patients" title="Real-data study: NHANES 2017–2018"
        description="Everything else here runs on textbook models or synthetic traces. This page does not: it analyses real measurements from 5,569 US adults in the CDC's national health survey — validates GlucoLab's own clinical engines, builds an honest non-invasive pre-screening model, and shows that model holds up on a separate survey year it never saw (the 'Does it hold up?' tab)."
        actions={<AskAI page="NHANES 2017–2018 real-data study" summary={summary} question="Explain this NHANES real-data study: what the weighted prevalence means, and whether a pre-screening AUC of 0.75 from body measurements alone is actually useful." />} />

      <Callout tone="warn" title="What this is, and is not">
        NHANES is a public-domain survey from the US CDC/NCHS. These are <b>real people's measurements</b>, analysed here for research and education. The prevalence figures are survey-weighted to represent US adults, but they are <b>our computation, not an official CDC statistic</b>, and cover a single 2017–2018 cycle. The pre-screening model is a triage aid — it points to who should get a blood test — <b>not a diagnosis and not a medical device</b>.
      </Callout>

      {q.isLoading && <p className="text-[13px] text-muted mt-6">Loading the study…</p>}
      {q.isError && <Callout tone="danger" className="mt-5">{(q.error as Error).message}</Callout>}

      {r && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-5">
            <Stat label="Adults analysed" value={r.n_adults.toLocaleString()} hint="age 20+, NHANES 2017–2018" />
            <Stat label="Diabetes (weighted)" value={fmt(r.prevalence.diabetes_pct, 1)} unit="%" tone="warn" hint="HbA1c ≥ 6.5% or FPG ≥ 126" />
            <Stat label="Prediabetes (weighted)" value={fmt(r.prevalence.prediabetes_pct, 1)} unit="%" tone="warn" hint="by either test" />
            <Stat label="Undiagnosed" value={fmt(r.prevalence.undiagnosed_pct, 1)} unit="%" tone="warn" hint="of lab-defined diabetes" />
          </div>

          <div className="mt-5 flex flex-wrap gap-1.5">
            {tabs.map((t) => (
              <button key={t.k} onClick={() => setTab(t.k)}
                className={cx("rounded-lg border px-3 py-1.5 text-[13px] transition-colors",
                  tab === t.k ? "bg-brand text-white border-brand font-medium" : "border-line-strong text-ink-2 hover:bg-surface-2")}>
                {t.label}
              </button>
            ))}
          </div>

          {tab === "prevalence" && (
            <>
              <ChartCard className="mt-4" title="Diabetes prevalence rises steeply with age" subtitle="Weighted share meeting the laboratory criteria, by age group">
                <Bars data={r.prevalence_by_age.map((a) => ({ group: a.group, pct: a.diabetes_pct }))} x="group" y="pct" xLabel="age group" height={240} />
              </ChartCard>
              <Callout className="mt-4">
                <b>The gap that matters.</b> About {fmt(r.prevalence.diabetes_pct, 1)}% of US adults meet the laboratory definition of diabetes, but only {fmt(r.prevalence.diagnosed_self_pct, 1)}% report having been diagnosed — so roughly <b>{fmt(r.prevalence.undiagnosed_pct, 0)}% of diabetes in this survey is undiagnosed</b>. On top of that, {fmt(r.prevalence.prediabetes_pct, 1)}% meet a prediabetes criterion. That undiagnosed fraction is exactly the problem a cheap, non-invasive screening step could help with — which is what the last tab builds.
              </Callout>
            </>
          )}

          {tab === "homa" && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                <Stat label="HOMA-IR median" value={fmt(r.homa.median, 2)} hint={`IQR ${fmt(r.homa.q1, 2)}–${fmt(r.homa.q3, 2)}`} />
                <Stat label="vs BMI (Pearson r)" value={fmt(r.homa.r_with_bmi, 2)} tone="ok" hint={`n = ${r.homa.n.toLocaleString()}`} />
              </div>
              <ChartCard className="mt-4" title="Insulin resistance climbs with body-mass category" subtitle="Median HOMA-IR on real fasting insulin and glucose — our clinical engine, real data">
                <Bars data={r.homa.by_bmi.map((b) => ({ group: b.group, homa: b.median_homa_ir }))} x="group" y="homa" xLabel="BMI category" height={240} />
              </ChartCard>
              <Callout className="mt-4">
                <b>A validation, not just a number.</b> HOMA-IR here is computed by the same formula GlucoLab's clinical-indices page uses, now applied to real fasting insulin and glucose. The median rises monotonically from the normal-weight to the obese group, reproducing the well-established link between adiposity and insulin resistance — evidence the engine behaves correctly on real measurements, not only on tidy textbook inputs.
              </Callout>
            </>
          )}

          {tab === "hba1c" && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                <Stat label="HbA1c vs FPG (r)" value={fmt(r.hba1c.r_hba1c_fpg, 2)} tone="ok" hint="Pearson correlation" />
                <Stat label="Mean fasting glucose" value={fmt(r.hba1c.mean_fpg, 0)} unit="mg/dl" />
                <Stat label="Mean eAG (from HbA1c)" value={fmt(r.hba1c.mean_eag_formula, 0)} unit="mg/dl" hint="ADAG formula" />
              </div>
              <ChartCard className="mt-4" title="Fasting glucose tracks HbA1c" subtitle="Mean fasting plasma glucose within each HbA1c bin (real data)">
                <TimeSeries data={r.hba1c.binned.map((b) => ({ hba1c: b.hba1c, fpg: b.mean_fpg }))} x="hba1c"
                  xLabel="HbA1c (%)" yLabel="mean fasting glucose (mg/dl)" height={260}
                  series={[{ key: "fpg", label: "Mean FPG", color: c.v1 }]} />
              </ChartCard>
              <Callout className="mt-4">
                <b>Two windows on the same disease.</b> HbA1c (a ~3-month average) and a single fasting glucose correlate strongly (r = {fmt(r.hba1c.r_hba1c_fpg, 2)}) but are not interchangeable — which is why the ADA accepts either for diagnosis. Our eAG formula turns HbA1c into an <i>average</i> glucose of {fmt(r.hba1c.mean_eag_formula, 0)} mg/dl, correctly sitting above the mean <i>fasting</i> glucose of {fmt(r.hba1c.mean_fpg, 0)} mg/dl, since glucose spends much of the day above its fasting low. The engine lands on the right side of that check.
              </Callout>
            </>
          )}

          {tab === "screen" && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                <Stat label="ROC AUC (no blood test)" value={fmt(r.screening.auc, 3)} tone="ok" hint={r.screening.features} />
                <Stat label="Age alone" value={fmt(r.screening.auc_age_only, 3)} hint="baseline" />
                <Stat label="At 80% sensitivity" value={`${fmt(r.screening.at_80_sensitivity.specificity * 100, 0)}%`} hint="specificity" />
                <Stat label="Cases / sample" value={`${r.screening.positives.toLocaleString()} / ${r.screening.n.toLocaleString()}`} hint={`${fmt(r.screening.prevalence_pct, 1)}% positive`} />
              </div>
              <ChartCard className="mt-4" title="Who should get a blood test?" subtitle="Out-of-fold ROC: predicting lab-defined diabetes from body measurements alone">
                <TimeSeries data={rocData} x="fpr" xLabel="false-positive rate" yLabel="true-positive rate" height={300}
                  xDomain={[0, 1]} yDomain={[0, 1]}
                  series={[
                    { key: "tpr", label: `Model (AUC ${fmt(r.screening.auc, 3)})`, color: c.v1 },
                    { key: "chance", label: "Chance", color: c.faint, dashed: true },
                  ]} />
              </ChartCard>
              <Callout className="mt-4">
                <b>Is 0.75 useful?</b> Yes — as a first filter. Using only age, sex, BMI and waist (no needle, no lab), the model reaches AUC {fmt(r.screening.auc, 3)}, clearly above age alone ({fmt(r.screening.auc_age_only, 3)}). Tuned to catch {fmt(r.screening.at_80_sensitivity.sensitivity * 100, 0)}% of people with undiagnosed diabetes, it would still clear about {fmt(r.screening.at_80_sensitivity.specificity * 100, 0)}% of people from needing a blood test. It is evaluated leak-free (out-of-fold cross-validation), so this is an honest estimate — not a diagnosis, but a defensible way to decide who to test first in a setting where testing everyone is not possible.
              </Callout>
            </>
          )}

          {tab === "validate" && (
            <>
              {!v && <p className="text-[13px] text-muted mt-4">Loading the validation…</p>}
              {v && (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                    <Stat label="Trained on" value="2017–18" hint="one survey cycle" />
                    <Stat label="Tested on" value="2015–16" hint="a different, unseen cycle" />
                    <Stat label="Held-out AUC" value={fmt(v.external.auc_external, 3)} tone="ok" hint="on the separate cohort" />
                    <Stat label="Within-sample AUC" value={fmt(v.external.auc_apparent, 3)} hint="for comparison" />
                  </div>
                  <Callout tone="ok" className="mt-4" title="It generalises to a different year">
                    The hard test for a model is data it has never seen. Fit only on the 2017–2018 survey and scored on the <b>entirely separate</b> {v.external.n_test.toLocaleString()} adults of the 2015–2016 cycle, the pre-screening model holds an AUC of <b>{fmt(v.external.auc_external, 3)}</b> — essentially the same as its within-sample {fmt(v.external.auc_apparent, 3)}. It isn't memorising one survey; the signal is real across years. Features: {v.external.features}.
                  </Callout>
                  <ChartCard className="mt-4" title="Diabetes across two survey cycles" subtitle="Weighted prevalence (%) — lab-defined diabetes, and the share of it still undiagnosed">
                    <Bars data={v.cycles.flatMap((cy) => [
                      { f: `${cy.cycle} · diabetes`, v: cy.diabetes_pct },
                      { f: `${cy.cycle} · undiagnosed`, v: cy.undiagnosed_pct },
                    ])} x="f" y="v" horizontal height={200} xLabel="%"
                      colorFn={(row) => (String(row.f).includes("undiagnosed") ? c.v3 : c.v1)} />
                  </ChartCard>
                  <Card className="mt-4" padded={false}>
                    <div className="px-5 pt-5"><CardHeader title="Cycle comparison" subtitle="Survey-weighted, fasting subsample" /></div>
                    <Table head={["Cycle", "Adults", "Diabetes", "Prediabetes", "Diagnosed", "Undiagnosed"]}>
                      {v.cycles.map((cy) => (
                        <tr key={cy.cycle}>
                          <td className="px-3 py-2.5 font-medium">{cy.cycle}</td>
                          <td className="px-3 py-2.5 num text-ink-2">{cy.n_adults.toLocaleString()}</td>
                          <td className="px-3 py-2.5 num">{fmt(cy.diabetes_pct, 1)}%</td>
                          <td className="px-3 py-2.5 num text-ink-2">{fmt(cy.prediabetes_pct, 1)}%</td>
                          <td className="px-3 py-2.5 num text-ink-2">{fmt(cy.diagnosed_pct, 1)}%</td>
                          <td className="px-3 py-2.5 num text-warn">{fmt(cy.undiagnosed_pct, 1)}%</td>
                        </tr>
                      ))}
                    </Table>
                  </Card>
                  <Callout className="mt-4">
                    <b>What the trend shows.</b> Lab-defined diabetes held steady at {fmt(v.cycles[1].diabetes_pct, 1)}% across both cycles, but the undiagnosed share fell from {fmt(v.cycles[0].undiagnosed_pct, 1)}% to {fmt(v.cycles[1].undiagnosed_pct, 1)}% — modest progress in catching it, with a large gap still open. Two cycles is a short window and the weighted estimates carry sampling uncertainty, so read the direction, not the decimals.
                  </Callout>

                  <h3 className="text-[15px] font-semibold tracking-[-0.01em] mt-8">Is the number honest? (calibration)</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
                    <Stat label="Calibration slope" value={fmt(v.calibration.slope, 2)} tone="ok" hint="ideal 1.00" />
                    <Stat label="Calibration intercept" value={fmt(v.calibration.intercept, 2)} tone="ok" hint="ideal 0.00" />
                    <Stat label="Brier score" value={fmt(v.calibration.brier, 3)} tone="ok" hint={`vs ${fmt(v.calibration.brier_baseline, 3)} base-rate`} />
                    <Stat label="Mean predicted vs real" value={`${fmt(v.calibration.mean_predicted * 100, 1)} / ${fmt(v.calibration.observed * 100, 1)}%`} hint="should match" />
                  </div>
                  <ChartCard className="mt-4" title="Reliability curve (held-out 2015–2016)" subtitle="Average predicted probability vs the fraction who actually had diabetes, by decile. On the line = honest.">
                    <TimeSeries data={v.calibration.bins.map((bn) => ({ pred: +(bn.pred * 100).toFixed(1), obs: +(bn.obs * 100).toFixed(1), ideal: +(bn.pred * 100).toFixed(1) }))}
                      x="pred" xLabel="predicted probability (%)" yLabel="observed (%)" height={280}
                      series={[
                        { key: "obs", label: "Observed", color: c.v1, points: true },
                        { key: "ideal", label: "Perfect calibration", color: c.faint, dashed: true },
                      ]} />
                  </ChartCard>
                  <Callout tone="ok" className="mt-4" title="The probabilities mean what they say">
                    Ranking (AUC) only tells you the model orders people correctly. Calibration asks the harder question: when it says "10%", is it really about 10%? On the unseen 2015–2016 cycle the reliability curve sits almost on the diagonal — calibration slope {fmt(v.calibration.slope, 2)} and intercept {fmt(v.calibration.intercept, 2)} (ideal 1.00 and 0.00), mean predicted {fmt(v.calibration.mean_predicted * 100, 1)}% against {fmt(v.calibration.observed * 100, 1)}% observed, expected calibration error just {fmt(v.calibration.ece * 100, 1)}%. The Brier score ({fmt(v.calibration.brier, 3)}) beats simply predicting the overall rate ({fmt(v.calibration.brier_baseline, 3)}). So the percentage the tool shows you is not just a ranking — it is close to a real probability.
                  </Callout>
                </>
              )}
            </>
          )}

          {tab === "fairness" && (
            <>
              {!f && <p className="text-[13px] text-muted mt-4">Loading the fairness audit…</p>}
              {f && (
                <>
                  <Callout className="mt-4" title="Why this tab exists">
                    A screening tool is only trustworthy if it works for the people it's meant to help — not just on average. Here is the model's accuracy (<b>AUC</b>, how well it ranks risk) and its <b>calibration gap</b> (predicted minus observed rate — positive means it over-states risk for that group, negative means it under-states it) within each subgroup. Showing this, including where the model is weaker, is the point.
                  </Callout>
                  <div className="grid lg:grid-cols-2 gap-5 mt-4">
                    <FairCard title="By sex" rows={f.by_sex} />
                    <FairCard title="By age band" rows={f.by_age} />
                  </div>
                  <div className="mt-5"><FairCard title="By race / ethnicity" rows={f.by_race} /></div>
                  <Callout tone="warn" className="mt-5" title="What the audit found — stated plainly">
                    <ul className="list-disc pl-5 flex flex-col gap-1.5">
                      {f.findings.map((fi, i) => <li key={i}>{fi}</li>)}
                    </ul>
                  </Callout>
                  <p className="text-[12px] text-faint mt-3">{f.note} Overall AUC {fmt(f.overall_auc, 3)} on {f.n.toLocaleString()} adults; features: {f.features}.</p>
                </>
              )}
            </>
          )}

          <Card className="mt-5">
            <CardHeader title="Caveats" subtitle={r.source} />
            <ul className="list-disc pl-5 flex flex-col gap-1.5 text-[13px] text-ink-2">
              {r.caveats.map((cv, i) => <li key={i}>{cv}</li>)}
            </ul>
            <div className="mt-3 flex flex-wrap gap-2">
              <Badge tone="accent">Public domain · CDC/NCHS</Badge>
              <Badge tone="brand">{r.prevalence.n_fasting.toLocaleString()} in fasting subsample</Badge>
            </div>
          </Card>

          <div className="mt-5"><MethodsPanel id="nhanes" /></div>
        </>
      )}
    </div>
  );
}

/** A subgroup table: accuracy (AUC) and the predicted-vs-observed calibration gap per group. */
function FairCard({ title, rows }: { title: string; rows: FairRow[] }) {
  return (
    <Card padded={false}>
      <div className="px-5 pt-5"><CardHeader title={title} subtitle="AUC = ranking accuracy · gap = predicted − observed risk" /></div>
      <Table head={["Group", "n", "Prev.", "AUC", "Pred.", "Obs.", "Gap"]}>
        {rows.map((r) => {
          const gap = r.calib_gap_pct;
          const tone = gap == null ? "text-faint" : Math.abs(gap) >= 4 ? "text-warn" : Math.abs(gap) >= 2 ? "text-ink-2" : "text-ok";
          return (
            <tr key={r.group}>
              <td className="px-3 py-2.5 font-medium">{r.group}</td>
              <td className="px-3 py-2.5 num text-faint">{r.n.toLocaleString()}</td>
              <td className="px-3 py-2.5 num text-ink-2">{r.prevalence_pct == null ? "–" : `${fmt(r.prevalence_pct, 1)}%`}</td>
              <td className="px-3 py-2.5 num">{r.auc == null ? <span className="text-faint" title="too few cases">n/a</span> : fmt(r.auc, 3)}</td>
              <td className="px-3 py-2.5 num text-ink-2">{r.predicted_pct == null ? "–" : `${fmt(r.predicted_pct, 1)}%`}</td>
              <td className="px-3 py-2.5 num text-ink-2">{r.observed_pct == null ? "–" : `${fmt(r.observed_pct, 1)}%`}</td>
              <td className={cx("px-3 py-2.5 num font-medium", tone)}>{gap == null ? "–" : `${gap > 0 ? "+" : ""}${fmt(gap, 1)}`}</td>
            </tr>
          );
        })}
      </Table>
    </Card>
  );
}
