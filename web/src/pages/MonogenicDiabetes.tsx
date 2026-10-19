import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { AskAI } from "@/components/AskAI";
import { ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { MethodsPanel } from "@/components/Methods";
import { Badge, Callout, Card, CardHeader, PageHeader, Slider, Table, cx } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

const MGDL = 18.016;

interface Membrane { ghk_vm_mv: number; pk_sweep: { pk_fraction: number; vm_mv: number }[] }
interface MealSim { series: { t_min: number[]; glucose_mg_dl: number[] } }

// Relative K+ permeability for each scenario. Illustrative values chosen to show the direction of each
// change; the voltages are computed by the GHK engine with textbook ion concentrations.
const SCENARIOS = [
  { key: "fasting", title: "Low glucose", pk: 1.0, tone: "neutral" as const,
    text: "Little ATP is made, so KATP channels stay open. Potassium leaks out and the membrane stays polarised: no insulin release." },
  { key: "fed", title: "High glucose, normal channels", pk: 0.1, tone: "ok" as const,
    text: "Glucose metabolism raises ATP, which closes KATP channels. The membrane depolarises, Ca²⁺ channels open and insulin is released." },
  { key: "mutant", title: "High glucose, KCNJ11 activating mutation", pk: 0.7, tone: "danger" as const,
    text: "The mutant channel responds poorly to ATP and stays open, so the cell cannot depolarise enough to release insulin. This causes neonatal diabetes." },
  { key: "su", title: "Mutation treated with a sulfonylurea", pk: 0.15, tone: "brand" as const,
    text: "Sulfonylurea tablets bind the SUR1 subunit and close the channel by another route, restoring depolarisation. Many patients have switched from insulin to tablets this way." },
];

const GENES = [
  { gene: "KCNJ11", product: "Kir6.2, pore of the KATP channel", presentation: "Neonatal diabetes, usually diagnosed before 6 months of age",
    treatment: "Many patients can switch from insulin to high-dose sulfonylurea tablets", refs: "Gloyn 2004; Pearson 2006" },
  { gene: "ABCC8", product: "SUR1, regulatory subunit of the KATP channel", presentation: "Neonatal diabetes",
    treatment: "Often responsive to sulfonylureas", refs: "Hattersley 2018 (ISPAD)" },
  { gene: "GCK", product: "Glucokinase, the β-cell glucose sensor", presentation: "Mild, stable fasting hyperglycaemia (about 5.5–8.0 mmol/l) from birth",
    treatment: "Usually no treatment outside pregnancy; complications are rare", refs: "Steele 2014; Hattersley 2018" },
  { gene: "HNF1A", product: "Transcription factor in β-cells and liver", presentation: "Diabetes that progresses from adolescence or young adulthood",
    treatment: "Often very sensitive to low-dose sulfonylureas", refs: "Pearson 2003" },
  { gene: "HNF4A", product: "Transcription factor in β-cells and liver", presentation: "Similar to HNF1A; can cause high birth weight and low blood sugar in newborns",
    treatment: "Often sensitive to sulfonylureas", refs: "Hattersley 2018 (ISPAD)" },
  { gene: "INS", product: "Insulin itself", presentation: "Neonatal diabetes", treatment: "Insulin", refs: "Hattersley 2018 (ISPAD)" },
];

function interp(sweep: Membrane["pk_sweep"], f: number) {
  const s = [...sweep].sort((a, b) => a.pk_fraction - b.pk_fraction);
  if (f <= s[0].pk_fraction) return s[0].vm_mv;
  for (let i = 1; i < s.length; i++) {
    if (f <= s[i].pk_fraction) {
      const a = s[i - 1], b = s[i];
      return a.vm_mv + ((f - a.pk_fraction) / (b.pk_fraction - a.pk_fraction)) * (b.vm_mv - a.vm_mv);
    }
  }
  return s[s.length - 1].vm_mv;
}

export default function MonogenicDiabetes() {
  const c = useChartColors();
  const [open, setOpen] = useState(40);
  const [gckBasal, setGckBasal] = useState(7.0);

  const scen = useQuery({
    queryKey: ["katp-scenarios"],
    queryFn: () => Promise.all(SCENARIOS.map((s) => api.post<Membrane>("/api/biophysics/membrane", { p_K: s.pk, p_Na: 0.04, p_Cl: 0.45 }))),
  });
  const sweep = scen.data?.[0]?.pk_sweep;
  const vmCustom = sweep ? interp(sweep, open / 100) : null;

  const meal = useMutation({
    mutationFn: async (basal: number) => {
      const body = (gb: number | null) => ({ meals: [{ time_min: 0, carbs_g: 75 }], duration_min: 300, phenotype: "normal", dt_min: 2, ...(gb ? { basal_glucose_mg_dl: gb } : {}) });
      const [n, g] = await Promise.all([api.post<MealSim>("/api/physiology/meal", body(null)), api.post<MealSim>("/api/physiology/meal", body(basal * MGDL))]);
      return { n, g };
    },
  });
  useEffect(() => { const t = setTimeout(() => meal.mutate(gckBasal), 250); return () => clearTimeout(t); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gckBasal]);
  const mealRows = useMemo(() => meal.data ? meal.data.n.series.t_min.map((t, i) => ({
    t, normal: meal.data!.n.series.glucose_mg_dl[i] / MGDL, gck: meal.data!.g.series.glucose_mg_dl[i] / MGDL,
  })) : [], [meal.data]);
  const inc = (k: "normal" | "gck") => mealRows.length ? Math.max(...mealRows.map((r) => r[k])) - mealRows[0][k] : null;

  const summary = scen.data ? `K_ATP scenarios (GHK, illustrative permeabilities): ${SCENARIOS.map((s, i) => `${s.title}: P_K ${s.pk * 100}% → V_m ${fmt(scen.data[i].ghk_vm_mv, 1)} mV`).join("; ")}. GCK-MODY illustration: fasting glucose ${gckBasal} mmol/l, meal increment ${fmt(inc("gck"), 1)} vs ${fmt(inc("normal"), 1)} mmol/l normal.` : "";

  return (
    <div>
      <PageHeader eyebrow="Genetics" title="Monogenic diabetes"
        description="Some diabetes is caused by a change in a single gene. Finding that gene can change treatment, sometimes from insulin injections to tablets. This page follows one example from gene to ion channel to cell, using GlucoLab's physics and physiology engines."
        actions={<AskAI page="Monogenic diabetes" summary={summary} question="Explain how a KCNJ11 mutation causes neonatal diabetes and why sulfonylureas can work." />} />

      <section className="mb-8">
        <h2 className="text-[15px] font-semibold mb-1">From gene to channel to cell: KCNJ11 and the K<sub>ATP</sub> channel</h2>
        <p className="text-[13.5px] text-muted mb-4 max-w-3xl leading-6">In β-cells, ATP-sensitive potassium (K<sub>ATP</sub>) channels carry much of the resting potassium current. The <i>KCNJ11</i> gene encodes Kir6.2, the channel's pore. Below, each scenario sets the cell's relative K⁺ permeability and the Goldman–Hodgkin–Katz equation computes the membrane voltage.</p>
        <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-3">
          {SCENARIOS.map((s, i) => (
            <Card key={s.key}>
              <div className="flex items-start justify-between gap-2">
                <div className="text-[13.5px] font-semibold leading-5">{s.title}</div>
                <Badge tone={s.tone === "neutral" ? "neutral" : s.tone}>P<sub>K</sub> {Math.round(s.pk * 100)}%</Badge>
              </div>
              <div className="num text-[26px] font-semibold mt-2 tracking-tight">{scen.data ? fmt(scen.data[i].ghk_vm_mv, 1) : "–"}<span className="text-[13px] text-muted font-medium ml-1">mV</span></div>
              <p className="text-[12.5px] text-muted leading-5 mt-2">{s.text}</p>
            </Card>
          ))}
        </div>
        <div className="grid lg:grid-cols-[1fr_1fr] gap-4 mt-4">
          <Card>
            <CardHeader title="Try it" subtitle="Share of KATP permeability still open" />
            <Slider label="K⁺ permeability (% of low-glucose value)" value={open} min={2} max={100} step={1} onChange={setOpen} format={(v) => `${v}%`} />
            <div className="mt-4 flex items-baseline gap-2">
              <span className="text-[13px] text-muted">Membrane potential</span>
              <span className="num text-[24px] font-semibold">{vmCustom !== null ? fmt(vmCustom, 1) : "–"}</span><span className="text-[13px] text-muted">mV</span>
            </div>
            <div className="mt-3 h-2 rounded-full bg-surface-3 overflow-hidden">
              {vmCustom !== null && <div className="h-full bg-brand transition-all" style={{ width: `${Math.max(0, Math.min(100, ((vmCustom + 70) / 30) * 100))}%` }} />}
            </div>
            <div className="flex justify-between text-[11.5px] text-faint mt-1"><span>−70 mV (polarised)</span><span>−40 mV (depolarised)</span></div>
          </Card>
          <Callout tone="warn" title="What these numbers are">
            Voltages come from the GHK equation with textbook ion concentrations and squid-axon permeability ratios (Hodgkin &amp; Katz 1949). The permeability for each scenario is illustrative, chosen to show the direction of the change. They are not measured β-cell values, and real β-cells fire bursts of action potentials that this steady-state calculation does not model.
          </Callout>
        </div>
      </section>

      <section className="mb-8">
        <h2 className="text-[15px] font-semibold mb-1">GCK-MODY: a higher set point, not a broken system</h2>
        <p className="text-[13.5px] text-muted mb-4 max-w-3xl leading-6">Glucokinase (<i>GCK</i>) acts as the β-cell's glucose sensor. Inactivating mutations in one copy reset the glucose level the body defends, so fasting glucose sits higher, but the response to a meal is largely preserved. Here the meal model runs with normal parameters and a raised basal glucose.</p>
        <div className="grid lg:grid-cols-[320px_1fr] gap-4 items-start">
          <Card>
            <Slider label="Fasting glucose in the GCK scenario" value={gckBasal} min={5.5} max={8.0} step={0.1} onChange={setGckBasal} format={(v) => `${v.toFixed(1)} mmol/l`} />
            <div className="grid grid-cols-2 gap-3 mt-4 text-[13px]">
              <div><div className="text-muted text-[12px]">Rise after meal, normal</div><div className="num font-semibold text-[18px]">{fmt(inc("normal"), 1)} <span className="text-[12px] text-muted font-medium">mmol/l</span></div></div>
              <div><div className="text-muted text-[12px]">Rise after meal, GCK</div><div className="num font-semibold text-[18px]">{fmt(inc("gck"), 1)} <span className="text-[12px] text-muted font-medium">mmol/l</span></div></div>
            </div>
            <p className="text-[12px] text-faint mt-3 leading-5">Illustrative: the published healthy-adult parameter set with only the basal glucose changed. The range 5.5–8.0 mmol/l is typical of GCK-MODY.</p>
          </Card>
          <ChartCard title="Glucose after a 75 g meal" subtitle="mmol/l; normal vs GCK-MODY illustration">
            <TimeSeries data={mealRows} x="t" xLabel="min" height={240}
              series={[{ key: "normal", label: "Normal", color: c.v1 }, { key: "gck", label: `GCK-MODY illustration (fasting ${gckBasal.toFixed(1)})`, color: c.v2 }]} />
          </ChartCard>
        </div>
      </section>

      <section className="mb-8">
        <Card padded={false}>
          <div className="px-5 pt-5"><CardHeader title="Main genes in monogenic diabetes" subtitle="Summary for learning; diagnosis and treatment decisions belong to specialist clinicians" /></div>
          <Table head={["Gene", "Encodes", "Typical presentation", "Treatment notes", "Sources"]}>
            {GENES.map((g) => (
              <tr key={g.gene} className="align-top">
                <td className="px-3 py-2.5 font-mono font-semibold italic whitespace-nowrap">{g.gene}</td>
                <td className="px-3 py-2.5 text-ink-2 min-w-[180px]">{g.product}</td>
                <td className="px-3 py-2.5 text-ink-2 min-w-[200px]">{g.presentation}</td>
                <td className="px-3 py-2.5 text-ink-2 min-w-[200px]">{g.treatment}</td>
                <td className="px-3 py-2.5 text-muted whitespace-nowrap text-[12px]">{g.refs}</td>
              </tr>
            ))}
          </Table>
        </Card>
      </section>

      <section className="mb-8 grid lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader title="When clinicians consider genetic testing" subtitle="From the ISPAD 2018 guideline (Hattersley et al.)" />
          <ul className="text-[13px] text-ink-2 leading-6 list-disc pl-5 space-y-1">
            <li>Diabetes diagnosed before 6 months of age: genetic testing for neonatal diabetes is recommended, whatever the person's current age.</li>
            <li>Features that suggest MODY: diabetes in a parent and across generations, no islet autoantibodies, the body still making insulin (C-peptide) years after diagnosis, or mild stable fasting hyperglycaemia.</li>
            <li>Clinical prediction tools such as the MODY probability calculator (Shields et al. 2012) help decide who to test.</li>
          </ul>
        </Card>
        <Callout tone="warn" title="Education, not diagnosis">
          GlucoLab does not analyse anyone's genes and cannot tell whether a person has monogenic diabetes. Never change diabetes treatment, including stopping insulin, without a specialist. If this page describes you or your child, ask your diabetes team about genetic testing.
        </Callout>
      </section>
      <div className={cx("flex flex-col gap-3")}>
        <MethodsPanel id="membrane" />
        <MethodsPanel id="meal" />
      </div>
    </div>
  );
}
