import { useMutation } from "@tanstack/react-query";
import { Play, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { Badge, Button, Callout, Card, CardHeader, PageHeader, Slider, Table, useToast } from "@/components/ui";
import { api, type BetaCellResult } from "@/lib/api";
import { fmt, rowsOf } from "@/lib/format";

export default function BetaCell() {
  const toast = useToast();
  const c = useChartColors();
  const [p, setP] = useState({ years: 10, si_final_fraction: 0.3, si_decline_years: 5, sigma_scale: 1, d0_scale: 1 });
  const [r, setR] = useState<BetaCellResult | null>(null);
  const run = useMutation({
    mutationFn: (body: typeof p) => api.post<BetaCellResult>("/api/physiology/beta-cell", { ...body, si_decline_years: Math.min(body.si_decline_years, body.years) }),
    onSuccess: setR,
    onError: (e: Error) => toast(e.message, "error"),
  });
  useEffect(() => { run.mutate(p); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const data = useMemo(() => r ? rowsOf({ t: r.t_years, g: r.glucose_mg_dl, b: r.beta_cell_mass_mg, si: r.si }) : [], [r]);
  const law = useMemo(() => {
    if (!r) return [];
    const P = r.params;
    const G = (P.r1 - Math.sqrt(P.r1 ** 2 - 4 * P.r2 * P.d0)) / (2 * P.r2);
    const out = [];
    for (let f = 0.1; f <= 1.5001; f += 0.02) {
      const I = (P.R0 / G - P.EG0) / (P.SI * f);
      out.push({ f: +f.toFixed(2), beta: (P.k * I * (P.alpha + G * G)) / (P.sigma * G * G) });
    }
    return out;
  }, [r]);

  return (
    <div>
      <PageHeader
        eyebrow="Physiology"
        title="β-cell mass dynamics"
        description={<>Topp et al. (J Theor Biol 2000): dG/dt = R₀ − (E<sub>G0</sub> + S<sub>I</sub>I)G · dI/dt = βσG²/(α+G²) − kI · dβ/dt = (−d₀ + r₁G − r₂G²)β, in days.</>}
      />
      <div className="grid xl:grid-cols-[320px_1fr] gap-5 items-start">
        <Card className="xl:sticky xl:top-8">
          <CardHeader title="Scenario" subtitle="Insulin sensitivity falls linearly, then stays constant." />
          <div className="flex flex-col gap-4">
            <Slider label="Simulated years" value={p.years} min={1} max={30} step={1} onChange={(v) => setP({ ...p, years: v })} />
            <Slider label="Final S_I (fraction of normal)" value={p.si_final_fraction} min={0.05} max={1.5} step={0.05} onChange={(v) => setP({ ...p, si_final_fraction: v })} format={(v) => v.toFixed(2)} />
            <Slider label="Decline duration (years)" value={p.si_decline_years} min={0} max={10} step={0.05} onChange={(v) => setP({ ...p, si_decline_years: v })} format={(v) => v.toFixed(2)} />
            <Slider label="Secretory capacity σ ×" value={p.sigma_scale} min={0.2} max={2} step={0.05} onChange={(v) => setP({ ...p, sigma_scale: v })} format={(v) => v.toFixed(2)} />
            <Slider label="β-cell death rate d₀ ×" value={p.d0_scale} min={0.5} max={3} step={0.05} onChange={(v) => setP({ ...p, d0_scale: v })} format={(v) => v.toFixed(2)} />
            <div className="flex gap-2">
              <Button variant="primary" className="flex-1" loading={run.isPending} icon={<Play className="size-4" />} onClick={() => run.mutate(p)}>Simulate</Button>
              <Button icon={<Zap className="size-4" />} onClick={() => { const q = { years: 5, si_final_fraction: 0.1, si_decline_years: 0.05, sigma_scale: 1, d0_scale: 1 }; setP(q); run.mutate(q); }}>Rapid decline</Button>
            </div>
            {r && <Callout tone={r.diabetic_at_end ? "danger" : "ok"} title={r.diabetic_at_end ? "Compensation failed" : "Compensated"}>{r.outcome}</Callout>}
          </div>
        </Card>
        <div className="flex flex-col gap-5 min-w-0">
          <div className="grid lg:grid-cols-2 gap-5">
            <ChartCard title="Fasting glucose" subtitle="mg/dl">
              <TimeSeries data={data} x="t" xLabel="years" height={250} yDomain={[0, "auto"]}
                series={[{ key: "g", label: "Glucose", color: c.v2 }]}
                refLines={[{ y: 126, label: "126 · diabetes (fasting)", color: c.v3 }, { y: 250, label: "250 · saddle", color: c.v4 }]} />
            </ChartCard>
            <ChartCard title="β-cell mass and insulin sensitivity" subtitle="mg (left) · ml/µU/day (right)">
              <TimeSeries data={data} x="t" xLabel="years" height={250}
                series={[{ key: "b", label: "β-cell mass", color: c.v1 }, { key: "si", label: "S_I", color: c.v5, axis: "right", dashed: true }]} />
            </ChartCard>
          </div>
          <ChartCard title="Compensation law" subtitle="Equilibrium β-cell mass at the physiological fixed point (closed form): β* ∝ 1/S_I">
            <TimeSeries data={law} x="f" xLabel="S_I / S_I,normal" height={220} series={[{ key: "beta", label: "β* (mg)", color: c.v4 }]} />
          </ChartCard>
          {r && (
            <Card>
              <CardHeader title="Fixed points at the final insulin sensitivity" subtitle="Linear stability from the eigenvalues of the Jacobian" />
              <Table head={["Fixed point", "G (mg/dl)", "I (µU/ml)", "β (mg)", "Eigenvalues (day⁻¹)", "Stability"]}>
                {r.final_fixed_points.map((fp) => (
                  <tr key={fp.label}>
                    <td className="px-3 py-2.5 capitalize">{fp.label}</td>
                    <td className="px-3 py-2.5 num font-mono text-[12.5px]">{fmt(fp.glucose_mg_dl, 1)}</td>
                    <td className="px-3 py-2.5 num font-mono text-[12.5px]">{fmt(fp.insulin_uU_ml, 2)}</td>
                    <td className="px-3 py-2.5 num font-mono text-[12.5px]">{fmt(fp.beta_cell_mass_mg, 1)}</td>
                    <td className="px-3 py-2.5 num font-mono text-[12px] text-ink-2">{fp.eigenvalues_per_day.map((e) => fmt(e.real, 4)).join(", ")}</td>
                    <td className="px-3 py-2.5"><Badge tone={fp.stability === "stable" ? "ok" : fp.stability === "saddle" ? "warn" : "danger"}>{fp.stability}</Badge></td>
                  </tr>
                ))}
              </Table>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
