import { useMutation } from "@tanstack/react-query";
import { Atom, Waves } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { AskAI } from "@/components/AskAI";
import { Bars, ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { MethodsPanel, TeX } from "@/components/Methods";
import { Button, Card, CardHeader, Field, Input, PageHeader, Slider, Stat, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

type Ion = "K" | "Na" | "Cl" | "Ca";
interface Membrane {
  temperature_c: number; thermal_voltage_mv: number; equilibrium_potentials_mv: Record<Ion, number>; ghk_vm_mv: number;
  driving_force_mv: Record<Ion, number>; pk_sweep: { pk_fraction: number; vm_mv: number }[]; notes: string[];
}
interface Diffusion { D_m2_s: number; D_um2_s: number; time_to_distance_s: number; rms_displacement_um: { t_s: number; rms_um: number }[]; notes: string[] }

const ION_LABEL: Record<Ion, string> = { K: "K⁺", Na: "Na⁺", Cl: "Cl⁻", Ca: "Ca²⁺" };

function fmtTime(s: number) {
  if (s < 1e-3) return `${fmt(s * 1e6, 1)} µs`;
  if (s < 1) return `${fmt(s * 1e3, 1)} ms`;
  if (s < 120) return `${fmt(s, 2)} s`;
  if (s < 7200) return `${fmt(s / 60, 1)} min`;
  return `${fmt(s / 3600, 1)} h`;
}

export default function Biophysics() {
  const toast = useToast();
  const c = useChartColors();
  const [T, setT] = useState(37);
  const [conc, setConc] = useState<Record<Ion, { in: number; out: number }>>({
    K: { in: 140, out: 5 }, Na: { in: 10, out: 145 }, Cl: { in: 10, out: 110 }, Ca: { in: 0.0001, out: 2 },
  });
  const [perm, setPerm] = useState({ p_K: 1, p_Na: 0.04, p_Cl: 0.45 });
  const [diff, setDiff] = useState({ radius_nm: 2.5, distance_um: 10, viscosity_mpa_s: 0.69, dims: 3 });

  const mem = useMutation({
    mutationFn: () => api.post<Membrane>("/api/biophysics/membrane", { temperature_c: T, ...conc, ...perm }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const dif = useMutation({
    mutationFn: () => api.post<Diffusion>("/api/biophysics/diffusion", { ...diff, temperature_c: T }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  useEffect(() => { mem.mutate(); dif.mutate(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const m = mem.data, d = dif.data;
  const eqRows = useMemo(() => m ? (Object.keys(m.equilibrium_potentials_mv) as Ion[]).map((k) => ({ ion: ION_LABEL[k], E: +m.equilibrium_potentials_mv[k].toFixed(1) })) : [], [m]);
  const sweep = useMemo(() => m ? m.pk_sweep.map((p) => ({ f: +(100 * p.pk_fraction).toFixed(1), vm: p.vm_mv })) : [], [m]);
  const summary = m ? `Temperature ${T} °C. Equilibrium potentials (mV): ${Object.entries(m.equilibrium_potentials_mv).map(([k, v]) => `${k} ${v.toFixed(1)}`).join(", ")}. GHK V_m = ${m.ghk_vm_mv.toFixed(1)} mV with P_K:P_Na:P_Cl = ${perm.p_K}:${perm.p_Na}:${perm.p_Cl}.` : "";

  return (
    <div>
      <PageHeader eyebrow="Physiology" title="Membrane biophysics"
        description="Exact physical laws behind cell electrical activity and transport: Nernst and Goldman–Hodgkin–Katz potentials, and Stokes–Einstein diffusion."
        actions={<AskAI page="Membrane biophysics" summary={summary} question="Explain these membrane potentials and what K_ATP closure does to the β-cell." />} />

      <div className="grid xl:grid-cols-[340px_1fr] gap-5 items-start mb-8">
        <Card>
          <CardHeader title="Ions and permeabilities" subtitle="Concentrations in mM. Defaults: typical mammalian cell (Alberts); squid-axon permeability ratios (Hodgkin & Katz 1949)." />
          <div className="flex flex-col gap-4">
            <Slider label="Temperature" value={T} min={0} max={45} step={1} onChange={setT} format={(v) => `${v} °C`} />
            <div className="grid grid-cols-[auto_1fr_1fr] gap-2 items-center text-[12.5px]">
              <span /><span className="text-muted font-medium">Inside</span><span className="text-muted font-medium">Outside</span>
              {(Object.keys(conc) as Ion[]).map((ion) => (
                <div key={ion} className="contents">
                  <span className="font-medium w-10">{ION_LABEL[ion]}</span>
                  <Input type="number" step="any" min={0} value={conc[ion].in} onChange={(e) => setConc({ ...conc, [ion]: { ...conc[ion], in: +e.target.value } })} aria-label={`${ion} inside`} />
                  <Input type="number" step="any" min={0} value={conc[ion].out} onChange={(e) => setConc({ ...conc, [ion]: { ...conc[ion], out: +e.target.value } })} aria-label={`${ion} outside`} />
                </div>
              ))}
            </div>
            <div className="grid grid-cols-3 gap-2">
              <Field label="P_K"><Input type="number" step="any" min={0} value={perm.p_K} onChange={(e) => setPerm({ ...perm, p_K: +e.target.value })} /></Field>
              <Field label="P_Na"><Input type="number" step="any" min={0} value={perm.p_Na} onChange={(e) => setPerm({ ...perm, p_Na: +e.target.value })} /></Field>
              <Field label="P_Cl"><Input type="number" step="any" min={0} value={perm.p_Cl} onChange={(e) => setPerm({ ...perm, p_Cl: +e.target.value })} /></Field>
            </div>
            <Button variant="primary" icon={<Atom className="size-4" />} loading={mem.isPending} onClick={() => mem.mutate()}>Compute potentials</Button>
          </div>
        </Card>
        <div className="flex flex-col gap-5 min-w-0">
          {m && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat label="GHK membrane potential" value={fmt(m.ghk_vm_mv, 1)} unit="mV" />
              <Stat label="RT/F" value={fmt(m.thermal_voltage_mv, 2)} unit="mV" hint={`at ${m.temperature_c} °C`} />
              <Stat label={`E_K`} value={fmt(m.equilibrium_potentials_mv.K, 1)} unit="mV" hint={`driving force ${fmt(m.driving_force_mv.K, 1)} mV`} />
              <Stat label={`E_Na`} value={fmt(m.equilibrium_potentials_mv.Na, 1)} unit="mV" hint={`driving force ${fmt(m.driving_force_mv.Na, 1)} mV`} />
            </div>
          )}
          <div className="grid lg:grid-cols-2 gap-5">
            <ChartCard title="Nernst equilibrium potentials" subtitle={<TeX>{String.raw`E_X = \tfrac{RT}{zF}\ln\tfrac{[X]_o}{[X]_i}`}</TeX>}>
              <Bars data={eqRows} x="ion" y="E" height={230} colorFn={(r) => (Number(r.E) >= 0 ? c.v2 : c.v1)} />
            </ChartCard>
            <ChartCard title="Depolarisation as K⁺ permeability falls" subtitle="GHK V_m vs P_K (% of initial), P_Na and P_Cl fixed: the physics of K_ATP-channel closure">
              <TimeSeries data={sweep} x="f" xLabel="P_K (% of initial)" height={230} xDomain={[100, 2]} series={[{ key: "vm", label: "V_m (mV)", color: c.v3 }]} />
            </ChartCard>
          </div>
          {m && <ul className="text-[12.5px] text-muted list-disc ml-5 space-y-0.5">{m.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        </div>
      </div>

      <div className="grid xl:grid-cols-[340px_1fr] gap-5 items-start">
        <Card>
          <CardHeader title="Diffusion" subtitle="How long does a molecule take to travel a distance by Brownian motion?" />
          <div className="flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Hydrodynamic radius (nm)"><Input type="number" step="any" min={0} value={diff.radius_nm} onChange={(e) => setDiff({ ...diff, radius_nm: +e.target.value })} /></Field>
              <Field label="Distance (µm)"><Input type="number" step="any" min={0} value={diff.distance_um} onChange={(e) => setDiff({ ...diff, distance_um: +e.target.value })} /></Field>
              <Field label="Viscosity (mPa·s)" hint="water at 37 °C ≈ 0.69"><Input type="number" step="any" min={0} value={diff.viscosity_mpa_s} onChange={(e) => setDiff({ ...diff, viscosity_mpa_s: +e.target.value })} /></Field>
              <Field label="Dimensions"><Input type="number" min={1} max={3} value={diff.dims} onChange={(e) => setDiff({ ...diff, dims: +e.target.value })} /></Field>
            </div>
            <Button variant="primary" icon={<Waves className="size-4" />} loading={dif.isPending} onClick={() => dif.mutate()}>Compute diffusion</Button>
          </div>
        </Card>
        <div className="flex flex-col gap-5 min-w-0">
          {d && (
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Diffusion coefficient" value={fmt(d.D_um2_s, 1)} unit="µm²/s" hint={`${d.D_m2_s.toExponential(3)} m²/s`} />
              <Stat label={`Time to diffuse ${diff.distance_um} µm`} value={fmtTime(d.time_to_distance_s)} hint={`t = L²/(2dD), d = ${diff.dims}`} />
            </div>
          )}
          {d && (
            <ChartCard title="Root-mean-square displacement" subtitle={<TeX>{String.raw`\sqrt{\langle x^2\rangle} = \sqrt{2dDt}`}</TeX>}>
              <TimeSeries data={d.rms_displacement_um.map((p) => ({ t: p.t_s, rms: p.rms_um }))} x="t" xLabel="s" height={220} series={[{ key: "rms", label: "RMS displacement (µm)", color: c.v5 }]} />
            </ChartCard>
          )}
          {d && <ul className="text-[12.5px] text-muted list-disc ml-5 space-y-0.5">{d.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        </div>
      </div>
      <div className="mt-6 flex flex-col gap-4"><MethodsPanel id="membrane" /><MethodsPanel id="diffusion" /></div>
    </div>
  );
}
