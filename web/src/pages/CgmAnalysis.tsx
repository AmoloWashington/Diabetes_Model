import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, FileUp, FlaskConical, XCircle } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { AskAI } from "@/components/AskAI";
import { ChartCard, TimeSeries, useChartColors } from "@/components/charts";
import { MethodsPanel } from "@/components/Methods";
import { Badge, Button, Callout, Card, CardHeader, Field, PageHeader, Segmented, Select, Stat, cx, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";
import { useI18n, type Key } from "@/lib/i18n";

type Unit = "mg/dl" | "mmol/l";
interface Band { key: string; label: string; pct: number }
interface Target { metric: string; target: string; value: number; met: boolean; description: string }
interface AgpBin { minute: number; p5: number | null; p25: number | null; p50: number | null; p75: number | null; p95: number | null; n: number }
interface Cgm {
  n_readings: number; unit_detected: Unit; days: number; sampling_interval_min: number; wear_pct: number | null;
  mean_mg_dl: number; mean_mmol_l: number; sd_mg_dl: number; cv_pct: number; gmi_pct: number; gmi_mmol_mol: number;
  bands: Band[]; below_70_pct: number; above_180_pct: number; targets: Target[]; agp: AgpBin[]; notes: string[];
  series: { day: number; g: number }[];
  source: { synthetic?: boolean; description?: string; phenotype?: string; time_column?: string; glucose_column?: string;
    skipped_non_numeric?: number; start?: string; end?: string; points?: number };
}

const MGDL = 18.016;
// Stacked-bar order from top (very high) to bottom (very low), as in the standard AGP report
const BAND_ORDER = ["very_high", "high", "in_range", "low", "very_low"];
const BAND_COLOR: Record<string, string> = {
  very_high: "var(--warn)", high: "var(--accent)", in_range: "var(--ok)", low: "var(--danger)", very_low: "#7a1410",
};

export default function CgmAnalysis() {
  const toast = useToast();
  const c = useChartColors();
  const { t } = useI18n();
  const fileRef = useRef<HTMLInputElement>(null);
  const [display, setDisplay] = useState<Unit>("mg/dl");
  const [inUnit, setInUnit] = useState<"auto" | Unit>("auto");
  const [phenotype, setPhenotype] = useState("type2");
  const [fileName, setFileName] = useState<string | null>(null);

  const sample = useMutation({
    mutationFn: (ph: string) => api.get<Cgm>(`/api/cgm/sample?phenotype=${ph}&days=14`),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const upload = useMutation({
    mutationFn: (csv: string) => api.post<Cgm>("/api/cgm/analyze", { csv, unit: inUnit }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  useEffect(() => { sample.mutate("type2"); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const r = (upload.isSuccess && fileName ? upload.data : null) ?? sample.data;
  const conv = (v: number) => (display === "mmol/l" ? v / MGDL : v);
  const dec = display === "mmol/l" ? 1 : 0;

  const agpRows = useMemo(() => (r?.agp ?? []).filter((b) => b.p50 !== null).map((b) => ({
    h: b.minute / 60,
    outer: [conv(b.p5!), conv(b.p95!)] as [number, number],
    inner: [conv(b.p25!), conv(b.p75!)] as [number, number],
    median: conv(b.p50!),
  })), [r, display]); // eslint-disable-line react-hooks/exhaustive-deps
  const trace = useMemo(() => (r?.series ?? []).map((p) => ({ day: p.day, g: conv(p.g) })), [r, display]); // eslint-disable-line react-hooks/exhaustive-deps

  async function onFile(f: File | undefined) {
    if (!f) return;
    if (f.size > 15 * 1024 * 1024) { toast(t("cgm.fileTooBig"), "error"); return; }
    setFileName(f.name);
    upload.mutate(await f.text());
  }

  const bandsByKey = Object.fromEntries((r?.bands ?? []).map((b) => [b.key, b]));
  const bandLabel = (k: string) => t(`${display === "mmol/l" ? "cgm.bandMmol" : "cgm.band"}.${k}` as Key);
  const summary = r ? cgmSummary(r) : "";
  const explainQ = t("cgm.explainQ");

  return (
    <div>
      <PageHeader eyebrow={t("nav.group.physiology")} title={t("cgm.title")}
        description={t("cgm.description")}
        actions={<AskAI page="Glucose data (CGM)" summary={summary} question={explainQ} label={t("cgm.askBtn")} />} />

      <div className="grid xl:grid-cols-[320px_1fr] gap-5 items-start">
        <div className="flex flex-col gap-4 xl:sticky xl:top-8">
          <Card>
            <CardHeader title={t("cgm.yourData")} subtitle={t("cgm.yourDataSub")} />
            <div className="flex flex-col gap-3">
              <Field label={t("cgm.valuesAre")}>
                <Select value={inUnit} onChange={(e) => setInUnit(e.target.value as "auto" | Unit)}>
                  <option value="auto">{t("cgm.detect")}</option>
                  <option value="mg/dl">mg/dl</option>
                  <option value="mmol/l">mmol/l</option>
                </Select>
              </Field>
              <input ref={fileRef} type="file" accept=".csv,.txt,text/csv" className="hidden" onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
              <Button variant="primary" icon={<FileUp className="size-4" />} loading={upload.isPending} onClick={() => fileRef.current?.click()}>{t("cgm.upload")}</Button>
              <p className="text-[12px] text-muted leading-5">{t("cgm.privacy")}</p>
            </div>
          </Card>
          <Card>
            <CardHeader title={t("cgm.sample")} subtitle={t("cgm.sampleSub")} />
            <div className="flex flex-col gap-3">
              <Select value={phenotype} onChange={(e) => setPhenotype(e.target.value)}>
                <option value="normal">{t("cgm.ph.normal")}</option>
                <option value="insulin_resistant">{t("cgm.ph.ir")}</option>
                <option value="type2">{t("cgm.ph.t2")}</option>
              </Select>
              <Button icon={<FlaskConical className="size-4" />} loading={sample.isPending} onClick={() => { setFileName(null); upload.reset(); sample.mutate(phenotype); }}>{t("cgm.generate")}</Button>
            </div>
          </Card>
        </div>

        <div className="flex flex-col gap-5 min-w-0">
          {r && (
            <>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 flex-wrap text-[13px] text-muted min-w-0">
                  {r.source.synthetic ? <Badge tone="accent">{t("cgm.syntheticBadge")}</Badge> : <Badge tone="brand">{fileName}</Badge>}
                  <span>{t("cgm.info", { days: fmt(r.days, 1), n: r.n_readings.toLocaleString(), m: fmt(r.sampling_interval_min, 0) })}{r.wear_pct !== null && t("cgm.coverage", { c: fmt(Math.min(r.wear_pct, 100), 0) })}</span>
                </div>
                <Segmented<Unit> value={display} onChange={setDisplay} options={[{ value: "mg/dl", label: "mg/dl" }, { value: "mmol/l", label: "mmol/l" }]} />
              </div>
              {r.notes.length > 0 && <Callout tone="warn">{r.notes.join(" ")}</Callout>}
              {r.source.time_column && <p className="text-[12px] text-faint -mt-2">Read columns “{r.source.time_column}” and “{r.source.glucose_column}”, from {r.source.start?.replace("T", " ")} to {r.source.end?.replace("T", " ")}.{r.source.skipped_non_numeric ? ` Skipped ${r.source.skipped_non_numeric} non-numeric readings.` : ""}</p>}

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <Stat label={display === "mmol/l" ? t("cgm.tirMmol") : t("cgm.tir")} value={fmt(bandsByKey.in_range?.pct, 0)} unit="%" tone={bandsByKey.in_range?.pct > 70 ? "ok" : "warn"} hint={t("cgm.tirTarget")} />
                <Stat label={t("cgm.mean")} value={fmt(conv(r.mean_mg_dl), dec)} unit={display} hint={`SD ${fmt(conv(r.sd_mg_dl), dec)} ${display}`} />
                <Stat label={t("cgm.gmi")} value={fmt(r.gmi_pct, 1)} unit="%" hint={`${fmt(r.gmi_mmol_mol, 0)} mmol/mol`} />
                <Stat label={t("cgm.cv")} value={fmt(r.cv_pct, 1)} unit="%" tone={r.cv_pct <= 36 ? "ok" : "warn"} hint={t("cgm.cvTarget")} />
              </div>

              <div className="grid lg:grid-cols-[300px_1fr] gap-5">
                <Card>
                  <CardHeader title={t("cgm.ranges")} />
                  <div className="flex gap-4 items-stretch">
                    <div className="w-10 h-60 flex flex-col rounded-md overflow-hidden border border-line shrink-0" aria-label="Stacked time in ranges">
                      {BAND_ORDER.map((k) => <div key={k} style={{ flexGrow: Math.max(bandsByKey[k]?.pct ?? 0, 0.0001), background: BAND_COLOR[k] }} title={`${bandLabel(k)}: ${fmt(bandsByKey[k]?.pct, 1)}%`} />)}
                    </div>
                    <ul className="flex flex-col justify-between text-[12.5px] py-0.5 min-w-0">
                      {BAND_ORDER.map((k) => (
                        <li key={k} className="flex items-center gap-2">
                          <span className="size-2.5 rounded-sm shrink-0" style={{ background: BAND_COLOR[k] }} />
                          <span className="text-ink-2 leading-4 min-w-0">{bandLabel(k)}</span>
                          <span className="num font-semibold ml-auto pl-2 shrink-0">{fmt(bandsByKey[k]?.pct, 1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </Card>
                <ChartCard title={t("cgm.agp")} subtitle={t("cgm.agpSub", { unit: display })}>
                  <div style={{ height: 260 }} className="w-full">
                    <ResponsiveContainer>
                      <ComposedChart data={agpRows} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                        <CartesianGrid stroke={c.grid} vertical={false} />
                        <ReferenceArea y1={conv(70)} y2={conv(180)} fill={c.v1} fillOpacity={0.07} strokeOpacity={0} />
                        <ReferenceLine y={conv(70)} stroke={c.v1} strokeDasharray="4 4" />
                        <ReferenceLine y={conv(180)} stroke={c.v1} strokeDasharray="4 4" />
                        <XAxis dataKey="h" type="number" domain={[0, 24]} ticks={[0, 3, 6, 9, 12, 15, 18, 21, 24]} tickFormatter={(v) => `${String(v).padStart(2, "0")}:00`}
                          tick={{ fontSize: 11.5, fill: c.axis }} tickLine={false} axisLine={{ stroke: c.line }} />
                        <YAxis tick={{ fontSize: 11.5, fill: c.axis }} tickLine={false} axisLine={false} width={44} domain={[0, conv(400)]} ticks={display === "mmol/l" ? [0, 4, 8, 12, 16, 20] : [0, 70, 180, 250, 400]} />
                        <Tooltip contentStyle={{ background: c.surface, border: `1px solid ${c.line}`, borderRadius: 8, fontSize: 12 }}
                          labelFormatter={(v: number) => `${String(Math.floor(v)).padStart(2, "0")}:${String(Math.round((v % 1) * 60)).padStart(2, "0")}`}
                          formatter={(v: number | number[], name: string) => [Array.isArray(v) ? `${fmt(v[0], dec)}–${fmt(v[1], dec)}` : fmt(v, dec), name === "outer" ? "5–95%" : name === "inner" ? "25–75%" : "Median"]} />
                        <Area dataKey="outer" stroke="none" fill={c.v5} fillOpacity={0.18} isAnimationActive={false} />
                        <Area dataKey="inner" stroke="none" fill={c.v5} fillOpacity={0.38} isAnimationActive={false} />
                        <Line dataKey="median" stroke={c.v1} strokeWidth={2} dot={false} isAnimationActive={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </ChartCard>
              </div>

              <Card>
                <CardHeader title={t("cgm.targets")} subtitle={t("cgm.targetsSub")} />
                <ul className="grid sm:grid-cols-2 gap-x-6 gap-y-2">
                  {r.targets.map((tg) => (
                    <li key={tg.metric} className="flex items-center gap-2 text-[13px]">
                      {tg.met ? <CheckCircle2 className="size-4 text-ok shrink-0" /> : <XCircle className="size-4 text-warn shrink-0" />}
                      <span className="text-ink-2 min-w-0">{t(`cgm.t.${tg.metric}` as Key)}</span>
                      <span className={cx("num font-semibold ml-auto", tg.met ? "text-ok" : "text-warn")}>{fmt(tg.value, 1)}%</span>
                    </li>
                  ))}
                </ul>
              </Card>

              <Card>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-[14px] font-semibold">{t("cgm.explainTitle")}</div>
                    <p className="text-[13px] text-muted mt-0.5">{t("cgm.explainText")}</p>
                  </div>
                  <AskAI page="Glucose data (CGM)" summary={summary} question={explainQ} label={t("cgm.explainBtn")} />
                </div>
              </Card>

              <ChartCard title={t("cgm.all")} subtitle={t("cgm.allSub", { unit: display })}>
                <TimeSeries data={trace} x="day" xLabel={t("cgm.day")} height={220}
                  refAreas={[{ y1: conv(70), y2: conv(180), color: c.v1 }]}
                  series={[{ key: "g", label: `Glucose (${display})`, color: c.v1 }]} />
              </ChartCard>
              {r.source.synthetic && <Callout>{t("cgm.syntheticNote")}</Callout>}
            </>
          )}
          <Callout tone="warn" title={t("cgm.notAdviceTitle")}>{t("cgm.notAdvice")}</Callout>
          <MethodsPanel id="cgm" />
        </div>
      </div>
    </div>
  );
}


/** Plain-text summary of the report for the AI assistant: summary statistics only, no raw readings. */
function cgmSummary(r: Cgm): string {
  const periods: [string, number, number][] = [["overnight (00–06)", 0, 360], ["morning (06–12)", 360, 720], ["afternoon (12–18)", 720, 1080], ["evening (18–24)", 1080, 1440]];
  const med = (lo: number, hi: number) => {
    const v = r.agp.filter((b) => b.minute >= lo && b.minute < hi && b.p50 !== null).map((b) => b.p50 as number).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : null;
  };
  const tod = periods.map(([name, lo, hi]) => { const m = med(lo, hi); return m === null ? null : `${name} ${fmt(m, 0)} mg/dl (${fmt(m / MGDL, 1)} mmol/l)`; }).filter(Boolean).join("; ");
  const src = r.source.synthetic ? `SYNTHETIC demonstration data (meal-model ${r.source.phenotype} sample), not a real person` : "data uploaded by the user";
  return [
    `CGM report from ${src}.`,
    `${fmt(r.days, 1)} days, ${r.n_readings} readings every ${fmt(r.sampling_interval_min, 0)} min${r.wear_pct !== null ? `, ${fmt(Math.min(r.wear_pct, 100), 0)}% coverage` : ""}.`,
    `Mean glucose ${fmt(r.mean_mg_dl, 0)} mg/dl (${fmt(r.mean_mmol_l, 1)} mmol/l), SD ${fmt(r.sd_mg_dl, 0)} mg/dl, coefficient of variation ${fmt(r.cv_pct, 1)}%, GMI ${fmt(r.gmi_pct, 1)}% (${fmt(r.gmi_mmol_mol, 0)} mmol/mol).`,
    `Time in ranges: ${r.bands.map((b) => `${b.label} ${fmt(b.pct, 1)}%`).join(", ")}; below 70 mg/dl total ${fmt(r.below_70_pct, 1)}%, above 180 mg/dl total ${fmt(r.above_180_pct, 1)}%.`,
    `Consensus targets: ${r.targets.map((t) => `${t.description}: ${fmt(t.value, 1)}% (${t.met ? "met" : "not met"})`).join("; ")}.`,
    tod ? `Median glucose by time of day: ${tod}.` : "",
    r.notes.length ? `Data notes: ${r.notes.join(" ")}` : "",
  ].filter(Boolean).join(" ");
}
