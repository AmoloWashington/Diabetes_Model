import type { ReactNode } from "react";
import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";
import { useTheme } from "@/lib/theme";

export type Series = { key: string; label: string; color: string; dashed?: boolean; axis?: "left" | "right"; area?: boolean; points?: boolean };

export function useChartColors() {
  const { colors } = useTheme();
  return {
    grid: colors["viz-grid"] || "#ebe8df",
    axis: colors.muted || "#676c7a",
    ink: colors.ink || "#15171c",
    surface: colors.surface || "#fff",
    line: colors.line || "#e6e3da",
    v1: colors["viz-1"] || "#1b563e",
    v2: colors["viz-2"] || "#c07a12",
    v3: colors["viz-3"] || "#b42318",
    v4: colors["viz-4"] || "#6a5acd",
    v5: colors["viz-5"] || "#3f7f7a",
    faint: colors.faint || "#9a9ea9",
  };
}

const tick = { fontSize: 11.5 };

/** Round axis bounds and ticks to 1/2/2.5/5 x 10^k steps (about 5 ticks). */
export function niceScale(lo: number, hi: number, target = 5): { domain: [number, number]; ticks: number[] } {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { domain: [0, 1], ticks: [0, 0.5, 1] };
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const raw = (hi - lo) / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw) ?? 10 * mag;
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push(+v.toFixed(10));
  return { domain: [a, b], ticks };
}

export function TimeSeries({
  data, x, series, xLabel, yLabel, yRightLabel, height = 280, refLines = [], refAreas = [], yDomain, xDomain, xType = "number",
}: {
  data: Record<string, number | null>[]; x: string; series: Series[]; xLabel?: string; yLabel?: string; yRightLabel?: string;
  height?: number; refLines?: { y?: number; x?: number; label?: string; color?: string }[];
  refAreas?: { y1: number; y2: number; color?: string }[]; yDomain?: [number | string, number | string];
  xDomain?: [number | string, number | string]; xType?: "number" | "category";
}) {
  const c = useChartColors();
  const hasRight = series.some((s) => s.axis === "right");
  const leftVals = data.flatMap((r) => series.filter((s) => (s.axis ?? "left") === "left").map((s) => r[s.key]))
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const extra = [...refLines.map((r) => r.y), ...refAreas.flatMap((a) => [a.y1, a.y2])].filter((v): v is number => typeof v === "number");
  const lo0 = typeof yDomain?.[0] === "number" ? yDomain[0] : Math.min(...leftVals, ...extra.filter((v) => v <= Math.max(...leftVals)));
  const hi0 = typeof yDomain?.[1] === "number" ? yDomain[1] : Math.max(...leftVals);
  const nice = leftVals.length ? niceScale(lo0, hi0) : null;
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: hasRight ? 8 : 16, bottom: xLabel ? 18 : 4, left: 0 }}>
          <CartesianGrid stroke={c.grid} vertical={false} />
          <XAxis dataKey={x} type={xType} domain={xDomain ?? ["dataMin", "dataMax"]} tick={{ ...tick, fill: c.axis }} tickLine={false}
            axisLine={{ stroke: c.line }} label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10, fill: c.axis, fontSize: 11.5 } : undefined} allowDecimals
            tickFormatter={(v) => (typeof v === "number" ? String(+v.toPrecision(3)) : String(v))} />
          <YAxis yAxisId="left" tick={{ ...tick, fill: c.axis }} tickLine={false} axisLine={false} width={52}
            domain={nice ? nice.domain : yDomain ?? ["auto", "auto"]} ticks={nice?.ticks} allowDataOverflow={false}
            label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", offset: 12, fill: c.axis, fontSize: 11.5, style: { textAnchor: "middle" } } : undefined} />
          {hasRight && (
            <YAxis yAxisId="right" orientation="right" tick={{ ...tick, fill: c.axis }} tickLine={false} axisLine={false} width={52}
              label={yRightLabel ? { value: yRightLabel, angle: 90, position: "insideRight", offset: 12, fill: c.axis, fontSize: 11.5, style: { textAnchor: "middle" } } : undefined} />
          )}
          {refAreas.map((a, i) => <ReferenceArea key={i} yAxisId="left" y1={a.y1} y2={a.y2} fill={a.color ?? c.v1} fillOpacity={0.06} strokeOpacity={0} />)}
          {refLines.map((r, i) => (
            <ReferenceLine key={i} yAxisId="left" y={r.y} x={r.x} stroke={r.color ?? c.faint} strokeDasharray="4 4"
              label={r.label ? { value: r.label, position: r.x !== undefined ? "insideTopRight" : "insideTopLeft", fill: r.color ?? c.axis, fontSize: 11 } : undefined} />
          ))}
          <Tooltip
            contentStyle={{ background: c.surface, border: `1px solid ${c.line}`, borderRadius: 8, fontSize: 12, boxShadow: "0 4px 16px rgb(0 0 0 / .08)" }}
            labelStyle={{ color: c.axis }} formatter={(v: number) => (typeof v === "number" ? v.toFixed(Math.abs(v) < 10 ? 3 : 1) : v)}
            labelFormatter={(v: number) => (xLabel ? `${typeof v === "number" ? +v.toPrecision(4) : v} ${xLabel}` : String(v))}
          />
          {series.length > 1 && <Legend verticalAlign="top" height={28} iconType="plainline" wrapperStyle={{ fontSize: 12 }} />}
          {series.map((s) =>
            s.points ? (
              <Scatter key={s.key} yAxisId={s.axis ?? "left"} dataKey={s.key} name={s.label} fill={s.color} />
            ) : s.area ? (
              <Area key={s.key} yAxisId={s.axis ?? "left"} dataKey={s.key} name={s.label} stroke={s.color} fill={s.color} fillOpacity={0.12} strokeWidth={2} dot={false} isAnimationActive={false} />
            ) : (
              <Line key={s.key} yAxisId={s.axis ?? "left"} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={false}
                strokeDasharray={s.dashed ? "5 4" : undefined} isAnimationActive={false} connectNulls />
            ),
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Bars({ data, x, y, color, height = 240, horizontal, xLabel, colorFn }: {
  data: Record<string, number | string>[]; x: string; y: string; color?: string; height?: number; horizontal?: boolean;
  xLabel?: string; colorFn?: (row: Record<string, number | string>) => string;
}) {
  const c = useChartColors();
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <BarChart data={data} layout={horizontal ? "vertical" : "horizontal"} margin={{ top: 4, right: 16, bottom: xLabel ? 18 : 4, left: horizontal ? 8 : 0 }}>
          <CartesianGrid stroke={c.grid} horizontal={!horizontal} vertical={!!horizontal} />
          <XAxis
            type={horizontal ? "number" : "category"} dataKey={horizontal ? undefined : x}
            tick={{ ...tick, fill: c.axis }} tickLine={false} axisLine={{ stroke: c.line }}
            label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10, fill: c.axis, fontSize: 11.5 } : undefined}
          />
          <YAxis
            type={horizontal ? "category" : "number"} dataKey={horizontal ? x : undefined}
            tick={{ ...tick, fill: horizontal ? c.ink : c.axis }} tickLine={false} axisLine={false}
            width={horizontal ? 200 : 44} interval={0}
          />
          <Tooltip cursor={{ fill: c.grid, opacity: 0.5 }} contentStyle={{ background: c.surface, border: `1px solid ${c.line}`, borderRadius: 8, fontSize: 12 }}
            formatter={(v: number) => (typeof v === "number" ? v.toFixed(3) : v)} />
          <Bar dataKey={y} radius={3} isAnimationActive={false} fill={color ?? c.v1}>
            {colorFn && data.map((row, i) => <Cell key={i} fill={colorFn(row)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ChartCard({ title, subtitle, children, actions }: { title: string; subtitle?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="bg-surface border border-line rounded-xl shadow-card p-5">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="text-[14px] font-semibold text-ink">{title}</h3>
          {subtitle && <p className="text-[12.5px] text-muted mt-0.5">{subtitle}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}
