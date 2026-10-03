import { useMutation } from "@tanstack/react-query";
import { MethodsPanel } from "@/components/Methods";
import { Calculator } from "lucide-react";
import { useState } from "react";
import { Badge, Button, Callout, Card, CardHeader, Empty, Field, Input, PageHeader, Stat, Table, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

interface Indices {
  homa1?: { homa_ir: number; homa_beta_pct: number | null; note: string };
  quicki?: number; eag?: { eag_mg_dl: number; eag_mmol_l: number }; tyg_index?: number;
  bmi?: { bmi: number; who_category: string; note: string };
  ada: { results: { test: string; value: number; unit: string; category: string; thresholds: string }[]; summary: string; disclaimer: string };
}

const FIELDS: [string, string, string][] = [
  ["fasting_glucose_mg_dl", "Fasting glucose", "mg/dl"],
  ["fasting_insulin_uU_ml", "Fasting insulin", "µU/ml"],
  ["hba1c_pct", "HbA1c", "%"],
  ["ogtt_2h_mg_dl", "2-h OGTT glucose", "mg/dl"],
  ["random_glucose_mg_dl", "Random glucose", "mg/dl"],
  ["triglycerides_mg_dl", "Triglycerides", "mg/dl"],
  ["weight_kg", "Weight", "kg"],
  ["height_cm", "Height", "cm"],
];

export default function Clinical() {
  const toast = useToast();
  const [v, setV] = useState<Record<string, string>>({ fasting_glucose_mg_dl: "105", fasting_insulin_uU_ml: "12", hba1c_pct: "5.9" });
  const [sym, setSym] = useState(false);
  const run = useMutation({
    mutationFn: () => api.post<Indices>("/api/clinical/indices", {
      ...Object.fromEntries(Object.entries(v).filter(([, x]) => x !== "").map(([k, x]) => [k, +x])), classic_symptoms: sym,
    }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const r = run.data;
  const tone = (c: string) => (c.startsWith("diabetes") ? "danger" : c.includes("prediabetes") ? "warn" : c === "normal" ? "ok" : "neutral") as "danger" | "warn" | "ok" | "neutral";

  return (
    <div>
      <PageHeader eyebrow="Physiology" title="Clinical indices"
        description="HOMA1-IR and %B (Matthews 1985), QUICKI (Katz 2000), eAG (ADAG 2008), TyG (Simental-Mendía 2008), BMI (WHO) and ADA diagnostic thresholds. Enter any subset." />
      <div className="grid xl:grid-cols-[340px_1fr] gap-5 items-start">
        <Card>
          <CardHeader title="Measurements" />
          <form className="flex flex-col gap-3" onSubmit={(e) => { e.preventDefault(); run.mutate(); }}>
            <div className="grid grid-cols-2 gap-3">
              {FIELDS.map(([k, label, unit]) => (
                <Field key={k} label={<>{label} <span className="text-faint font-normal">({unit})</span></>}>
                  <Input type="number" step="any" value={v[k] ?? ""} onChange={(e) => setV({ ...v, [k]: e.target.value })} />
                </Field>
              ))}
            </div>
            <label className="flex items-center gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={sym} onChange={(e) => setSym(e.target.checked)} /> Classic symptoms of hyperglycaemia</label>
            <Button type="submit" variant="primary" loading={run.isPending} icon={<Calculator className="size-4" />}>Compute</Button>
          </form>
        </Card>
        <div className="flex flex-col gap-5 min-w-0">
          {!r ? (
            <Card><Empty icon={<Calculator className="size-5" />} title="No results yet">Enter measurements and select Compute.</Empty></Card>
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {r.homa1 && <Stat label="HOMA1-IR" value={fmt(r.homa1.homa_ir, 2)} />}
                {r.homa1 && <Stat label="HOMA1-%B" value={r.homa1.homa_beta_pct == null ? "n/a" : fmt(r.homa1.homa_beta_pct, 0)} unit={r.homa1.homa_beta_pct == null ? undefined : "%"} />}
                {r.quicki !== undefined && <Stat label="QUICKI" value={fmt(r.quicki, 3)} />}
                {r.eag && <Stat label="Estimated average glucose" value={fmt(r.eag.eag_mg_dl, 0)} unit="mg/dl" hint={`${fmt(r.eag.eag_mmol_l, 1)} mmol/l`} />}
                {r.tyg_index !== undefined && <Stat label="TyG index" value={fmt(r.tyg_index, 2)} />}
                {r.bmi && <Stat label="BMI" value={fmt(r.bmi.bmi, 1)} unit="kg/m²" hint={r.bmi.who_category} />}
              </div>
              <Card>
                <CardHeader title="ADA diagnostic classification" />
                {r.ada.results.length > 0 && (
                  <Table head={["Test", "Value", "Category", "Thresholds"]}>
                    {r.ada.results.map((x) => (
                      <tr key={x.test}>
                        <td className="px-3 py-2.5">{x.test}</td>
                        <td className="px-3 py-2.5 num font-mono text-[12.5px]">{fmt(x.value, 1)} {x.unit}</td>
                        <td className="px-3 py-2.5"><Badge tone={tone(x.category)}>{x.category}</Badge></td>
                        <td className="px-3 py-2.5 text-muted text-[12.5px]">{x.thresholds}</td>
                      </tr>
                    ))}
                  </Table>
                )}
                <div className="mt-4"><Callout tone={r.ada.summary.startsWith("Meets") ? "danger" : r.ada.summary.includes("prediabetes") || r.ada.summary.includes("confirmation") ? "warn" : "info"}>{r.ada.summary}</Callout></div>
                <p className="text-[12px] text-faint mt-3">{r.ada.disclaimer} {r.homa1?.note} {r.bmi?.note}</p>
              </Card>
            </>
          )}
        </div>
      </div>
      <div className="mt-6"><MethodsPanel id="clinical" /></div>
    </div>
  );
}
