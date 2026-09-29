import { useMutation, useQuery } from "@tanstack/react-query";
import { BarChart3, Box, Cpu, Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Bars, ChartCard } from "@/components/charts";
import { CONF_BINS, ConfidenceLegend, MolViewer } from "@/components/rna/MolViewer";
import { Badge, Button, Callout, Card, CardHeader, Empty, Field, PageHeader, Segmented, Select, Stat, Table, Textarea, useToast } from "@/components/ui";
import { api, type DatasetMeta, type PredictResult, type RnaSequence } from "@/lib/api";
import { fmt } from "@/lib/format";
import { dotBracketPairs } from "@/lib/rna";

type Method = "auto" | "template" | "denovo";
interface Bench { n_targets: number; n_with_template: number; mean_tm_score: number | null; confidence_tm_correlation: number | null; max_template_identity: number;
  rows: { target_id: string; length: number; template: string | null; identity: number; mean_confidence: number; tm_score: number | null }[] }

function download(name: string, text: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "chemical/x-pdb" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
}

export default function RnaPredict() {
  const toast = useToast();
  const [params] = useSearchParams();
  const [source, setSource] = useState<"stored" | "paste">("stored");
  const [sid, setSid] = useState<number | "">(params.get("sequence_id") ? Number(params.get("sequence_id")) : "");
  const [raw, setRaw] = useState("GGGCGCAAGCCUAUGCGCUUCGGCGCAUAGGCUUGCGCCC");
  const [method, setMethod] = useState<Method>("auto");
  const seqs = useQuery({ queryKey: ["rna-seqs", ""], queryFn: () => api.get<{ items: RnaSequence[] }>("/api/rna/sequences?limit=500") });
  const ds = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<{ items: DatasetMeta[] }>("/api/datasets") });

  useEffect(() => { if (!params.get("sequence_id") && seqs.data && !seqs.data.items.length) setSource("paste"); }, [seqs.data, params]);

  const predict = useMutation({
    mutationFn: () => api.post<PredictResult>("/api/rna/predict", source === "stored" ? { sequence_id: sid, method } : { sequence: raw, method }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const bench = useMutation({
    mutationFn: () => api.post<Bench>("/api/datasets/benchmark", { n_targets: 20, max_identity: 0.9 }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const r = predict.data;
  const confRows = useMemo(() => r ? r.confidence.map((v, i) => ({ pos: i + 1, conf: +(100 * v).toFixed(1) })) : [], [r]);
  const pairs = useMemo(() => dotBracketPairs(r?.secondary_structure), [r]);
  const binColor = (v: number) => (CONF_BINS.find((b) => v >= b.min) ?? CONF_BINS[3]).color;

  return (
    <div>
      <PageHeader eyebrow="RNA structure" title="3D structure prediction"
        description="C1'-level models with per-residue confidence. Template-based modelling uses structures from loaded datasets; without a suitable template a coarse-grained de novo model is built from the ViennaRNA secondary structure." />
      <div className="grid xl:grid-cols-[340px_1fr] gap-5 items-start">
        <Card className="xl:sticky xl:top-8">
          <CardHeader title="Input" />
          <div className="flex flex-col gap-4">
            <Segmented value={source} onChange={setSource} options={[{ value: "stored", label: "Stored sequence" }, { value: "paste", label: "Paste" }]} />
            {source === "stored" ? (
              <Field label="Sequence" hint={<Link to="/rna/sequences" className="text-brand-text hover:underline">Manage sequences</Link>}>
                <Select value={sid} onChange={(e) => setSid(e.target.value ? Number(e.target.value) : "")}>
                  <option value="">Select a sequence…</option>
                  {seqs.data?.items.map((s) => <option key={s.id} value={s.id}>{s.name} ({s.length} nt)</option>)}
                </Select>
              </Field>
            ) : (
              <Field label="RNA sequence"><Textarea rows={5} className="font-mono text-[12px] break-all" value={raw} onChange={(e) => setRaw(e.target.value)} spellCheck={false} /></Field>
            )}
            <Field label="Method">
              <Select value={method} onChange={(e) => setMethod(e.target.value as Method)}>
                <option value="auto">Auto (template → de novo)</option>
                <option value="template">Template-based</option>
                <option value="denovo">Coarse-grained de novo</option>
              </Select>
            </Field>
            <Button variant="primary" icon={<Cpu className="size-4" />} loading={predict.isPending}
              disabled={source === "stored" ? sid === "" : !raw.trim()} onClick={() => predict.mutate()}>Predict structure</Button>
            <div className="rounded-lg bg-surface-2 border border-line p-3 text-[12.5px] text-muted">
              Template library: <b className="text-ink">{ds.data?.items.length ? `${ds.data.items.length} dataset(s) loaded` : "empty"}</b>.{" "}
              <Link to="/data" className="text-brand-text hover:underline">Load Kaggle structures</Link> to enable template-based modelling.
            </div>
          </div>
        </Card>
        <div className="flex flex-col gap-5 min-w-0">
          {!r ? (
            <Card><Empty icon={<Box className="size-5" />} title="No prediction yet">Choose a sequence and method, then select Predict structure.</Empty></Card>
          ) : (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <Stat label="Mean confidence" value={fmt(100 * r.mean_confidence, 0)} hint="0–100 per residue" />
                {r.template_id ? (
                  <>
                    <Stat label="Template" value={<span className="text-[16px]">{r.template_id}</span>} />
                    <Stat label="Sequence identity" value={fmt(100 * (r.identity ?? 0), 1)} unit="%" />
                    <Stat label="Coverage" value={fmt(100 * (r.coverage ?? 0), 1)} unit="%" />
                  </>
                ) : (
                  <>
                    <Stat label="MFE" value={fmt(r.mfe_kcal_mol, 2)} unit="kcal/mol" />
                    <Stat label="Restraint RMSE" value={fmt(r.restraint_rmse_A, 2)} unit="Å" />
                    <Stat label="Length" value={r.sequence.length} unit="nt" />
                  </>
                )}
              </div>
              <Card padded={false}>
                <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-line">
                  <div className="flex items-center gap-2"><h3 className="text-[14px] font-semibold">Predicted model</h3><Badge tone={r.template_id ? "brand" : "accent"}>{r.method}</Badge></div>
                  <div className="flex gap-2">
                    <Button size="sm" icon={<Download className="size-4" />} onClick={() => download(`glucolab_prediction${r.structure_id ? `_${r.structure_id}` : ""}.pdb`, r.pdb)}>PDB</Button>
                    {r.structure_id && <Link to={`/rna/structures?id=${r.structure_id}`}><Button size="sm">Open in viewer</Button></Link>}
                  </div>
                </div>
                <MolViewer data={r.pdb} colorBy="confidence" height={460} pairs={pairs} />
                <div className="px-5 py-3 border-t border-line flex flex-wrap justify-between gap-2"><ConfidenceLegend /><span className="text-[12px] text-faint">Spheres: C1' atoms · dark sticks: backbone · grey rungs: base pairs</span></div>
              </Card>
              <Callout tone="warn" title="How to read this model">{r.caveat}</Callout>
              {r.secondary_structure && (
                <div className="font-mono text-[12px] leading-5 rounded-lg bg-surface border border-line p-3 overflow-x-auto whitespace-pre shadow-card">
                  <div className="text-faint">{r.sequence}</div><div>{r.secondary_structure}</div>
                </div>
              )}
              <ChartCard title="Per-residue confidence" subtitle="0–100; coloured by confidence bin">
                <Bars data={confRows} x="pos" y="conf" height={200} xLabel="residue" colorFn={(row) => binColor(Number(row.conf))} />
              </ChartCard>
            </>
          )}
          <Card>
            <CardHeader title="Calibration benchmark" subtitle="Leave-one-out on the loaded template library: each target is predicted from the others, excluding near-identical templates, and scored with TM-score."
              actions={<Button size="sm" icon={<BarChart3 className="size-4" />} loading={bench.isPending} onClick={() => bench.mutate()} disabled={!ds.data?.items.length}>Run benchmark</Button>} />
            {!bench.data ? (
              <p className="text-[13px] text-muted">{ds.data?.items.length ? "Run the benchmark to see how template confidence relates to actual accuracy." : "Load a structure dataset first."}</p>
            ) : (
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-3 gap-3">
                  <Stat label="Targets with a template" value={`${bench.data.n_with_template}/${bench.data.n_targets}`} />
                  <Stat label="Mean TM-score" value={fmt(bench.data.mean_tm_score, 3)} hint="0–1; 1 = identical after superposition" />
                  <Stat label="Confidence–TM correlation" value={fmt(bench.data.confidence_tm_correlation, 2)} hint="Pearson r" />
                </div>
                <Table head={["Target", "Length", "Template", "Identity", "Confidence", "TM-score"]}>
                  {bench.data.rows.map((row) => (
                    <tr key={row.target_id}>
                      <td className="px-3 py-2 font-mono text-[12px]">{row.target_id}</td><td className="px-3 py-2 num">{row.length}</td>
                      <td className="px-3 py-2 font-mono text-[12px]">{row.template ?? "–"}</td><td className="px-3 py-2 num">{fmt(100 * row.identity, 0)}%</td>
                      <td className="px-3 py-2 num">{fmt(100 * row.mean_confidence, 0)}</td>
                      <td className="px-3 py-2 num font-medium">{fmt(row.tm_score, 3)}</td>
                    </tr>
                  ))}
                </Table>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
