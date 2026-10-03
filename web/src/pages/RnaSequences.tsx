import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MethodsPanel } from "@/components/Methods";
import { Cpu, FileUp, Plus, Search, TestTubes, Trash2, Wand2, X } from "lucide-react";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArcDiagram, SequenceBlocks } from "@/components/rna/Sequence";
import { ConfidenceLegend } from "@/components/rna/MolViewer";
import { Badge, Button, Callout, Card, CardHeader, Empty, Input, PageHeader, Slider, Spinner, Stat, Table, Textarea, cx, useToast } from "@/components/ui";
import { api, type FoldResult, type RnaSequence, type StructureSummary } from "@/lib/api";
import { date, fmt } from "@/lib/format";

type AddResult = { records: { sequence: RnaSequence; created: boolean; warnings: string[] }[]; created: number; duplicates: number };
type Detail = RnaSequence & { composition: { length: number; counts: Record<string, number>; gc_percent: number | null }; structures: StructureSummary[] };

const EXAMPLE = `>tRNA-like_hairpin example
GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA
>GNRA_tetraloop_hairpin
GGGCGCAAGCCUAUGCGCUUCGGCGCAUAGGCUUGCGCCC`;

function AddPanel({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [text, setText] = useState("");
  const [warnings, setWarnings] = useState<string[]>([]);
  const add = useMutation({
    mutationFn: () => api.post<AddResult>("/api/rna/sequences", { fasta: text, source: "paste" }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["rna-seqs"] });
      const w = r.records.flatMap((x) => x.warnings.map((m) => `${x.sequence.name}: ${m}`));
      setWarnings(w);
      toast(`${r.created} added${r.duplicates ? `, ${r.duplicates} already stored` : ""}`);
      if (!w.length) onClose();
    },
    onError: (e: Error) => toast(e.message, "error"),
  });
  return (
    <Card className="mb-5">
      <CardHeader title="Add sequences" subtitle="FASTA (one or many records) or a single raw sequence. T is converted to U; IUPAC codes are accepted and flagged."
        actions={<button onClick={onClose} className="size-8 grid place-items-center rounded-md text-muted hover:bg-surface-3" aria-label="Close"><X className="size-4" /></button>} />
      <Textarea rows={7} className="font-mono text-[12px]" value={text} onChange={(e) => setText(e.target.value)} placeholder={">my_rna description\nGGGAAACUUCGGUUUCCC"} spellCheck={false} />
      <div className="flex gap-2 mt-3">
        <Button variant="primary" loading={add.isPending} disabled={!text.trim()} onClick={() => add.mutate()} icon={<Plus className="size-4" />}>Validate and store</Button>
        <Button variant="ghost" onClick={() => setText(EXAMPLE)}>Insert example</Button>
      </div>
      {warnings.length > 0 && <div className="mt-3"><Callout tone="warn" title="Stored with warnings">{warnings.map((w) => <div key={w}>{w}</div>)}</Callout></div>}
    </Card>
  );
}

function DetailPanel({ id, onClose }: { id: number; onClose: () => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const nav = useNavigate();
  const [temp, setTemp] = useState(37);
  const d = useQuery({ queryKey: ["rna-seq", id], queryFn: () => api.get<Detail>(`/api/rna/sequences/${id}`) });
  const fold = useMutation({
    mutationFn: () => api.post<FoldResult>("/api/rna/fold", { sequence_id: id, temperature_c: temp }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const del = useMutation({
    mutationFn: () => api.del(`/api/rna/sequences/${id}`),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["rna-seqs"] }); toast("Sequence deleted"); onClose(); },
  });
  if (d.isLoading || !d.data) return <Card><Spinner label="Loading sequence…" /></Card>;
  const s = d.data;
  const f = fold.data;
  return (
    <Card>
      <CardHeader
        title={<span className="flex items-center gap-2">{s.name} <Badge>{s.source}</Badge></span>}
        subtitle={s.description || `Added ${date(s.created_at)}`}
        actions={<>
          <Button size="sm" variant="primary" icon={<Cpu className="size-4" />} onClick={() => nav(`/rna/predict?sequence_id=${id}`)}>Predict 3D</Button>
          <Button size="sm" variant="danger" icon={<Trash2 className="size-4" />} loading={del.isPending} onClick={() => { if (confirm(`Delete ${s.name}?`)) del.mutate(); }}>Delete</Button>
          <button onClick={onClose} className="size-8 grid place-items-center rounded-md text-muted hover:bg-surface-3" aria-label="Close"><X className="size-4" /></button>
        </>}
      />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <Stat label="Length" value={s.length} unit="nt" />
        <Stat label="GC content" value={fmt(s.composition.gc_percent, 1)} unit="%" />
        <Stat label="A · U" value={`${s.composition.counts.A} · ${s.composition.counts.U}`} />
        <Stat label="G · C" value={`${s.composition.counts.G} · ${s.composition.counts.C}`} hint={s.composition.counts.other ? `${s.composition.counts.other} ambiguous` : undefined} />
      </div>
      <div className="rounded-lg border border-line bg-surface-2 p-3 mb-5 max-h-48 overflow-y-auto"><SequenceBlocks sequence={s.sequence} /></div>

      <div className="flex flex-wrap items-end gap-4 mb-4">
        <div className="w-60"><Slider label="Folding temperature" value={temp} min={0} max={100} step={1} onChange={setTemp} format={(v) => `${v} °C`} /></div>
        <Button variant="primary" icon={<Wand2 className="size-4" />} loading={fold.isPending} onClick={() => fold.mutate()}>Fold with ViennaRNA</Button>
      </div>
      {f && (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="MFE" value={fmt(f.mfe_kcal_mol, 2)} unit="kcal/mol" />
            <Stat label="Ensemble free energy" value={fmt(f.ensemble_free_energy_kcal_mol, 2)} unit="kcal/mol" />
            <Stat label="MFE frequency" value={f.mfe_frequency_in_ensemble !== undefined ? fmt(100 * f.mfe_frequency_in_ensemble, 1) : "–"} unit="%" />
            <Stat label="Mean confidence" value={f.mean_confidence !== undefined ? fmt(100 * f.mean_confidence, 0) : "–"} hint={`${f.n_pairs} pairs · ${f.n_stems} stems`} />
          </div>
          <div className="rounded-lg border border-line p-3">
            <ArcDiagram sequence={f.sequence} pairs={f.pairs} probs={f.pair_probabilities} confidence={f.confidence} />
            <div className="mt-2 flex flex-wrap justify-between gap-2"><ConfidenceLegend /><span className="text-[12px] text-faint">Arcs: MFE pairs coloured by pair probability; faint arcs: other ensemble pairs (p ≥ 0.1). Bar: per-nucleotide confidence.</span></div>
          </div>
          <div className="font-mono text-[12px] leading-5 rounded-lg bg-surface-2 border border-line p-3 overflow-x-auto whitespace-pre">
            <div className="text-faint">sequence  {f.sequence}</div>
            <div>MFE       {f.mfe_structure}</div>
            {f.centroid_structure && <div className="text-ink-2">centroid  {f.centroid_structure}</div>}
          </div>
          <p className="text-[12px] text-faint">{f.method} · {f.energy_model} · {f.temperature_c} °C. Partition function computed for sequences up to 1000 nt.</p>
        </div>
      )}
      {s.structures.length > 0 && (
        <div className="mt-5">
          <h4 className="text-[13px] font-semibold mb-2">3D models</h4>
          <div className="flex flex-wrap gap-2">
            {s.structures.map((st) => (
              <button key={st.id} onClick={() => nav(`/rna/structures?id=${st.id}`)} className="text-left rounded-lg border border-line px-3 py-2 hover:bg-surface-2">
                <div className="text-[13px] font-medium">{st.name}</div>
                <div className="text-[12px] text-muted">confidence {st.mean_confidence != null ? fmt(100 * st.mean_confidence, 0) : "–"} · {date(st.created_at)}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

export default function RnaSequences() {
  const qc = useQueryClient();
  const toast = useToast();
  const file = useRef<HTMLInputElement>(null);
  const [q, setQ] = useState("");
  const [adding, setAdding] = useState(false);
  const [sel, setSel] = useState<number | null>(null);
  const list = useQuery({ queryKey: ["rna-seqs", q], queryFn: () => api.get<{ items: RnaSequence[]; total: number }>(`/api/rna/sequences?q=${encodeURIComponent(q)}`) });
  const upload = useMutation({
    mutationFn: (f: File) => { const fd = new FormData(); fd.append("file", f); return api.upload<AddResult>("/api/rna/sequences/upload", fd); },
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["rna-seqs"] }); toast(`${r.created} added from file${r.duplicates ? `, ${r.duplicates} duplicates skipped` : ""}`); },
    onError: (e: Error) => toast(e.message, "error"),
  });

  return (
    <div>
      <PageHeader eyebrow="RNA structure" title="Sequences"
        description="Upload, validate and store RNA sequences. Identical sequences are de-duplicated by SHA-256. Fold them with ViennaRNA or send them to 3D prediction."
        actions={<>
          <input ref={file} type="file" accept=".fa,.fasta,.fna,.txt" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) upload.mutate(f); e.target.value = ""; }} />
          <Button icon={<FileUp className="size-4" />} loading={upload.isPending} onClick={() => file.current?.click()}>Upload FASTA</Button>
          <Button variant="primary" icon={<Plus className="size-4" />} onClick={() => setAdding(true)}>Add sequences</Button>
        </>}
      />
      {adding && <AddPanel onClose={() => setAdding(false)} />}
      <div className={cx("grid gap-5 items-start", sel ? "2xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]" : "")}>
        <Card padded={false}>
          <div className="p-4 border-b border-line flex items-center gap-3">
            <div className="relative flex-1 max-w-sm">
              <Search className="size-4 text-faint absolute left-3 top-1/2 -translate-y-1/2" />
              <Input className="pl-9" placeholder="Search by name or description" value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            <span className="text-[12.5px] text-muted">{list.data?.total ?? 0} sequences</span>
          </div>
          {list.isLoading ? <div className="p-6"><Spinner label="Loading…" /></div> : !list.data?.items.length ? (
            <Empty icon={<TestTubes className="size-5" />} title="No sequences yet" action={<Button variant="primary" onClick={() => setAdding(true)} icon={<Plus className="size-4" />}>Add sequences</Button>}>
              Paste FASTA, upload a file, or import sequences from a dataset in the Dataset explorer.
            </Empty>
          ) : (
            <Table head={["Name", "Length", "GC", "Source", "Added"]}>
              {list.data.items.map((s) => (
                <tr key={s.id} onClick={() => setSel(s.id)} className={cx("cursor-pointer hover:bg-surface-2", sel === s.id && "bg-brand-soft/60")}>
                  <td className="px-3 py-2.5"><div className="font-medium text-ink">{s.name}</div><div className="font-mono text-[11.5px] text-faint truncate max-w-[260px]">{s.sequence.slice(0, 40)}{s.length > 40 ? "…" : ""}</div></td>
                  <td className="px-3 py-2.5 num">{s.length}</td>
                  <td className="px-3 py-2.5 num">{fmt(s.gc_percent, 1)}%</td>
                  <td className="px-3 py-2.5"><Badge>{s.source}</Badge></td>
                  <td className="px-3 py-2.5 text-muted text-[12.5px] whitespace-nowrap">{date(s.created_at)}</td>
                </tr>
              ))}
            </Table>
          )}
        </Card>
        {sel && <DetailPanel key={sel} id={sel} onClose={() => setSel(null)} />}
      </div>
      <p className="mt-4 text-[12px] text-faint">Sequences are stored in a local SQLite database in the server's data directory.</p>
      <div className="mt-6"><MethodsPanel id="rnafold" /></div>
    </div>
  );
}
