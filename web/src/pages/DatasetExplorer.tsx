import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, CloudDownload, Database, FileUp, KeyRound, Search, Table2, TestTubes, Trash2 } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Bars, ChartCard, useChartColors } from "@/components/charts";
import { Badge, Button, Callout, Card, CardHeader, Empty, Field, Input, PageHeader, Select, Spinner, Stat, Table, cx, useToast } from "@/components/ui";
import { api, type DatasetDetail, type DatasetMeta } from "@/lib/api";
import { bytes, date, fmt } from "@/lib/format";

interface KaggleStatus { configured: boolean; suggested: { handle: string; kind: string; title: string; description: string }[] }
interface Rows { total: number; offset: number; columns: string[]; rows: (string | number | null)[][] }
interface Rna {
  sequence_files: { file: string; n_sequences: number; length: { min: number; max: number; mean: number; median: number };
    length_histogram: { counts: number[]; edges: number[] }; gc_percent: { mean: number; std: number };
    composition: Record<string, number>; by_year?: Record<string, number> }[];
}

function KagglePanel() {
  const qc = useQueryClient();
  const toast = useToast();
  const st = useQuery({ queryKey: ["kaggle-status"], queryFn: () => api.get<KaggleStatus>("/api/datasets/kaggle/status") });
  const [handle, setHandle] = useState("");
  const [kind, setKind] = useState<"competition" | "dataset">("dataset");
  const imp = useMutation({
    mutationFn: (b: { handle: string; kind: string }) => api.post<DatasetMeta>("/api/datasets/kaggle", b),
    onSuccess: (m) => { qc.invalidateQueries({ queryKey: ["datasets"] }); toast(`Imported ${m.title}`); },
    onError: (e: Error) => toast(e.message, "error"),
  });
  return (
    <Card>
      <CardHeader title={<span className="flex items-center gap-2">Kaggle {st.data && <Badge tone={st.data.configured ? "ok" : "warn"}>{st.data.configured ? "Connected" : "Token required"}</Badge>}</span>}
        subtitle="Downloads use the official kagglehub client with your own API token." />
      {st.data && !st.data.configured && (
        <div className="mb-4"><Callout tone="warn" title="Connect your Kaggle account">
          Create an API token at <b>kaggle.com → Settings → API</b>, then add <code className="font-mono">KAGGLE_API_TOKEN</code> (or <code className="font-mono">KAGGLE_USERNAME</code> and <code className="font-mono">KAGGLE_KEY</code>) to the git-ignored <code className="font-mono">.env</code> file and restart the server. For competitions, accept the rules on kaggle.com first.
        </Callout></div>
      )}
      <div className="flex flex-col gap-2.5 mb-4">
        {st.data?.suggested.map((s) => (
          <div key={s.handle} className="rounded-lg border border-line p-3 flex items-start justify-between gap-3">
            <div className="min-w-0"><div className="text-[13px] font-semibold">{s.title} <Badge>{s.kind}</Badge></div>
              <div className="text-[12px] text-muted mt-0.5">{s.description}</div>
              <div className="font-mono text-[11.5px] text-faint mt-1">{s.handle}</div></div>
            <Button size="sm" icon={<CloudDownload className="size-4" />} disabled={!st.data?.configured} loading={imp.isPending && imp.variables?.handle === s.handle}
              onClick={() => imp.mutate({ handle: s.handle, kind: s.kind })}>Import</Button>
          </div>
        ))}
      </div>
      <form className="grid grid-cols-[1fr_auto] gap-2 items-end" onSubmit={(e) => { e.preventDefault(); imp.mutate({ handle: handle.trim(), kind }); }}>
        <Field label="Any Kaggle handle"><Input placeholder="owner/dataset or competition-slug" value={handle} onChange={(e) => setHandle(e.target.value)} className="font-mono" /></Field>
        <Select value={kind} onChange={(e) => setKind(e.target.value as "competition" | "dataset")} className="w-36" aria-label="Kind"><option value="dataset">Dataset</option><option value="competition">Competition</option></Select>
        <Button type="submit" className="col-span-2" icon={<KeyRound className="size-4" />} disabled={!handle.trim() || !st.data?.configured} loading={imp.isPending && imp.variables?.handle === handle.trim()}>Download from Kaggle</Button>
      </form>
    </Card>
  );
}

function UploadPanel() {
  const qc = useQueryClient();
  const toast = useToast();
  const fileRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const up = useMutation({
    mutationFn: () => { const fd = new FormData(); fd.append("name", name); files.forEach((f) => fd.append("files", f)); return api.upload<DatasetMeta>("/api/datasets/upload", fd); },
    onSuccess: (m) => { qc.invalidateQueries({ queryKey: ["datasets"] }); toast(`Uploaded ${m.title}`); setFiles([]); setName(""); },
    onError: (e: Error) => toast(e.message, "error"),
  });
  return (
    <Card>
      <CardHeader title="Upload files" subtitle="CSV, TSV, FASTA, PDB, mmCIF or JSON; up to 200 MB. Works offline." />
      <div className="flex flex-col gap-3">
        <Field label="Dataset name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. rna-3d-train" /></Field>
        <input ref={fileRef} type="file" multiple hidden accept=".csv,.tsv,.txt,.fasta,.fa,.fna,.pdb,.cif,.json" onChange={(e) => setFiles([...(e.target.files ?? [])])} />
        <button type="button" onClick={() => fileRef.current?.click()} className="rounded-lg border border-dashed border-line-strong px-4 py-5 text-center hover:bg-surface-2">
          <FileUp className="size-5 mx-auto text-muted" />
          <div className="text-[13px] font-medium mt-1.5">{files.length ? `${files.length} file(s) selected` : "Choose files"}</div>
          <div className="text-[12px] text-faint">{files.length ? files.map((f) => f.name).join(", ") : "Stanford RNA 3D Folding files work directly"}</div>
        </button>
        <Button variant="primary" disabled={!name.trim() || !files.length} loading={up.isPending} onClick={() => up.mutate()}>Upload dataset</Button>
      </div>
    </Card>
  );
}

function RowsTable({ slug, file }: { slug: string; file: string }) {
  const [offset, setOffset] = useState(0);
  const [q, setQ] = useState("");
  const [query, setQuery] = useState("");
  const limit = 25;
  const rows = useQuery({ queryKey: ["rows", slug, file, offset, query], queryFn: () => api.get<Rows>(`/api/datasets/${slug}/rows?file=${encodeURIComponent(file)}&offset=${offset}&limit=${limit}&q=${encodeURIComponent(query)}`) });
  const d = rows.data;
  return (
    <div>
      <form className="flex items-center gap-2 mb-3" onSubmit={(e) => { e.preventDefault(); setOffset(0); setQuery(q); }}>
        <div className="relative flex-1 max-w-sm"><Search className="size-4 text-faint absolute left-3 top-1/2 -translate-y-1/2" /><Input className="pl-9" placeholder="Filter rows" value={q} onChange={(e) => setQ(e.target.value)} /></div>
        <span className="text-[12.5px] text-muted ml-auto num">{d ? `${d.total ? offset + 1 : 0}–${Math.min(offset + limit, d.total)} of ${d.total.toLocaleString()}` : ""}</span>
        <Button size="sm" variant="ghost" icon={<ChevronLeft className="size-4" />} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))} aria-label="Previous" />
        <Button size="sm" variant="ghost" icon={<ChevronRight className="size-4" />} disabled={!d || offset + limit >= d.total} onClick={() => setOffset(offset + limit)} aria-label="Next" />
      </form>
      {rows.isLoading ? <Spinner label="Loading rows…" /> : d && (
        <div className="overflow-auto max-h-[440px] rounded-lg border border-line">
          <table className="text-[12px] w-full">
            <thead className="sticky top-0 bg-surface-2"><tr>{d.columns.map((c) => <th key={c} className="text-left font-medium text-muted px-3 py-2 whitespace-nowrap border-b border-line">{c}</th>)}</tr></thead>
            <tbody className="divide-y divide-line">
              {d.rows.map((r, i) => (
                <tr key={i} className="hover:bg-surface-2">
                  {r.map((v, j) => <td key={j} className="px-3 py-1.5 font-mono whitespace-nowrap max-w-[320px] truncate" title={String(v ?? "")}>{v === null ? <span className="text-faint">null</span> : String(v)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function DatasetView({ slug, onDeleted }: { slug: string; onDeleted: () => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const c = useChartColors();
  const d = useQuery({ queryKey: ["dataset", slug], queryFn: () => api.get<DatasetDetail>(`/api/datasets/${slug}`) });
  const rna = useQuery({ queryKey: ["dataset-rna", slug], queryFn: () => api.get<Rna>(`/api/datasets/${slug}/rna`) });
  const [file, setFile] = useState<string | null>(null);
  const del = useMutation({ mutationFn: () => api.del(`/api/datasets/${slug}`), onSuccess: () => { qc.invalidateQueries({ queryKey: ["datasets"] }); onDeleted(); } });
  const importSeqs = useMutation({
    mutationFn: (f: string) => api.post<{ created: number; duplicates: number }>(`/api/datasets/${slug}/import-sequences`, { file: f, limit: 200 }),
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["rna-seqs"] }); toast(`${r.created} sequences added to the workspace${r.duplicates ? ` (${r.duplicates} already stored)` : ""}`); },
    onError: (e: Error) => toast(e.message, "error"),
  });
  const tables = d.data?.files.filter((f) => f.rows !== undefined) ?? [];
  const active = file ?? tables[0]?.name ?? null;
  const activeInfo = d.data?.files.find((f) => f.name === active);
  const r0 = rna.data?.sequence_files[0];
  const hist = useMemo(() => r0 ? r0.length_histogram.counts.map((n, i) => ({ bin: `${Math.round(r0.length_histogram.edges[i])}`, n })) : [], [r0]);

  if (d.isLoading || !d.data) return <Card><Spinner label="Reading dataset…" /></Card>;
  return (
    <div className="flex flex-col gap-5 min-w-0">
      <Card>
        <CardHeader title={d.data.title} subtitle={`${d.data.source} · added ${date(d.data.created_at)}`}
          actions={<Button size="sm" variant="danger" icon={<Trash2 className="size-4" />} onClick={() => { if (confirm(`Delete dataset ${d.data!.title}?`)) del.mutate(); }}>Delete</Button>} />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Files" value={d.data.files.length} />
          <Stat label="Tables" value={tables.length} />
          <Stat label="3D templates" value={d.data.n_templates} hint="from structure-label files" />
          <Stat label="Size" value={bytes(d.data.files.reduce((a, f) => a + f.bytes, 0))} />
        </div>
      </Card>

      {r0 && (
        <div className="grid lg:grid-cols-[1.4fr_1fr] gap-5">
          <ChartCard title="Sequence length distribution" subtitle={`${r0.file} · ${r0.n_sequences.toLocaleString()} sequences · median ${fmt(r0.length.median, 0)} nt (range ${r0.length.min}–${r0.length.max})`}>
            <Bars data={hist} x="bin" y="n" height={220} xLabel="length (nt, bin start)" />
          </ChartCard>
          <ChartCard title="Composition" subtitle={`GC ${fmt(r0.gc_percent.mean, 1)} ± ${fmt(r0.gc_percent.std, 1)}%`}>
            <Bars data={Object.entries(r0.composition).map(([b, n]) => ({ b, n }))} x="b" y="n" height={220}
              colorFn={(row) => ({ A: c.v1, C: c.v4, G: c.v2, U: c.v3 } as Record<string, string>)[String(row.b)] ?? c.v5} />
          </ChartCard>
          {r0.by_year && (
            <ChartCard title="Structures by temporal cutoff" subtitle="Year of the temporal_cutoff column">
              <Bars data={Object.entries(r0.by_year).map(([y, n]) => ({ y, n }))} x="y" y="n" height={200} color={c.v5} />
            </ChartCard>
          )}
        </div>
      )}

      <Card>
        <CardHeader title="Files" subtitle="Select a table to inspect its schema and rows." />
        <div className="flex flex-wrap gap-2 mb-4">
          {d.data.files.map((f) => (
            <button key={f.name} onClick={() => f.rows !== undefined && setFile(f.name)} disabled={f.rows === undefined}
              className={cx("rounded-lg border px-3 py-2 text-left", active === f.name ? "border-brand bg-brand-soft/60" : "border-line hover:bg-surface-2", f.rows === undefined && "opacity-60 cursor-default")}>
              <div className="text-[13px] font-medium flex items-center gap-1.5"><Table2 className="size-3.5 text-muted" />{f.name}</div>
              <div className="text-[12px] text-muted">{f.rows !== undefined ? `${f.rows.toLocaleString()} rows · ` : ""}{bytes(f.bytes)}{f.kind && f.kind !== "table" && <> · <span className="text-brand-text">{f.kind.replace("_", " ")}</span></>}</div>
            </button>
          ))}
        </div>
        {activeInfo?.error && <Callout tone="danger">{activeInfo.error}</Callout>}
        {active && activeInfo?.columns && (
          <div className="flex flex-col gap-5">
            {activeInfo.kind === "rna_sequences" && (
              <div><Button icon={<TestTubes className="size-4" />} loading={importSeqs.isPending} onClick={() => importSeqs.mutate(active)}>Import first 200 sequences into workspace</Button></div>
            )}
            <details open>
              <summary className="cursor-pointer text-[13px] font-semibold mb-2">Schema ({activeInfo.columns.length} columns)</summary>
              <Table head={["Column", "Type", "Missing", "Unique", "Min", "Mean", "Max"]}>
                {activeInfo.columns.map((col) => (
                  <tr key={col.name}>
                    <td className="px-3 py-2 font-mono text-[12px]">{col.name}</td><td className="px-3 py-2 text-muted">{col.dtype}</td>
                    <td className="px-3 py-2 num">{col.missing.toLocaleString()}</td><td className="px-3 py-2 num">{col.unique.toLocaleString()}</td>
                    <td className="px-3 py-2 num font-mono text-[12px]">{fmt(col.min, 2)}</td><td className="px-3 py-2 num font-mono text-[12px]">{fmt(col.mean, 2)}</td><td className="px-3 py-2 num font-mono text-[12px]">{fmt(col.max, 2)}</td>
                  </tr>
                ))}
              </Table>
            </details>
            <RowsTable key={active} slug={slug} file={active} />
          </div>
        )}
      </Card>
    </div>
  );
}

export default function DatasetExplorer() {
  const list = useQuery({ queryKey: ["datasets"], queryFn: () => api.get<{ items: DatasetMeta[] }>("/api/datasets") });
  const [sel, setSel] = useState<string | null>(null);
  const active = sel ?? list.data?.items[0]?.slug ?? null;
  return (
    <div>
      <PageHeader eyebrow="Data & models" title="Dataset explorer"
        description="Load RNA datasets from Kaggle or upload files, inspect schemas and distributions, import sequences into the workspace, and build the template library used for 3D prediction." />
      <div className="grid lg:grid-cols-2 gap-5 mb-8"><KagglePanel /><UploadPanel /></div>
      <h2 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-faint mb-3">Loaded datasets</h2>
      {!list.data?.items.length ? (
        <Card><Empty icon={<Database className="size-5" />} title="No datasets loaded">Import the Stanford RNA 3D Folding competition from Kaggle or upload CSV files above.</Empty></Card>
      ) : (
        <div className="grid xl:grid-cols-[260px_1fr] gap-5 items-start">
          <Card padded={false}>
            <ul className="divide-y divide-line">
              {list.data.items.map((m) => (
                <li key={m.slug}>
                  <button onClick={() => setSel(m.slug)} className={cx("w-full text-left px-4 py-3 hover:bg-surface-2", active === m.slug && "bg-brand-soft/60")}>
                    <div className="text-[13px] font-medium truncate">{m.title}</div>
                    <div className="text-[12px] text-muted">{m.files.length} files · {m.source}</div>
                  </button>
                </li>
              ))}
            </ul>
          </Card>
          {active && <DatasetView key={active} slug={active} onDeleted={() => setSel(null)} />}
        </div>
      )}
    </div>
  );
}
