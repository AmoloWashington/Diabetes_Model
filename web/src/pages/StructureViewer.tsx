import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Box, Copy, Download, FileUp, GitCompare, Globe, Trash2 } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ConfidenceLegend, MolViewer, type ColorBy, type StyleKind } from "@/components/rna/MolViewer";
import { Badge, Button, Card, CardHeader, Empty, Field, Input, PageHeader, Select, Spinner, Stat, cx, useToast } from "@/components/ui";
import { api, type StructureFull, type StructureSummary } from "@/lib/api";
import { date, fmt } from "@/lib/format";
import { dotBracketPairs } from "@/lib/rna";

interface Compare { reference_length: number; model_length: number; aligned: number; sequence_identity: number; tm_score: number; rmsd_A: number; tm_score_definition: string }

export default function StructureViewer() {
  const qc = useQueryClient();
  const toast = useToast();
  const [params, setParams] = useSearchParams();
  const selId = params.get("id") ? Number(params.get("id")) : null;
  const [pdbId, setPdbId] = useState("1EHZ");
  const [colorBy, setColorBy] = useState<ColorBy>("confidence");
  const [style, setStyle] = useState<StyleKind>("auto");
  const [ref, setRef] = useState<number | "">("");
  const fileRef = useRef<HTMLInputElement>(null);

  const list = useQuery({ queryKey: ["structures"], queryFn: () => api.get<{ items: StructureSummary[] }>("/api/rna/structures") });
  const sel = useQuery({ queryKey: ["structure", selId], queryFn: () => api.get<StructureFull>(`/api/rna/structures/${selId}`), enabled: selId != null });
  const snippet = useQuery({ queryKey: ["py3dmol", selId], queryFn: () => api.get<{ python: string }>(`/api/rna/structures/${selId}/py3dmol`), enabled: selId != null });
  const select = (id: number) => setParams({ id: String(id) });
  const refresh = () => qc.invalidateQueries({ queryKey: ["structures"] });

  const fetchPdb = useMutation({
    mutationFn: () => api.post<StructureFull>("/api/rna/structures/fetch", { pdb_id: pdbId.trim() }),
    onSuccess: (s) => { refresh(); select(s.id); setColorBy("nucleotide"); toast(`Loaded ${s.name} from RCSB PDB`); },
    onError: (e: Error) => toast(e.message, "error"),
  });
  const upload = useMutation({
    mutationFn: (f: File) => { const fd = new FormData(); fd.append("file", f); return api.upload<StructureFull>("/api/rna/structures/upload", fd); },
    onSuccess: (s) => { refresh(); select(s.id); toast(`Uploaded ${s.name}`); },
    onError: (e: Error) => toast(e.message, "error"),
  });
  const del = useMutation({
    mutationFn: (id: number) => api.del(`/api/rna/structures/${id}`),
    onSuccess: () => { refresh(); setParams({}); },
  });
  const compare = useMutation({
    mutationFn: () => api.post<Compare>("/api/rna/structures/compare", { reference_id: ref, model_id: selId }),
    onError: (e: Error) => toast(e.message, "error"),
  });

  const s = sel.data;
  const isPredicted = s?.kind === "predicted";
  const ss = s?.meta?.secondary_structure as string | undefined;
  const pairs = useMemo(() => dotBracketPairs(ss), [ss]);
  const chains = (s?.meta?.rna_chains ?? {}) as Record<string, { length: number; sequence: string }>;

  return (
    <div>
      <PageHeader eyebrow="RNA structure" title="Structure viewer"
        description="Interactive 3D viewer (3Dmol.js, the engine behind py3Dmol) for predicted models, uploaded coordinates and entries from the RCSB Protein Data Bank." />
      <div className="grid xl:grid-cols-[300px_1fr] gap-5 items-start">
        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader title="Load a structure" />
            <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); fetchPdb.mutate(); }}>
              <Input value={pdbId} onChange={(e) => setPdbId(e.target.value.toUpperCase())} maxLength={4} placeholder="PDB ID" className="font-mono uppercase" aria-label="PDB ID" />
              <Button type="submit" icon={<Globe className="size-4" />} loading={fetchPdb.isPending}>RCSB</Button>
            </form>
            <p className="text-[12px] text-faint mt-1.5">e.g. 1EHZ (yeast tRNA<sup>Phe</sup>). Requires internet access.</p>
            <input ref={fileRef} type="file" accept=".pdb,.ent,.cif,.mmcif" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) upload.mutate(f); e.target.value = ""; }} />
            <Button className="w-full mt-3" icon={<FileUp className="size-4" />} loading={upload.isPending} onClick={() => fileRef.current?.click()}>Upload PDB / mmCIF</Button>
          </Card>
          <Card padded={false}>
            <div className="px-4 py-3 border-b border-line text-[13px] font-semibold">Library <span className="text-muted font-normal">({list.data?.items.length ?? 0})</span></div>
            <ul className="max-h-[520px] overflow-y-auto divide-y divide-line">
              {list.data?.items.map((it) => (
                <li key={it.id}>
                  <button onClick={() => select(it.id)} className={cx("w-full text-left px-4 py-2.5 hover:bg-surface-2", selId === it.id && "bg-brand-soft/60")}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[13px] font-medium truncate">{it.name}</span>
                      <Badge tone={it.kind === "predicted" ? "accent" : "brand"}>{it.kind}</Badge>
                    </div>
                    <div className="text-[12px] text-muted mt-0.5">{it.mean_confidence != null ? `confidence ${fmt(100 * it.mean_confidence, 0)} · ` : ""}{date(it.created_at)}</div>
                  </button>
                </li>
              ))}
              {!list.data?.items.length && <li className="px-4 py-6 text-[13px] text-muted">No structures yet. Predict one or load from the PDB.</li>}
            </ul>
          </Card>
        </div>

        <div className="flex flex-col gap-5 min-w-0">
          {!selId ? (
            <Card><Empty icon={<Box className="size-5" />} title="Select a structure">Choose one from the library, fetch a PDB entry, or upload coordinates.</Empty></Card>
          ) : sel.isLoading || !s ? (
            <Card><Spinner label="Loading structure…" /></Card>
          ) : (
            <>
              <Card padded={false}>
                <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3.5 border-b border-line">
                  <div className="min-w-0"><h3 className="text-[14px] font-semibold truncate">{s.name}</h3><p className="text-[12px] text-muted">{s.method}</p></div>
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="w-48"><Select value={colorBy} onChange={(e) => setColorBy(e.target.value as ColorBy)} className="h-8 text-[13px]" aria-label="Colour by">
                      <option value="confidence">Colour: confidence</option><option value="nucleotide">Colour: nucleotide</option>
                      <option value="chain">Colour: chain</option><option value="spectrum">Colour: 5′→3′ spectrum</option>
                    </Select></div>
                    <div className="w-36"><Select value={style} onChange={(e) => setStyle(e.target.value as StyleKind)} className="h-8 text-[13px]" aria-label="Style">
                      <option value="auto">Style: auto</option><option value="cartoon">Cartoon</option><option value="trace">Trace</option><option value="stick">Sticks</option><option value="sphere">Spheres</option><option value="surface">Molecular surface</option>
                    </Select></div>
                    <Button size="sm" icon={<Download className="size-4" />} onClick={() => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([s.pdb])); a.download = `glucolab_structure_${s.id}.${s.meta?.format === "mmcif" ? "cif" : "pdb"}`; a.click(); }}>Download</Button>
                    <Button size="sm" variant="danger" icon={<Trash2 className="size-4" />} onClick={() => { if (confirm("Delete this structure?")) del.mutate(s.id); }} />
                  </div>
                </div>
                <MolViewer data={s.pdb} format={s.meta?.format === "mmcif" ? "cif" : "pdb"} colorBy={colorBy} style={style} height={520} pairs={pairs} />
                {colorBy === "confidence" && <div className="px-5 py-3 border-t border-line"><ConfidenceLegend />{!isPredicted && <p className="text-[12px] text-faint mt-1">For experimental structures the B-factor column holds atomic displacement, not confidence; prefer nucleotide colouring.</p>}</div>}
              </Card>
              <div className="grid lg:grid-cols-2 gap-5">
                <Card>
                  <CardHeader title="Compare with a reference" subtitle="Sequence-aligned TM-score (US-align RNA d0) and RMSD after optimal superposition." />
                  <div className="flex gap-2 items-end">
                    <Field label="Reference structure" className="flex-1">
                      <Select value={ref} onChange={(e) => setRef(e.target.value ? Number(e.target.value) : "")}>
                        <option value="">Select…</option>
                        {list.data?.items.filter((x) => x.id !== s.id).map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
                      </Select>
                    </Field>
                    <Button icon={<GitCompare className="size-4" />} disabled={ref === ""} loading={compare.isPending} onClick={() => compare.mutate()}>Compare</Button>
                  </div>
                  {compare.data && (
                    <div className="grid grid-cols-2 gap-3 mt-4">
                      <Stat label="TM-score" value={fmt(compare.data.tm_score, 3)} hint="0–1, normalised by reference length" />
                      <Stat label="RMSD" value={fmt(compare.data.rmsd_A, 2)} unit="Å" hint={`${compare.data.aligned} aligned C1' atoms`} />
                    </div>
                  )}
                </Card>
                <Card>
                  <CardHeader title="Use in Jupyter (py3Dmol)" actions={snippet.data && <Button size="sm" icon={<Copy className="size-4" />} onClick={() => { navigator.clipboard?.writeText(snippet.data!.python); toast("Copied"); }}>Copy</Button>} />
                  <pre className="font-mono text-[11.5px] leading-5 bg-surface-2 border border-line rounded-lg p-3 overflow-x-auto">{snippet.data?.python ?? "…"}</pre>
                </Card>
              </div>
              {Object.keys(chains).length > 0 && (
                <Card>
                  <CardHeader title="RNA chains (C1' atoms)" />
                  <div className="flex flex-col gap-2">
                    {Object.entries(chains).map(([c, v]) => (
                      <div key={c} className="text-[12.5px]"><Badge>Chain {c}</Badge> <span className="text-muted ml-1">{v.length} nt</span>
                        <div className="font-mono text-[11.5px] text-ink-2 break-all mt-1">{v.sequence}</div></div>
                    ))}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
