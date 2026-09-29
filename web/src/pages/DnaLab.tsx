import { useMutation, useQuery } from "@tanstack/react-query";
import { Circle, Dna, Sparkles, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge, Button, Callout, Card, CardHeader, PageHeader, Stat, Textarea, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";
import { useHelix } from "@/lib/useHelix";
import { useRecorder } from "@/lib/useRecorder";

interface Analysis {
  length_nt: number; counts: Record<string, number>; gc_percent: number | null; reverse_complement: string;
  translations: Record<string, string>; orfs: { frame: number; start_nt: number; end_nt: number; length_aa: number; protein: string }[];
  melting_temperature: { tm_c: number; method: string }; contains_insulin_b_chain: boolean;
  helix_geometry: { turns: number; length_nm: number };
}

export default function DnaLab() {
  const toast = useToast();
  const demo = useQuery({ queryKey: ["dna-demo"], queryFn: () => api.get<{ sequence: string; note: string }>("/api/dna/demo") });
  const [seq, setSeq] = useState("");
  const [shown, setShown] = useState<string | undefined>();
  const { ref, helix, error } = useHelix(shown, { interactive: true, maxBp: 150 });
  const rec = useRecorder(() => helix.current?.canvas, "glucolab-dna");
  const analyze = useMutation({
    mutationFn: (s: string) => api.post<Analysis>("/api/dna/analyze", { sequence: s }),
    onSuccess: (_, s) => setShown(s),
    onError: (e: Error) => toast(e.message, "error"),
  });
  useEffect(() => {
    if (demo.data && !seq) { setSeq(demo.data.sequence); analyze.mutate(demo.data.sequence); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo.data]);
  const a = analyze.data;

  return (
    <div>
      <PageHeader eyebrow="Cell & molecular" title="DNA lab"
        description="Paste a DNA sequence to build a B-DNA helix base by base and translate it in three frames with the standard genetic code (NCBI table 1)."
        actions={<>
          <Button icon={<Sparkles className="size-4" />} onClick={() => helix.current?.startBubble()}>Transcription bubble</Button>
          <Button variant={rec.recording ? "danger" : "secondary"} icon={rec.recording ? <Square className="size-4" /> : <Circle className="size-4 fill-current text-danger" />}
            onClick={() => { const e = rec.toggle(); if (e) toast(e, "error"); }}>{rec.recording ? "Stop & save" : "Record video"}</Button>
        </>}
      />
      <div className="grid xl:grid-cols-[380px_1fr] gap-5 items-start">
        <Card>
          <CardHeader title="Sequence" subtitle="A, C, G, T, N; U is read as T" />
          <Textarea rows={7} className="font-mono text-[12px] break-all" value={seq} onChange={(e) => setSeq(e.target.value)} spellCheck={false} />
          {demo.data && <p className="text-[12px] text-faint mt-2">{demo.data.note}</p>}
          <div className="flex gap-2 mt-3">
            <Button variant="primary" icon={<Dna className="size-4" />} loading={analyze.isPending} onClick={() => analyze.mutate(seq)}>Analyse and build</Button>
            <Button onClick={() => { if (demo.data) { setSeq(demo.data.sequence); analyze.mutate(demo.data.sequence); } }}>Insulin demo</Button>
          </div>
          {a && (
            <div className="grid grid-cols-2 gap-3 mt-5">
              <Stat label="Length" value={a.length_nt} unit="nt" hint={`${fmt(a.helix_geometry.turns, 1)} turns · ${fmt(a.helix_geometry.length_nm, 1)} nm`} />
              <Stat label="GC content" value={fmt(a.gc_percent, 1)} unit="%" />
              <Stat label="Melting temperature" value={fmt(a.melting_temperature.tm_c, 1)} unit="°C" hint={a.melting_temperature.method} />
              <Stat label="Longest ORF" value={a.orfs[0] ? a.orfs[0].length_aa : "–"} unit={a.orfs[0] ? "aa" : undefined} hint={a.orfs[0] ? `frame ${a.orfs[0].frame}, nt ${a.orfs[0].start_nt}–${a.orfs[0].end_nt}` : "none ≥ 5 aa"} />
            </div>
          )}
        </Card>
        <div className="flex flex-col gap-5 min-w-0">
          <div className="rounded-xl border border-line bg-gradient-to-b from-surface to-surface-2 shadow-card overflow-hidden">
            {error ? <div className="p-6"><Callout tone="warn">{error}</Callout></div> : <div ref={ref} className="h-[440px]" />}
            <div className="flex flex-wrap items-center gap-3 px-4 py-2.5 border-t border-line text-[12px] text-muted">
              {[["A", "#3e9b74"], ["T", "#c4553f"], ["G", "#d29a3c"], ["C", "#5a7fb0"]].map(([b, col]) => (
                <span key={b} className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm" style={{ background: col }} />{b}</span>
              ))}
              <span className="ml-auto">Drag to rotate · scroll to zoom · schematic groove geometry</span>
            </div>
          </div>
          {a && (
            <Card>
              <CardHeader title="Translation" actions={a.contains_insulin_b_chain ? <Badge tone="brand">Encodes human insulin B chain</Badge> : undefined} />
              <div className="flex flex-col gap-3 font-mono text-[12px] break-all">
                {Object.entries(a.translations).map(([k, v]) => (
                  <div key={k}><div className="text-muted mb-0.5">{k.replace("_", " ")}</div>
                    <div>{v.split("").map((ch, i) => <span key={i} className={ch === "*" ? "text-danger font-semibold" : ch === "M" ? "text-brand-text font-semibold" : "text-ink-2"}>{ch}</span>)}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
