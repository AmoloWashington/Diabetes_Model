import { useMutation, useQuery } from "@tanstack/react-query";
import { Scissors, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { AskAI } from "@/components/AskAI";
import { MethodsPanel } from "@/components/Methods";
import { Badge, Button, Callout, Card, CardHeader, Field, PageHeader, Slider, Stat, Table, Textarea, cx, useToast } from "@/components/ui";
import { api } from "@/lib/api";
import { fmt } from "@/lib/format";

interface Guide {
  strand: "+" | "-";
  spacer: string;
  pam: string;
  full_target: string;
  gc_percent: number;
  spacer_start: number;
  spacer_end: number;
  pam_start: number;
  pam_end: number;
  cut_site: number;
  flags: string[];
  good: boolean;
}
interface Result {
  length: number; n_guides: number; n_good: number; pam: string; spacer_length: number;
  gc_window: [number, number]; guides: Guide[]; note: string;
}

export default function CrisprFinder() {
  const toast = useToast();
  const [seq, setSeq] = useState("");
  const [gc, setGc] = useState<[number, number]>([40, 60]);
  const [onlyGood, setOnlyGood] = useState(false);

  const demo = useQuery({ queryKey: ["dna-demo"], queryFn: () => api.get<{ sequence: string; note: string }>("/api/dna/demo") });
  const find = useMutation({
    mutationFn: (s: string) => api.post<Result>("/api/crispr/guides", { sequence: s, gc_low: gc[0], gc_high: gc[1] }),
    onError: (e: Error) => toast(e.message, "error"),
  });
  const r = find.data;
  const shown = useMemo(() => (r ? r.guides.filter((g) => !onlyGood || g.good) : []), [r, onlyGood]);

  const summary = r ? `CRISPR-Cas9 (SpCas9, NGG PAM) scan of a ${r.length} nt sequence: ${r.n_guides} candidate guide sites, ${r.n_good} with no design flags. Flags used: GC outside ${gc[0]}–${gc[1]}%, poly-T (TTTT), and non-uniqueness within this sequence (not a genome-wide off-target search).` : "";

  return (
    <div>
      <PageHeader eyebrow="Genetics" title="CRISPR guide finder"
        description="Paste a DNA sequence and find where the standard CRISPR enzyme (SpCas9) can be targeted. Each site shows the 20-letter guide, its PAM, where the DNA would be cut, and common design warnings — with the rules explained. This is an educational tool, not for planning real gene editing."
        actions={<AskAI page="CRISPR guide finder" summary={summary} question="Explain how CRISPR-Cas9 finds and cuts its target, and how to read these guide results." />} />

      <div className="grid xl:grid-cols-[340px_1fr] gap-5 items-start">
        <Card className="xl:sticky xl:top-8">
          <CardHeader title="Sequence" subtitle="A, C, G, T, N; U is read as T" />
          <div className="flex flex-col gap-4">
            <Textarea rows={6} className="font-mono text-[12px] break-all" spellCheck={false} value={seq} onChange={(e) => setSeq(e.target.value)}
              placeholder="Paste a DNA sequence (at least 23 letters)…" />
            <Field label="Preferred GC window for the guide">
              <div className="pt-1">
                <Slider label={`${gc[0]}% – ${gc[1]}%`} value={gc[0]} min={0} max={gc[1] - 5} step={5} onChange={(v) => setGc([v, gc[1]])} format={(v) => `low ${v}%`} />
                <div className="h-2" />
                <Slider label="" value={gc[1]} min={gc[0] + 5} max={100} step={5} onChange={(v) => setGc([gc[0], v])} format={(v) => `high ${v}%`} />
              </div>
            </Field>
            <Button variant="primary" icon={<Search className="size-4" />} loading={find.isPending} disabled={!seq.trim()} onClick={() => find.mutate(seq)}>Find guides</Button>
            <Button variant="secondary" icon={<Scissors className="size-4" />} onClick={() => { if (demo.data) { setSeq(demo.data.sequence); find.mutate(demo.data.sequence); } }}>Insulin B-chain demo</Button>
            {demo.data && <p className="text-[12px] text-faint">{demo.data.note}</p>}
          </div>
        </Card>

        <div className="flex flex-col gap-5 min-w-0">
          <Callout tone="warn" title="Education only">
            This finds candidate target sites and flags common design problems within the sequence you paste. It does not predict off-target effects anywhere in a genome, does not design therapies, and must not be used to plan editing of any living organism. Human germline (heritable) editing is banned in most countries.
          </Callout>

          {r && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Stat label="Sequence length" value={r.length} unit="nt" />
                <Stat label="Candidate guides" value={r.n_guides} hint="both strands" />
                <Stat label="No flags" value={r.n_good} tone={r.n_good > 0 ? "ok" : "warn"} hint="pass the rules of thumb" />
                <Stat label="PAM" value={r.pam} hint="SpCas9" />
              </div>

              <Card>
                <SequenceMap seq={seq} guides={r.guides} />
              </Card>

              <Card padded={false}>
                <div className="px-5 pt-5 flex flex-wrap items-center justify-between gap-2">
                  <CardHeader title="Guide sites" subtitle={`${shown.length} shown · sorted by cut position`} />
                  <label className="flex items-center gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={onlyGood} onChange={(e) => setOnlyGood(e.target.checked)} />Only guides with no flags</label>
                </div>
                <Table head={["Strand", "Guide (spacer, 5′→3′)", "PAM", "GC", "Cut @", "Notes"]}>
                  {shown.map((g, i) => (
                    <tr key={i} className="align-top">
                      <td className="px-3 py-2.5"><Badge tone={g.strand === "+" ? "brand" : "accent"}>{g.strand}</Badge></td>
                      <td className="px-3 py-2.5 font-mono text-[12px] break-all">{g.spacer}</td>
                      <td className="px-3 py-2.5 font-mono text-[12px] text-muted">{g.pam}</td>
                      <td className={cx("px-3 py-2.5 num", g.gc_percent < r.gc_window[0] || g.gc_percent > r.gc_window[1] ? "text-warn" : "text-ink-2")}>{fmt(g.gc_percent, 0)}%</td>
                      <td className="px-3 py-2.5 num text-ink-2">{g.cut_site}</td>
                      <td className="px-3 py-2.5">
                        {g.flags.length === 0
                          ? <span className="text-ok text-[12.5px]">No flags</span>
                          : <div className="flex flex-col gap-1">{g.flags.map((f, j) => <span key={j} className="text-[12px] text-warn">{f}</span>)}</div>}
                      </td>
                    </tr>
                  ))}
                  {shown.length === 0 && <tr><td colSpan={6} className="px-3 py-6 text-center text-muted text-[13px]">No guides to show.</td></tr>}
                </Table>
              </Card>
            </>
          )}

          <Callout>
            <b>How to read a guide.</b> SpCas9 is directed by a 20-letter guide RNA that matches the DNA just before a short "PAM" signal (the letters NGG). The enzyme makes a clean cut in the DNA 3 letters before the PAM. The cell then repairs the break, which is how a gene can be switched off or changed. The warnings here (GC content, poly-T runs, repeats) are simple design rules; real guide design also needs a genome-wide off-target search, which this tool does not do.
          </Callout>
          <MethodsPanel id="crispr" />
        </div>
      </div>
    </div>
  );
}

/** A compact ruler showing where each guide's cut falls along the sequence. */
function SequenceMap({ seq, guides }: { seq: string; guides: Guide[] }) {
  const n = seq.length || 1;
  return (
    <div>
      <div className="text-[12.5px] font-semibold text-ink-2 mb-2">Cut sites along the sequence</div>
      <div className="relative h-16 rounded-md bg-surface-2 border border-line overflow-hidden">
        {guides.map((g, i) => (
          <span key={i} title={`${g.strand} strand, cut at ${g.cut_site}${g.flags.length ? ` — ${g.flags.join("; ")}` : ""}`}
            className={cx("absolute w-px", g.good ? "bg-brand" : "bg-warn/70")}
            style={{ left: `${(g.cut_site / n) * 100}%`, top: g.strand === "+" ? "4px" : "50%", height: "46%" }} />
        ))}
        <div className="absolute inset-x-0 top-1/2 h-px bg-line-strong" />
      </div>
      <div className="flex justify-between text-[11px] text-faint mt-1">
        <span>1</span><span>+ strand above · − strand below · green = no flags</span><span>{n}</span>
      </div>
    </div>
  );
}
