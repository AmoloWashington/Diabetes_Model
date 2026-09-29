import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowUp, BrainCircuit, Wrench } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge, Callout, PageHeader, cx } from "@/components/ui";
import { api, type Health } from "@/lib/api";

type Msg = { role: "user" | "assistant"; content: string };
interface Reply { text: string; model: string; usage: { input_tokens: number; output_tokens: number }; tool_calls: { name: string; input: unknown; is_error: boolean; output_preview: string }[] }
type Entry = Msg & { reply?: Reply; error?: string };

const SUGGESTIONS = [
  "Simulate a 75 g meal in the healthy and type 2 phenotypes and explain the mechanistic differences.",
  "Fold GGGCGCAAGCCUAUGCGCUUCGGCGCAUAGGCUUGCGCCC and explain how confident the structure is.",
  "Why does a rapid fall in insulin sensitivity cause β-cell failure in the Topp model while a slow one does not?",
  "How reliable is the symptom risk model, and what do the duplicate records do to naive accuracy?",
];

export default function Assistant() {
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get<Health>("/api/health") });
  const [log, setLog] = useState<Entry[]>([]);
  const [input, setInput] = useState("");
  const end = useRef<HTMLDivElement>(null);
  const send = useMutation({
    mutationFn: (history: Msg[]) => api.post<Reply>("/api/ai/chat", { messages: history.slice(-40) }),
    onSuccess: (r) => setLog((l) => [...l, { role: "assistant", content: r.text, reply: r }]),
    onError: (e: Error) => setLog((l) => [...l.slice(0, -1), { ...l[l.length - 1], error: e.message }]),
  });
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [log, send.isPending]);

  const submit = (text: string) => {
    const t = text.trim();
    if (!t || send.isPending) return;
    const history: Msg[] = [...log.filter((e) => !e.error).map(({ role, content }) => ({ role, content })), { role: "user", content: t }];
    setLog((l) => [...l.filter((e) => !e.error), { role: "user", content: t }]);
    setInput("");
    send.mutate(history);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] min-h-[560px]">
      <PageHeader eyebrow="AI" title="Research assistant"
        description="Claude answers by calling GlucoLab's validated engines as tools. Every call is listed under the answer so each number can be audited."
        actions={health.data?.ai_configured ? <Badge tone="brand">{health.data.ai_model}</Badge> : undefined} />
      {health.data && !health.data.ai_configured && (
        <div className="mb-4"><Callout tone="warn" title="AI is not configured on this server">
          Add <code className="font-mono">ANTHROPIC_API_KEY</code> to the git-ignored <code className="font-mono">.env</code> file in the repository root and restart the backend. Every other feature works without it.
        </Callout></div>
      )}
      <div className="flex-1 overflow-y-auto rounded-xl border border-line bg-surface shadow-card">
        <div className="max-w-3xl mx-auto px-5 py-6 flex flex-col gap-6">
          {log.length === 0 && (
            <div className="text-center py-8">
              <div className="size-11 rounded-xl bg-brand-soft text-brand-text grid place-items-center mx-auto"><BrainCircuit className="size-5" /></div>
              <h3 className="mt-3 font-semibold">Ask about physiology, RNA structure or the models</h3>
              <p className="text-[13px] text-muted mt-1">The assistant runs simulations, folds RNA, predicts 3D structures and explains the results.</p>
              <div className="grid sm:grid-cols-2 gap-2 mt-6 text-left">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => submit(s)} className="rounded-lg border border-line p-3 text-[13px] text-ink-2 hover:bg-surface-2 hover:border-line-strong">{s}</button>
                ))}
              </div>
            </div>
          )}
          {log.map((e, i) => e.role === "user" ? (
            <div key={i} className="self-end max-w-[85%]">
              <div className="rounded-2xl rounded-br-md bg-brand-soft px-4 py-2.5 text-[14px] whitespace-pre-wrap">{e.content}</div>
              {e.error && <div className="text-[12.5px] text-danger mt-1.5 text-right">{e.error}</div>}
            </div>
          ) : (
            <div key={i} className="flex gap-3">
              <div className="size-7 shrink-0 rounded-lg bg-brand text-white dark:text-[#0e1013] grid place-items-center"><BrainCircuit className="size-4" /></div>
              <div className="min-w-0 flex-1">
                <div className="prose-ai text-[14px] leading-6 text-ink"><ReactMarkdown remarkPlugins={[remarkGfm]}>{e.content}</ReactMarkdown></div>
                {e.reply && e.reply.tool_calls.length > 0 && (
                  <details className="mt-3 rounded-lg border border-line bg-surface-2">
                    <summary className="cursor-pointer px-3 py-2 text-[12.5px] text-muted flex flex-wrap items-center gap-1.5">
                      <Wrench className="size-3.5" /> {e.reply.tool_calls.length} tool call{e.reply.tool_calls.length > 1 ? "s" : ""}:
                      {e.reply.tool_calls.map((t, j) => <Badge key={j} tone={t.is_error ? "danger" : "neutral"} className="font-mono">{t.name}</Badge>)}
                    </summary>
                    <div className="px-3 pb-3 flex flex-col gap-3">
                      {e.reply.tool_calls.map((t, j) => (
                        <div key={j} className="text-[12px]">
                          <div className="font-medium font-mono">{t.name}</div>
                          <pre className="mt-1 font-mono text-[11px] bg-surface border border-line rounded p-2 overflow-x-auto max-h-40">{JSON.stringify(t.input, null, 2)}</pre>
                          <pre className="mt-1 font-mono text-[11px] bg-surface border border-line rounded p-2 overflow-x-auto max-h-48 whitespace-pre-wrap break-all">{t.output_preview}</pre>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                {e.reply && <div className="text-[11.5px] text-faint mt-2">{e.reply.model} · {(e.reply.usage.input_tokens + e.reply.usage.output_tokens).toLocaleString()} tokens</div>}
              </div>
            </div>
          ))}
          {send.isPending && (
            <div className="flex gap-3 items-center text-[13px] text-muted">
              <div className="size-7 rounded-lg bg-brand-soft grid place-items-center"><BrainCircuit className="size-4 text-brand-text animate-pulse" /></div>
              Thinking and running models…
            </div>
          )}
          <div ref={end} />
        </div>
      </div>
      <form className="mt-3 flex gap-2 items-end" onSubmit={(e) => { e.preventDefault(); submit(input); }}>
        <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={2} maxLength={8000}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(input); } }}
          placeholder="Ask a question… (Enter to send, Shift+Enter for a new line)"
          className="flex-1 resize-none rounded-xl border border-line-strong bg-surface px-4 py-3 text-[14px] focus:outline-none focus:border-brand focus:ring-3 focus:ring-[var(--brand-ring)] shadow-card" />
        <button type="submit" disabled={!input.trim() || send.isPending} aria-label="Send"
          className={cx("size-11 rounded-xl grid place-items-center bg-brand text-white dark:text-[#0e1013] hover:bg-brand-hover disabled:opacity-40 shadow-card")}>
          <ArrowUp className="size-5" />
        </button>
      </form>
    </div>
  );
}
