import { useQuery } from "@tanstack/react-query";
import "katex/dist/katex.min.css";
import { ArrowUp, BrainCircuit, CheckCircle2, ChevronDown, Loader2, MessageSquarePlus, RotateCcw, Square, Trash2, Wrench, XCircle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Badge, Callout, PageHeader, cx } from "@/components/ui";
import { api, type Health } from "@/lib/api";
import { loadConversations, newId, saveConversations, takePendingQuestion, type ChatMsg, type Conversation } from "@/lib/chatStore";

const SUGGESTIONS = [
  "Simulate a 75 g meal in the healthy and type 2 phenotypes and explain the mechanistic differences.",
  "Derive the β-cell resting potential with the GHK equation and show how K_ATP closure depolarises it.",
  "Fold GGGCGCAAGCCUAUGCGCUUCGGCGCAUAGGCUUGCGCCC and explain how confident the structure is.",
  "Why does a rapid fall in insulin sensitivity cause β-cell failure in the Topp model while a slow one does not?",
];

type StreamEvent =
  | { type: "text"; text: string } | { type: "thinking"; text: string }
  | { type: "tool_start"; name: string; input: unknown }
  | { type: "tool_result"; name: string; is_error: boolean; output: string }
  | { type: "done"; model: string; usage: { input_tokens: number; output_tokens: number } }
  | { type: "error"; message: string };

async function* sse(res: Response): AsyncGenerator<StreamEvent> {
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of chunk.split("\n")) if (line.startsWith("data: ")) yield JSON.parse(line.slice(6));
    }
  }
}

function Markdown({ text }: { text: string }) {
  return <div className="prose-ai text-[14px] leading-6 text-ink"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{text}</ReactMarkdown></div>;
}

function AssistantMessage({ m, live }: { m: ChatMsg; live: boolean }) {
  const [showThinking, setShowThinking] = useState(false);
  return (
    <div className="flex gap-3">
      <div className="size-7 shrink-0 rounded-lg bg-brand text-white dark:text-[#0e1013] grid place-items-center"><BrainCircuit className="size-4" /></div>
      <div className="min-w-0 flex-1 flex flex-col gap-2.5">
        {m.thinking && (
          <div className="rounded-lg border border-line bg-surface-2">
            <button onClick={() => setShowThinking(!showThinking)} className="w-full flex items-center gap-2 px-3 py-2 text-[12.5px] text-muted">
              {live && !m.content ? <Loader2 className="size-3.5 animate-spin" /> : <BrainCircuit className="size-3.5" />}
              Reasoning summary
              <ChevronDown className={cx("size-3.5 ml-auto transition", showThinking && "rotate-180")} />
            </button>
            {showThinking && <div className="px-3 pb-3 text-[12.5px] text-muted whitespace-pre-wrap">{m.thinking}</div>}
          </div>
        )}
        {m.tools && m.tools.length > 0 && (
          <div className="flex flex-col gap-1.5">
            {m.tools.map((t, i) => (
              <details key={i} className="rounded-lg border border-line bg-surface-2 group">
                <summary className="cursor-pointer list-none px-3 py-2 flex items-center gap-2 text-[12.5px]">
                  {t.status === "running" ? <Loader2 className="size-3.5 animate-spin text-brand-text" /> : t.status === "ok" ? <CheckCircle2 className="size-3.5 text-ok" /> : <XCircle className="size-3.5 text-danger" />}
                  <Wrench className="size-3.5 text-muted" /><span className="font-mono text-ink-2">{t.name}</span>
                  <span className="text-faint">{t.status === "running" ? "running…" : t.status === "ok" ? "computed" : "error"}</span>
                  <ChevronDown className="size-3.5 ml-auto text-muted group-open:rotate-180 transition" />
                </summary>
                <div className="px-3 pb-3 flex flex-col gap-1.5 text-[11.5px]">
                  <pre className="font-mono bg-surface border border-line rounded p-2 overflow-x-auto max-h-40">{JSON.stringify(t.input, null, 2)}</pre>
                  {t.output && <pre className="font-mono bg-surface border border-line rounded p-2 overflow-x-auto max-h-48 whitespace-pre-wrap break-all">{t.output}</pre>}
                </div>
              </details>
            ))}
          </div>
        )}
        {m.content ? <Markdown text={m.content} /> : live && !m.thinking && !m.tools?.length && <div className="text-[13px] text-muted flex items-center gap-2"><Loader2 className="size-4 animate-spin" />Starting…</div>}
        {m.error && <Callout tone="danger">{m.error}</Callout>}
        {m.model && <div className="text-[11.5px] text-faint">{m.model}{m.tokens ? ` · ${m.tokens.toLocaleString()} tokens` : ""}</div>}
      </div>
    </div>
  );
}

export default function Assistant() {
  const health = useQuery({ queryKey: ["health"], queryFn: () => api.get<Health>("/api/health") });
  const [convs, setConvs] = useState<Conversation[]>(() => loadConversations());
  const [activeId, setActiveId] = useState<string | null>(() => loadConversations()[0]?.id ?? null);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const active = convs.find((c) => c.id === activeId) ?? null;

  useEffect(() => { saveConversations(convs); }, [convs]);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [active?.messages]);

  const update = useCallback((id: string, fn: (msgs: ChatMsg[]) => ChatMsg[]) => {
    setConvs((cs) => cs.map((c) => (c.id === id ? { ...c, messages: fn(c.messages), updated: Date.now() } : c)));
  }, []);

  const send = useCallback(async (text: string, context?: { page: string; summary: string }) => {
    const t = text.trim();
    if (!t || streaming) return;
    let conv = active;
    if (!conv) {
      conv = { id: newId(), title: t.slice(0, 70), updated: Date.now(), messages: [] };
      setConvs((cs) => [conv!, ...cs]);
      setActiveId(conv.id);
    }
    const id = conv.id;
    const history = [...conv.messages.filter((m) => !m.error && m.content).map((m) => ({ role: m.role, content: m.content })), { role: "user" as const, content: t }];
    update(id, (ms) => [...ms.filter((m) => !(m.role === "assistant" && m.error && !m.content)), { role: "user", content: t, context }, { role: "assistant", content: "", tools: [] }]);
    setInput("");
    setStreaming(true);
    const ctrl = new AbortController();
    abort.current = ctrl;
    const patchLast = (fn: (m: ChatMsg) => ChatMsg) => update(id, (ms) => [...ms.slice(0, -1), fn(ms[ms.length - 1])]);
    try {
      const res = await fetch("/api/ai/chat/stream", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: ctrl.signal,
        body: JSON.stringify({ messages: history.slice(-40), ...(context ? { context } : {}) }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => null);
        throw new Error((d && typeof d.detail === "string" && d.detail) || `Request failed (${res.status})`);
      }
      for await (const ev of sse(res)) {
        if (ev.type === "text") patchLast((m) => ({ ...m, content: m.content + ev.text }));
        else if (ev.type === "thinking") patchLast((m) => ({ ...m, thinking: (m.thinking ?? "") + ev.text }));
        else if (ev.type === "tool_start") patchLast((m) => ({ ...m, tools: [...(m.tools ?? []), { name: ev.name, input: ev.input, status: "running" }] }));
        else if (ev.type === "tool_result") patchLast((m) => {
          const tools = [...(m.tools ?? [])];
          const k = tools.map((x) => x.name === ev.name && x.status === "running").lastIndexOf(true);
          if (k >= 0) tools[k] = { ...tools[k], status: ev.is_error ? "error" : "ok", output: ev.output };
          return { ...m, tools };
        });
        else if (ev.type === "done") patchLast((m) => ({ ...m, model: ev.model, tokens: ev.usage.input_tokens + ev.usage.output_tokens }));
        else if (ev.type === "error") patchLast((m) => ({ ...m, error: ev.message }));
      }
    } catch (e) {
      const msg = (e as Error).name === "AbortError" ? "Stopped." : (e as Error).message || "Cannot reach the GlucoLab server.";
      patchLast((m) => ({ ...m, error: msg }));
    } finally {
      setStreaming(false);
      abort.current = null;
    }
  }, [active, streaming, update]);

  // Hand-off from "Ask AI about this" buttons on other pages
  useEffect(() => {
    const p = takePendingQuestion();
    if (p) {
      setActiveId(null);
      setTimeout(() => send(p.question, p.context), 0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const lastUser = [...(active?.messages ?? [])].reverse().find((m) => m.role === "user");
  const lastFailed = active?.messages.at(-1)?.error;

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] min-h-[600px]">
      <PageHeader eyebrow="AI" title="Research assistant"
        description="Streams its answer live and calls GlucoLab's validated engines as tools; each call and its output is shown as it runs."
        actions={health.data?.ai_configured ? <Badge tone="brand">{health.data.ai_model}</Badge> : undefined} />
      {health.data && !health.data.ai_configured && (
        <div className="mb-4"><Callout tone="warn" title="AI is not configured on this server">
          Add <code className="font-mono">ANTHROPIC_API_KEY</code> to the git-ignored <code className="font-mono">.env</code> file in the repository root and restart the backend. Every other feature works without it.
        </Callout></div>
      )}
      <div className="flex-1 min-h-0 grid lg:grid-cols-[230px_1fr] gap-4">
        <aside className="hidden lg:flex flex-col rounded-xl border border-line bg-surface shadow-card min-h-0">
          <button onClick={() => setActiveId(null)} className="m-2 flex items-center gap-2 rounded-lg border border-line-strong px-3 h-9 text-[13px] font-medium hover:bg-surface-2"><MessageSquarePlus className="size-4" />New conversation</button>
          <ul className="flex-1 overflow-y-auto px-2 pb-2 flex flex-col gap-0.5">
            {convs.map((c) => (
              <li key={c.id} className={cx("group flex items-center rounded-lg", c.id === activeId ? "bg-brand-soft/70" : "hover:bg-surface-2")}>
                <button onClick={() => setActiveId(c.id)} className="flex-1 min-w-0 text-left px-2.5 py-2 text-[12.5px] truncate">{c.title}</button>
                <button onClick={() => { setConvs((cs) => cs.filter((x) => x.id !== c.id)); if (c.id === activeId) setActiveId(null); }}
                  className="opacity-0 group-hover:opacity-100 p-1.5 text-faint hover:text-danger" aria-label="Delete conversation"><Trash2 className="size-3.5" /></button>
              </li>
            ))}
            {!convs.length && <li className="px-2.5 py-2 text-[12px] text-faint">Conversations are saved in this browser.</li>}
          </ul>
        </aside>
        <div className="flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto rounded-xl border border-line bg-surface shadow-card">
            <div className="max-w-3xl mx-auto px-5 py-6 flex flex-col gap-6">
              {!active?.messages.length && (
                <div className="text-center py-8">
                  <div className="size-11 rounded-xl bg-brand-soft text-brand-text grid place-items-center mx-auto"><BrainCircuit className="size-5" /></div>
                  <h3 className="mt-3 font-semibold">Ask about physiology, biophysics, RNA or the models</h3>
                  <p className="text-[13px] text-muted mt-1">Use "Ask AI about this" on any page to bring its results here.</p>
                  <div className="grid sm:grid-cols-2 gap-2 mt-6 text-left">
                    {SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)} className="min-w-0 break-words [overflow-wrap:anywhere] rounded-lg border border-line p-3 text-[13px] text-ink-2 hover:bg-surface-2 hover:border-line-strong">{s}</button>)}
                  </div>
                </div>
              )}
              {active?.messages.map((m, i) => m.role === "user" ? (
                <div key={i} className="self-end max-w-[85%] flex flex-col items-end gap-1">
                  {m.context && <Badge tone="brand">Context: {m.context.page}</Badge>}
                  <div className="rounded-2xl rounded-br-md bg-brand-soft px-4 py-2.5 text-[14px] whitespace-pre-wrap">{m.content}</div>
                </div>
              ) : (
                <AssistantMessage key={i} m={m} live={streaming && i === active.messages.length - 1} />
              ))}
              {lastFailed && lastUser && !streaming && (
                <button onClick={() => send(lastUser.content, lastUser.context)} className="self-start flex items-center gap-1.5 text-[12.5px] text-brand-text hover:underline"><RotateCcw className="size-3.5" />Retry</button>
              )}
              <div ref={end} />
            </div>
          </div>
          <form className="mt-3 flex gap-2 items-end" onSubmit={(e) => { e.preventDefault(); send(input); }}>
            <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={2} maxLength={8000}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); } }}
              placeholder="Ask a question… (Enter to send, Shift+Enter for a new line)"
              className="flex-1 resize-none rounded-xl border border-line-strong bg-surface px-4 py-3 text-[14px] focus:outline-none focus:border-brand focus:ring-3 focus:ring-[var(--brand-ring)] shadow-card" />
            {streaming ? (
              <button type="button" onClick={() => abort.current?.abort()} aria-label="Stop" className="size-11 rounded-xl grid place-items-center bg-surface border border-line-strong text-ink hover:bg-surface-2 shadow-card"><Square className="size-4 fill-current" /></button>
            ) : (
              <button type="submit" disabled={!input.trim()} aria-label="Send" className="size-11 rounded-xl grid place-items-center bg-brand text-white dark:text-[#0e1013] hover:bg-brand-hover disabled:opacity-40 shadow-card"><ArrowUp className="size-5" /></button>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}
