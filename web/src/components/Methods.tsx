import katex from "katex";
import "katex/dist/katex.min.css";
import { ChevronDown, FileCode2, Sigma } from "lucide-react";
import { useMemo, useState } from "react";
import { METHODS } from "@/lib/methods";
import { cx } from "./ui";

export function TeX({ children, display = false }: { children: string; display?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: display, throwOnError: false, strict: "ignore" }), [children, display]);
  return <span className={display ? "block overflow-x-auto py-1" : undefined} dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Collapsible "Model & methods" panel showing the governing equations of an engine. */
export function MethodsPanel({ id, defaultOpen = false }: { id: keyof typeof METHODS | string; defaultOpen?: boolean }) {
  const m = METHODS[id];
  const [open, setOpen] = useState(defaultOpen);
  if (!m) return null;
  return (
    <section className="bg-surface border border-line rounded-xl shadow-card min-w-0">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between gap-3 px-5 py-4 text-left" aria-expanded={open}>
        <span className="flex items-center gap-2.5 min-w-0">
          <span className="size-8 rounded-lg bg-brand-soft text-brand-text grid place-items-center shrink-0"><Sigma className="size-4" /></span>
          <span className="min-w-0">
            <span className="block text-[14px] font-semibold text-ink">Model & methods · {m.title}</span>
            <span className="block text-[12px] text-muted truncate">{m.source}</span>
          </span>
        </span>
        <ChevronDown className={cx("size-4 text-muted transition-transform shrink-0", open && "rotate-180")} />
      </button>
      {open && (
        <div className="px-5 pb-5 border-t border-line pt-4 flex flex-col gap-5">
          {m.sections.map((s) => (
            <div key={s.title}>
              <h4 className="text-[13px] font-semibold text-ink-2 mb-2">{s.title}</h4>
              {s.eqs?.map((e, i) => (
                <div key={i} className="rounded-lg bg-surface-2 border border-line px-4 py-2 mb-2 text-ink text-[15px]"><TeX display>{e}</TeX></div>
              ))}
              {s.text && <p className="text-[13px] text-muted leading-6">{s.text}</p>}
            </div>
          ))}
          <div className="grid md:grid-cols-2 gap-3 text-[12.5px]">
            <div className="rounded-lg border border-line p-3"><div className="font-semibold text-ink-2 mb-1">Numerical method</div><div className="text-muted leading-5">{m.numerics}</div></div>
            <div className="rounded-lg border border-line p-3"><div className="font-semibold text-ink-2 mb-1 flex items-center gap-1.5"><FileCode2 className="size-3.5" />Implementation</div><code className="font-mono text-[12px] text-muted">{m.code}</code></div>
          </div>
        </div>
      )}
    </section>
  );
}
