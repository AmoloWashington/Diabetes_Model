import { useQuery } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { Card, PageHeader, Spinner } from "@/components/ui";
import { api } from "@/lib/api";

interface Ref { id: string; used_for: string; citation: string; doi?: string }

export default function References() {
  const q = useQuery({ queryKey: ["refs"], queryFn: () => api.get<{ references: Ref[] }>("/api/references") });
  return (
    <div>
      <PageHeader title="References" description="Primary sources for every model, formula, dataset and mechanism implemented or animated in GlucoLab." />
      <Card padded={false}>
        {!q.data ? <div className="p-6"><Spinner /></div> : (
          <ol className="divide-y divide-line">
            {q.data.references.map((r, i) => (
              <li key={r.id} className="px-5 py-4 flex gap-4">
                <span className="num text-[13px] text-faint w-6 shrink-0">{i + 1}</span>
                <div className="min-w-0">
                  <div className="text-[12px] font-medium text-brand-text">{r.used_for}</div>
                  <div className="text-[13.5px] text-ink mt-0.5 leading-6">{r.citation}</div>
                  {r.doi && (
                    <a href={`https://doi.org/${r.doi}`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-[12.5px] text-muted hover:text-brand-text mt-1">
                      doi:{r.doi} <ExternalLink className="size-3" />
                    </a>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
      </Card>
    </div>
  );
}
