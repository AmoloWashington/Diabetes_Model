import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Box, BrainCircuit, Database, Microscope, Timer } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Badge, Card, Table } from "@/components/ui";
import { api } from "@/lib/api";
import { useHelix } from "@/lib/useHelix";

function Module({ to, icon, title, source, children }: { to: string; icon: ReactNode; title: string; source: string; children: ReactNode }) {
  return (
    <Link to={to} className="group bg-surface border border-line rounded-xl p-5 shadow-card hover:border-line-strong hover:shadow-pop transition-all flex flex-col">
      <div className="flex items-center justify-between">
        <div className="size-9 rounded-lg bg-brand-soft text-brand-text grid place-items-center [&_svg]:size-[18px]">{icon}</div>
        <ArrowRight className="size-4 text-faint group-hover:text-brand-text group-hover:translate-x-0.5 transition" />
      </div>
      <h3 className="mt-4 text-[15px] font-semibold text-ink">{title}</h3>
      <p className="text-[12px] font-medium text-brand-text mt-0.5">{source}</p>
      <p className="text-[13px] text-muted mt-2 leading-5">{children}</p>
    </Link>
  );
}

export default function Overview() {
  const demo = useQuery({ queryKey: ["dna-demo"], queryFn: () => api.get<{ sequence: string }>("/api/dna/demo") });
  const { ref } = useHelix(demo.data?.sequence, { interactive: false, maxBp: 42 });

  return (
    <div>
      <section className="grid lg:grid-cols-[1.15fr_1fr] gap-8 items-center mb-12">
        <div>
          <Badge tone="brand">Research workbench</Badge>
          <h1 className="mt-4 text-[34px] sm:text-[40px] leading-[1.1] font-semibold tracking-[-0.025em] text-ink">
            Glucose physiology and RNA structure, computed from first principles.
          </h1>
          <p className="mt-4 text-[15.5px] leading-7 text-muted max-w-xl">
            GlucoLab implements peer-reviewed models of the glucose–insulin system, RNA folding and 3D structure,
            validates each against its publication, and adds an AI assistant that reports only numbers it has computed
            with these engines.
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            <Link to="/physiology/meal" className="inline-flex items-center gap-1.5 h-10 px-4 rounded-lg bg-brand text-white dark:text-[#0e1013] text-[14px] font-medium hover:bg-brand-hover shadow-card">
              Run a meal simulation <ArrowRight className="size-4" />
            </Link>
            <Link to="/rna/predict" className="inline-flex items-center h-10 px-4 rounded-lg border border-line-strong bg-surface text-[14px] font-medium hover:bg-surface-2">
              Predict an RNA structure
            </Link>
          </div>
        </div>
        <div className="relative rounded-2xl border border-line bg-gradient-to-b from-surface to-surface-2 overflow-hidden shadow-card">
          <div ref={ref} className="h-[340px]" aria-label="Rotating 3D DNA double helix" />
          <div className="absolute bottom-3 left-4 right-4 flex justify-between text-[11.5px] text-muted">
            <span>B-DNA · 10.5 bp/turn · 0.34 nm rise</span>
            <span>Insulin B-chain coding sequence (synthetic)</span>
          </div>
        </div>
      </section>

      <h2 className="text-[13px] font-semibold uppercase tracking-[0.08em] text-faint mb-3">Modules</h2>
      <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4 mb-12">
        <Module to="/physiology/meal" icon={<Activity />} title="Meal simulation" source="Dalla Man, Rizza & Cobelli · IEEE TBME 2007">
          12-state nonlinear model of absorption, hepatic production, utilisation and β-cell secretion. The core of the FDA-accepted UVA/Padova simulator.
        </Module>
        <Module to="/physiology/beta-cell" icon={<Timer />} title="β-cell dynamics" source="Topp et al. · J Theor Biol 2000">
          Slow–fast dynamics of β-cell mass with fixed-point and eigenvalue analysis: why the rate of insulin-resistance onset matters.
        </Module>
        <Module to="/rna/predict" icon={<Box />} title="RNA 3D prediction" source="ViennaRNA 2 · US-align TM-score">
          Template-based modelling from structure datasets, coarse-grained de novo fallback, and per-residue confidence.
        </Module>
        <Module to="/data" icon={<Database />} title="Dataset explorer" source="Kaggle · CSV / FASTA / PDB">
          Load the Stanford RNA 3D Folding data from Kaggle or upload files, inspect schemas and build a template library.
        </Module>
        <Module to="/cells" icon={<Microscope />} title="Cell theatre" source="Rorsman & Ashcroft 2018 · Saltiel & Kahn 2001">
          Model-driven animations of stimulus–secretion coupling, insulin signalling to GLUT4, and insulin biosynthesis.
        </Module>
        <Module to="/assistant" icon={<BrainCircuit />} title="Research assistant" source="Claude · tool-grounded">
          Ask questions in plain language. Every number the assistant reports comes from a visible call to a validated engine.
        </Module>
      </div>

      <div className="grid lg:grid-cols-[1.4fr_1fr] gap-4">
        <Card>
          <h3 className="text-[15px] font-semibold mb-1">Validation against primary publications</h3>
          <p className="text-[13px] text-muted mb-4">Enforced by the automated test suite on every change.</p>
          <Table head={["Model", "Quantity", "Published", "GlucoLab"]}>
            {[
              ["Dalla Man 2007", "k_p1 (derived at steady state)", "2.70", "2.698"],
              ["Dalla Man 2007", "m6 (hepatic extraction)", "0.6471", "0.6469"],
              ["Topp 2000", "Physiological fixed point G, I, β", "100, 10, 300", "100, 10, 300"],
              ["Topp 2000", "Saddle glucose (mg/dl)", "250", "250"],
              ["ADAG 2008", "eAG at HbA1c 7%", "154 mg/dl", "154.2 mg/dl"],
              ["Minimal model", "S_I recovery (noise-free)", "5.00e-4", "5.00e-4"],
            ].map((r, i) => (
              <tr key={i}>{r.map((c, j) => <td key={j} className={j >= 2 ? "px-3 py-2 num font-mono text-[12.5px]" : "px-3 py-2"}>{c}</td>)}</tr>
            ))}
          </Table>
        </Card>
        <Card>
          <h3 className="text-[15px] font-semibold mb-1">A finding worth presenting</h3>
          <p className="text-[13px] text-muted mb-4">The public symptom dataset contains 269 exact duplicates among 520 records.</p>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-danger-soft p-4">
              <div className="text-[12px] text-muted">Naive random split</div>
              <div className="num text-[26px] font-semibold text-danger mt-1">96.9%</div>
              <div className="text-[12px] text-muted">accuracy (inflated)</div>
            </div>
            <div className="rounded-lg bg-ok-soft p-4">
              <div className="text-[12px] text-muted">Leak-free grouped CV</div>
              <div className="num text-[26px] font-semibold text-ok mt-1">90.8%</div>
              <div className="text-[12px] text-muted">AUC 0.972 [0.95–0.99]</div>
            </div>
          </div>
          <Link to="/risk" className="mt-4 inline-flex items-center gap-1 text-[13px] font-medium text-brand-text hover:underline">
            Open the model card <ArrowRight className="size-3.5" />
          </Link>
        </Card>
      </div>
    </div>
  );
}
