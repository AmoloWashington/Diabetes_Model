import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { MethodsPanel, TeX } from "@/components/Methods";
import { Card, CardHeader, PageHeader } from "@/components/ui";

// Exact SI values (2019 redefinition); identical to backend/app/physiology/biophysics.py.
const CONSTANTS: [string, string, string][] = [
  [String.raw`k_B`, String.raw`1.380649\times10^{-23}\ \mathrm{J\,K^{-1}}`, "Boltzmann constant (exact)"],
  [String.raw`N_A`, String.raw`6.02214076\times10^{23}\ \mathrm{mol^{-1}}`, "Avogadro constant (exact)"],
  [String.raw`e`, String.raw`1.602176634\times10^{-19}\ \mathrm{C}`, "Elementary charge (exact)"],
  [String.raw`R = k_B N_A`, String.raw`8.314462618\ldots\ \mathrm{J\,mol^{-1}K^{-1}}`, "Gas constant"],
  [String.raw`F = e N_A`, String.raw`96485.33212\ldots\ \mathrm{C\,mol^{-1}}`, "Faraday constant"],
];

const GROUPS: { title: string; items: { id: string; to: string; page: string }[] }[] = [
  { title: "Physiology (ordinary differential equations)", items: [
    { id: "meal", to: "/physiology/meal", page: "Meal simulation" },
    { id: "betacell", to: "/physiology/beta-cell", page: "β-cell dynamics" },
    { id: "minimal", to: "/physiology/minimal-model", page: "Insulin sensitivity" },
    { id: "clinical", to: "/physiology/clinical", page: "Clinical indices" },
  ] },
  { title: "Physics of the cell", items: [
    { id: "membrane", to: "/physiology/biophysics", page: "Membrane biophysics" },
    { id: "diffusion", to: "/physiology/biophysics", page: "Membrane biophysics" },
  ] },
  { title: "RNA structure (statistical mechanics and geometry)", items: [
    { id: "rnafold", to: "/rna/sequences", page: "Sequences" },
    { id: "rna3d", to: "/rna/predict", page: "3D prediction" },
  ] },
  { title: "Statistics and machine learning", items: [
    { id: "risk", to: "/risk", page: "Diabetes risk model" },
  ] },
];

export default function MathPhysics() {
  return (
    <div>
      <PageHeader eyebrow="Foundations" title="Math & physics"
        description="Every equation GlucoLab solves, in one place. Each block mirrors the backend code that produces the numbers on screen, with its numerical method, source file and primary reference. Open a panel to read it, or jump to the page that runs it." />
      <Card className="mb-6">
        <CardHeader title="Physical constants" subtitle="SI 2019 exact values; R and F are derived from them, never typed in" />
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <tbody>
              {CONSTANTS.map(([sym, val, name]) => (
                <tr key={name} className="border-t border-line first:border-0">
                  <td className="py-2 pr-4 whitespace-nowrap"><TeX>{sym}</TeX></td>
                  <td className="py-2 pr-4 whitespace-nowrap"><TeX>{val}</TeX></td>
                  <td className="py-2 text-muted">{name}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="flex flex-col gap-8">
        {GROUPS.map((g, gi) => (
          <section key={g.title}>
            <h2 className="text-[12px] font-semibold tracking-wider uppercase text-muted mb-3">{g.title}</h2>
            <div className="flex flex-col gap-3">
              {g.items.map((it, i) => (
                <div key={it.id}>
                  <MethodsPanel id={it.id} defaultOpen={gi === 0 && i === 0} />
                  <Link to={it.to} className="inline-flex items-center gap-1 text-[12.5px] text-brand-text hover:underline mt-1.5 ml-1">
                    Run it in {it.page}<ArrowRight className="size-3.5" />
                  </Link>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
