import { useMemo } from "react";
import { CONF_BINS } from "./MolViewer";

const BASE_CLS: Record<string, string> = {
  A: "text-[#2f7d5b] dark:text-[#6fc29b]",
  U: "text-[#b3452f] dark:text-[#f0795f]",
  G: "text-[#a8690f] dark:text-[#e0a64a]",
  C: "text-[#3f6594] dark:text-[#8fb0da]",
};

/** Sequence in blocks of 10 with position ruler; bases coloured by identity. */
export function SequenceBlocks({ sequence, perLine = 60 }: { sequence: string; perLine?: number }) {
  const lines = useMemo(() => {
    const out: { start: number; chunk: string }[] = [];
    for (let i = 0; i < sequence.length; i += perLine) out.push({ start: i, chunk: sequence.slice(i, i + perLine) });
    return out;
  }, [sequence, perLine]);
  return (
    <div className="font-mono text-[12.5px] leading-6 overflow-x-auto">
      {lines.map(({ start, chunk }) => (
        <div key={start} className="flex gap-3 whitespace-nowrap">
          <span className="w-12 text-right text-faint select-none num">{start + 1}</span>
          <span>
            {chunk.match(/.{1,10}/g)?.map((blk, bi) => (
              <span key={bi} className="mr-2">
                {blk.split("").map((b, i) => <span key={i} className={BASE_CLS[b] ?? "text-faint"}>{b}</span>)}
              </span>
            ))}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Arc diagram: MFE pairs as solid arcs coloured by pair probability; ensemble pairs faint. */
export function ArcDiagram({ sequence, pairs, probs, confidence, height = 220 }: {
  sequence: string; pairs: [number, number][]; probs?: [number, number, number][]; confidence?: number[]; height?: number;
}) {
  const n = sequence.length;
  const W = 1000;
  const x = (i: number) => 12 + (i / Math.max(n - 1, 1)) * (W - 24);
  const base = height - 34;
  const pmap = useMemo(() => new Map((probs ?? []).map(([i, j, p]) => [`${i}-${j}`, p])), [probs]);
  const colorFor = (p: number) => (CONF_BINS.find((b) => p * 100 >= b.min) ?? CONF_BINS[3]).color;
  const arc = (i: number, j: number) => {
    const x1 = x(i), x2 = x(j), r = (x2 - x1) / 2;
    const h = Math.min(base - 8, r);
    return `M ${x1} ${base} C ${x1} ${base - h * 1.33}, ${x2} ${base - h * 1.33}, ${x2} ${base}`;
  };
  const mfe = new Set(pairs.map(([i, j]) => `${i}-${j}`));
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full" role="img" aria-label="Arc diagram of base pairs">
      {(probs ?? []).filter(([i, j, p]) => !mfe.has(`${i}-${j}`) && p >= 0.1).map(([i, j, p]) => (
        <path key={`e${i}-${j}`} d={arc(i, j)} fill="none" stroke="currentColor" className="text-faint" strokeOpacity={p * 0.6} strokeWidth={1} />
      ))}
      {pairs.map(([i, j]) => {
        const p = pmap.get(`${i}-${j}`);
        return <path key={`${i}-${j}`} d={arc(i, j)} fill="none" stroke={p === undefined ? "#1b563e" : colorFor(p)} strokeWidth={1.8} strokeOpacity={0.9} />;
      })}
      <line x1={x(0)} x2={x(n - 1)} y1={base} y2={base} stroke="currentColor" className="text-line-strong" strokeWidth={2} />
      {confidence && confidence.map((c, i) => (
        <rect key={i} x={x(i) - (W - 24) / Math.max(n, 1) / 2} y={base + 8} width={Math.max(1, (W - 24) / Math.max(n, 1))} height={10} fill={colorFor(c)} />
      ))}
      {n <= 120 && sequence.split("").map((b, i) => (
        <text key={i} x={x(i)} y={base + 30} textAnchor="middle" fontSize={n > 60 ? 8 : 11} className="fill-muted font-mono">{b}</text>
      ))}
    </svg>
  );
}
