export const fmt = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(d);

export const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : `${(100 * v).toFixed(d)}%`;

export const bytes = (n: number) =>
  n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 ** 2).toFixed(1)} MB`;

export const date = (ts: number) => new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

/** Zip parallel arrays into row objects for charts, optionally downsampled. */
export function rowsOf<K extends string>(cols: Record<K, number[]>, every = 1): Record<K, number>[] {
  const keys = Object.keys(cols) as K[];
  const n = cols[keys[0]].length;
  const out: Record<K, number>[] = [];
  for (let i = 0; i < n; i += every) {
    const r = {} as Record<K, number>;
    for (const k of keys) r[k] = cols[k][i];
    out.push(r);
  }
  if ((n - 1) % every !== 0) {
    const r = {} as Record<K, number>;
    for (const k of keys) r[k] = cols[k][n - 1];
    out.push(r);
  }
  return out;
}
