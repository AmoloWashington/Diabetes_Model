/** Base pairs (0-based i<j) from a dot-bracket string; unbalanced input returns []. */
export function dotBracketPairs(db: string | undefined | null): [number, number][] {
  if (!db) return [];
  const stack: number[] = [];
  const out: [number, number][] = [];
  for (let i = 0; i < db.length; i++) {
    if (db[i] === "(") stack.push(i);
    else if (db[i] === ")") {
      const j = stack.pop();
      if (j === undefined) return [];
      out.push([j, i]);
    }
  }
  return stack.length ? [] : out;
}
