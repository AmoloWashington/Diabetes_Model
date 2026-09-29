import { useEffect, useRef, useState } from "react";
import type { DNAHelix } from "./viz/dna3d";

/** Mounts a DNAHelix into the returned ref'd element; disposes it on unmount. */
export function useHelix(sequence: string | undefined, opts: { interactive?: boolean; maxBp?: number } = {}) {
  const ref = useRef<HTMLDivElement>(null);
  const helixRef = useRef<DNAHelix | null>(null);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef(sequence);
  latest.current = sequence;
  const { interactive = true, maxBp = 120 } = opts;

  useEffect(() => {
    let disposed = false;
    import("./viz/dna3d")
      .then(({ DNAHelix }) => {
        if (disposed || !ref.current) return;
        try {
          helixRef.current = new DNAHelix(ref.current, { interactive, maxBp });
          if (latest.current) helixRef.current.setSequence(latest.current);
        } catch {
          setError("3D rendering (WebGL) is not available in this browser.");
        }
      })
      .catch(() => setError("Could not load the 3D renderer."));
    return () => {
      disposed = true;
      helixRef.current?.dispose();
      helixRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [interactive, maxBp]);

  useEffect(() => {
    if (sequence) helixRef.current?.setSequence(sequence);
  }, [sequence]);

  return { ref, helix: helixRef, error };
}
