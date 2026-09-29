"""RNA secondary structure with ViennaRNA.

    R. Lorenz, S. H. Bernhart, C. Hoener zu Siederdissen, H. Tafer, C. Flamm,
    P. F. Stadler, I. L. Hofacker. "ViennaRNA Package 2.0".
    Algorithms for Molecular Biology 6:26, 2011. doi:10.1186/1748-7188-6-26

Minimum free energy (MFE) structures use the Turner 2004 nearest-neighbour
energy parameters (ViennaRNA default). Base-pair probabilities come from the
McCaskill partition function; they provide a per-nucleotide confidence:
for a nucleotide paired in the MFE structure, the probability of that pair;
for an unpaired nucleotide, the probability of being unpaired.
"""

from __future__ import annotations

import RNA

MAX_FOLD_LENGTH = 4000
MAX_PF_LENGTH = 1000


def _foldable(seq: str) -> str:
    # ViennaRNA treats unknown letters as unpairable; map ambiguity codes to N.
    return "".join(c if c in "ACGU" else "N" for c in seq)


def pair_table(db: str) -> list[int]:
    """0-based partner index for each position, -1 if unpaired."""
    stack: list[int] = []
    pt = [-1] * len(db)
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if not stack:
                raise ValueError("Unbalanced dot-bracket")
            j = stack.pop()
            pt[i], pt[j] = j, i
    if stack:
        raise ValueError("Unbalanced dot-bracket")
    return pt


def stems(pt: list[int]) -> list[list[tuple[int, int]]]:
    """Group base pairs (i<j) into stacked helices."""
    pairs = sorted((i, j) for i, j in enumerate(pt) if j > i)
    out: list[list[tuple[int, int]]] = []
    for i, j in pairs:
        if out and out[-1][-1][0] == i - 1 and out[-1][-1][1] == j + 1:
            out[-1].append((i, j))
        else:
            out.append([(i, j)])
    return out


def fold(seq: str, temperature_c: float = 37.0, with_pf: bool | None = None) -> dict:
    if not seq:
        raise ValueError("Empty sequence")
    if len(seq) > MAX_FOLD_LENGTH:
        raise ValueError(f"Folding is limited to {MAX_FOLD_LENGTH} nt")
    if not (0.0 <= temperature_c <= 100.0):
        raise ValueError("Temperature must be within 0-100 degC")
    md = RNA.md()
    md.temperature = temperature_c
    fc = RNA.fold_compound(_foldable(seq), md)
    structure, mfe = fc.mfe()
    pt = pair_table(structure)
    out: dict = {
        "sequence": seq,
        "temperature_c": temperature_c,
        "mfe_structure": structure,
        "mfe_kcal_mol": float(mfe),
        "pairs": [[i, j] for i, j in enumerate(pt) if j > i],
        "n_pairs": sum(1 for i, j in enumerate(pt) if j > i),
        "n_stems": len(stems(pt)),
        "energy_model": "Turner 2004 (ViennaRNA default)",
        "method": f"ViennaRNA {RNA.__version__}",
    }
    do_pf = (len(seq) <= MAX_PF_LENGTH) if with_pf is None else with_pf
    if do_pf:
        if len(seq) > MAX_PF_LENGTH:
            raise ValueError(f"Partition function is limited to {MAX_PF_LENGTH} nt")
        fc.exp_params_rescale(mfe)
        _, ensemble_energy = fc.pf()
        bpp = fc.bpp()
        n = len(seq)
        p_paired = [0.0] * n
        dot = []
        for i in range(1, n + 1):
            row = bpp[i]
            for j in range(i + 1, n + 1):
                p = row[j]
                if p > 1e-5:
                    p_paired[i - 1] += p
                    p_paired[j - 1] += p
                    if p >= 0.01:
                        dot.append([i - 1, j - 1, round(float(p), 4)])
        conf = [
            float(bpp[min(i, pt[i]) + 1][max(i, pt[i]) + 1]) if pt[i] >= 0 else max(0.0, 1.0 - p_paired[i])
            for i in range(n)
        ]
        centroid, _ = fc.centroid()
        out.update({
            "ensemble_free_energy_kcal_mol": float(ensemble_energy),
            "mfe_frequency_in_ensemble": float(fc.pr_structure(structure)),
            "centroid_structure": centroid,
            "confidence": conf,
            "mean_confidence": sum(conf) / n,
            "pair_probabilities": dot,
            "confidence_definition": (
                "Per nucleotide: probability of its MFE pair (paired) or of being unpaired "
                "(unpaired), from the McCaskill partition function."
            ),
        })
    return out
