"""Deterministic DNA sequence analysis for the molecular lab.

* Genetic code: NCBI translation table 1 (the standard code).
* Melting temperature: two basic textbook approximations that ignore salt
  and nearest-neighbour effects, and are labelled as such:
    Wallace rule (< 14 nt):  Tm = 2 (A+T) + 4 (G+C)                   degC
    Basic formula (>= 14 nt): Tm = 64.9 + 41 (G+C - 16.4) / N          degC
* B-DNA geometry used by the 3D renderer: ~10.5 bp per helical turn,
  0.34 nm rise per base pair, ~2.0 nm diameter, right-handed.

Human insulin chains (mature hormone, UniProt P01308):
    B chain (30 aa): FVNQHLCGSHLVEALYLVCGERGFFYTPKT
    A chain (21 aa): GIVEQCCTSICSLYQLENYCN
Preproinsulin is 110 aa: a 24-aa signal peptide, the B chain, a 35-aa
connecting region (the 31-aa C-peptide flanked by the dibasic cleavage
sites Arg-Arg and Lys-Arg), and the A chain.
"""

from __future__ import annotations

import re

_BASES = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE: dict[str, str] = {
    a + b + c: _AA[16 * i + 4 * j + k]
    for i, a in enumerate(_BASES)
    for j, b in enumerate(_BASES)
    for k, c in enumerate(_BASES)
}

INSULIN_B_CHAIN = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
INSULIN_A_CHAIN = "GIVEQCCTSICSLYQLENYCN"

# One fixed codon per amino acid, used only to build a clearly-labelled
# synthetic demonstration sequence (not the native INS gene sequence).
_DEMO_CODON = {
    "A": "GCC", "R": "CGC", "N": "AAC", "D": "GAC", "C": "TGC", "Q": "CAG",
    "E": "GAG", "G": "GGC", "H": "CAC", "I": "ATC", "L": "CTG", "K": "AAG",
    "M": "ATG", "F": "TTC", "P": "CCC", "S": "AGC", "T": "ACC", "W": "TGG",
    "Y": "TAC", "V": "GTG", "*": "TGA",
}

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def back_translate(protein: str) -> str:
    try:
        return "".join(_DEMO_CODON[aa] for aa in protein.upper())
    except KeyError as e:
        raise ValueError(f"Unknown amino acid {e.args[0]!r}") from None


DEMO_SEQUENCE = "ATG" + back_translate(INSULIN_B_CHAIN) + "TGA"
DEMO_SEQUENCE_NOTE = (
    "Synthetic back-translation of the human insulin B chain with a start codon "
    "and a stop codon, using one fixed codon per amino acid. It translates to the "
    "true B-chain sequence but is NOT the native INS gene sequence."
)


def clean(seq: str) -> str:
    s = re.sub(r"\s+", "", seq).upper().replace("U", "T")
    if not s:
        raise ValueError("Empty sequence")
    bad = sorted(set(s) - set("ACGTN"))
    if bad:
        raise ValueError(f"Invalid characters in DNA sequence: {''.join(bad)}")
    return s


def reverse_complement(seq: str) -> str:
    return clean(seq).translate(_COMPLEMENT)[::-1]


def translate(seq: str, frame: int = 0) -> str:
    s = clean(seq)[frame:]
    return "".join(
        CODON_TABLE.get(s[i:i + 3], "X") for i in range(0, len(s) - len(s) % 3, 3)
    )


def melting_temperature(seq: str) -> dict:
    s = clean(seq)
    n = len(s)
    gc = s.count("G") + s.count("C")
    at = s.count("A") + s.count("T")
    if n < 14:
        return {"tm_c": 2.0 * at + 4.0 * gc, "method": "Wallace rule (< 14 nt)"}
    return {"tm_c": 64.9 + 41.0 * (gc - 16.4) / n, "method": "Basic GC formula (>= 14 nt)"}


def analyze(seq: str) -> dict:
    s = clean(seq)
    n = len(s)
    counts = {b: s.count(b) for b in "ACGTN"}
    acgt = n - counts["N"]
    orfs = []
    for frame in range(3):
        prot = translate(s, frame)
        for m in re.finditer(r"M[^*]*\*", prot):
            if m.end() - m.start() - 1 >= 5:
                orfs.append({
                    "frame": frame + 1,
                    "start_nt": frame + 3 * m.start() + 1,
                    "end_nt": frame + 3 * m.end(),
                    "length_aa": m.end() - m.start() - 1,
                    "protein": m.group(0)[:-1],
                })
    orfs.sort(key=lambda o: -o["length_aa"])
    return {
        "length_nt": n,
        "counts": counts,
        "gc_percent": 100.0 * (counts["G"] + counts["C"]) / acgt if acgt else None,
        "reverse_complement": reverse_complement(s),
        "translations": {f"frame_{k + 1}": translate(s, k) for k in range(3)},
        "orfs": orfs[:10],
        "melting_temperature": melting_temperature(s),
        "contains_insulin_b_chain": INSULIN_B_CHAIN in "".join(translate(s, k) for k in range(3)),
        "helix_geometry": {
            "form": "B-DNA", "bp_per_turn": 10.5, "rise_nm_per_bp": 0.34,
            "diameter_nm": 2.0, "handedness": "right",
            "length_nm": n * 0.34, "turns": n / 10.5,
        },
    }
