"""CRISPR-Cas9 guide RNA (gRNA) finder for the standard SpCas9 enzyme.

Educational tool only. It finds candidate target sites in a DNA sequence and flags common
design problems. It does not design therapies and cannot predict off-target effects anywhere
outside the sequence you paste, so it must never be used to plan editing of real organisms.

How SpCas9 targeting works (the rules implemented here):

* SpCas9 cuts next to a protospacer-adjacent motif (PAM) of the form 5'-NGG-3' on the DNA.
  Jinek M, Chylinski K, Fonfara I, Hauer M, Doudna JA, Charpentier E. A programmable dual-RNA-
  guided DNA endonuclease in adaptive bacterial immunity. Science 2012;337:816-821.
* The guide (spacer) is the 20 nt immediately 5' of the PAM, on the same strand as the PAM.
* The blunt double-strand break falls between positions 17 and 18 of the 20-nt protospacer,
  i.e. 3 bp upstream (5') of the PAM.
  Ran FA, Hsu PD, Wright J, Agarwala V, Scott DA, Zhang F. Genome engineering using the
  CRISPR-Cas9 system. Nat Protoc 2013;8:2281-2308.
* Both strands are searched, because a PAM can occur on either.

Simple quality flags (rules of thumb, not a scoring model):

* GC content near 40-60% tends to work better; very low or very high GC is flagged.
* A run of four or more T's (TTTT) in the spacer can act as a Pol III terminator and is flagged.
* Guides whose 20-nt protospacer occurs more than once within the pasted sequence are flagged
  as non-unique here. This is NOT a genome-wide off-target search.

No PAM-site scoring algorithm (such as Doench 2016) is implemented; the flags above are
transparent heuristics so the output can be fully explained.
"""

from __future__ import annotations

from .dna import clean, reverse_complement

PAM = "NGG"
SPACER_LEN = 20
GC_LOW = 40.0
GC_HIGH = 60.0
MAX_SEQUENCE_LEN = 50_000


def _gc_percent(s: str) -> float:
    return 100.0 * (s.count("G") + s.count("C")) / len(s) if s else 0.0


def _find_pam_guides(seq: str, strand: str) -> list[dict]:
    """Find every 5'-NGG-3' PAM with 20 nt of sequence available 5' of it, on one strand.

    Coordinates returned are 0-based indices into `seq` (the strand actually scanned)."""
    guides: list[dict] = []
    n = len(seq)
    # PAM occupies positions [i, i+3); the 'GG' is at i+1, i+2. Spacer is [i-20, i).
    for i in range(SPACER_LEN, n - 2):
        if seq[i + 1] == "G" and seq[i + 2] == "G":
            spacer = seq[i - SPACER_LEN:i]
            if "N" in spacer or "N" in seq[i:i + 3]:
                continue
            guides.append({
                "strand": strand,
                "spacer": spacer,
                "pam": seq[i:i + 3],
                # cut site is between spacer positions 17 and 18 (0-based 16|17): 3 bp 5' of the PAM
                "_spacer_start": i - SPACER_LEN,
                "_pam_start": i,
                "_cut_on_strand": i - 3,
            })
    return guides


def find_guides(seq: str, gc_low: float = GC_LOW, gc_high: float = GC_HIGH) -> dict:
    s = clean(seq)
    n = len(s)
    if n < SPACER_LEN + 3:
        raise ValueError(f"Sequence must be at least {SPACER_LEN + 3} nt to contain a guide and PAM.")
    if n > MAX_SEQUENCE_LEN:
        raise ValueError(f"Sequence must be at most {MAX_SEQUENCE_LEN:,} nt.")
    if not (0 <= gc_low < gc_high <= 100):
        raise ValueError("GC bounds must satisfy 0 <= low < high <= 100.")

    rc = reverse_complement(s)
    fwd = _find_pam_guides(s, "+")
    rev = _find_pam_guides(rc, "-")

    # Count how often each spacer occurs across both strands, to flag non-unique guides.
    counts: dict[str, int] = {}
    for g in fwd + rev:
        counts[g["spacer"]] = counts.get(g["spacer"], 0) + 1

    out: list[dict] = []
    for g in fwd + rev:
        if g["strand"] == "+":
            cut = g["_cut_on_strand"]  # index on the forward strand, break is just 5' of this base
            pam_start, pam_end = g["_pam_start"], g["_pam_start"] + 3
            spacer_start, spacer_end = g["_spacer_start"], g["_spacer_start"] + SPACER_LEN
        else:
            # Map reverse-strand coordinates back onto the forward strand for display.
            cut = n - g["_cut_on_strand"]
            pam_end, pam_start = n - g["_pam_start"], n - (g["_pam_start"] + 3)
            spacer_end, spacer_start = n - g["_spacer_start"], n - (g["_spacer_start"] + SPACER_LEN)

        gc = _gc_percent(g["spacer"])
        flags = []
        if gc < gc_low:
            flags.append(f"Low GC ({gc:.0f}%)")
        elif gc > gc_high:
            flags.append(f"High GC ({gc:.0f}%)")
        if "TTTT" in g["spacer"]:
            flags.append("Poly-T (TTTT) may terminate transcription")
        if counts[g["spacer"]] > 1:
            flags.append("Not unique within this sequence")

        out.append({
            "strand": g["strand"],
            "spacer": g["spacer"],
            "pam": g["pam"],
            "full_target": g["spacer"] + g["pam"],
            "gc_percent": round(gc, 1),
            "spacer_start": spacer_start,  # 0-based, forward-strand, half-open [start, end)
            "spacer_end": spacer_end,
            "pam_start": pam_start,
            "pam_end": pam_end,
            "cut_site": cut,  # blunt break on the forward strand falls just 5' of this index
            "flags": flags,
            "good": not flags,
        })

    out.sort(key=lambda g: (g["cut_site"], g["strand"]))
    good = sum(1 for g in out if g["good"])
    return {
        "length": n,
        "n_guides": len(out),
        "n_good": good,
        "pam": "NGG",
        "spacer_length": SPACER_LEN,
        "gc_window": [gc_low, gc_high],
        "guides": out,
        "note": (
            "SpCas9 (5'-NGG-3' PAM). The cut site is a blunt double-strand break 3 bp "
            "upstream of the PAM. Flags are design rules of thumb; uniqueness is checked only "
            "within this sequence, which is not a genome-wide off-target search."
        ),
    }
