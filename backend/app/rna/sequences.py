"""RNA sequence parsing and validation.

Accepted alphabet: A, C, G, U. T is converted to U (reported as a warning).
IUPAC ambiguity codes (N, R, Y, S, W, K, M, B, D, H, V) are accepted for
storage but flagged; structure prediction treats them as unpairable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CANONICAL = set("ACGU")
AMBIGUOUS = set("NRYSWKMBDHV")
MAX_LENGTH = 10_000
NAME_RE = re.compile(r"^[\w .:|/()+,-]{1,120}$")


class SequenceError(ValueError):
    pass


@dataclass
class ParsedSequence:
    name: str
    sequence: str
    description: str = ""
    warnings: list[str] = field(default_factory=list)


def normalize(raw: str) -> tuple[str, list[str]]:
    """Uppercase, strip whitespace/digits, T->U. Raises SequenceError if invalid."""
    s = re.sub(r"[\s\d]+", "", raw).upper()
    warnings: list[str] = []
    if not s:
        raise SequenceError("Empty sequence")
    if len(s) > MAX_LENGTH:
        raise SequenceError(f"Sequence longer than {MAX_LENGTH} nt")
    if "T" in s:
        s = s.replace("T", "U")
        warnings.append("T converted to U (DNA alphabet detected).")
    bad = sorted(set(s) - CANONICAL - AMBIGUOUS - {"-"})
    if bad:
        raise SequenceError(f"Invalid characters: {''.join(bad)}")
    if "-" in s:
        s = s.replace("-", "")
        warnings.append("Gap characters '-' removed.")
        if not s:
            raise SequenceError("Sequence contained only gaps")
    amb = sorted(set(s) & AMBIGUOUS)
    if amb:
        warnings.append(f"Ambiguity codes present ({''.join(amb)}); treated as unpairable in folding.")
    return s, warnings


def parse_fasta(text: str, max_records: int = 500) -> list[ParsedSequence]:
    """Parse FASTA (or a single raw sequence without header)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise SequenceError("No sequence data")
    if not text.startswith(">"):
        seq, w = normalize(text)
        return [ParsedSequence(name="sequence_1", sequence=seq, warnings=w)]
    records: list[ParsedSequence] = []
    for i, block in enumerate(text.split("\n>")):
        block = block.lstrip(">")
        header, _, body = block.partition("\n")
        header = header.strip()
        name, _, desc = header.partition(" ")
        name = name or f"sequence_{i + 1}"
        if not NAME_RE.match(name):
            raise SequenceError(f"Invalid record name {name[:40]!r}")
        try:
            seq, w = normalize(body)
        except SequenceError as e:
            raise SequenceError(f"Record {name!r}: {e}") from None
        records.append(ParsedSequence(name=name, sequence=seq, description=desc.strip()[:500], warnings=w))
        if len(records) > max_records:
            raise SequenceError(f"More than {max_records} records")
    return records


def composition(seq: str) -> dict:
    n = len(seq)
    counts = {b: seq.count(b) for b in "ACGU"}
    other = n - sum(counts.values())
    gc = counts["G"] + counts["C"]
    canon = n - other
    return {
        "length": n,
        "counts": {**counts, "other": other},
        "gc_percent": 100.0 * gc / canon if canon else None,
    }
