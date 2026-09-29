"""RNA 3D structure: prediction, PDB I/O and structural comparison.

Coordinates are represented at one bead per nucleotide, the C1' atom, the
representation used by the Stanford RNA 3D Folding Kaggle competition.

Prediction methods
------------------
1. Template-based modelling (TBM). The query is aligned to every structure in
   the loaded template library (e.g. Kaggle training labels); C1' coordinates
   of aligned positions are copied from the best template and gaps are
   interpolated. Per-residue confidence is a heuristic of alignment quality
   (identity x match type) and is reported together with an empirical
   leave-one-out benchmark so users can see how well it is calibrated.

2. Coarse-grained de novo model (fallback when no template is found). The
   ViennaRNA MFE secondary structure is embedded in 3D by minimising a stress
   function whose targets are: consecutive C1'-C1' distances, Watson-Crick
   C1'-C1' distances (~10.5 A), ideal A-form helix geometry inside stems
   (rise 2.81 A and twist 32.7 deg per base pair, i.e. ~11 bp per turn), and
   steric repulsion. The helix radius of the C1' atoms and the consecutive
   distance are approximations. Per-residue confidence is the ViennaRNA
   ensemble probability of the secondary-structure state, which measures
   secondary-structure certainty only; tertiary packing of such models is
   low-accuracy and is labelled accordingly.

Structural comparison
---------------------
TM-score with a fixed residue correspondence, using the RNA d0 of US-align
(Zhang C, Shine M, Pyle AM, Zhang Y. Nat Methods 19:1109-1115, 2022):
d0 = 0.6 sqrt(L - 0.5) - 2.5 for L >= 30, and 0.3/0.4/0.5/0.6/0.7 for
L < 12, 12-15, 16-19, 20-23, 24-29.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from . import folding

# Geometry (Angstrom)
RISE = 2.81
TWIST = math.radians(32.7)
C1_RADIUS = 8.7  # approximate radial distance of C1' atoms from the A-form helix axis
PAIR_C1C1 = 10.5  # C1'-C1' distance across a Watson-Crick pair
CONSEC = 5.9  # approximate consecutive C1'-C1' distance
MIN_NONBONDED = 6.0
# Angular offset between paired C1' atoms that reproduces PAIR_C1C1 at C1_RADIUS
PAIR_PHASE = 2 * math.asin(PAIR_C1C1 / (2 * C1_RADIUS))
MAX_DENOVO_LENGTH = 1000


# ============================================================ PDB I/O


def write_pdb(seq: str, coords: np.ndarray, bfactors: list[float] | None = None, chain: str = "A",
              remarks: list[str] | None = None) -> str:
    """C1'-only PDB with CONECT records linking consecutive residues.

    B-factor column holds confidence x 100 (pLDDT-like convention)."""
    lines = [f"REMARK   1 {r[:68]}" for r in (remarks or [])]
    serial = 0
    serials: list[int | None] = []
    for i, (res, xyz) in enumerate(zip(seq, coords)):
        if not np.all(np.isfinite(xyz)):
            serials.append(None)
            continue
        serial += 1
        serials.append(serial)
        b = 100.0 * bfactors[i] if bfactors is not None else 0.0
        resn = res if res in "ACGU" else "N"
        lines.append(
            f"ATOM  {serial:5d}  C1' {resn:>3s} {chain:1s}{i + 1:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.00:6.2f}{b:6.2f}           C"
        )
    for a, b in zip(serials, serials[1:]):
        if a is not None and b is not None:
            lines.append(f"CONECT{a:5d}{b:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


_ATOM_RE = re.compile(r"^(ATOM  |HETATM)")


def parse_pdb_c1(pdb_text: str) -> dict[str, dict]:
    """Extract C1' coordinates per chain from a PDB file (first model only)."""
    chains: dict[str, dict] = {}
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not _ATOM_RE.match(line) or len(line) < 54:
            continue
        name = line[12:16].strip()
        if name not in ("C1'", "C1*"):
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        resn = line[17:20].strip()
        ch = line[21]
        try:
            resi = int(line[22:26])
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        c = chains.setdefault(ch, {"resnames": [], "resids": [], "coords": []})
        if c["resids"] and c["resids"][-1] == resi:
            continue  # insertion code / duplicate
        c["resnames"].append(resn)
        c["resids"].append(resi)
        c["coords"].append(xyz)
    for c in chains.values():
        c["coords"] = np.array(c["coords"], dtype=float)
        c["sequence"] = "".join(r[-1] if r and r[-1] in "ACGU" else "N" for r in c["resnames"])
    return chains


# ============================================================ comparison


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |R P + t - Q| (rows are points)."""
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    R, t = kabsch(P, Q)
    return float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))


def tm_d0(L: int) -> float:
    if L < 12:
        return 0.3
    if L < 16:
        return 0.4
    if L < 20:
        return 0.5
    if L < 24:
        return 0.6
    if L < 30:
        return 0.7
    return 0.6 * math.sqrt(L - 0.5) - 2.5


def tm_score(pred: np.ndarray, ref: np.ndarray) -> float:
    """TM-score of pred against ref with a fixed 1:1 correspondence (NaN rows skipped).

    Normalised by the number of reference residues with coordinates. Uses the
    TM-score heuristic search: superpositions seeded from fragments of several
    lengths, each refined iteratively on residues within a distance cut-off."""
    ok = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(ref), axis=1)
    L = int(np.all(np.isfinite(ref), axis=1).sum())
    P, Q = pred[ok], ref[ok]
    n = len(P)
    if n < 3 or L == 0:
        return 0.0
    d0 = tm_d0(L)
    best = 0.0
    for frag in sorted({n, max(n // 2, 3), max(n // 4, 3), min(n, 8), 4} - {0}):
        step = max(1, frag // 2)
        for start in range(0, n - frag + 1, step):
            idx = np.arange(start, start + frag)
            for _ in range(20):
                R, t = kabsch(P[idx], Q[idx])
                d = np.linalg.norm(P @ R.T + t - Q, axis=1)
                best = max(best, float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / L))
                new = np.flatnonzero(d < max(d0, 1.0) + 0.5)
                if len(new) < 3 or np.array_equal(new, idx):
                    break
                idx = new
    return best


# ============================================================ de novo


def _ideal_stem(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Ideal C1' coordinates for an m-bp stem: strand 1 (5'->3') and its partners."""
    k = np.arange(m)
    th = k * TWIST
    s1 = np.c_[C1_RADIUS * np.cos(th), C1_RADIUS * np.sin(th), k * RISE]
    s2 = np.c_[C1_RADIUS * np.cos(th + PAIR_PHASE), C1_RADIUS * np.sin(th + PAIR_PHASE), k * RISE]
    return s1, s2


def predict_denovo(seq: str, seed: int = 0) -> dict:
    n = len(seq)
    if n < 4:
        raise ValueError("Sequence too short for 3D modelling (need >= 4 nt)")
    if n > MAX_DENOVO_LENGTH:
        raise ValueError(f"De novo modelling is limited to {MAX_DENOVO_LENGTH} nt")
    f = folding.fold(seq)
    pt = folding.pair_table(f["mfe_structure"])

    # Distance restraints (i, j, target, weight)
    I, J, T, Wt = [], [], [], []

    def add(i, j, d, w):
        I.append(i); J.append(j); T.append(d); Wt.append(w)

    for i in range(n - 1):
        add(i, i + 1, CONSEC, 10.0)
    for stem in folding.stems(pt):
        s1, s2 = _ideal_stem(len(stem))
        idx = [p[0] for p in stem] + [p[1] for p in stem]
        pts = np.vstack([s1, s2])
        D = squareform(pdist(pts))
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                add(idx[a], idx[b], float(D[a, b]), 5.0)
    I, J, T, Wt = map(np.array, (I, J, T, Wt))
    # de-duplicate (keep strongest weight)
    key = np.minimum(I, J) * n + np.maximum(I, J)
    _, first = np.unique(key[::-1], return_index=True)
    sel = len(key) - 1 - first
    I, J, T, Wt = I[sel], J[sel], T[sel], Wt[sel]

    iu, ju = np.triu_indices(n, k=2)

    def energy(x):
        X = x.reshape(n, 3)
        diff = X[I] - X[J]
        d = np.linalg.norm(diff, axis=1) + 1e-9
        r = d - T
        e = np.sum(Wt * r * r)
        g = np.zeros_like(X)
        coef = (2 * Wt * r / d)[:, None] * diff
        np.add.at(g, I, coef)
        np.add.at(g, J, -coef)
        # steric repulsion for non-bonded pairs closer than MIN_NONBONDED
        dv = X[iu] - X[ju]
        dd = np.linalg.norm(dv, axis=1) + 1e-9
        close = dd < MIN_NONBONDED
        if close.any():
            rr = dd[close] - MIN_NONBONDED
            e += np.sum(rr * rr)
            c2 = (2 * rr / dd[close])[:, None] * dv[close]
            np.add.at(g, iu[close], c2)
            np.add.at(g, ju[close], -c2)
        return e, g.ravel()

    rng = np.random.default_rng(seed)
    # initial guess: a loose spiral, which avoids knots better than random init
    t = np.arange(n)
    x0 = np.c_[12 * np.cos(t * 0.35), 12 * np.sin(t * 0.35), t * 1.5] + rng.normal(0, 0.5, (n, 3))
    res = minimize(energy, x0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    X = res.x.reshape(n, 3)
    X -= X.mean(axis=0)

    conf = f.get("confidence") or [0.5] * n
    # tertiary arrangement of such models is uncertain: cap confidence for unpaired nucleotides
    conf = [c if pt[i] >= 0 else min(c, 0.5) for i, c in enumerate(conf)]
    viol = np.abs(np.linalg.norm(X[I] - X[J], axis=1) - T)
    return {
        "method": "coarse-grained de novo (ViennaRNA MFE + A-form restraints)",
        "coords": X,
        "confidence": conf,
        "mean_confidence": float(np.mean(conf)),
        "secondary_structure": f["mfe_structure"],
        "mfe_kcal_mol": f["mfe_kcal_mol"],
        "restraint_rmse_A": float(np.sqrt(np.mean(viol**2))),
        "caveat": "Low-accuracy tertiary model: only secondary structure and helix geometry are restrained.",
    }


# ============================================================ template-based


@dataclass
class Template:
    target_id: str
    sequence: str
    coords: np.ndarray  # (L, 3), NaN where unresolved


def _kmers(s: str, k: int) -> set[str]:
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def _aligner():
    from Bio import Align

    a = Align.PairwiseAligner()
    a.mode = "global"
    a.match_score = 2
    a.mismatch_score = -1
    a.open_gap_score = -5
    a.extend_gap_score = -1
    # end gaps are free (semi-global); attribute names changed in Biopython 1.85
    for new, old in (
        ("open_end_insertion_score", "target_end_open_gap_score"),
        ("extend_end_insertion_score", "target_end_extend_gap_score"),
        ("open_end_deletion_score", "query_end_open_gap_score"),
        ("extend_end_deletion_score", "query_end_extend_gap_score"),
    ):
        setattr(a, new if hasattr(a, new) else old, 0)
    return a


def align_to_template(query: str, tmpl: Template) -> dict:
    aln = _aligner().align(query, tmpl.sequence)[0]
    mapping = np.full(len(query), -1, dtype=int)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        mapping[qs:qe] = np.arange(ts, te)
    aligned = mapping >= 0
    ident = np.array([aligned[i] and query[i] == tmpl.sequence[mapping[i]] for i in range(len(query))])
    return {
        "mapping": mapping,
        "identical": ident,
        "identity": float(ident.sum() / len(query)),
        "coverage": float(aligned.sum() / len(query)),
    }


def _fill_gaps(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation for missing interior positions; extension at ends."""
    X = X.copy()
    known = np.all(np.isfinite(X), axis=1)
    filled = ~known
    idx = np.flatnonzero(known)
    if len(idx) == 0:
        return X, filled
    for d in range(3):
        X[~known, d] = np.interp(np.flatnonzero(~known), idx, X[idx, d])
    # extend termini along the terminal direction with CONSEC spacing
    for end in (0, 1):
        k = idx[0] if end == 0 else idx[-1]
        nb = idx[1] if end == 0 and len(idx) > 1 else (idx[-2] if len(idx) > 1 else None)
        direction = (X[k] - X[nb]) if nb is not None else np.array([1.0, 0, 0])
        direction /= np.linalg.norm(direction) + 1e-9
        rng = range(k - 1, -1, -1) if end == 0 else range(k + 1, len(X))
        for step, i in enumerate(rng, start=1):
            X[i] = X[k] + direction * CONSEC * step
    return X, filled


def predict_template(query: str, library: list[Template], exclude_ids: set[str] | None = None,
                     min_coverage: float = 0.5, top_k: int = 25) -> dict | None:
    """Best template model, or None if no template meets the coverage threshold."""
    exclude_ids = exclude_ids or set()
    cands = [t for t in library if t.target_id not in exclude_ids and len(t.sequence) >= 4]
    if not cands:
        return None
    k = 4 if len(query) >= 20 else 3
    qk = _kmers(query, k)
    scored = sorted(cands, key=lambda t: -len(qk & _kmers(t.sequence, k)) / (len(qk) + 1e-9))[:top_k]
    best = None
    for t in scored:
        a = align_to_template(query, t)
        score = a["identity"] * min(1.0, a["coverage"])
        if best is None or score > best[0]:
            best = (score, t, a)
    if best is None or best[2]["coverage"] < min_coverage:
        return None
    _, t, a = best
    X = np.full((len(query), 3), np.nan)
    m = a["mapping"]
    for i in np.flatnonzero(m >= 0):
        X[i] = t.coords[m[i]]
    X, filled = _fill_gaps(X)
    g = a["identity"]
    conf = [float(g * (0.2 if filled[i] else (1.0 if a["identical"][i] else 0.6))) for i in range(len(query))]
    return {
        "method": "template-based (alignment + C1' coordinate transfer)",
        "template_id": t.target_id,
        "identity": a["identity"],
        "coverage": a["coverage"],
        "coords": X - np.nanmean(X, axis=0),
        "confidence": conf,
        "mean_confidence": float(np.mean(conf)),
        "caveat": "Confidence is a heuristic of alignment quality; see the leave-one-out benchmark for calibration.",
    }


def benchmark_templates(library: list[Template], n_targets: int = 30, max_identity: float = 0.95,
                        seed: int = 0) -> dict:
    """Leave-one-out: predict each target from the rest of the library, excluding
    near-identical templates (identity > max_identity), and score with TM-score."""
    rng = np.random.default_rng(seed)
    usable = [t for t in library if 10 <= len(t.sequence) <= 400 and np.isfinite(t.coords).all(axis=1).sum() >= 10]
    if len(usable) < 3:
        raise ValueError("Need at least 3 structures of 10-400 nt to benchmark")
    picks = rng.choice(len(usable), size=min(n_targets, len(usable)), replace=False)
    rows = []
    for p in picks:
        tgt = usable[p]
        others = [t for t in library if t.target_id != tgt.target_id
                  and align_identity_fast(tgt.sequence, t.sequence) <= max_identity]
        pred = predict_template(tgt.sequence, others)
        if pred is None:
            rows.append({"target_id": tgt.target_id, "length": len(tgt.sequence), "template": None,
                         "identity": 0.0, "mean_confidence": 0.0, "tm_score": None})
            continue
        rows.append({
            "target_id": tgt.target_id, "length": len(tgt.sequence), "template": pred["template_id"],
            "identity": pred["identity"], "mean_confidence": pred["mean_confidence"],
            "tm_score": tm_score(pred["coords"], tgt.coords),
        })
    scored = [r for r in rows if r["tm_score"] is not None]
    conf = np.array([r["mean_confidence"] for r in scored])
    tms = np.array([r["tm_score"] for r in scored])
    corr = float(np.corrcoef(conf, tms)[0, 1]) if len(scored) >= 3 and conf.std() > 0 and tms.std() > 0 else None
    return {
        "n_targets": len(rows),
        "n_with_template": len(scored),
        "mean_tm_score": float(tms.mean()) if len(scored) else None,
        "confidence_tm_correlation": corr,
        "max_template_identity": max_identity,
        "rows": rows,
    }


def align_identity_fast(a: str, b: str) -> float:
    """Cheap upper bound on identity via shared 4-mers, used only to exclude near-duplicates."""
    if len(a) < 8 or len(b) < 8:
        return 1.0 if a == b else 0.0
    ka, kb = _kmers(a, 4), _kmers(b, 4)
    return len(ka & kb) / max(len(ka | kb), 1)
