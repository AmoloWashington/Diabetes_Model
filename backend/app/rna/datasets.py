"""Dataset registry: Kaggle import, file upload, exploration and RNA analytics.

Datasets live under ``$GLUCOLAB_DATA_DIR/datasets/<slug>/`` with a
``meta.json``. Kaggle downloads use ``kagglehub`` (official Kaggle client);
credentials are read by kagglehub from ``KAGGLE_API_TOKEN``, or
``KAGGLE_USERNAME`` + ``KAGGLE_KEY``, or ``~/.kaggle/kaggle.json``.
Competition data requires accepting the competition rules on kaggle.com first.

Structure label files in the Stanford RNA 3D Folding format (columns ``ID``,
``resname``, ``resid``, ``x_1``, ``y_1``, ``z_1``; ID = ``<target_id>_<resid>``)
are converted into a template library for template-based 3D prediction.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .store import DEFAULT_DB
from .structure3d import Template

DATA_DIR = DEFAULT_DB.parent / "datasets"
ALLOWED_EXT = {".csv", ".tsv", ".txt", ".fasta", ".fa", ".fna", ".pdb", ".cif", ".json"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_FILES = 50
SLUG_RE = re.compile(r"[^a-z0-9._-]+")

SUGGESTED_KAGGLE = [
    {
        "handle": "stanford-rna-3d-folding",
        "kind": "competition",
        "title": "Stanford RNA 3D Folding",
        "description": "RNA sequences with experimentally determined C1' coordinates (train/validation labels). "
                       "Accept the competition rules on kaggle.com before downloading.",
    },
    {
        "handle": "stanford-ribonanza-rna-folding",
        "kind": "competition",
        "title": "Stanford Ribonanza RNA Folding",
        "description": "Chemical-mapping reactivity profiles (2A3, DMS) for RNA sequences; large download.",
    },
]

_lock = threading.Lock()
_template_cache: dict[str, list[Template]] = {}


class DatasetError(ValueError):
    pass


class KaggleUnavailable(Exception):
    pass


def _slug(name: str) -> str:
    s = SLUG_RE.sub("-", name.strip().lower()).strip("-.")[:60]
    if not s:
        raise DatasetError("Invalid dataset name")
    return s


def _safe_filename(name: str) -> str:
    base = Path(name).name
    if not base or base.startswith(".") or Path(base).suffix.lower() not in ALLOWED_EXT:
        raise DatasetError(f"Unsupported file {name!r}; allowed: {', '.join(sorted(ALLOWED_EXT))}")
    return base


def _dir(slug: str) -> Path:
    d = DATA_DIR / _slug(slug)
    if not d.is_dir():
        raise DatasetError(f"Dataset {slug!r} not found")
    return d


def _write_meta(d: Path, meta: dict) -> None:
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def list_datasets() -> list[dict]:
    if not DATA_DIR.is_dir():
        return []
    out = []
    for d in sorted(DATA_DIR.iterdir()):
        m = d / "meta.json"
        if d.is_dir() and m.is_file():
            out.append(json.loads(m.read_text()))
    return sorted(out, key=lambda x: -x.get("created_at", 0))


def _register(slug: str, title: str, source: str, files: list[Path], extra: dict | None = None) -> dict:
    d = DATA_DIR / slug
    meta = {
        "slug": slug, "title": title, "source": source, "created_at": time.time(),
        "files": [{"name": f.name, "bytes": f.stat().st_size} for f in files],
        **(extra or {}),
    }
    _write_meta(d, meta)
    _template_cache.pop(slug, None)
    return meta


def import_upload(name: str, files: list[tuple[str, bytes]]) -> dict:
    if not files:
        raise DatasetError("No files uploaded")
    if len(files) > MAX_FILES:
        raise DatasetError(f"At most {MAX_FILES} files per dataset")
    total = sum(len(b) for _, b in files)
    if total > MAX_UPLOAD_BYTES:
        raise DatasetError(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    slug = _slug(name)
    with _lock:
        d = DATA_DIR / slug
        if d.exists():
            raise DatasetError(f"Dataset {slug!r} already exists")
        d.mkdir(parents=True)
        try:
            paths = []
            for fname, data in files:
                p = d / _safe_filename(fname)
                p.write_bytes(data)
                paths.append(p)
            return _register(slug, name, "upload", paths)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise


def kaggle_configured() -> bool:
    try:
        from kagglehub.config import get_kaggle_credentials

        return get_kaggle_credentials() is not None
    except Exception:
        return False


def import_kaggle(handle: str, kind: str = "competition", files: list[str] | None = None,
                  downloader: Callable[..., str] | None = None) -> dict:
    """Download a Kaggle competition or dataset into the registry.

    ``downloader`` is injectable for tests; by default kagglehub is used."""
    if kind not in ("competition", "dataset"):
        raise DatasetError("kind must be 'competition' or 'dataset'")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?(/versions/\d+)?", handle):
        raise DatasetError("Invalid Kaggle handle")
    if kind == "dataset" and "/" not in handle:
        raise DatasetError("Dataset handles look like 'owner/dataset'")
    if downloader is None:
        if not kaggle_configured():
            raise KaggleUnavailable(
                "Kaggle credentials not found. Create an API token at kaggle.com/settings and set "
                "KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY) in .env, or place kaggle.json in ~/.kaggle/."
            )
        import kagglehub

        downloader = kagglehub.competition_download if kind == "competition" else kagglehub.dataset_download

    slug = _slug(("kaggle-" + handle).replace("/", "-"))
    with _lock:
        d = DATA_DIR / slug
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    try:
        srcs: list[Path] = []
        if files:
            for f in files:
                srcs.append(Path(downloader(handle, path=f)))
        else:
            root = Path(downloader(handle))
            srcs = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        copied = []
        for s in srcs:
            if s.suffix.lower() not in ALLOWED_EXT:
                continue
            dst = d / s.name
            shutil.copy2(s, dst)
            copied.append(dst)
        if not copied:
            raise DatasetError("The Kaggle download contained no supported files")
        return _register(slug, handle, f"kaggle:{kind}", copied, {"kaggle_handle": handle, "kaggle_kind": kind})
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def delete_dataset(slug: str) -> None:
    d = _dir(slug)
    with _lock:
        shutil.rmtree(d)
        _template_cache.pop(d.name, None)


# ------------------------------------------------------------ exploration


def _read_table(p: Path, nrows: int | None = None) -> pd.DataFrame:
    sep = "\t" if p.suffix.lower() in (".tsv", ".txt") else ","
    return pd.read_csv(p, sep=sep, nrows=nrows, low_memory=False)


def _is_table(p: Path) -> bool:
    return p.suffix.lower() in (".csv", ".tsv", ".txt")


def describe(slug: str) -> dict:
    d = _dir(slug)
    meta = json.loads((d / "meta.json").read_text())
    files = []
    for f in meta["files"]:
        p = d / f["name"]
        info = {"name": f["name"], "bytes": f["bytes"], "type": p.suffix.lower().lstrip(".")}
        if _is_table(p):
            try:
                df = _read_table(p)
                info.update(_table_summary(df))
            except Exception as e:  # malformed file: report, don't crash
                info["error"] = f"Could not parse: {e}"
        files.append(info)
    meta["files"] = files
    meta["n_templates"] = len(templates_for(slug))
    return meta


def _table_summary(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns[:200]:
        s = df[c]
        col = {"name": str(c), "dtype": str(s.dtype), "missing": int(s.isna().sum()),
               "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            v = s.replace([np.inf, -np.inf], np.nan).dropna()
            v = v[np.abs(v) < 1e17]  # Kaggle RNA labels use -1e18 for missing coordinates
            if len(v):
                col.update({"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                            "std": float(v.std()) if len(v) > 1 else 0.0})
        cols.append(col)
    kind = "rna_sequences" if _sequence_column(df) else "structure_labels" if _is_label_table(df) else "table"
    return {"rows": int(len(df)), "columns": cols, "kind": kind}


def rows(slug: str, filename: str, offset: int = 0, limit: int = 50, q: str = "") -> dict:
    p = _dir(slug) / _safe_filename(filename)
    if not p.is_file() or not _is_table(p):
        raise DatasetError("Not a tabular file in this dataset")
    df = _read_table(p)
    if q:
        mask = df.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
        df = df[mask]
    total = len(df)
    page = df.iloc[offset: offset + min(limit, 500)]
    page = page.replace([np.inf, -np.inf], np.nan).astype(object).where(page.notna(), None)
    return {"total": int(total), "offset": offset, "columns": [str(c) for c in df.columns],
            "rows": page.values.tolist()}


def _sequence_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "seq" in str(c).lower() and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            sample = df[c].dropna().astype(str).head(50)
            if len(sample) and np.mean([bool(re.fullmatch(r"[ACGUTNacgutn]+", s)) for s in sample]) > 0.9:
                return str(c)
    return None


def _is_label_table(df: pd.DataFrame) -> bool:
    return {"ID", "resname", "resid", "x_1", "y_1", "z_1"}.issubset(df.columns)


def rna_analytics(slug: str) -> dict:
    d = _dir(slug)
    meta = json.loads((d / "meta.json").read_text())
    out = []
    for f in meta["files"]:
        p = d / f["name"]
        if not _is_table(p):
            continue
        df = _read_table(p)
        col = _sequence_column(df)
        if not col:
            continue
        seqs = df[col].dropna().astype(str).str.upper().str.replace("T", "U")
        lengths = seqs.str.len()
        gc = seqs.apply(lambda s: 100.0 * (s.count("G") + s.count("C")) / max(len(s), 1))
        comp = {b: int(seqs.str.count(b).sum()) for b in "ACGU"}
        hist_counts, edges = np.histogram(lengths, bins=min(30, max(5, int(np.sqrt(len(lengths))))))
        entry = {
            "file": f["name"], "sequence_column": col, "n_sequences": int(len(seqs)),
            "length": {"min": int(lengths.min()), "max": int(lengths.max()), "mean": float(lengths.mean()),
                       "median": float(lengths.median())},
            "length_histogram": {"counts": hist_counts.tolist(), "edges": edges.tolist()},
            "gc_percent": {"mean": float(gc.mean()), "std": float(gc.std()) if len(gc) > 1 else 0.0},
            "composition": comp,
        }
        if "temporal_cutoff" in df.columns:
            years = pd.to_datetime(df["temporal_cutoff"], errors="coerce").dt.year.dropna().astype(int)
            entry["by_year"] = {str(k): int(v) for k, v in years.value_counts().sort_index().items()}
        out.append(entry)
    return {"slug": slug, "sequence_files": out}


# ------------------------------------------------------------ templates


def _labels_to_templates(df: pd.DataFrame) -> list[Template]:
    df = df.copy()
    df["target_id"] = df["ID"].astype(str).str.rsplit("_", n=1).str[0]
    out = []
    for tid, g in df.groupby("target_id", sort=False):
        g = g.sort_values("resid")
        seq = "".join(str(r).upper()[:1] if str(r).upper()[:1] in "ACGU" else "N" for r in g["resname"])
        xyz = np.array(g[["x_1", "y_1", "z_1"]].to_numpy(dtype=float), copy=True)
        xyz[(np.abs(xyz) > 1e17).any(axis=1)] = np.nan
        if np.isfinite(xyz).all(axis=1).sum() >= 4:
            out.append(Template(target_id=str(tid), sequence=seq, coords=xyz))
    return out


def templates_for(slug: str) -> list[Template]:
    if slug in _template_cache:
        return _template_cache[slug]
    d = _dir(slug)
    meta = json.loads((d / "meta.json").read_text())
    tmpls: list[Template] = []
    for f in meta["files"]:
        p = d / f["name"]
        if not _is_table(p):
            continue
        try:
            head = _read_table(p, nrows=5)
        except Exception:
            continue
        if _is_label_table(head):
            tmpls.extend(_labels_to_templates(_read_table(p)))
    _template_cache[slug] = tmpls
    return tmpls


def template_library() -> list[Template]:
    lib: list[Template] = []
    for m in list_datasets():
        lib.extend(templates_for(m["slug"]))
    return lib
