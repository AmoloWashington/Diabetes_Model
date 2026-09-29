"""REST API for RNA sequences, folding, 3D structures and datasets."""

from __future__ import annotations

import re
from typing import Literal

import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from . import datasets, folding, sequences, structure3d
from .store import get_store

router = APIRouter(prefix="/api", tags=["rna"])
MAX_STRUCTURE_BYTES = 20 * 1024 * 1024


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SequencesIn(_M):
    fasta: str = Field(min_length=1, max_length=2_000_000)
    source: str = Field("paste", max_length=60)


class SequencePatch(_M):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=500)
    tags: list[str] | None = Field(None, max_length=20)


class FoldIn(_M):
    sequence: str | None = Field(None, max_length=folding.MAX_FOLD_LENGTH)
    sequence_id: int | None = None
    temperature_c: float = Field(37.0, ge=0, le=100)


class PredictIn(_M):
    sequence: str | None = Field(None, max_length=structure3d.MAX_DENOVO_LENGTH * 4)
    sequence_id: int | None = None
    method: Literal["auto", "template", "denovo"] = "auto"
    save: bool = True


class FetchIn(_M):
    pdb_id: str = Field(pattern=r"^[0-9][A-Za-z0-9]{3}$")


class CompareIn(_M):
    reference_id: int
    model_id: int


class KaggleIn(_M):
    handle: str = Field(min_length=1, max_length=200)
    kind: Literal["competition", "dataset"] = "competition"
    files: list[str] | None = Field(None, max_length=20)


class BenchmarkIn(_M):
    n_targets: int = Field(20, ge=3, le=200)
    max_identity: float = Field(0.9, ge=0.3, le=1.0)


class ImportSeqIn(_M):
    file: str
    limit: int = Field(100, ge=1, le=500)


# ------------------------------------------------------------ helpers


def _resolve_sequence(sequence: str | None, sequence_id: int | None) -> tuple[str, dict | None]:
    if (sequence is None) == (sequence_id is None):
        raise HTTPException(422, "Provide exactly one of 'sequence' or 'sequence_id'")
    if sequence_id is not None:
        row = get_store().get_sequence(sequence_id)
        if not row:
            raise HTTPException(404, "Sequence not found")
        return row["sequence"], row
    try:
        seq, _ = sequences.normalize(sequence)
    except sequences.SequenceError as e:
        raise HTTPException(422, str(e)) from e
    return seq, None


def _dataset_error(e: Exception) -> HTTPException:
    return HTTPException(404 if "not found" in str(e) else 422, str(e))


# ------------------------------------------------------------ sequences


@router.post("/rna/sequences")
def add_sequences(body: SequencesIn) -> dict:
    return _store_records(body.fasta, body.source)


@router.post("/rna/sequences/upload")
async def upload_sequences(file: UploadFile = File(...)) -> dict:
    data = await file.read(5_000_001)
    if len(data) > 5_000_000:
        raise HTTPException(413, "FASTA file larger than 5 MB")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(422, "File must be UTF-8 text (FASTA)") from e
    return _store_records(text, f"file:{(file.filename or 'upload')[:40]}")


def _store_records(text: str, source: str) -> dict:
    try:
        recs = sequences.parse_fasta(text)
    except sequences.SequenceError as e:
        raise HTTPException(422, str(e)) from e
    store = get_store()
    out = []
    for r in recs:
        row, created = store.add_sequence(r.name, r.sequence, r.description, source)
        out.append({"sequence": row, "created": created, "warnings": r.warnings})
    return {"records": out, "created": sum(o["created"] for o in out), "duplicates": sum(not o["created"] for o in out)}


@router.get("/rna/sequences")
def list_sequences(q: str = Query("", max_length=100), limit: int = Query(100, ge=1, le=500),
                   offset: int = Query(0, ge=0)) -> dict:
    items, total = get_store().list_sequences(q, limit, offset)
    for it in items:
        it["gc_percent"] = sequences.composition(it["sequence"])["gc_percent"]
    return {"items": items, "total": total}


@router.get("/rna/sequences/{sid}")
def get_sequence(sid: int) -> dict:
    row = get_store().get_sequence(sid)
    if not row:
        raise HTTPException(404, "Sequence not found")
    row["composition"] = sequences.composition(row["sequence"])
    row["structures"] = get_store().list_structures(sid)
    return row


@router.patch("/rna/sequences/{sid}")
def patch_sequence(sid: int, body: SequencePatch) -> dict:
    row = get_store().update_sequence(sid, body.name, body.description, body.tags)
    if not row:
        raise HTTPException(404, "Sequence not found")
    return row


@router.delete("/rna/sequences/{sid}")
def delete_sequence(sid: int) -> dict:
    if not get_store().delete_sequence(sid):
        raise HTTPException(404, "Sequence not found")
    return {"deleted": sid}


# ------------------------------------------------------------ folding & prediction


@router.post("/rna/fold")
async def fold(body: FoldIn) -> dict:
    seq, _ = _resolve_sequence(body.sequence, body.sequence_id)
    return await run_in_threadpool(folding.fold, seq, body.temperature_c)


def predict_structure(seq: str, method: str = "auto") -> dict:
    """Shared by the REST API and the AI assistant."""
    result = None
    if method in ("auto", "template"):
        lib = datasets.template_library()
        result = structure3d.predict_template(seq, lib) if lib else None
        if result is None and method == "template":
            raise ValueError(
                "No template covers this sequence. Load structure labels (e.g. the Kaggle Stanford "
                "RNA 3D Folding dataset) in the Dataset Explorer, or use the de novo method."
            )
    if result is None:
        result = structure3d.predict_denovo(seq)
    return result


@router.post("/rna/predict")
async def predict(body: PredictIn) -> dict:
    seq, row = _resolve_sequence(body.sequence, body.sequence_id)
    try:
        result = await run_in_threadpool(predict_structure, seq, body.method)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    remarks = [f"GlucoLab RNA model: {result['method']}", "B-factor = per-residue confidence x 100"]
    if "template_id" in result:
        remarks.append(f"Template {result['template_id']} identity {result['identity']:.2f}")
    pdb = structure3d.write_pdb(seq, result["coords"], result["confidence"], remarks=remarks)
    meta = {k: v for k, v in result.items() if k not in ("coords", "confidence")}
    out = {"pdb": pdb, "confidence": result["confidence"], "sequence": seq, **meta}
    if body.save:
        name = f"{row['name'] if row else 'query'} · {'template' if 'template_id' in result else 'de novo'}"
        saved = get_store().add_structure(row["id"] if row else None, name, "predicted", result["method"],
                                          pdb, result["confidence"], meta)
        out["structure_id"] = saved["id"]
    return out


# ------------------------------------------------------------ structures


@router.get("/rna/structures")
def list_structures(sequence_id: int | None = None) -> dict:
    return {"items": get_store().list_structures(sequence_id)}


@router.get("/rna/structures/{stid}")
def get_structure(stid: int) -> dict:
    st = get_store().get_structure(stid)
    if not st:
        raise HTTPException(404, "Structure not found")
    return st


@router.delete("/rna/structures/{stid}")
def delete_structure(stid: int) -> dict:
    if not get_store().delete_structure(stid):
        raise HTTPException(404, "Structure not found")
    return {"deleted": stid}


def _register_structure_text(text: str, name: str, kind: str, method: str) -> dict:
    if "ATOM" not in text and "_atom_site" not in text:
        raise HTTPException(422, "Not a PDB or mmCIF coordinate file")
    meta: dict = {"format": "mmcif" if "_atom_site" in text else "pdb"}
    if meta["format"] == "pdb":
        chains = structure3d.parse_pdb_c1(text)
        meta["rna_chains"] = {c: {"length": len(v["sequence"]), "sequence": v["sequence"]} for c, v in chains.items()}
    return get_store().add_structure(None, name, kind, method, text, None, meta)


@router.post("/rna/structures/upload")
async def upload_structure(file: UploadFile = File(...), name: str = Form("")) -> dict:
    data = await file.read(MAX_STRUCTURE_BYTES + 1)
    if len(data) > MAX_STRUCTURE_BYTES:
        raise HTTPException(413, "Structure file larger than 20 MB")
    fname = file.filename or "structure"
    if not re.search(r"\.(pdb|ent|cif|mmcif)$", fname, re.I):
        raise HTTPException(422, "Upload a .pdb or .cif file")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(422, "Structure file must be text") from e
    return _register_structure_text(text, name or fname, "uploaded", "experimental (uploaded)")


@router.post("/rna/structures/fetch")
async def fetch_structure(body: FetchIn) -> dict:
    pid = body.pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pid}.pdb"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            r = await client.get(url)
    except httpx.HTTPError as e:
        raise HTTPException(503, f"Could not reach RCSB PDB: {e.__class__.__name__}") from e
    if r.status_code == 404:
        raise HTTPException(404, f"PDB entry {pid} not found (or not available in PDB format; try uploading mmCIF)")
    if r.status_code != 200:
        raise HTTPException(502, f"RCSB returned HTTP {r.status_code}")
    if len(r.content) > MAX_STRUCTURE_BYTES:
        raise HTTPException(413, "Entry too large")
    return _register_structure_text(r.text, pid, "pdb", f"experimental (RCSB PDB {pid})")


@router.post("/rna/structures/compare")
def compare(body: CompareIn) -> dict:
    store = get_store()
    ref, mod = store.get_structure(body.reference_id), store.get_structure(body.model_id)
    if not ref or not mod:
        raise HTTPException(404, "Structure not found")
    cr, cm = structure3d.parse_pdb_c1(ref["pdb"]), structure3d.parse_pdb_c1(mod["pdb"])
    if not cr or not cm:
        raise HTTPException(422, "Both structures need C1' atoms in PDB format")
    a = max(cr.values(), key=lambda c: len(c["sequence"]))
    b = max(cm.values(), key=lambda c: len(c["sequence"]))
    tmpl = structure3d.Template("ref", a["sequence"], a["coords"])
    aln = structure3d.align_to_template(b["sequence"], tmpl)
    m = aln["mapping"]
    keep = m >= 0
    if keep.sum() < 3:
        raise HTTPException(422, "Sequences do not align")
    P = b["coords"][keep]
    Q = a["coords"][m[keep]]
    ref_full = np.full((len(a["sequence"]), 3), np.nan)
    pred_full = np.full_like(ref_full, np.nan)
    ref_full[:] = a["coords"]
    pred_full[m[keep]] = P
    return {
        "reference_length": len(a["sequence"]), "model_length": len(b["sequence"]),
        "aligned": int(keep.sum()), "sequence_identity": aln["identity"],
        "tm_score": structure3d.tm_score(pred_full, ref_full),
        "rmsd_A": structure3d.rmsd(P, Q),
        "tm_score_definition": "TM-score normalised by reference length, RNA d0 of US-align (Zhang et al. 2022)",
    }


@router.get("/rna/structures/{stid}/py3dmol")
def py3dmol_snippet(stid: int) -> dict:
    st = get_store().get_structure(stid)
    if not st:
        raise HTTPException(404, "Structure not found")
    code = (
        "# pip install py3Dmol  (Jupyter)\n"
        "import py3Dmol\n"
        f"pdb = open('glucolab_structure_{stid}.pdb').read()\n"
        "view = py3Dmol.view(width=800, height=600)\n"
        "view.addModel(pdb, 'pdb')\n"
        "view.setStyle({'sphere': {'radius': 1.2, 'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 0, 'max': 100}}})\n"
        "view.zoomTo()\n"
        "view.show()\n"
    )
    return {"python": code, "pdb_filename": f"glucolab_structure_{stid}.pdb"}


# ------------------------------------------------------------ datasets


@router.get("/datasets")
def list_datasets() -> dict:
    return {"items": datasets.list_datasets()}


@router.get("/datasets/kaggle/status")
def kaggle_status() -> dict:
    return {"configured": datasets.kaggle_configured(), "suggested": datasets.SUGGESTED_KAGGLE}


@router.post("/datasets/kaggle")
async def kaggle_import(body: KaggleIn) -> dict:
    try:
        return await run_in_threadpool(datasets.import_kaggle, body.handle, body.kind, body.files)
    except datasets.KaggleUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except datasets.DatasetError as e:
        raise _dataset_error(e) from e
    except Exception as e:  # network/auth/rules errors from kagglehub
        raise HTTPException(502, f"Kaggle download failed: {e}") from e


@router.post("/datasets/upload")
async def dataset_upload(name: str = Form(...), files: list[UploadFile] = File(...)) -> dict:
    payload = []
    total = 0
    for f in files:
        data = await f.read(datasets.MAX_UPLOAD_BYTES + 1)
        total += len(data)
        if total > datasets.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Upload exceeds 200 MB")
        payload.append((f.filename or "file", data))
    try:
        return await run_in_threadpool(datasets.import_upload, name, payload)
    except datasets.DatasetError as e:
        raise _dataset_error(e) from e


@router.get("/datasets/{slug}")
async def dataset_describe(slug: str) -> dict:
    try:
        return await run_in_threadpool(datasets.describe, slug)
    except datasets.DatasetError as e:
        raise _dataset_error(e) from e


@router.get("/datasets/{slug}/rows")
async def dataset_rows(slug: str, file: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500),
                       q: str = Query("", max_length=100)) -> dict:
    try:
        return await run_in_threadpool(datasets.rows, slug, file, offset, limit, q)
    except datasets.DatasetError as e:
        raise _dataset_error(e) from e


@router.get("/datasets/{slug}/rna")
async def dataset_rna(slug: str) -> dict:
    try:
        return await run_in_threadpool(datasets.rna_analytics, slug)
    except datasets.DatasetError as e:
        raise _dataset_error(e) from e


@router.post("/datasets/{slug}/import-sequences")
async def dataset_import_sequences(slug: str, body: ImportSeqIn) -> dict:
    try:
        page = await run_in_threadpool(datasets.rows, slug, body.file, 0, body.limit)
    except datasets.DatasetError as e:
        raise _dataset_error(e) from e
    cols = page["columns"]
    seq_col = next((c for c in cols if "seq" in c.lower()), None)
    id_col = next((c for c in cols if c.lower() in ("target_id", "id", "name", "sequence_id")), None)
    if not seq_col:
        raise HTTPException(422, "No sequence column in this file")
    def rec_name(r, i):
        raw = str(r[cols.index(id_col)]) if id_col else f"{slug}_{i}"
        return re.sub(r"[^\w.:|/()+,-]", "_", raw)[:120] or f"{slug}_{i}"

    fasta = "\n".join(
        f">{rec_name(r, i)}\n{r[cols.index(seq_col)]}"
        for i, r in enumerate(page["rows"]) if r[cols.index(seq_col)]
    )
    return _store_records(fasta, f"dataset:{slug}")


@router.delete("/datasets/{slug}")
def dataset_delete(slug: str) -> dict:
    try:
        datasets.delete_dataset(slug)
    except datasets.DatasetError as e:
        raise _dataset_error(e) from e
    return {"deleted": slug}


@router.post("/datasets/benchmark")
async def dataset_benchmark(body: BenchmarkIn) -> dict:
    lib = datasets.template_library()
    try:
        return await run_in_threadpool(structure3d.benchmark_templates, lib, body.n_targets, body.max_identity)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
