"""RNA sequence management, folding, 3D prediction, datasets and Kaggle import (mocked)."""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rna import datasets, folding, sequences, store, structure3d as s3

HAIRPIN = "GGGCGCAAGCCUAUGCGCUUCGGCGCAUAGGCUUGCGCCC"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    store.set_store(store.Store(tmp_path / "test.db"))
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path / "datasets")
    datasets._template_cache.clear()
    with TestClient(app) as c:
        yield c
    datasets._template_cache.clear()


def _labels_csv(targets: dict[str, tuple[str, np.ndarray]]) -> str:
    rows = []
    for tid, (seq, X) in targets.items():
        for i, (b, xyz) in enumerate(zip(seq, X), start=1):
            rows.append({"ID": f"{tid}_{i}", "resname": b, "resid": i, "x_1": xyz[0], "y_1": xyz[1], "z_1": xyz[2]})
    return pd.DataFrame(rows).to_csv(index=False)


# ------------------------------------------------------------ sequences


def test_normalize_and_validation():
    s, w = sequences.normalize("acgt tgca\n12")
    assert s == "ACGUUGCA" and any("T converted" in x for x in w)
    s, w = sequences.normalize("ACGUN")
    assert any("Ambiguity" in x for x in w)
    with pytest.raises(sequences.SequenceError):
        sequences.normalize("ACGUX")
    with pytest.raises(sequences.SequenceError):
        sequences.normalize("   ")


def test_fasta_parsing():
    recs = sequences.parse_fasta(">a first\nACGU\nACGU\n>b\nGGCC\n")
    assert [(r.name, r.sequence, r.description) for r in recs] == [("a", "ACGUACGU", "first"), ("b", "GGCC", "")]
    assert sequences.parse_fasta("ACGU")[0].name == "sequence_1"
    with pytest.raises(sequences.SequenceError):
        sequences.parse_fasta(">bad\nACGQ")


def test_sequence_crud_and_dedup(client):
    r = client.post("/api/rna/sequences", json={"fasta": ">h1 hairpin\n" + HAIRPIN + "\n>h2\nGGGAAACCC"})
    assert r.status_code == 200 and r.json()["created"] == 2
    again = client.post("/api/rna/sequences", json={"fasta": ">dup\n" + HAIRPIN}).json()
    assert again["created"] == 0 and again["duplicates"] == 1
    items = client.get("/api/rna/sequences").json()
    assert items["total"] == 2
    sid = next(i["id"] for i in items["items"] if i["name"] == "h1")
    detail = client.get(f"/api/rna/sequences/{sid}").json()
    assert detail["composition"]["length"] == len(HAIRPIN)
    assert client.patch(f"/api/rna/sequences/{sid}", json={"name": "renamed"}).json()["name"] == "renamed"
    assert client.get("/api/rna/sequences?q=renamed").json()["total"] == 1
    assert client.delete(f"/api/rna/sequences/{sid}").status_code == 200
    assert client.get(f"/api/rna/sequences/{sid}").status_code == 404


def test_fasta_file_upload(client):
    f = io.BytesIO(b">u1\nACGUACGUAC\n")
    r = client.post("/api/rna/sequences/upload", files={"file": ("seqs.fasta", f, "text/plain")})
    assert r.status_code == 200 and r.json()["created"] == 1
    bad = client.post("/api/rna/sequences/upload", files={"file": ("x.fa", io.BytesIO(b">x\nAXGU"), "text/plain")})
    assert bad.status_code == 422


# ------------------------------------------------------------ folding


def test_fold_known_hairpin():
    r = folding.fold("GGGAAAUCCCGCGCAAAGCGC")
    assert r["mfe_structure"] == "(((....)))((((...))))"
    assert r["mfe_kcal_mol"] < 0
    assert len(r["confidence"]) == 21 and all(0 <= c <= 1 for c in r["confidence"])
    assert r["ensemble_free_energy_kcal_mol"] <= r["mfe_kcal_mol"] + 1e-6  # ensemble includes MFE
    assert 0 < r["mfe_frequency_in_ensemble"] <= 1


def test_pair_table_and_stems():
    pt = folding.pair_table("((..))..(.)")
    assert pt == [5, 4, -1, -1, 1, 0, -1, -1, 10, -1, 8]
    assert folding.stems(pt) == [[(0, 5), (1, 4)], [(8, 10)]]
    with pytest.raises(ValueError):
        folding.pair_table("((.)")


# ------------------------------------------------------------ 3D


def test_denovo_geometry_and_confidence():
    m = s3.predict_denovo(HAIRPIN)
    X = m["coords"]
    pt = folding.pair_table(m["secondary_structure"])
    pair_d = [np.linalg.norm(X[i] - X[j]) for i, j in enumerate(pt) if j > i]
    assert np.allclose(pair_d, s3.PAIR_C1C1, atol=0.3)
    assert m["restraint_rmse_A"] < 0.5
    assert 0 <= m["mean_confidence"] <= 1


def test_tm_score_properties():
    X = s3.predict_denovo(HAIRPIN)["coords"]
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    assert s3.tm_score(X @ R.T + 7.0, X) == pytest.approx(1.0, abs=1e-6)
    assert s3.tm_score(np.random.default_rng(1).normal(scale=30, size=X.shape), X) < 0.2
    assert s3.tm_d0(10) == 0.3 and s3.tm_d0(29) == 0.7
    assert s3.tm_d0(100) == pytest.approx(0.6 * np.sqrt(99.5) - 2.5)


def test_pdb_roundtrip_and_confidence_bfactor():
    X = s3.predict_denovo(HAIRPIN)["coords"]
    pdb = s3.write_pdb(HAIRPIN, X, [0.5] * len(HAIRPIN))
    ch = s3.parse_pdb_c1(pdb)["A"]
    assert ch["sequence"] == HAIRPIN
    assert np.abs(ch["coords"] - X).max() < 1e-3
    assert " 50.00" in pdb.splitlines()[0]
    assert "CONECT" in pdb


def test_template_prediction_transfers_coordinates():
    X = s3.predict_denovo(HAIRPIN)["coords"]
    lib = [s3.Template("T1", HAIRPIN, X)]
    mutant = HAIRPIN[:20] + "A" + HAIRPIN[21:]
    p = s3.predict_template(mutant, lib)
    assert p["template_id"] == "T1" and p["identity"] == pytest.approx(39 / 40)
    assert s3.tm_score(p["coords"], X) == pytest.approx(1.0, abs=1e-6)
    assert s3.predict_template("AAAAAAAAAAAAAAAAAAAA", [s3.Template("T2", "GC" * 5, X[:10])]) is None


# ------------------------------------------------------------ API: predict / structures


def test_predict_endpoint_denovo_then_template(client, monkeypatch):
    sid = client.post("/api/rna/sequences", json={"fasta": ">h\n" + HAIRPIN}).json()["records"][0]["sequence"]["id"]
    r = client.post("/api/rna/predict", json={"sequence_id": sid})
    body = r.json()
    assert r.status_code == 200 and body["method"].startswith("coarse-grained de novo")
    assert body["structure_id"] and body["pdb"].startswith("REMARK")
    assert client.get(f"/api/rna/sequences/{sid}").json()["structures"]
    # template requested but no library -> clear 422
    assert client.post("/api/rna/predict", json={"sequence": HAIRPIN, "method": "template"}).status_code == 422
    # load a Kaggle-format labels file and predict again
    X = s3.predict_denovo(HAIRPIN)["coords"]
    up = client.post("/api/datasets/upload", data={"name": "labels"},
                     files=[("files", ("train_labels.csv", _labels_csv({"1ABC_A": (HAIRPIN, X)}).encode(), "text/csv"))])
    assert up.status_code == 200
    r2 = client.post("/api/rna/predict", json={"sequence": HAIRPIN, "method": "auto", "save": False}).json()
    assert r2["method"].startswith("template-based") and r2["template_id"] == "1ABC_A"


def test_structure_upload_compare_and_py3dmol(client):
    X = s3.predict_denovo(HAIRPIN)["coords"]
    pdb = s3.write_pdb(HAIRPIN, X).encode()
    a = client.post("/api/rna/structures/upload", files={"file": ("ref.pdb", io.BytesIO(pdb), "chemical/x-pdb")}).json()
    rot = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))[0]
    b = client.post("/api/rna/structures/upload",
                    files={"file": ("m.pdb", io.BytesIO(s3.write_pdb(HAIRPIN, X @ rot.T).encode()), "chemical/x-pdb")}).json()
    assert a["meta"]["rna_chains"]["A"]["length"] == len(HAIRPIN)
    cmp = client.post("/api/rna/structures/compare", json={"reference_id": a["id"], "model_id": b["id"]}).json()
    assert cmp["tm_score"] == pytest.approx(1.0, abs=1e-6) and cmp["rmsd_A"] < 1e-3
    snip = client.get(f"/api/rna/structures/{a['id']}/py3dmol").json()
    assert "import py3Dmol" in snip["python"]
    assert client.post("/api/rna/structures/upload", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")}).status_code == 422
    assert client.post("/api/rna/structures/fetch", json={"pdb_id": "not-an-id"}).status_code == 422


def test_rcsb_fetch_uses_pdb_download(client, monkeypatch):
    X = s3.predict_denovo(HAIRPIN)["coords"]
    pdb_text = s3.write_pdb(HAIRPIN, X)
    seen = {}

    class FakeResp:
        status_code = 200
        text = pdb_text
        content = pdb_text.encode()

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            seen["url"] = url
            return FakeResp()

    import app.rna.api as rna_api
    monkeypatch.setattr(rna_api.httpx, "AsyncClient", FakeClient)
    r = client.post("/api/rna/structures/fetch", json={"pdb_id": "1ehz"})
    assert r.status_code == 200 and r.json()["kind"] == "pdb"
    assert seen["url"] == "https://files.rcsb.org/download/1EHZ.pdb"


# ------------------------------------------------------------ datasets & Kaggle


def test_kaggle_import_with_mock_downloader(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path / "datasets")
    datasets._template_cache.clear()
    src = tmp_path / "kaggle_cache"
    src.mkdir()
    X = s3.predict_denovo(HAIRPIN)["coords"]
    (src / "train_sequences.csv").write_text(
        "target_id,sequence,temporal_cutoff,description\n1ABC_A," + HAIRPIN + ",2020-01-01,hairpin\n2XYZ_B,GGGAAACCC,2021-05-01,short\n")
    (src / "train_labels.csv").write_text(_labels_csv({"1ABC_A": (HAIRPIN, X)}))
    (src / "README.md").write_text("ignored")
    calls = []

    def fake_download(handle, path=None):
        calls.append((handle, path))
        return str(src)

    meta = datasets.import_kaggle("stanford-rna-3d-folding", "competition", downloader=fake_download)
    assert calls == [("stanford-rna-3d-folding", None)]
    assert meta["source"] == "kaggle:competition"
    assert {f["name"] for f in meta["files"]} == {"train_sequences.csv", "train_labels.csv"}
    desc = datasets.describe(meta["slug"])
    kinds = {f["name"]: f["kind"] for f in desc["files"]}
    assert kinds == {"train_sequences.csv": "rna_sequences", "train_labels.csv": "structure_labels"}
    assert desc["n_templates"] == 1
    rna = datasets.rna_analytics(meta["slug"])["sequence_files"][0]
    assert rna["n_sequences"] == 2 and rna["by_year"] == {"2020": 1, "2021": 1}
    page = datasets.rows(meta["slug"], "train_sequences.csv", q="short")
    assert page["total"] == 1
    with pytest.raises(datasets.DatasetError):
        datasets.import_kaggle("../etc/passwd", downloader=fake_download)
    with pytest.raises(datasets.DatasetError):
        datasets.rows(meta["slug"], "../meta.json")


def test_kaggle_unconfigured_returns_503(client, monkeypatch):
    monkeypatch.setattr(datasets, "kaggle_configured", lambda: False)
    r = client.post("/api/datasets/kaggle", json={"handle": "stanford-rna-3d-folding"})
    assert r.status_code == 503 and "KAGGLE" in r.json()["detail"]
    st = client.get("/api/datasets/kaggle/status").json()
    assert st["configured"] is False and st["suggested"][0]["handle"] == "stanford-rna-3d-folding"


def test_missing_coordinates_are_handled():
    df = pd.DataFrame({"ID": [f"T_{i}" for i in range(1, 7)], "resname": list("GGGCCC"), "resid": range(1, 7),
                       "x_1": [0, 5, 10, -1e18, 20, 25.0], "y_1": [0.0] * 6, "z_1": [0.0] * 6})
    t = datasets._labels_to_templates(df)[0]
    assert np.isnan(t.coords[3]).all() and np.isfinite(t.coords[4]).all()


def test_dataset_upload_rejects_bad_files(client):
    r = client.post("/api/datasets/upload", data={"name": "evil"}, files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 422


def test_benchmark_and_import_sequences(client):
    lib = {}
    seqs = [HAIRPIN, "GGCGCUUCGGCGCC", "GGGAAACCCAAAGGGAAACCC", "GCGCGCAAAAGCGCGC"]
    for k, sq in enumerate(seqs):
        lib[f"T{k}_A"] = (sq, s3.predict_denovo(sq)["coords"])
    csv = _labels_csv(lib).encode()
    seq_csv = ("target_id,sequence\n" + "\n".join(f"T{k}_A,{sq}" for k, sq in enumerate(seqs))).encode()
    client.post("/api/datasets/upload", data={"name": "mini"},
                files=[("files", ("train_labels.csv", csv, "text/csv")), ("files", ("train_sequences.csv", seq_csv, "text/csv"))])
    b = client.post("/api/datasets/benchmark", json={"n_targets": 4, "max_identity": 0.9}).json()
    assert b["n_targets"] == 4
    imp = client.post("/api/datasets/mini/import-sequences", json={"file": "train_sequences.csv"}).json()
    assert imp["created"] == 4
