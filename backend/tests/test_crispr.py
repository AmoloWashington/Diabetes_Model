import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.molecular import crispr
from app.molecular.dna import reverse_complement

client = TestClient(app)


def test_finds_forward_guide_with_ngg_pam():
    # 20-nt spacer immediately 5' of an NGG PAM, nothing else.
    spacer = "ACGTACGTACGTACGTACGT"
    r = crispr.find_guides(spacer + "TGG")
    fwd = [g for g in r["guides"] if g["strand"] == "+"]
    assert len(fwd) == 1
    g = fwd[0]
    assert g["spacer"] == spacer and g["pam"] == "TGG"
    assert g["full_target"] == spacer + "TGG"
    assert g["spacer_start"] == 0 and g["spacer_end"] == 20 and g["pam_start"] == 20


def test_cut_site_is_three_bp_upstream_of_pam():
    # Forward guide: PAM starts at index 20, so the blunt break is 3 bp upstream -> index 17.
    g = next(g for g in crispr.find_guides("A" * 20 + "CGG")["guides"] if g["strand"] == "+")
    assert g["cut_site"] == 17
    assert g["pam_start"] - g["cut_site"] == 3


def test_reverse_strand_guide_detected_and_mapped():
    # Put the guide only on the reverse strand: forward = revcomp(spacer + PAM).
    spacer = "ACGTACGTACGTACGTAAAA"  # distinct from its complement
    forward = reverse_complement(spacer + "AGG")
    r = crispr.find_guides(forward)
    rev = [g for g in r["guides"] if g["strand"] == "-"]
    assert any(g["spacer"] == spacer and g["pam"] == "AGG" for g in rev)
    g = next(g for g in rev if g["spacer"] == spacer)
    # On the forward strand the PAM lies 5' of the spacer; revcomp of [pam_start, spacer_end) is spacer+pam.
    assert reverse_complement(forward[g["pam_start"]:g["spacer_end"]]) == g["spacer"] + g["pam"]
    assert 0 <= g["cut_site"] <= r["length"]


def test_non_ngg_pam_is_rejected():
    # "TAG" is not NGG, so no forward guide should be found here.
    assert all(g["strand"] != "+" for g in crispr.find_guides("A" * 20 + "TAG")["guides"])


def test_flags_low_gc_polyt_and_non_unique():
    r = crispr.find_guides("A" * 20 + "TGG")
    g = next(g for g in r["guides"] if g["strand"] == "+")
    assert any("Low GC" in f for f in g["flags"])  # 0% GC
    poly = crispr.find_guides("GGGGGTTTTGGGGGCCCCCA" + "TGG")
    gg = next(g for g in poly["guides"] if g["strand"] == "+" and g["spacer"] == "GGGGGTTTTGGGGGCCCCCA")
    assert any("Poly-T" in f for f in gg["flags"])


def test_non_unique_flag_when_spacer_repeats():
    spacer = "ACGTACGTACGTACGTACGT"
    # Two identical target sites separated by a spacer region.
    seq = spacer + "TGG" + "AAAAAAAA" + spacer + "TGG"
    r = crispr.find_guides(seq)
    repeated = [g for g in r["guides"] if g["spacer"] == spacer]
    assert len(repeated) == 2
    assert all(any("Not unique" in f for f in g["flags"]) for g in repeated)


def test_short_sequence_rejected():
    with pytest.raises(ValueError):
        crispr.find_guides("ACGT")


def test_gc_window_validation():
    with pytest.raises(ValueError):
        crispr.find_guides("A" * 20 + "TGG", gc_low=60, gc_high=40)


def test_api_endpoint():
    spacer = "GCTAGCTAGCCGATCGATCG"
    r = client.post("/api/crispr/guides", json={"sequence": spacer + "AGG"})
    assert r.status_code == 200
    body = r.json()
    assert body["pam"] == "NGG" and body["spacer_length"] == 20
    assert any(g["spacer"] == spacer for g in body["guides"])
    assert client.post("/api/crispr/guides", json={"sequence": "ACGT"}).status_code == 422
    assert client.post("/api/crispr/guides", json={"sequence": "A" * 25 + "TGG", "gc_low": 70, "gc_high": 50}).status_code == 422
