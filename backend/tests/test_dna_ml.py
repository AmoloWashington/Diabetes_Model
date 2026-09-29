import numpy as np
import pytest

from app.ml import risk_model
from app.molecular import dna


# ------------------------------------------------------------ DNA

def test_standard_genetic_code():
    assert len(dna.CODON_TABLE) == 64
    assert {c for c, aa in dna.CODON_TABLE.items() if aa == "*"} == {"TAA", "TAG", "TGA"}
    assert dna.CODON_TABLE["ATG"] == "M"
    assert dna.CODON_TABLE["TGG"] == "W"
    assert sum(aa == "L" for aa in dna.CODON_TABLE.values()) == 6
    assert sum(aa == "S" for aa in dna.CODON_TABLE.values()) == 6


def test_demo_sequence_encodes_insulin_b_chain():
    assert dna.translate(dna.DEMO_SEQUENCE) == "M" + dna.INSULIN_B_CHAIN + "*"
    r = dna.analyze(dna.DEMO_SEQUENCE)
    assert r["contains_insulin_b_chain"]
    assert r["orfs"][0]["protein"] == "M" + dna.INSULIN_B_CHAIN


def test_insulin_chain_facts():
    assert len(dna.INSULIN_B_CHAIN) == 30 and len(dna.INSULIN_A_CHAIN) == 21
    cys_a = [i + 1 for i, c in enumerate(dna.INSULIN_A_CHAIN) if c == "C"]
    cys_b = [i + 1 for i, c in enumerate(dna.INSULIN_B_CHAIN) if c == "C"]
    assert cys_a == [6, 7, 11, 20] and cys_b == [7, 19]
    # preproinsulin: signal 24 + B 30 + connecting 35 + A 21 = 110 aa
    assert 24 + 30 + 35 + 21 == 110


def test_reverse_complement_and_tm():
    assert dna.reverse_complement("ATGCN") == "NGCAT"
    assert dna.reverse_complement(dna.reverse_complement("ACGTTGCA")) == "ACGTTGCA"
    assert dna.melting_temperature("ATGC")["tm_c"] == 2 * 2 + 4 * 2
    long = "ACGT" * 5
    assert dna.melting_temperature(long)["tm_c"] == pytest.approx(64.9 + 41 * (10 - 16.4) / 20)
    assert dna.analyze("augc")["length_nt"] == 4  # RNA input read as DNA


def test_invalid_sequence():
    with pytest.raises(ValueError):
        dna.analyze("ACGTXZ")
    with pytest.raises(ValueError):
        dna.analyze("   ")


# ------------------------------------------------------------ risk model

@pytest.fixture(scope="module")
def bundle():
    return risk_model.get_bundle()


def test_dataset_integrity():
    X, y, groups, raw = risk_model.load_dataset()
    assert X.shape == (520, 16)
    assert int(y.sum()) == 320
    assert len(np.unique(groups)) == 251  # 269 duplicates


def test_leak_free_metrics_are_reported_and_leakage_is_visible(bundle):
    ev = bundle.report["evaluation"]
    ens = ev["models"]["ensemble"]
    assert 0.9 < ens["pooled"]["auc"] <= 1.0
    lo, hi = ens["ci95"]["auc"]
    assert lo <= ens["pooled"]["auc"] <= hi
    assert ev["naive_leaky_cv"]["accuracy"] > ens["pooled"]["accuracy"]
    assert bundle.report["dataset"]["duplicate_rows"] == 269


def _patient(**kw):
    base = {"age": 45, "male": False, **{f: False for f in risk_model.BINARY_FEATURES}}
    base.update(kw)
    return base


def test_prediction_properties(bundle):
    none = risk_model.predict(bundle, _patient())
    classic = risk_model.predict(bundle, _patient(polyuria=True, polydipsia=True))
    assert classic["probability"] > none["probability"]
    lo, hi = classic["logistic_95ci"]
    assert 0 <= lo <= hi <= 1
    # additive log-odds contributions reproduce the logistic prediction exactly
    logit = classic["explanation"]["reference_log_odds"] + sum(c["log_odds"] for c in classic["explanation"]["contributions"])
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(classic["probability_logistic"], rel=1e-9)


def test_prior_shift_is_bayes_consistent(bundle):
    r = risk_model.predict(bundle, _patient(polyuria=True), target_prevalence=bundle.train_prevalence)
    assert r["prevalence_adjusted"]["probability"] == pytest.approx(r["probability"], rel=1e-6)
    low = risk_model.predict(bundle, _patient(polyuria=True), target_prevalence=0.05)
    assert low["prevalence_adjusted"]["probability"] < low["probability"]


def test_out_of_range_age_warns(bundle):
    assert any("outside the training range" in w for w in risk_model.predict(bundle, _patient(age=100))["warnings"])
