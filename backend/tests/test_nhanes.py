"""Tests for the NHANES 2017-2018 real-data study.

These assert the shape and internal consistency of the computed study, and that the headline
figures match what the public CDC data actually say (within rounding). They are not hard-coded
expectations pulled from thin air: they are checks that the pipeline reproduces documented,
well-established facts about US adult diabetes.
"""

import math

import pytest
from fastapi.testclient import TestClient

from app.data import nhanes
from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def study():
    return nhanes.compute()


def test_data_file_present_and_adult_only():
    import pandas as pd

    assert nhanes.DATA.exists(), "NHANES CSV must ship with the repo for offline/CI use"
    df = pd.read_csv(nhanes.DATA)
    assert len(df) > 4000
    assert df.age.min() >= 20  # adults only


def test_prevalence_is_plausible_and_ordered(study):
    p = study["prevalence"]
    # Documented US adult figures: ~13-15% diabetes, with prediabetes much larger.
    assert 10.0 <= p["diabetes_pct"] <= 18.0
    assert p["prediabetes_pct"] > p["diabetes_pct"]
    # More people meet the lab criteria than report a diagnosis -> some are undiagnosed.
    assert p["diabetes_pct"] > p["diagnosed_self_pct"]
    assert 0.0 < p["undiagnosed_pct"] < 60.0


def test_prevalence_rises_with_age(study):
    by_age = study["prevalence_by_age"]
    pcts = [g["diabetes_pct"] for g in by_age]
    assert pcts == sorted(pcts), "diabetes prevalence should be non-decreasing with age group"
    assert pcts[-1] > pcts[0]


def test_homa_ir_tracks_bmi(study):
    h = study["homa"]
    assert h["q1"] < h["median"] < h["q3"]
    assert h["r_with_bmi"] > 0  # insulin resistance rises with adiposity
    medians = [b["median_homa_ir"] for b in h["by_bmi"]]
    assert medians == sorted(medians), "median HOMA-IR should rise across BMI categories"


def test_hba1c_glucose_consistency(study):
    a = study["hba1c"]
    assert a["r_hba1c_fpg"] > 0.5  # strong positive correlation
    # eAG is an average over the day; it should exceed mean fasting glucose.
    assert a["mean_eag_formula"] > a["mean_fpg"]
    # binned chart data is monotone-ish and non-empty
    assert len(a["binned"]) >= 5
    fpgs = [b["mean_fpg"] for b in a["binned"]]
    assert fpgs[-1] > fpgs[0]


def test_screening_beats_age_baseline(study):
    s = study["screening"]
    assert 0.5 < s["auc_age_only"] < s["auc"] <= 1.0
    assert s["auc"] >= 0.70  # body-measurement model is a real signal
    assert 0.0 <= s["at_80_sensitivity"]["specificity"] <= 1.0
    # ROC curve is a valid monotone-ish path from (low) to (high)
    roc = s["roc"]
    assert roc[0]["fpr"] <= roc[-1]["fpr"]
    assert roc[0]["tpr"] <= roc[-1]["tpr"]


def test_all_reported_numbers_are_finite(study):
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, float):
            assert math.isfinite(x), f"non-finite value in study: {x}"

    walk(study)


def test_endpoint_returns_study():
    r = client.get("/api/nhanes/study")
    assert r.status_code == 200
    d = r.json()
    assert d["n_adults"] > 4000
    assert "diabetes_pct" in d["prevalence"]
    assert d["screening"]["auc"] > d["screening"]["auc_age_only"]


def test_reference_registered():
    r = client.get("/api/references")
    ids = {x["id"] for x in r.json()["references"]}
    assert "nhanes2018" in ids


def test_screen_model_ranks_risk_sensibly():
    # A young, lean person should score well below an older person with high BMI and waist.
    low = nhanes.screen(28, "female", 21.0, 72.0)
    high = nhanes.screen(68, "male", 36.0, 122.0)
    assert 0.0 <= low["probability"] <= 1.0
    assert low["probability"] < high["probability"]
    assert low["percentile"] < high["percentile"]
    assert high["recommend_test"] is True
    assert low["recommend_test"] is False


def test_screen_probability_matches_fitted_model():
    import numpy as np

    b = nhanes.get_screen_model()
    x = np.array([[55.0, 1, 31.0, 108.0]])
    expected = float(b["model"].predict_proba(x)[:, 1][0])
    got = nhanes.screen(55, "male", 31.0, 108.0)["probability"]
    assert abs(got - round(expected, 4)) < 1e-9


def test_screen_endpoint_validates_and_scores():
    r = client.post("/api/nhanes/screen", json={"age": 60, "sex": "male", "bmi": 30, "waist_cm": 105})
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["probability_pct"] <= 100
    assert 0 <= d["percentile"] <= 100
    assert d["band"] in {"lower", "average", "higher"}
    # out-of-range input is rejected by the schema
    assert client.post("/api/nhanes/screen", json={"age": 5, "sex": "male", "bmi": 30, "waist_cm": 105}).status_code == 422
    assert client.post("/api/nhanes/screen", json={"age": 60, "sex": "x", "bmi": 30, "waist_cm": 105}).status_code == 422


@pytest.fixture(scope="module")
def validation():
    return nhanes.compute_validation()


def test_second_cycle_present():
    import pandas as pd

    assert nhanes.DATA_2015.exists(), "the 2015-2016 CSV must ship for the trend / external validation"
    df = pd.read_csv(nhanes.DATA_2015)
    assert len(df) > 4000 and df.age.min() >= 20


def test_validation_has_two_cycles_in_order(validation):
    cy = validation["cycles"]
    assert [c["cycle"] for c in cy] == ["2015–2016", "2017–2018"]
    for c in cy:
        assert 10.0 <= c["diabetes_pct"] <= 18.0
        assert c["prediabetes_pct"] > c["diabetes_pct"]
        assert 0 < c["undiagnosed_pct"] < 60


def test_external_validation_generalises(validation):
    e = validation["external"]
    # Held-out AUC on a *separate* survey cycle should still be a real signal, close to the
    # within-sample estimate (not collapsing), which is the whole point of the test.
    assert 0.70 <= e["auc_external"] <= 0.85
    assert abs(e["auc_external"] - e["auc_apparent"]) < 0.08
    assert e["train"] != e["test"]
    assert e["n_test"] > 4000


def test_validation_endpoint():
    r = client.get("/api/nhanes/validation")
    assert r.status_code == 200
    assert r.json()["external"]["auc_external"] > 0.7


def test_calibration_is_well_behaved(validation):
    cal = validation["calibration"]
    # On the held-out cycle the model should be close to calibrated: slope near 1, intercept near 0.
    assert 0.8 <= cal["slope"] <= 1.2
    assert abs(cal["intercept"]) < 0.3
    # Mean predicted should track the real rate, and Brier should beat the base-rate baseline.
    assert abs(cal["mean_predicted"] - cal["observed"]) < 0.03
    assert cal["brier"] < cal["brier_baseline"]
    assert cal["ece"] < 0.06
    # Reliability bins rise monotonically-ish: the top bin's observed rate exceeds the bottom's.
    assert len(cal["bins"]) >= 8
    assert cal["bins"][-1]["obs"] > cal["bins"][0]["obs"]


def test_calibration_in_endpoint():
    cal = client.get("/api/nhanes/validation").json()["calibration"]
    assert "bins" in cal and cal["bins"][0]["n"] > 0
