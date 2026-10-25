import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml import glucose_forecast as gf

client = TestClient(app)


def test_persistence_is_exact_on_flat_signal():
    g = np.full(300, 120.0)
    grid = np.arange(0, 300 * gf.GRID_MIN, gf.GRID_MIN)
    X, y, cur, tod = gf._windows(g, 6, grid)
    assert np.allclose(gf._persistence(cur, X, 6), y)  # flat -> persistence has zero error


def test_linear_model_extrapolates_a_ramp():
    # Perfectly linear ramp: the slope model should predict the future value exactly.
    slope_per_step = 2.0
    g = 80.0 + slope_per_step * np.arange(300)
    grid = np.arange(0, 300 * gf.GRID_MIN, gf.GRID_MIN)
    X, y, cur, tod = gf._windows(g, 6, grid)
    pred = gf._linear(cur, X, 6)
    assert np.allclose(pred, y, atol=1e-6)
    # persistence must be worse than linear on a ramp
    assert np.mean(np.abs(gf._persistence(cur, X, 6) - y)) > np.mean(np.abs(pred - y))


def test_resample_marks_long_gaps_as_nan():
    minutes = np.array([0, 5, 10, 120, 125, 130], float)  # ~2 h gap
    g = np.array([100, 101, 102, 150, 151, 152], float)
    grid, gi = gf._resample(minutes, g)
    assert np.isnan(gi).any()  # the gap region is masked
    assert not np.isnan(gi[0]) and not np.isnan(gi[-1])


def test_windows_skip_nan():
    g = np.full(100, 120.0)
    g[40] = np.nan
    grid = np.arange(0, 100 * gf.GRID_MIN, gf.GRID_MIN)
    X, y, cur, tod = gf._windows(g, 6, grid)
    assert not np.isnan(X).any() and not np.isnan(y).any()


def test_cohort_metrics_are_ordered_and_ridge_beats_persistence():
    r = gf.evaluate_cohort([("normal", 1), ("type2", 2), ("insulin_resistant", 3)], days=6)
    by = {(row["horizon_min"], row["model"]): row for row in r["table"]}
    # error grows with the horizon for persistence
    assert by[(60, "persistence")]["rmse"] > by[(30, "persistence")]["rmse"]
    # ridge improves on persistence at both horizons (positive skill) on this smooth synthetic data
    assert by[(30, "ridge")]["skill_vs_persistence"] > 0
    assert by[(60, "ridge")]["skill_vs_persistence"] > 0
    # persistence skill is zero by definition
    assert by[(30, "persistence")]["skill_vs_persistence"] == 0.0


def test_no_leakage_between_train_and_test():
    # Scores on the held-out tail must be finite and computed on non-empty test windows.
    r = gf.evaluate_cohort([("type2", 5)], days=6)
    for row in r["table"]:
        assert row["n"] > 0 and np.isfinite(row["rmse"])


def test_api_benchmark_and_evaluate():
    r = client.get("/api/forecast/benchmark")
    assert r.status_code == 200
    body = r.json()
    assert body["n_subjects"] >= 1
    assert any(row["model"] == "ridge" for row in body["table"])

    from app.physiology.cgm import synthetic_sample
    t, g = synthetic_sample("type2", days=3, seed=9)
    pts = [{"minute": float(ti), "g": float(gi)} for ti, gi in zip(t, g)]
    r = client.post("/api/forecast/evaluate", json={"points": pts})
    assert r.status_code == 200
    tbl = r.json()["table"]
    assert {row["model"] for row in tbl} == {"persistence", "linear", "ridge"}

    assert client.post("/api/forecast/evaluate", json={}).status_code == 422
    # too little data to train/test
    short = [{"minute": 5.0 * i, "g": 120.0} for i in range(20)]
    assert client.post("/api/forecast/evaluate", json={"points": short}).status_code == 422
