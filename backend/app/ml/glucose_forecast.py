"""Short-term glucose forecasting benchmark.

Task: from the most recent hour of continuous glucose monitoring (CGM) readings, predict the
glucose value 30 and 60 minutes ahead. This is the standard prediction-horizon task in the CGM
forecasting literature.

Models compared (all transparent, all implemented here):

* persistence  - predict that glucose stays at its current value (the standard strong baseline;
  short-term glucose is highly autocorrelated, so this is hard to beat at 30 min).
* linear       - extrapolate the least-squares slope of the last 30 minutes.
* ridge        - L2 linear regression on lagged values, the recent slope, and time-of-day,
  fitted per subject on that subject's training period (scikit-learn).

Evaluation protocol (designed to avoid data leakage):

* Each virtual subject's series is split in time: the first 70% is training, the last 30% is the
  held-out test period. The ridge model never sees the test period of the subject it is scored on.
* Windows are built on a uniform 5-minute grid. A window is used only if its history and its
  target are both free of long gaps.
* Metrics are root-mean-square error (RMSE) and mean absolute error (MAE) in mg/dl, pooled over
  all subjects, reported per horizon. "Skill vs persistence" = 1 - RMSE_model / RMSE_persistence
  (positive means better than the persistence baseline).

IMPORTANT: the built-in cohort is SYNTHETIC, generated from the Dalla Man meal model, not real
patients, and not the OhioT1DM benchmark. Numbers here measure the models against each other on
reproducible in-silico data; they are not a clinical accuracy claim and are not comparable to
published results on real CGM. Upload your own CGM file to evaluate on real data.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

GRID_MIN = 5.0            # uniform sampling grid (minutes)
HISTORY_MIN = 60.0        # input window length
HORIZONS_MIN = (30, 60)   # prediction horizons
SLOPE_WINDOW_MIN = 30.0   # window for the linear-slope model
MAX_GAP_MIN = 15.0        # a window spanning a longer gap is discarded
VERSION = "1"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".model_cache"

_H = int(round(HISTORY_MIN / GRID_MIN))        # history samples (12)
_SLOPE_N = int(round(SLOPE_WINDOW_MIN / GRID_MIN))
_result: dict | None = None
_lock = threading.Lock()


def _resample(minutes: np.ndarray, glucose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put readings on a uniform 5-minute grid. Returns (grid_minutes, glucose) with NaN where a
    gap is longer than MAX_GAP_MIN so those windows can be excluded."""
    t = np.asarray(minutes, float)
    g = np.asarray(glucose, float)
    order = np.argsort(t)
    t, g = t[order], g[order]
    grid = np.arange(t[0], t[-1] + 1e-9, GRID_MIN)
    gi = np.interp(grid, t, g)
    # Mark grid points whose nearest real reading is too far away (inside a big gap).
    idx = np.searchsorted(t, grid)
    idx = np.clip(idx, 1, len(t) - 1)
    nearest = np.minimum(np.abs(grid - t[idx]), np.abs(grid - t[idx - 1]))
    gi[nearest > MAX_GAP_MIN] = np.nan
    return grid, gi


def _windows(g: np.ndarray, horizon_steps: int, grid: np.ndarray | None = None):
    """Build (history[_H], target, now, tod) arrays on a uniform grid, skipping windows with NaN.
    `grid` gives the absolute minute of each sample so time-of-day lines up with each window."""
    n = g.size
    X, y, cur, tod = [], [], [], []
    # "now" is sample i (the last of the history window); the target is horizon_steps ahead of it.
    for i in range(_H - 1, n - horizon_steps):
        hist = g[i - _H + 1:i + 1]
        target = g[i + horizon_steps]
        now = g[i]
        if np.isnan(hist).any() or np.isnan(target) or np.isnan(now):
            continue
        X.append(hist)
        y.append(target)
        cur.append(now)
        tod.append((grid[i] % 1440.0) if grid is not None else 0.0)
    if not X:
        return np.empty((0, _H)), np.empty(0), np.empty(0), np.empty(0)
    return np.array(X), np.array(y), np.array(cur), np.array(tod)


def _features(hist: np.ndarray, tod_min: np.ndarray | None = None) -> np.ndarray:
    """Features for the ridge model: the lagged values, the recent slope, and time of day."""
    # recent slope over the last SLOPE_WINDOW_MIN, per row, via least squares on the tail columns
    tail = hist[:, -_SLOPE_N:]
    x = np.arange(_SLOPE_N, dtype=float)
    x -= x.mean()
    slope = (tail - tail.mean(axis=1, keepdims=True)) @ x / (x @ x)
    feats = [hist, slope[:, None]]
    if tod_min is not None:
        ang = 2 * np.pi * (tod_min / 1440.0)
        feats.append(np.c_[np.sin(ang), np.cos(ang)])
    return np.hstack(feats)


def _persistence(cur: np.ndarray, hist: np.ndarray, horizon_steps: int) -> np.ndarray:
    return cur.copy()


def _linear(cur: np.ndarray, hist: np.ndarray, horizon_steps: int) -> np.ndarray:
    tail = hist[:, -_SLOPE_N:]
    x = np.arange(_SLOPE_N, dtype=float)
    x -= x.mean()
    slope = (tail - tail.mean(axis=1, keepdims=True)) @ x / (x @ x)  # mg/dl per grid step
    return cur + slope * horizon_steps


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))), "n": int(y.size)}


def _subject_series(phenotype: str, seed: int, days: int) -> tuple[np.ndarray, np.ndarray]:
    from ..physiology.cgm import synthetic_sample

    return synthetic_sample(phenotype=phenotype, days=days, seed=seed)


def evaluate_cohort(subjects: list[tuple[str, int]], days: int = 10) -> dict:
    from sklearn.linear_model import Ridge

    horizons = {h: {"persistence": [], "linear": [], "ridge": [], "y": []} for h in HORIZONS_MIN}
    per_subject = []
    example = None

    for phen, seed in subjects:
        t, g = _subject_series(phen, seed, days)
        grid, gi = _resample(t, g)
        tod = grid % 1440.0
        split = int(len(gi) * 0.70)
        subj_row = {"phenotype": phen, "seed": seed}
        for h in HORIZONS_MIN:
            hs = int(round(h / GRID_MIN))
            Xtr, ytr, ctr, todtr = _windows(gi[:split], hs, grid[:split])
            lead = split - _H - hs
            Xte, yte, cte, todte = _windows(gi[lead:], hs, grid[lead:])  # lead-in so test windows are complete
            if Xte.shape[0] == 0 or Xtr.shape[0] == 0:
                continue
            # ridge fitted on this subject's training period only
            model = Ridge(alpha=10.0)
            model.fit(_features(Xtr, todtr), ytr)
            preds = {
                "persistence": _persistence(cte, Xte, hs),
                "linear": _linear(cte, Xte, hs),
                "ridge": model.predict(_features(Xte, todte)),
            }
            for name, p in preds.items():
                horizons[h][name].append((yte, p))
            horizons[h]["y"].append(yte)
            subj_row[f"rmse_persistence_{h}"] = _metrics(yte, preds["persistence"])["rmse"]
            subj_row[f"rmse_ridge_{h}"] = _metrics(yte, preds["ridge"])["rmse"]
            if example is None and h == 30 and phen == "type2":
                example = _example_trace(grid, gi, split, hs, model)
        per_subject.append(subj_row)

    table = []
    for h in HORIZONS_MIN:
        yall = np.concatenate(horizons[h]["y"]) if horizons[h]["y"] else np.empty(0)
        base = None
        for name in ("persistence", "linear", "ridge"):
            pairs = horizons[h][name]
            if not pairs:
                continue
            y = np.concatenate([p[0] for p in pairs])
            pred = np.concatenate([p[1] for p in pairs])
            m = _metrics(y, pred)
            if name == "persistence":
                base = m["rmse"]
            m.update({"horizon_min": h, "model": name,
                      "skill_vs_persistence": None if not base else round(1 - m["rmse"] / base, 3)})
            table.append(m)
    return {"table": table, "per_subject": per_subject, "example": example,
            "n_subjects": len(subjects), "days": days}




def _example_trace(grid, gi, split, hs, model) -> dict:
    """A slice of the test period with actual glucose and the ridge 30-min-ahead prediction."""
    seg_start = split
    end = min(seg_start + int(12 * 60 / GRID_MIN), len(gi))  # up to 12 hours
    times, actual, pred = [], [], []
    for i in range(seg_start, end - hs):
        hist = gi[i - _H + 1:i + 1]
        if np.isnan(hist).any() or np.isnan(gi[i + hs]):
            continue
        tod = np.array([grid[i] % 1440.0])
        yhat = float(model.predict(_features(hist[None, :], tod))[0])
        times.append(float((grid[i + hs] - grid[seg_start]) / 60.0))  # hours into the window
        actual.append(float(gi[i + hs]))
        pred.append(yhat)
    return {"horizon_min": int(hs * GRID_MIN), "t_hours": times, "actual": actual, "predicted": pred}


def _default_cohort() -> list[tuple[str, int]]:
    # A small, reproducible cohort: a mix of phenotypes with distinct random seeds.
    phenos = ["normal", "insulin_resistant", "type2"]
    return [(phenos[i % 3], 100 + i) for i in range(9)]


def _cache_path() -> Path:
    key = hashlib.sha256(repr((_default_cohort(), VERSION, GRID_MIN, HORIZONS_MIN)).encode()).hexdigest()[:16]
    return CACHE_DIR / f"glucose_forecast_v{VERSION}_{key}.joblib"


def get_benchmark() -> dict:
    """Run the cohort benchmark once and cache it to disk (it is a fixed experiment)."""
    global _result
    if _result is None:
        with _lock:
            if _result is None:
                cp = _cache_path()
                if cp.exists():
                    try:
                        _result = joblib.load(cp)
                    except Exception:
                        _result = None
                if _result is None:
                    _result = evaluate_cohort(_default_cohort(), days=10)
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        joblib.dump(_result, cp)
                    except OSError:
                        pass
    return _result


def is_ready() -> bool:
    return _result is not None


def evaluate_series(minutes: np.ndarray, glucose: np.ndarray) -> dict:
    """Benchmark the models on one uploaded CGM series (temporal 70/30 split)."""
    from sklearn.linear_model import Ridge

    grid, gi = _resample(minutes, glucose)
    if np.sum(~np.isnan(gi)) < 12 * 24:  # need at least ~1 day of usable data
        raise ValueError("Need at least about a day of readings to train and test a forecast.")
    tod = grid % 1440.0
    split = int(len(gi) * 0.70)
    table = []
    example = None
    for h in HORIZONS_MIN:
        hs = int(round(h / GRID_MIN))
        Xtr, ytr, ctr, todtr = _windows(gi[:split], hs, grid[:split])
        lead = split - _H - hs
        Xte, yte, cte, todte = _windows(gi[lead:], hs, grid[lead:])
        if Xtr.shape[0] < 50 or Xte.shape[0] < 10:
            raise ValueError("Not enough clean windows after removing gaps; a longer or denser recording is needed.")
        model = Ridge(alpha=10.0)
        model.fit(_features(Xtr, todtr), ytr)
        preds = {
            "persistence": _persistence(cte, Xte, hs),
            "linear": _linear(cte, Xte, hs),
            "ridge": model.predict(_features(Xte, todte)),
        }
        base = _metrics(yte, preds["persistence"])["rmse"]
        for name, p in preds.items():
            m = _metrics(yte, p)
            m.update({"horizon_min": h, "model": name,
                      "skill_vs_persistence": round(1 - m["rmse"] / base, 3) if base else None})
            table.append(m)
        if h == 30:
            example = _example_trace(grid, gi, split, hs, model)
    return {"table": table, "example": example, "n_subjects": 1, "source": "uploaded"}
