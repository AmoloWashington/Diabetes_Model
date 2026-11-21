"""A real-data study on NHANES 2017-2018 (US adults, 20+).

This is the project's first analysis on real people rather than synthetic or textbook data. It
validates GlucoLab's clinical engines against a nationally representative survey and reports a
simple, honest diabetes pre-screening result.

What it computes:

* Diabetes and prediabetes prevalence by the ADA laboratory criteria (HbA1c >= 6.5% or fasting
  plasma glucose >= 126 mg/dl for diabetes; HbA1c 5.7-6.4% or FPG 100-125 for prediabetes),
  weighted with the NHANES fasting-subsample weight as NCHS recommends.
* The fraction of laboratory-defined diabetes that is undiagnosed (no self-reported diagnosis).
* HOMA-IR on real fasting insulin and glucose, and its correlation with BMI (our HOMA engine,
  applied to real data).
* The HbA1c-to-fasting-glucose relationship, and a check that our eAG formula (28.7*A1c - 46.7,
  average glucose) sits above fasting glucose as it should.
* A pre-screening model that predicts laboratory-defined diabetes from non-laboratory features
  only (age, sex, BMI, waist), scored with honest stratified cross-validation (pooled out-of-fold
  ROC AUC), against an age-only baseline. This mirrors published non-invasive risk scores and is
  genuinely useful: it points to who should get a blood test.

NHANES data are public domain (CDC/NCHS). Numbers here are research and education output, not
official CDC statistics and not a clinical tool.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "nhanes"
DATA = _DATA_DIR / "nhanes_2017_2018.csv"
DATA_2015 = _DATA_DIR / "nhanes_2015_2016.csv"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".model_cache"
VERSION = "2"

_result: dict | None = None
_lock = threading.Lock()


def _load(path: Path = DATA):
    import pandas as pd

    df = pd.read_csv(path)
    return df


def _weighted_pct(mask: np.ndarray, w: np.ndarray) -> float:
    tot = np.nansum(w)
    return float(100.0 * np.nansum(w[mask]) / tot) if tot > 0 else float("nan")


def compute() -> dict:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve

    df = _load()
    n_adults = int(len(df))

    df["lab_dm"] = (df.hba1c_pct >= 6.5) | (df.fpg_mgdl >= 126)
    df["lab_pre"] = (((df.hba1c_pct >= 5.7) & (df.hba1c_pct < 6.5)) |
                     ((df.fpg_mgdl >= 100) & (df.fpg_mgdl < 126)))
    df["dx_self"] = df.diag_self == 1

    # Prevalence: fasting subsample with both HbA1c and FPG, weighted
    sub = df[df.fast_wt.notna() & df.hba1c_pct.notna() & df.fpg_mgdl.notna()].copy()
    w = sub.fast_wt.to_numpy()
    prev = {
        "n_fasting": int(len(sub)),
        "diabetes_pct": round(_weighted_pct(sub.lab_dm.to_numpy(), w), 1),
        "prediabetes_pct": round(_weighted_pct(sub.lab_pre.to_numpy(), w), 1),
        "diagnosed_self_pct": round(_weighted_pct(sub.dx_self.to_numpy(), w), 1),
    }
    labpos = sub[sub.lab_dm]
    wl = labpos.fast_wt.to_numpy()
    prev["undiagnosed_pct"] = round(_weighted_pct((~labpos.dx_self).to_numpy(), wl), 1)

    # HOMA-IR on real fasting insulin + glucose
    h = sub[sub.insulin_uuml.notna() & sub.fpg_mgdl.notna()].copy()
    h["homa_ir"] = h.insulin_uuml * (h.fpg_mgdl / 18.016) / 22.5
    hb = h[h.bmi.notna()]
    homa = {
        "n": int(len(h)),
        "median": round(float(h.homa_ir.median()), 2),
        "q1": round(float(h.homa_ir.quantile(0.25)), 2),
        "q3": round(float(h.homa_ir.quantile(0.75)), 2),
        "r_with_bmi": round(float(np.corrcoef(hb.homa_ir, hb.bmi)[0, 1]), 3),
    }
    # HOMA-IR by BMI category (WHO)
    cats = [("Normal (<25)", hb.bmi < 25), ("Overweight (25–30)", (hb.bmi >= 25) & (hb.bmi < 30)),
            ("Obese (≥30)", hb.bmi >= 30)]
    homa["by_bmi"] = [{"group": g, "median_homa_ir": round(float(hb.homa_ir[m].median()), 2), "n": int(m.sum())}
                      for g, m in cats]

    # HbA1c vs FPG
    cc = sub[sub.hba1c_pct.notna() & sub.fpg_mgdl.notna()]
    a1c = {
        "r_hba1c_fpg": round(float(np.corrcoef(cc.hba1c_pct, cc.fpg_mgdl)[0, 1]), 3),
        "mean_fpg": round(float(cc.fpg_mgdl.mean()), 0),
        "mean_eag_formula": round(float((28.7 * cc.hba1c_pct - 46.7).mean()), 0),
    }
    # binned mean FPG per HbA1c bin for a chart
    bins = np.arange(4.5, 10.1, 0.5)
    cc = cc.assign(bin=pd.cut(cc.hba1c_pct, bins))
    a1c["binned"] = [{"hba1c": round((iv.left + iv.right) / 2, 2), "mean_fpg": round(float(g.fpg_mgdl.mean()), 0),
                      "n": int(len(g))} for iv, g in cc.groupby("bin", observed=True) if len(g) >= 10]

    # Prevalence by age group (weighted)
    ages = [("20–39", (sub.age >= 20) & (sub.age < 40)), ("40–59", (sub.age >= 40) & (sub.age < 60)),
            ("60+", sub.age >= 60)]
    by_age = [{"group": g, "diabetes_pct": round(_weighted_pct((sub.lab_dm & m).to_numpy(), w) /
              max(_weighted_pct(m.to_numpy(), w), 1e-9) * 100, 1), "n": int(m.sum())} for g, m in ages]

    # Pre-screening model: predict lab diabetes from non-lab features only
    feat = ["age", "sex_male", "bmi", "waist_cm"]
    m = df.copy()
    m["sex_male"] = (m.sex == "male").astype(int)
    m = m.dropna(subset=["age", "bmi", "waist_cm"])
    m["y"] = ((m.hba1c_pct >= 6.5) | (m.fpg_mgdl >= 126)).astype(int)
    m = m[m.hba1c_pct.notna() | m.fpg_mgdl.notna()]
    X, y = m[feat].to_numpy(float), m.y.to_numpy()
    oof = np.zeros(len(y))
    oof_age = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof_age[te] = LogisticRegression(max_iter=1000).fit(X[tr][:, :1], y[tr]).predict_proba(X[te][:, :1])[:, 1]
    auc = roc_auc_score(y, oof)
    # operating points
    fpr, tpr, thr = roc_curve(y, oof)
    def at_sensitivity(target):
        i = int(np.argmin(np.abs(tpr - target)))
        return {"sensitivity": round(float(tpr[i]), 2), "specificity": round(float(1 - fpr[i]), 2)}
    screen = {
        "n": int(len(m)), "positives": int(y.sum()), "prevalence_pct": round(100 * float(y.mean()), 1),
        "auc": round(float(auc), 3), "auc_age_only": round(float(roc_auc_score(y, oof_age)), 3),
        "features": "age, sex, BMI, waist",
        "at_80_sensitivity": at_sensitivity(0.80),
        "roc": [{"fpr": round(float(f), 3), "tpr": round(float(t), 3)} for f, t in
                zip(fpr[:: max(1, len(fpr) // 60)], tpr[:: max(1, len(tpr) // 60)])],
    }

    return {
        "source": "NHANES 2017–2018, US adults 20+ (CDC/NCHS, public domain)",
        "n_adults": n_adults,
        "prevalence": prev,
        "prevalence_by_age": by_age,
        "homa": homa,
        "hba1c": a1c,
        "screening": screen,
        "caveats": [
            "Prevalence is weighted with the NHANES fasting-subsample weight; other statistics are unweighted.",
            "One survey cycle (2017–2018); prediabetes prevalence depends strongly on the criteria used.",
            "Research and education output, not an official CDC statistic and not a clinical screening tool.",
        ],
    }


_screen: dict | None = None
_screen_lock = threading.Lock()

# Plausible input ranges for the pre-screening form (adults). Used for validation and to clamp.
SCREEN_RANGES = {"age": (20, 100), "bmi": (12.0, 70.0), "waist_cm": (50.0, 200.0)}
_FEATURES = ["age", "sex_male", "bmi", "waist_cm"]


def _fit_screen() -> dict:
    """Fit the final pre-screening model on all adults and build the reference distribution.

    The reported AUC (from compute()) is the honest out-of-fold estimate; this fits on the full
    sample so predictions use every data point, which is standard once a model has been validated.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    df = _load()
    df["sex_male"] = (df.sex == "male").astype(int)
    df = df.dropna(subset=["age", "bmi", "waist_cm"])
    df["y"] = ((df.hba1c_pct >= 6.5) | (df.fpg_mgdl >= 126)).astype(int)
    df = df[df.hba1c_pct.notna() | df.fpg_mgdl.notna()]
    X, y = df[_FEATURES].to_numpy(float), df.y.to_numpy()

    # Out-of-fold probabilities give an honest reference distribution (percentiles) and the
    # probability cut that reaches ~80% sensitivity, without the person's own row leaking in.
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    order = np.argsort(oof)
    oof_sorted = oof[order]
    pos_sorted = y[order]
    # Threshold reaching ~80% sensitivity: scan cut points, pick the highest cut that still
    # catches >= 80% of true cases (fewest false alarms at that sensitivity).
    total_pos = int(y.sum())
    thr80 = 0.0
    for cut in np.unique(oof_sorted):
        sens = float((oof >= cut)[y == 1].mean()) if total_pos else 0.0
        if sens >= 0.80:
            thr80 = float(cut)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return {
        "model": model,
        "oof_sorted": oof_sorted,
        "threshold80": thr80,
        "prevalence": float(y.mean()),
        "n": int(len(y)),
    }


def get_screen_model() -> dict:
    import joblib

    global _screen
    if _screen is None:
        with _screen_lock:
            if _screen is None:
                cp = CACHE_DIR / f"nhanes_screen_v{VERSION}_{_data_hash()}.joblib"
                if cp.exists():
                    try:
                        _screen = joblib.load(cp)
                    except Exception:
                        _screen = None
                if _screen is None:
                    _screen = _fit_screen()
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        joblib.dump(_screen, cp)
                    except OSError:
                        pass
    return _screen


def screen(age: float, sex: str, bmi: float, waist_cm: float) -> dict:
    """Score one person with the non-invasive pre-screening model (no blood test)."""
    import numpy as np

    b = get_screen_model()
    sex_male = 1 if str(sex).lower().startswith("m") else 0
    x = np.array([[float(age), sex_male, float(bmi), float(waist_cm)]])
    prob = float(b["model"].predict_proba(x)[:, 1][0])
    oof = b["oof_sorted"]
    percentile = float((np.searchsorted(oof, prob, side="right") / len(oof)) * 100)
    recommend = prob >= b["threshold80"]
    rel = prob / b["prevalence"] if b["prevalence"] else 1.0
    band = "higher" if rel >= 1.5 else "lower" if rel <= 0.667 else "average"
    return {
        "probability": round(prob, 4),
        "probability_pct": round(prob * 100, 1),
        "population_rate_pct": round(b["prevalence"] * 100, 1),
        "percentile": round(percentile, 0),
        "fold_vs_population": round(rel, 2),
        "band": band,
        "recommend_test": bool(recommend),
        "test_threshold_pct": round(b["threshold80"] * 100, 1),
        "features": {"age": float(age), "sex": "male" if sex_male else "female",
                     "bmi": float(bmi), "waist_cm": float(waist_cm)},
        "note": ("This is a triage aid built from NHANES body measurements, not a diagnosis. "
                 "A high estimate means a blood test (HbA1c or fasting glucose) is worthwhile; "
                 "a low estimate does not rule diabetes out."),
    }


_validation: dict | None = None
_validation_lock = threading.Lock()


def _cycle_summary(df, label: str) -> dict:
    """Weighted prevalence summary for one NHANES cycle (fasting subsample)."""
    sub = df[df.fast_wt.notna() & df.hba1c_pct.notna() & df.fpg_mgdl.notna()]
    w = sub.fast_wt.to_numpy()
    dm = (sub.hba1c_pct >= 6.5) | (sub.fpg_mgdl >= 126)
    pre = (((sub.hba1c_pct >= 5.7) & (sub.hba1c_pct < 6.5)) | ((sub.fpg_mgdl >= 100) & (sub.fpg_mgdl < 126)))
    dx = sub.diag_self == 1
    labpos = sub[dm]
    return {
        "cycle": label,
        "n_adults": int(len(df)),
        "n_fasting": int(len(sub)),
        "diabetes_pct": round(_weighted_pct(dm.to_numpy(), w), 1),
        "prediabetes_pct": round(_weighted_pct(pre.to_numpy(), w), 1),
        "diagnosed_pct": round(_weighted_pct(dx.to_numpy(), w), 1),
        "undiagnosed_pct": round(_weighted_pct((~(labpos.diag_self == 1)).to_numpy(), labpos.fast_wt.to_numpy()), 1),
    }


def compute_validation() -> dict:
    """Cross-cycle trend and external (temporal) validation of the pre-screening model.

    The model is fit on 2017-2018 and scored on the *entirely separate* 2015-2016 survey
    cohort. A held-out AUC on data from a different year is a far stronger test than any
    within-sample estimate: it shows the model generalises beyond the sample it was built on.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    newer, older = _load(DATA), _load(DATA_2015)
    cycles = [_cycle_summary(older, "2015–2016"), _cycle_summary(newer, "2017–2018")]

    feat = ["age", "sex_male", "bmi", "waist_cm"]

    def prep(d):
        d = d.copy()
        d["sex_male"] = (d.sex == "male").astype(int)
        d = d.dropna(subset=["age", "bmi", "waist_cm"])
        d["y"] = ((d.hba1c_pct >= 6.5) | (d.fpg_mgdl >= 126)).astype(int)
        return d[d.hba1c_pct.notna() | d.fpg_mgdl.notna()]

    tr, te = prep(newer), prep(older)
    Xtr, ytr = tr[feat].to_numpy(float), tr.y.to_numpy()
    model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    p = model.predict_proba(te[feat].to_numpy(float))[:, 1]
    y = te.y.to_numpy()
    auc_ext = float(roc_auc_score(y, p))
    auc_app = float(roc_auc_score(ytr, model.predict_proba(Xtr)[:, 1]))

    return {
        "cycles": cycles,
        "external": {
            "train": "2017–2018", "test": "2015–2016",
            "features": "age, sex, BMI, waist",
            "auc_external": round(auc_ext, 3),
            "auc_apparent": round(auc_app, 3),
            "n_test": int(len(te)), "positives_test": int(y.sum()),
        },
        "calibration": _calibration(p, y),
        "note": ("The model is trained on one survey cycle and scored on a different one. A held-out "
                 "AUC on a separate year is strong evidence the model generalises, not just memorises."),
    }


def _calibration(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Does a predicted probability mean what it says, on the held-out cycle?

    Returns the reliability curve (predicted vs observed by equal-count bins), the Brier score
    against a base-rate baseline, the logistic calibration slope/intercept (ideal 1.0 / 0.0), and
    the expected calibration error. Discrimination (AUC) is about ranking; this is about whether the
    numbers are honest.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1.0, 2.0
    idx = np.digitize(p, edges) - 1
    bins = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        bins.append({"pred": round(pred, 3), "obs": round(obs, 3), "n": int(m.sum())})
        ece += abs(pred - obs) * m.sum()
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(max_iter=1000).fit(logit.reshape(-1, 1), y)
    base = np.full(len(y), y.mean())
    return {
        "bins": bins,
        "brier": round(float(brier_score_loss(y, p)), 4),
        "brier_baseline": round(float(brier_score_loss(y, base)), 4),
        "mean_predicted": round(float(p.mean()), 3),
        "observed": round(float(y.mean()), 3),
        "slope": round(float(cal.coef_[0, 0]), 3),
        "intercept": round(float(cal.intercept_[0]), 3),
        "ece": round(float(ece / len(y)), 4),
    }


def get_validation() -> dict:
    import joblib

    global _validation
    if _validation is None:
        with _validation_lock:
            if _validation is None:
                h = hashlib.sha256(
                    (DATA.read_bytes() if DATA.exists() else b"")
                    + (DATA_2015.read_bytes() if DATA_2015.exists() else b"")
                    + VERSION.encode()).hexdigest()[:16]
                cp = CACHE_DIR / f"nhanes_validation_v{VERSION}_{h}.joblib"
                if cp.exists():
                    try:
                        _validation = joblib.load(cp)
                    except Exception:
                        _validation = None
                if _validation is None:
                    _validation = compute_validation()
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        joblib.dump(_validation, cp)
                    except OSError:
                        pass
    return _validation


def _data_hash() -> str:
    return hashlib.sha256((DATA.read_bytes() if DATA.exists() else b"") + VERSION.encode()).hexdigest()[:16]


def _cache_path() -> Path:
    return CACHE_DIR / f"nhanes_study_v{VERSION}_{_data_hash()}.joblib"


def get_study() -> dict:
    import joblib

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
                    _result = compute()
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        joblib.dump(_result, cp)
                    except OSError:
                        pass
    return _result


def is_ready() -> bool:
    return _result is not None and _screen is not None and _validation is not None
