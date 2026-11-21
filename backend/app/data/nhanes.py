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

DATA = Path(__file__).resolve().parents[2] / "data" / "nhanes" / "nhanes_2017_2018.csv"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".model_cache"
VERSION = "1"

_result: dict | None = None
_lock = threading.Lock()


def _load():
    import pandas as pd

    df = pd.read_csv(DATA)
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


def _cache_path() -> Path:
    h = hashlib.sha256((DATA.read_bytes() if DATA.exists() else b"") + VERSION.encode()).hexdigest()[:16]
    return CACHE_DIR / f"nhanes_study_v{VERSION}_{h}.joblib"


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
    return _result is not None
