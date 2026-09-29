"""Symptom-based early diabetes risk model, evaluated without data leakage.

Data: UCI Machine Learning Repository, "Early stage diabetes risk prediction"
(Islam MMF, Ferdousi R, Rahman S, Bushra HY, 2020), collected by
questionnaire from patients of Sylhet Diabetes Hospital, Bangladesh.

Methodological choices, each addressing a concrete flaw in naive pipelines:

* 269 of the 520 rows are exact duplicates of another row. Random K-fold
  splitting places copies of the same record in both training and test folds
  and inflates accuracy. All evaluation here uses StratifiedGroupKFold with
  groups defined by identical feature vectors, repeated with several seeds.
* Metrics are computed on pooled out-of-fold predictions and reported with
  95% bootstrap confidence intervals that resample duplicate groups.
* Discrimination (ROC AUC), calibration (Brier score, reliability curve) and
  threshold metrics (sensitivity, specificity at 0.5) are all reported.
* The primary model is a soft-voting ensemble of an L2 logistic regression
  (interpretable, exact additive log-odds explanations) and a random forest.
* Per-patient uncertainty: a group bootstrap ensemble of logistic models gives
  a 95% interval for the individual predicted probability.
* The training prevalence (61.5% positive) is a hospital sample. A Bayes
  prior-shift correction re-expresses the probability for any target
  prevalence, assuming symptom likelihoods transport between populations.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "diabetes_symptoms_data.csv"

BINARY_FEATURES = [
    "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
    "genital_thrush", "visual_blurring", "itching", "irritability",
    "delayed_healing", "partial_paresis", "muscle_stiffness", "alopecia", "obesity",
]
FEATURES = ["age", "male", *BINARY_FEATURES]
FEATURE_LABELS = {
    "age": "Age (years)",
    "male": "Male sex",
    "polyuria": "Polyuria (excessive urination)",
    "polydipsia": "Polydipsia (excessive thirst)",
    "sudden_weight_loss": "Sudden weight loss",
    "weakness": "Weakness",
    "polyphagia": "Polyphagia (excessive hunger)",
    "genital_thrush": "Genital thrush",
    "visual_blurring": "Visual blurring",
    "itching": "Itching",
    "irritability": "Irritability",
    "delayed_healing": "Delayed wound healing",
    "partial_paresis": "Partial paresis (muscle weakness)",
    "muscle_stiffness": "Muscle stiffness",
    "alopecia": "Alopecia (hair loss)",
    "obesity": "Obesity (self-reported)",
}
EXPECTED_COLUMNS = ["age", "gender", *BINARY_FEATURES, "class"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    if df[EXPECTED_COLUMNS].isna().any().any():
        raise ValueError("Dataset contains missing values")
    bad = {c for c in BINARY_FEATURES if not set(df[c].unique()) <= {"Yes", "No"}}
    if bad or not set(df["gender"].unique()) <= {"Male", "Female"} or not set(df["class"].unique()) <= {"Positive", "Negative"}:
        raise ValueError(f"Unexpected category values in {sorted(bad) or 'gender/class'}")

    X = pd.DataFrame({"age": df["age"].astype(float), "male": (df["gender"] == "Male").astype(float)})
    for c in BINARY_FEATURES:
        X[c] = (df[c] == "Yes").astype(float)
    y = (df["class"] == "Positive").astype(int).to_numpy()
    groups = pd.factorize(X.astype(str).agg("|".join, axis=1))[0]
    return X[FEATURES], y, groups, df


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class _LogReg:
    """Logistic regression on standardised age (binary features left as 0/1)."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogReg":
        self.age_mu = float(X[:, 0].mean())
        self.age_sd = float(X[:, 0].std()) or 1.0
        self.m = LogisticRegression(C=self.C, max_iter=5000).fit(self._t(X), y)
        return self

    def _t(self, X: np.ndarray) -> np.ndarray:
        Z = X.astype(float).copy()
        Z[:, 0] = (Z[:, 0] - self.age_mu) / self.age_sd
        return Z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.m.predict_proba(self._t(X))[:, 1]

    def logit_contributions(self, x: np.ndarray, ref: np.ndarray) -> np.ndarray:
        z = self._t(x[None, :])[0]
        zr = self._t(ref[None, :])[0]
        return self.m.coef_[0] * (z - zr)


def _rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=1,
    )


class _Ensemble:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Ensemble":
        self.lr = _LogReg().fit(X, y)
        self.rf = _rf(self.seed).fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 0.5 * self.lr.predict_proba(X) + 0.5 * self.rf.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": (tp + tn) / len(y),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def _group_bootstrap_ci(y, p, groups, rng, n_boot=500) -> dict:
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    samples = {k: [] for k in ["auc", "brier", "accuracy", "sensitivity", "specificity"]}
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[idx])) < 2:
            continue
        m = _metrics(y[idx], p[idx])
        for k in samples:
            samples[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}


@dataclass
class ModelBundle:
    ensemble: _Ensemble
    lr: _LogReg
    bootstrap_lrs: list[_LogReg]
    reference: np.ndarray
    train_prevalence: float
    report: dict
    age_range: tuple[float, float]
    trained_at: float = field(default_factory=time.time)


def train(path: Path = DATA_PATH, n_repeats: int = 5, n_splits: int = 5, n_bootstrap_models: int = 100, seed: int = 0) -> ModelBundle:
    Xdf, y, groups, raw = load_dataset(path)
    X = Xdf.to_numpy()
    rng = np.random.default_rng(seed)

    # --- Leak-free repeated grouped cross-validation (out-of-fold predictions)
    oof = {name: np.zeros((n_repeats, len(y))) for name in ["logistic", "random_forest", "ensemble"]}
    perm_drops = np.zeros((0, len(FEATURES)))
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(X, y, groups):
            lr = _LogReg().fit(X[tr], y[tr])
            rf = _rf(seed + r).fit(X[tr], y[tr])
            p_lr = lr.predict_proba(X[te])
            p_rf = rf.predict_proba(X[te])[:, 1]
            oof["logistic"][r, te] = p_lr
            oof["random_forest"][r, te] = p_rf
            oof["ensemble"][r, te] = 0.5 * (p_lr + p_rf)
            if r == 0 and len(np.unique(y[te])) == 2:
                # Held-out permutation importance (first repeat only, for cost)
                base = roc_auc_score(y[te], 0.5 * (p_lr + p_rf))
                drops = np.zeros(len(FEATURES))
                for j in range(len(FEATURES)):
                    d = []
                    for _ in range(5):
                        Xp = X[te].copy()
                        Xp[:, j] = rng.permutation(Xp[:, j])
                        pp = 0.5 * (lr.predict_proba(Xp) + rf.predict_proba(Xp)[:, 1])
                        d.append(base - roc_auc_score(y[te], pp))
                    drops[j] = np.mean(d)
                perm_drops = np.vstack([perm_drops, drops])

    # --- Naive (leaky) CV, computed only to quantify the inflation
    from sklearn.model_selection import StratifiedKFold

    naive = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        naive[te] = _rf(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

    models_report = {}
    for name, P in oof.items():
        p_mean = P.mean(axis=0)
        per_repeat = [_metrics(y, P[r]) for r in range(n_repeats)]
        models_report[name] = {
            "pooled": _metrics(y, p_mean),
            "ci95": _group_bootstrap_ci(y, p_mean, groups, rng),
            "repeat_sd": {k: float(np.std([m[k] for m in per_repeat])) for k in per_repeat[0]},
        }

    p_ens = oof["ensemble"].mean(axis=0)
    fpr, tpr, _ = roc_curve(y, p_ens)
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(p_ens, bins) - 1, 0, 9)
    calib = [
        {"bin_mid": float((bins[b] + bins[b + 1]) / 2), "mean_pred": float(p_ens[which == b].mean()),
         "observed": float(y[which == b].mean()), "n": int((which == b).sum())}
        for b in range(10) if (which == b).any()
    ]

    # --- Final models on all data
    ens = _Ensemble(seed).fit(X, y)
    lr_full = ens.lr
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = []
    for _ in range(n_bootstrap_models):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[idx])) == 2:
            boots.append(_LogReg().fit(X[idx], y[idx]))

    # --- Global explanations
    coefs = lr_full.m.coef_[0]
    odds = [
        {"feature": f, "label": FEATURE_LABELS[f], "odds_ratio": float(np.exp(c)),
         "per": "1 SD of age (%.1f y)" % lr_full.age_sd if f == "age" else "presence vs absence"}
        for f, c in zip(FEATURES, coefs)
    ]
    odds.sort(key=lambda d: -abs(np.log(d["odds_ratio"])))

    perm_list = sorted(
        [{"feature": f, "label": FEATURE_LABELS[f], "auc_drop": float(m), "sd_across_folds": float(sd)}
         for f, m, sd in zip(FEATURES, perm_drops.mean(axis=0), perm_drops.std(axis=0))],
        key=lambda d: -d["auc_drop"],
    )

    data_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    report = {
        "dataset": {
            "name": "UCI Early Stage Diabetes Risk Prediction",
            "source": "Islam MMF, Ferdousi R, Rahman S, Bushra HY (2020). Sylhet Diabetes Hospital, Bangladesh.",
            "rows": int(len(y)),
            "unique_rows": int(len(uniq)),
            "duplicate_rows": int(len(y) - len(uniq)),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "prevalence": float(y.mean()),
            "age_range": [float(X[:, 0].min()), float(X[:, 0].max())],
            "sha256_16": data_hash,
        },
        "evaluation": {
            "scheme": f"{n_repeats}x repeated {n_splits}-fold StratifiedGroupKFold (groups = identical records)",
            "models": models_report,
            "naive_leaky_cv": _metrics(y, naive),
            "leakage_note": (
                "Naive random K-fold lets duplicate records appear in both training and test "
                "folds; the difference between naive and grouped metrics is optimistic bias."
            ),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "calibration": calib,
        },
        "explanations": {"odds_ratios": odds, "permutation_importance": perm_list},
        "limitations": [
            "Symptoms were self-reported by patients at a single diabetes hospital; the model is not a screening test for the general population.",
            "Class labels are clinical diagnoses at that hospital; diabetes type is not recorded.",
            "Associations are not causal. Odds ratios are conditional on the other features.",
            "No laboratory measurements (glucose, HbA1c) are used; diagnosis requires them.",
            "Probabilities reflect the 61.5% training prevalence unless a prevalence adjustment is applied.",
        ],
    }
    return ModelBundle(
        ensemble=ens, lr=lr_full, bootstrap_lrs=boots,
        reference=np.r_[X[:, 0].mean(), np.zeros(len(FEATURES) - 1)],
        train_prevalence=float(y.mean()), report=report,
        age_range=(float(X[:, 0].min()), float(X[:, 0].max())),
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def _prior_shift(p: float, prev_train: float, prev_target: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    odds = p / (1 - p) * (prev_target / (1 - prev_target)) / (prev_train / (1 - prev_train))
    return odds / (1 + odds)


def predict(bundle: ModelBundle, patient: dict, target_prevalence: float | None = None) -> dict:
    x = np.array([float(patient["age"]), 1.0 if patient["male"] else 0.0,
                  *[1.0 if patient[f] else 0.0 for f in BINARY_FEATURES]])
    p_ens = float(bundle.ensemble.predict_proba(x[None, :])[0])
    p_lr = float(bundle.lr.predict_proba(x[None, :])[0])
    p_rf = float(bundle.ensemble.rf.predict_proba(x[None, :])[0, 1])
    boot = np.array([m.predict_proba(x[None, :])[0] for m in bundle.bootstrap_lrs])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    contrib = bundle.lr.logit_contributions(x, bundle.reference)
    base_logit = float(np.log(p_ref := bundle.lr.predict_proba(bundle.reference[None, :])[0]) - np.log(1 - p_ref))
    contributions = sorted(
        [{"feature": f, "label": FEATURE_LABELS[f], "log_odds": float(c)} for f, c in zip(FEATURES, contrib) if abs(c) > 1e-12],
        key=lambda d: -abs(d["log_odds"]),
    )

    warnings = []
    a0, a1 = bundle.age_range
    if not (a0 <= x[0] <= a1):
        warnings.append(f"Age {x[0]:.0f} lies outside the training range {a0:.0f}-{a1:.0f}; prediction is an extrapolation.")
    if abs(p_lr - p_rf) > 0.3:
        warnings.append("Logistic and random-forest models disagree by more than 30 points; treat the estimate as uncertain.")

    out = {
        "probability": p_ens,
        "probability_logistic": p_lr,
        "probability_random_forest": p_rf,
        "logistic_95ci": [float(lo), float(hi)],
        "explanation": {
            "method": "Exact additive log-odds contributions of the logistic model relative to a reference patient of mean age with no symptoms (female).",
            "reference_log_odds": base_logit,
            "contributions": contributions,
        },
        "train_prevalence": bundle.train_prevalence,
        "warnings": warnings,
        "disclaimer": "Research/education tool. Not a diagnosis. Confirm with fasting glucose, OGTT or HbA1c.",
    }
    if target_prevalence is not None:
        if not (0.001 <= target_prevalence <= 0.999):
            raise ValueError("target_prevalence must be within 0.001-0.999")
        out["prevalence_adjusted"] = {
            "target_prevalence": target_prevalence,
            "probability": _prior_shift(p_ens, bundle.train_prevalence, target_prevalence),
            "assumption": "Bayes prior shift: symptom likelihood ratios assumed transportable to the target population.",
        }
    return out


# --------------------------------------------------------------------------
# Lazy, thread-safe singleton
# --------------------------------------------------------------------------

MODEL_VERSION = "2"  # bump when training code changes to invalidate caches
CACHE_DIR = Path(__file__).resolve().parents[2] / ".model_cache"

_bundle: ModelBundle | None = None
_lock = threading.Lock()


def _cache_path(path: Path) -> Path:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return CACHE_DIR / f"risk_model_v{MODEL_VERSION}_{h}.joblib"


def get_bundle() -> ModelBundle:
    """Train once per process; reuse a disk cache keyed on data hash and code version."""
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                cp = _cache_path(DATA_PATH)
                if cp.exists():
                    try:
                        _bundle = joblib.load(cp)
                    except Exception:  # corrupt or incompatible cache -> retrain
                        _bundle = None
                if _bundle is None:
                    _bundle = train()
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        joblib.dump(_bundle, cp)
                    except OSError:
                        pass  # read-only filesystem: keep in memory only
    return _bundle


def is_ready() -> bool:
    return _bundle is not None
