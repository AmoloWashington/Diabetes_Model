"""Clinical indices and diagnostic classification.

Every formula below is quoted from its primary source:

* HOMA1-IR  = FPI [microU/ml] x FPG [mmol/l] / 22.5
  HOMA1-%B  = 20 x FPI [microU/ml] / (FPG [mmol/l] - 3.5)
  Matthews DR et al., Diabetologia 28:412-419, 1985.
* QUICKI    = 1 / (log10 FPI [microU/ml] + log10 FPG [mg/dl])
  Katz A et al., J Clin Endocrinol Metab 85:2402-2410, 2000.
* eAG       = 28.7 x HbA1c [%] - 46.7  (mg/dl)
  Nathan DM et al. (ADAG), Diabetes Care 31:1473-1478, 2008.
* TyG index = ln(TG [mg/dl] x FPG [mg/dl] / 2)
  Simental-Mendia LE et al., Metab Syndr Relat Disord 6:299-304, 2008.
* Diagnostic thresholds: American Diabetes Association, Standards of Care in
  Diabetes, section 2 (Diagnosis and Classification). In the absence of
  unequivocal hyperglycaemia a diagnosis requires two abnormal results.
* BMI categories: World Health Organization adult classification.
"""

from __future__ import annotations

import math

MGDL_PER_MMOLL = 18.016  # glucose, molar mass 180.16 g/mol
PMOLL_PER_UUML = 6.0  # insulin; some laboratories use 6.945


def mgdl_to_mmoll(g_mgdl: float) -> float:
    return g_mgdl / MGDL_PER_MMOLL


def homa1(fpg_mgdl: float, fpi_uuml: float) -> dict:
    _check(fpg_mgdl, 20, 600, "Fasting glucose (mg/dl)")
    _check(fpi_uuml, 0.5, 300, "Fasting insulin (microU/ml)")
    g = mgdl_to_mmoll(fpg_mgdl)
    ir = fpi_uuml * g / 22.5
    beta = 20.0 * fpi_uuml / (g - 3.5) if g > 3.5 else None
    return {
        "homa_ir": ir,
        "homa_beta_pct": beta,
        "note": (
            "HOMA1 (1985) linear approximations. HOMA-%B is undefined when "
            "FPG <= 3.5 mmol/l. Population cut-offs for insulin resistance vary "
            "by ethnicity and assay; no universal threshold exists."
        ),
    }


def quicki(fpg_mgdl: float, fpi_uuml: float) -> float:
    _check(fpg_mgdl, 20, 600, "Fasting glucose (mg/dl)")
    _check(fpi_uuml, 0.5, 300, "Fasting insulin (microU/ml)")
    return 1.0 / (math.log10(fpi_uuml) + math.log10(fpg_mgdl))


def eag_from_a1c(a1c_pct: float) -> dict:
    _check(a1c_pct, 3.0, 20.0, "HbA1c (%)")
    mgdl = 28.7 * a1c_pct - 46.7
    return {"eag_mg_dl": mgdl, "eag_mmol_l": mgdl / MGDL_PER_MMOLL}


def tyg_index(tg_mgdl: float, fpg_mgdl: float) -> float:
    _check(tg_mgdl, 10, 5000, "Triglycerides (mg/dl)")
    _check(fpg_mgdl, 20, 600, "Fasting glucose (mg/dl)")
    return math.log(tg_mgdl * fpg_mgdl / 2.0)


def bmi(weight_kg: float, height_cm: float) -> dict:
    _check(weight_kg, 2, 400, "Weight (kg)")
    _check(height_cm, 40, 260, "Height (cm)")
    v = weight_kg / (height_cm / 100.0) ** 2
    if v < 18.5:
        cat = "Underweight"
    elif v < 25:
        cat = "Normal weight"
    elif v < 30:
        cat = "Overweight"
    else:
        cat = "Obesity"
    return {
        "bmi": v,
        "who_category": cat,
        "note": "ADA recommends screening from BMI >= 23 kg/m2 in Asian American adults.",
    }


def ada_classification(
    fpg_mgdl: float | None = None,
    ogtt_2h_mgdl: float | None = None,
    a1c_pct: float | None = None,
    random_pg_mgdl: float | None = None,
    classic_symptoms: bool = False,
) -> dict:
    """Classify each supplied test against ADA diagnostic thresholds."""
    results = []
    if fpg_mgdl is not None:
        _check(fpg_mgdl, 20, 600, "Fasting glucose (mg/dl)")
        cat = "diabetes" if fpg_mgdl >= 126 else "prediabetes (IFG)" if fpg_mgdl >= 100 else "normal"
        results.append({"test": "Fasting plasma glucose", "value": fpg_mgdl, "unit": "mg/dl",
                        "category": cat, "thresholds": "normal <100; IFG 100-125; diabetes >=126"})
    if ogtt_2h_mgdl is not None:
        _check(ogtt_2h_mgdl, 20, 800, "2-h OGTT glucose (mg/dl)")
        cat = "diabetes" if ogtt_2h_mgdl >= 200 else "prediabetes (IGT)" if ogtt_2h_mgdl >= 140 else "normal"
        results.append({"test": "2-h plasma glucose, 75-g OGTT", "value": ogtt_2h_mgdl, "unit": "mg/dl",
                        "category": cat, "thresholds": "normal <140; IGT 140-199; diabetes >=200"})
    if a1c_pct is not None:
        _check(a1c_pct, 3.0, 20.0, "HbA1c (%)")
        cat = "diabetes" if a1c_pct >= 6.5 else "prediabetes" if a1c_pct >= 5.7 else "normal"
        results.append({"test": "HbA1c", "value": a1c_pct, "unit": "%",
                        "category": cat, "thresholds": "normal <5.7; prediabetes 5.7-6.4; diabetes >=6.5"})
    if random_pg_mgdl is not None:
        _check(random_pg_mgdl, 20, 1000, "Random plasma glucose (mg/dl)")
        if random_pg_mgdl >= 200 and classic_symptoms:
            cat = "diabetes"
        elif random_pg_mgdl >= 200:
            cat = "abnormal - diagnostic only with classic symptoms or crisis"
        else:
            cat = "not diagnostic"
        results.append({"test": "Random plasma glucose", "value": random_pg_mgdl, "unit": "mg/dl",
                        "category": cat, "thresholds": ">=200 with classic symptoms of hyperglycaemia"})

    n_diab = sum(r["category"] == "diabetes" for r in results)
    if n_diab >= 2 or any(r["test"] == "Random plasma glucose" and r["category"] == "diabetes" for r in results):
        summary = "Meets ADA criteria for diabetes."
    elif n_diab == 1:
        summary = ("One test in the diabetes range. ADA requires confirmation by a second "
                   "abnormal test (same sample or repeat) unless hyperglycaemia is unequivocal.")
    elif any("prediabetes" in r["category"] for r in results):
        summary = "Results in the prediabetes (increased risk) range."
    elif results:
        summary = "No test in the diabetes or prediabetes range."
    else:
        summary = "No test values supplied."
    return {"results": results, "summary": summary,
            "disclaimer": "Educational classification only; diagnosis requires a clinician."}


def _check(v: float, lo: float, hi: float, name: str) -> None:
    if v is None or not math.isfinite(v) or not (lo <= v <= hi):
        raise ValueError(f"{name} must be within {lo}-{hi}")
