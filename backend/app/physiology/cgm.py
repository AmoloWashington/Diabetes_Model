"""Continuous glucose monitoring (CGM) metrics.

Definitions and targets follow the international consensus on time in range:
Battelino T, et al. Clinical targets for continuous glucose monitoring data
interpretation. Diabetes Care 42:1593-1603, 2019.

* Ranges (mg/dl): very low < 54, low 54-69, in range 70-180, high 181-250,
  very high > 250.
* Targets for most adults with type 1 or type 2 diabetes: time in range > 70%,
  below 70 mg/dl < 4%, below 54 mg/dl < 1%, above 180 mg/dl < 25%,
  above 250 mg/dl < 5%. Glycaemic variability: coefficient of variation <= 36%.
* Data sufficiency: at least 14 days with at least 70% of possible readings.

Glucose management indicator (Bergenstal RM, et al. Diabetes Care
41:2275-2280, 2018):  GMI (%) = 3.31 + 0.02392 x mean glucose (mg/dl).

The ambulatory glucose profile (AGP) here is the 5th/25th/50th/75th/95th
percentile of all readings in each 30-minute bin of the day, without smoothing.
Percentages are fractions of readings, which equal fractions of time when the
sampling interval is regular.
"""

from __future__ import annotations

import io
import math

import numpy as np

from .indices import MGDL_PER_MMOLL

BANDS = [  # (key, label, predicate on glucose in mg/dl); boundaries hold for non-integer values too
    ("very_low", "Very low (<54)", lambda g: g < 54.0),
    ("low", "Low (54–69)", lambda g: (g >= 54.0) & (g < 70.0)),
    ("in_range", "In range (70–180)", lambda g: (g >= 70.0) & (g <= 180.0)),
    ("high", "High (181–250)", lambda g: (g > 180.0) & (g <= 250.0)),
    ("very_high", "Very high (>250)", lambda g: g > 250.0),
]

TARGETS = [  # (metric key, comparison, value %, description)
    ("in_range", ">", 70.0, "Time in range 70–180 mg/dl above 70%"),
    ("below_70", "<", 4.0, "Time below 70 mg/dl under 4%"),
    ("very_low", "<", 1.0, "Time below 54 mg/dl under 1%"),
    ("above_180", "<", 25.0, "Time above 180 mg/dl under 25%"),
    ("very_high", "<", 5.0, "Time above 250 mg/dl under 5%"),
    ("cv", "<=", 36.0, "Coefficient of variation at or below 36%"),
]

TIME_WORDS = ("timestamp", "time", "date")
GLUCOSE_WORDS = ("glucose", "sgv", "bg", "mg/dl", "mmol")


def gmi_percent(mean_mgdl: float) -> float:
    return 3.31 + 0.02392 * mean_mgdl


def _detect_unit(g: np.ndarray, unit: str) -> str:
    if unit in ("mg/dl", "mmol/l"):
        return unit
    # Median CGM glucose in mg/dl is essentially never below 35; in mmol/l it rarely exceeds 25.
    return "mmol/l" if float(np.median(g)) < 35.0 else "mg/dl"


def parse_csv(text: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Find a timestamp column and a glucose column in a CGM export.

    Generic CSVs need a header with a time-like and a glucose-like column name.
    Exports with metadata lines above the header (e.g. LibreView) are handled by
    searching the first 20 lines for the header. Non-numeric glucose entries
    (e.g. "Low"/"High") are skipped and counted.
    """
    import pandas as pd

    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("The file has too few lines to contain CGM readings.")
    header_idx = None
    for i, line in enumerate(lines[:20]):
        low = line.lower()
        if any(w in low for w in TIME_WORDS) and any(w in low for w in GLUCOSE_WORDS):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No header with a time column and a glucose column was found in the first 20 lines.")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), sep=None, engine="python")
    cols = {c: str(c).lower() for c in df.columns}
    tcol = next((c for c, l in cols.items() if "timestamp" in l), None) or next(
        (c for c, l in cols.items() if any(w in l for w in TIME_WORDS)), None)
    gcands = [c for c, l in cols.items() if any(w in l for w in GLUCOSE_WORDS) and c != tcol]
    # Prefer automatic/historic sensor readings over manual scans or strips
    gcol = next((c for c in gcands if any(k in cols[c] for k in ("historic", "value", "sgv"))), None) or (gcands[0] if gcands else None)
    if tcol is None or gcol is None:
        raise ValueError("Could not identify the time and glucose columns.")
    g_raw = pd.to_numeric(df[gcol], errors="coerce")
    t = pd.to_datetime(df[tcol], errors="coerce", format="mixed")
    if t.isna().mean() > 0.5:
        t = pd.to_datetime(df[tcol], errors="coerce", format="mixed", dayfirst=True)
    keep = g_raw.notna() & t.notna()
    skipped_text = int((df[gcol].notna() & g_raw.isna()).sum())
    if keep.sum() < 10:
        raise ValueError("Fewer than 10 readings with both a valid time and a numeric glucose value.")
    order = np.argsort(t[keep].values)
    ts = t[keep].values[order]
    g = g_raw[keep].to_numpy(dtype=float)[order]
    minutes = (ts - ts[0]).astype("timedelta64[s]").astype(float) / 60.0
    start = pd.Timestamp(ts[0])
    meta = {"time_column": str(tcol), "glucose_column": str(gcol), "skipped_non_numeric": skipped_text,
            "start": start.isoformat(), "end": pd.Timestamp(ts[-1]).isoformat(),
            "start_minute_of_day": start.hour * 60 + start.minute + start.second / 60.0}
    return minutes, g, meta


def analyze(minutes: np.ndarray, glucose: np.ndarray, unit: str = "auto", start_minute_of_day: float = 0.0) -> dict:
    t = np.asarray(minutes, dtype=float)
    g = np.asarray(glucose, dtype=float)
    if t.shape != g.shape or t.size < 10:
        raise ValueError("Need at least 10 paired readings.")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(t)):
        raise ValueError("Readings must be finite numbers.")
    used_unit = _detect_unit(g, unit)
    g_mgdl = g * MGDL_PER_MMOLL if used_unit == "mmol/l" else g.copy()
    if g_mgdl.min() < 10 or g_mgdl.max() > 700:
        raise ValueError("Glucose values outside 10–700 mg/dl; check the unit.")
    order = np.argsort(t)
    t, g_mgdl = t[order], g_mgdl[order]

    n = g_mgdl.size
    mean = float(g_mgdl.mean())
    sd = float(g_mgdl.std(ddof=1))
    dt = np.diff(t)
    interval = float(np.median(dt[dt > 0])) if np.any(dt > 0) else float("nan")
    span_min = float(t[-1] - t[0]) + (interval if math.isfinite(interval) else 0.0)
    expected = span_min / interval if interval and math.isfinite(interval) else float("nan")
    pct = {k: 100.0 * float(np.mean(f(g_mgdl))) for k, _, f in BANDS}
    pct["below_70"] = pct["very_low"] + pct["low"]
    pct["above_180"] = pct["high"] + pct["very_high"]
    cv = 100.0 * sd / mean

    checks = []
    for key, op, val, desc in TARGETS:
        x = cv if key == "cv" else pct[key]
        ok = x > val if op == ">" else x < val if op == "<" else x <= val
        checks.append({"metric": key, "target": f"{op} {val:g}%", "value": x, "met": bool(ok), "description": desc})

    # AGP: percentiles per 30-minute bin of the day
    tod = (start_minute_of_day + t) % 1440.0
    bins = (tod // 30).astype(int)
    agp = []
    for b in range(48):
        v = g_mgdl[bins == b]
        if v.size >= 3:
            p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
            agp.append({"minute": b * 30 + 15, "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95, "n": int(v.size)})
        else:
            agp.append({"minute": b * 30 + 15, "p5": None, "p25": None, "p50": None, "p75": None, "p95": None, "n": int(v.size)})

    days = span_min / 1440.0
    wear = 100.0 * n / expected if expected and math.isfinite(expected) else None
    notes = []
    if days < 14:
        notes.append(f"Only {days:.1f} days of data; the consensus recommends at least 14 days.")
    if wear is not None and wear < 70:
        notes.append(f"Readings cover {wear:.0f}% of the period; the consensus recommends at least 70%.")
    if used_unit != unit and unit == "auto":
        notes.append(f"Values were interpreted as {used_unit}.")
    return {
        "n_readings": n, "unit_detected": used_unit, "days": days,
        "sampling_interval_min": interval, "wear_pct": wear,
        "mean_mg_dl": mean, "mean_mmol_l": mean / MGDL_PER_MMOLL, "sd_mg_dl": sd, "cv_pct": cv,
        "gmi_pct": gmi_percent(mean), "gmi_mmol_mol": 10.929 * (gmi_percent(mean) - 2.15),
        "bands": [{"key": k, "label": lab, "pct": pct[k]} for k, lab, _ in BANDS],
        "below_70_pct": pct["below_70"], "above_180_pct": pct["above_180"],
        "targets": checks, "agp": agp, "notes": notes,
        "series": _downsample(t, g_mgdl, start_minute_of_day),
    }


def _downsample(t: np.ndarray, g: np.ndarray, start_tod: float, max_points: int = 2000) -> list[dict]:
    step = max(1, int(math.ceil(t.size / max_points)))
    return [{"day": float((start_tod + ti) / 1440.0), "g": float(gi)} for ti, gi in zip(t[::step], g[::step])]


def synthetic_sample(phenotype: str = "type2", days: int = 14, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic CGM trace from the Dalla Man meal model: three meals a day with randomised
    timing and size, simulated day by day from the previous day's end state, sampled every
    5 minutes with 5% multiplicative noise. For demonstration only; not patient data."""
    from .uva_padova import Meal, simulate_meals

    if not 1 <= days <= 28:
        raise ValueError("days must be within 1–28")
    rng = np.random.default_rng(seed)
    ts, gs, x = [], [], None
    for d in range(days):
        meals = [Meal(float(np.clip(h * 60 + rng.normal(0, 25), 0, 1430)), float(max(10.0, c * rng.uniform(0.6, 1.4))))
                 for h, c in ((7.5, 50), (13.0, 70), (19.5, 80))]
        sim = simulate_meals(meals, duration_min=1440.0 - 5.0, phenotype=phenotype, dt_min=5.0, initial_x=x)
        x = sim.x_end
        ts.append(sim.t + d * 1440.0)
        gs.append(sim.G * rng.normal(1.0, 0.05, sim.G.size))
    return np.concatenate(ts), np.clip(np.concatenate(gs), 40.0, 400.0)
