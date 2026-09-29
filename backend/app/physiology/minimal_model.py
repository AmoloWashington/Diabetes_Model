"""Bergman minimal model of glucose kinetics (IVGTT).

    R. N. Bergman, Y. Z. Ider, C. R. Bowden, C. Cobelli,
    "Quantitative estimation of insulin sensitivity",
    American Journal of Physiology 236(6):E667-E677, 1979.

Glucose minimal model (the part used to estimate insulin sensitivity):

    dG/dt = -(SG + X) * G + SG * Gb           G(0) = G0
    dX/dt = -p2 * X + p3 * (I(t) - Ib)        X(0) = 0
    SI    = p3 / p2

Two uses are provided:

1. ``simulate_ivgtt`` - forward simulation of an intravenous glucose tolerance
   test. Insulin is generated with a second-phase secretion term of the form
   gamma * (G - h)+ * t (Toffolo et al. 1980), written relative to basal
   insulin, plus an optional first-phase insulin spike. Default parameter
   values are illustrative and of typical order of magnitude for healthy
   adults; they are user-adjustable and are not taken from a single study.

2. ``estimate_insulin_sensitivity`` - the classical use of the model: given
   measured glucose and insulin samples, estimate SG, SI and p2 (and G0) by
   nonlinear least squares with the measured insulin as a forcing function.
   Precision of the estimates is reported as coefficient of variation from the
   Gauss-Newton approximation of the parameter covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

GLUCOSE_VOLUME_DL_PER_KG = 1.88  # VG of Dalla Man et al. 2007, used for G0


@dataclass(frozen=True)
class IVGTTParams:
    Gb: float = 90.0  # mg/dl
    Ib: float = 10.0  # microU/ml
    SG: float = 0.025  # 1/min, glucose effectiveness
    SI: float = 5.0e-4  # 1/min per microU/ml, insulin sensitivity
    p2: float = 0.025  # 1/min, remote insulin action rate
    n: float = 0.14  # 1/min, insulin clearance rate
    gamma: float = 0.004  # microU/ml/min^2 per mg/dl, second-phase secretion
    h: float | None = None  # mg/dl, glucose threshold (defaults to Gb)
    first_phase_uU_ml: float = 80.0  # microU/ml, first-phase insulin spike above Ib
    dose_g_per_kg: float = 0.3  # glucose bolus


def simulate_ivgtt(p: IVGTTParams = IVGTTParams(), duration_min: float = 180.0, dt: float = 1.0) -> dict:
    if not (0.05 <= p.dose_g_per_kg <= 0.5):
        raise ValueError("Dose must be within 0.05-0.5 g/kg")
    if min(p.Gb, p.Ib, p.SG, p.SI, p.p2, p.n) <= 0 or p.gamma < 0:
        raise ValueError("Parameters must be positive")
    h = p.Gb if p.h is None else p.h
    G0 = p.Gb + p.dose_g_per_kg * 1000.0 / GLUCOSE_VOLUME_DL_PER_KG
    I0 = p.Ib + p.first_phase_uU_ml
    p3 = p.SI * p.p2

    def rhs(t, y):
        G, X, I = y
        return [
            -(p.SG + X) * G + p.SG * p.Gb,
            -p.p2 * X + p3 * (I - p.Ib),
            -p.n * (I - p.Ib) + p.gamma * max(G - h, 0.0) * t,
        ]

    t = np.arange(0.0, duration_min + 1e-9, dt)
    sol = solve_ivp(rhs, (0.0, duration_min), [G0, 0.0, I0], t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    G, X, I = sol.y
    return {
        "t_min": sol.t.tolist(),
        "glucose_mg_dl": G.tolist(),
        "insulin_uU_ml": I.tolist(),
        "remote_insulin_X_per_min": X.tolist(),
        "G0_mg_dl": float(G0),
        # Glucose disappearance rate Kg (%/min) from log-linear fit, 10-40 min
        "Kg_pct_per_min": _kg(sol.t, G, p.Gb),
    }


def _kg(t: np.ndarray, G: np.ndarray, Gb: float) -> float | None:
    """Glucose disappearance constant (%/min) from the 10-40 min slope of ln(G - Gb).

    Kg is classically computed on ln(G); using the above-basal excess is a
    documented variant. Returns None if the excess is not strictly positive.
    """
    m = (t >= 10) & (t <= 40)
    exc = G[m] - Gb
    if m.sum() < 3 or np.any(exc <= 0):
        return None
    slope = np.polyfit(t[m], np.log(exc), 1)[0]
    return float(-100.0 * slope)


def estimate_insulin_sensitivity(
    t: list[float],
    glucose: list[float],
    insulin: list[float],
    Gb: float | None = None,
    Ib: float | None = None,
    exclude_before_min: float = 8.0,
) -> dict:
    """Fit SG, SI, p2 and G0 of the glucose minimal model to IVGTT samples.

    ``Gb``/``Ib`` default to the first sample (t <= 0) if present, otherwise
    to the last sample. Samples before ``exclude_before_min`` are excluded from
    the glucose fit because of incomplete intravascular mixing.
    """
    t_arr = np.asarray(t, dtype=float)
    g_arr = np.asarray(glucose, dtype=float)
    i_arr = np.asarray(insulin, dtype=float)
    if not (len(t_arr) == len(g_arr) == len(i_arr)):
        raise ValueError("t, glucose and insulin must have equal length")
    if len(t_arr) < 8:
        raise ValueError("At least 8 samples are required")
    if np.any(np.diff(t_arr) <= 0):
        raise ValueError("Sample times must be strictly increasing")
    if np.any(g_arr <= 0) or np.any(i_arr < 0) or not np.all(np.isfinite(np.r_[g_arr, i_arr, t_arr])):
        raise ValueError("Samples must be finite; glucose > 0, insulin >= 0")

    basal_idx = int(np.argmax(t_arr <= 0)) if np.any(t_arr <= 0) else len(t_arr) - 1
    Gb = float(g_arr[basal_idx]) if Gb is None else float(Gb)
    Ib = float(i_arr[basal_idx]) if Ib is None else float(Ib)

    post = t_arr >= 0
    tp, gp, ip = t_arr[post], g_arr[post], i_arr[post]
    fit_mask = tp >= exclude_before_min
    if fit_mask.sum() < 5:
        raise ValueError("Too few samples after the mixing-exclusion window")

    def insulin_at(tt: float) -> float:
        return float(np.interp(tt, tp, ip))

    def model(theta: np.ndarray) -> np.ndarray:
        SG, SI, p2, G0 = np.exp(theta)
        p3 = SI * p2

        def rhs(tt, y):
            G, X = y
            return [-(SG + X) * G + SG * Gb, -p2 * X + p3 * (insulin_at(tt) - Ib)]

        sol = solve_ivp(rhs, (tp[0], tp[-1]), [G0, 0.0], t_eval=tp, method="LSODA",
                        rtol=1e-10, atol=1e-12, max_step=1.0)
        if not sol.success or sol.y.shape[1] != len(tp):
            return np.full(len(tp), 1e6)
        return sol.y[0]

    g_fit = gp[fit_mask]
    sd = np.maximum(0.02 * g_fit, 1.0)  # 2% CV measurement error, floor 1 mg/dl

    def residuals(theta):
        return (model(theta)[fit_mask] - g_fit) / sd

    G0_guess = max(float(gp[fit_mask][0]) * 1.1, Gb + 1.0)
    # Multi-start over physiologically plausible ranges; keep the best fit.
    res = None
    for sg0 in (0.01, 0.03):
        for p20 in (0.01, 0.05):
            theta0 = np.log([sg0, 5e-4, p20, G0_guess])
            cand = least_squares(residuals, theta0, method="trf", diff_step=1e-4,
                                 xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
            if res is None or cand.cost < res.cost:
                res = cand
    SG, SI, p2, G0 = np.exp(res.x)

    # Covariance of log-parameters -> CV% (delta method: CV ~ SD of log param)
    cv = {}
    try:
        JTJ = res.jac.T @ res.jac
        dof = max(len(g_fit) - 4, 1)
        s2 = float(np.sum(res.fun**2) / dof)
        cov = np.linalg.inv(JTJ) * s2
        for name, v in zip(["SG", "SI", "p2", "G0"], np.sqrt(np.clip(np.diag(cov), 0, None))):
            cv[name] = float(100.0 * v)
    except np.linalg.LinAlgError:
        cv = {k: None for k in ["SG", "SI", "p2", "G0"]}

    fitted = model(res.x)
    return {
        "SG_per_min": float(SG),
        "SI_per_min_per_uU_ml": float(SI),
        "p2_per_min": float(p2),
        "G0_mg_dl": float(G0),
        "cv_percent": cv,
        "Gb_mg_dl": Gb,
        "Ib_uU_ml": Ib,
        "fitted_t_min": tp.tolist(),
        "fitted_glucose_mg_dl": fitted.tolist(),
        "converged": bool(res.success),
        "rmse_mg_dl": float(np.sqrt(np.mean((fitted[fit_mask] - g_fit) ** 2))),
        "note": (
            "Minimal-model estimates are precise only with a frequently sampled "
            "IVGTT (FSIVGTT); CV above ~50% indicates poor identifiability."
        ),
    }
