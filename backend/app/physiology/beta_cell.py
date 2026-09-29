"""Long-term beta-cell mass, insulin and glucose dynamics (the betaIG model).

Implementation of

    B. Topp, K. Promislow, G. deVries, R. M. Miura, D. T. Finegood,
    "A model of beta-cell mass, insulin, and glucose kinetics:
    pathways to diabetes",
    Journal of Theoretical Biology 206(4):605-619, 2000.
    doi:10.1006/jtbi.2000.2150

    dG/dt    = R0 - (EG0 + SI*I) * G
    dI/dt    = beta * sigma * G^2 / (alpha + G^2) - k * I
    dbeta/dt = (-d0 + r1*G - r2*G^2) * beta

Time is in days, G in mg/dl, I in microU/ml, beta (beta-cell mass) in mg.

With the published parameters the beta-cell equation has two non-trivial
glucose fixed points, the roots of r2*G^2 - r1*G + d0 = 0: G = 100 mg/dl
(physiological, stable) and G = 250 mg/dl (saddle). A third fixed point with
beta = 0 (G = R0/EG0 = 600 mg/dl) is the pathological, diabetic state.
Because the non-trivial glucose fixed points do not depend on SI, beta-cell
mass compensates for insulin resistance (beta* is proportional to 1/SI) until
the compensation is outrun, which is the model's "pathway to diabetes".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class TopParams:
    R0: float = 864.0  # mg/dl/day, net glucose production at zero glucose
    EG0: float = 1.44  # 1/day, glucose effectiveness at zero insulin
    SI: float = 0.72  # ml/microU/day, insulin sensitivity
    sigma: float = 43.2  # microU/ml/day/mg, maximal secretion rate per beta-cell mass
    alpha: float = 20000.0  # mg^2/dl^2, Hill coefficient scale
    k: float = 432.0  # 1/day, insulin clearance
    d0: float = 0.06  # 1/day, beta-cell death rate at zero glucose
    r1: float = 0.84e-3  # dl/mg/day
    r2: float = 0.24e-5  # dl^2/mg^2/day


TOPP_PARAMS = TopParams()


def rhs(_t: float, y: np.ndarray, p: TopParams, si: float) -> np.ndarray:
    G, I, B = y
    dG = p.R0 - (p.EG0 + si * I) * G
    dI = B * p.sigma * G**2 / (p.alpha + G**2) - p.k * I
    dB = (-p.d0 + p.r1 * G - p.r2 * G**2) * B
    return np.array([dG, dI, dB])


def jacobian(y: np.ndarray, p: TopParams, si: float) -> np.ndarray:
    G, I, B = y
    hill = G**2 / (p.alpha + G**2)
    dhill = 2.0 * p.alpha * G / (p.alpha + G**2) ** 2
    return np.array([
        [-(p.EG0 + si * I), -si * G, 0.0],
        [B * p.sigma * dhill, -p.k, p.sigma * hill],
        [(p.r1 - 2.0 * p.r2 * G) * B, 0.0, -p.d0 + p.r1 * G - p.r2 * G**2],
    ])


def fixed_points(p: TopParams = TOPP_PARAMS, si: float | None = None) -> list[dict]:
    """All biologically meaningful fixed points with linear stability."""
    si = p.SI if si is None else si
    if si <= 0:
        raise ValueError("SI must be positive")
    pts: list[tuple[str, np.ndarray]] = []
    disc = p.r1**2 - 4.0 * p.r2 * p.d0
    if disc >= 0:
        for label, G in (
            ("physiological", (p.r1 - np.sqrt(disc)) / (2.0 * p.r2)),
            ("saddle (threshold)", (p.r1 + np.sqrt(disc)) / (2.0 * p.r2)),
        ):
            I = (p.R0 / G - p.EG0) / si
            if I <= 0:
                continue
            B = p.k * I * (p.alpha + G**2) / (p.sigma * G**2)
            pts.append((label, np.array([G, I, B])))
    pts.append(("pathological (beta-cell loss)", np.array([p.R0 / p.EG0, 0.0, 0.0])))

    out = []
    for label, y in pts:
        eig = np.linalg.eigvals(jacobian(y, p, si))
        re = eig.real
        if np.all(re < 0):
            stability = "stable"
        elif np.all(re > 0):
            stability = "unstable"
        else:
            stability = "saddle"
        out.append({
            "label": label,
            "glucose_mg_dl": float(y[0]),
            "insulin_uU_ml": float(y[1]),
            "beta_cell_mass_mg": float(y[2]),
            "eigenvalues_per_day": [
                {"real": float(e.real), "imag": float(e.imag)} for e in eig
            ],
            "stability": stability,
        })
    return out


def simulate(
    years: float = 10.0,
    si_final_fraction: float = 0.3,
    si_decline_years: float = 5.0,
    initial: tuple[float, float, float] | None = None,
    p: TopParams = TOPP_PARAMS,
    samples_per_year: int = 73,
) -> dict:
    """Simulate a progressive decline in insulin sensitivity.

    SI falls linearly from its published value to ``si_final_fraction`` times
    that value over ``si_decline_years`` and then stays constant. The system
    starts at the physiological fixed point unless ``initial`` is given.
    """
    if not (0.1 <= years <= 50.0):
        raise ValueError("years must be within 0.1-50")
    if not (0.01 <= si_final_fraction <= 2.0):
        raise ValueError("si_final_fraction must be within 0.01-2")
    if not (0.0 <= si_decline_years <= years):
        raise ValueError("si_decline_years must be within 0 and years")

    si0 = p.SI

    def si_of_t(t_days: float) -> float:
        if si_decline_years == 0:
            return si0 * si_final_fraction
        frac = min(t_days / (si_decline_years * 365.0), 1.0)
        return si0 * (1.0 + (si_final_fraction - 1.0) * frac)

    if initial is None:
        fp = fixed_points(p, si0)[0]
        y0 = np.array([fp["glucose_mg_dl"], fp["insulin_uU_ml"], fp["beta_cell_mass_mg"]])
    else:
        y0 = np.array(initial, dtype=float)
        if np.any(y0 < 0):
            raise ValueError("Initial state must be non-negative")

    T = years * 365.0
    n = max(int(years * samples_per_year), 50)
    t_eval = np.linspace(0.0, T, n + 1)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, p, si_of_t(t)),
        (0.0, T), y0, method="Radau", t_eval=t_eval, rtol=1e-8, atol=1e-8,
        jac=lambda t, y: jacobian(y, p, si_of_t(t)),
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    G, I, B = sol.y
    si_t = np.array([si_of_t(t) for t in sol.t])
    final_fp = fixed_points(p, si_of_t(T))
    diabetic = bool(G[-1] >= 126.0)
    return {
        "t_years": (sol.t / 365.0).tolist(),
        "glucose_mg_dl": G.tolist(),
        "insulin_uU_ml": I.tolist(),
        "beta_cell_mass_mg": B.tolist(),
        "si": si_t.tolist(),
        "final_fixed_points": final_fp,
        "outcome": (
            "Glucose exceeded the 126 mg/dl fasting threshold: compensation failed."
            if diabetic else
            "Beta-cell mass compensated; fasting glucose remains below 126 mg/dl."
        ),
        "diabetic_at_end": diabetic,
        "params": asdict(p),
    }


def with_overrides(**kw: float) -> TopParams:
    return replace(TOPP_PARAMS, **{k: v for k, v in kw.items() if v is not None})
