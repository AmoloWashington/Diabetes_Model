"""Meal simulation model of the glucose-insulin system.

Implementation of

    C. Dalla Man, R. A. Rizza, C. Cobelli,
    "Meal Simulation Model of the Glucose-Insulin System",
    IEEE Transactions on Biomedical Engineering 54(10):1740-1749, 2007.
    doi:10.1109/TBME.2007.893506

This model is the physiological core of the UVA/Padova simulator, accepted by
the US FDA in 2008 as a substitute for pre-clinical animal trials in the
testing of closed-loop insulin control algorithms.

Units follow the paper:
    glucose masses      mg/kg          plasma glucose G     mg/dl
    insulin masses      pmol/kg        plasma insulin I     pmol/l
    meal amounts        mg             time                 min

Parameters in ``NORMAL_PARAMS`` are the average normal-subject values of
Table 1 of the paper. Basal-state quantities that the paper derives from
steady-state constraints (kp1, S_b, I_lb, I_pb, G_tb, EGP_b, m6, m3(0)) are
*computed* here from the chosen basal glucose and insulin, so that the model
starts exactly at equilibrium for any physiologically valid basal state.

Phenotypes other than "normal" are produced by scaling insulin-action and
beta-cell parameters of the normal set. They are illustrative, transparent
parameter perturbations and are NOT the published type 2 diabetes parameter
set of the paper; they are labelled as such in every API response.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UVAPadovaParams:
    # Glucose kinetics
    VG: float = 1.88  # dl/kg, distribution volume of glucose
    k1: float = 0.065  # 1/min
    k2: float = 0.079  # 1/min
    # Insulin kinetics
    VI: float = 0.05  # l/kg, distribution volume of insulin
    m1: float = 0.190  # 1/min
    m2: float = 0.484  # 1/min
    m4: float = 0.194  # 1/min
    m5: float = 0.0304  # min*kg/pmol
    HEb: float = 0.6  # basal hepatic insulin extraction (dimensionless)
    # Rate of appearance (gastro-intestinal tract)
    kmax: float = 0.0558  # 1/min
    kmin: float = 0.0080  # 1/min
    kabs: float = 0.057  # 1/min
    kgri: float = 0.0558  # 1/min
    f: float = 0.90  # fraction of intestinal absorption appearing in plasma
    b: float = 0.82  # fraction of dose at which kempt decreases to (kmax+kmin)/2
    d: float = 0.010  # fraction of dose at which kempt recovers to (kmax+kmin)/2
    # Endogenous glucose production
    kp2: float = 0.0021  # 1/min, liver glucose effectiveness
    kp3: float = 0.009  # mg/kg/min per pmol/l, amplitude of insulin action on liver
    kp4: float = 0.0618  # mg/kg/min per pmol/kg, portal insulin action on liver
    ki: float = 0.0079  # 1/min, delay between insulin signal and action on liver
    # Glucose utilisation
    Fcns: float = 1.0  # mg/kg/min, insulin-independent utilisation (brain, RBC)
    Vm0: float = 2.50  # mg/kg/min
    Vmx: float = 0.047  # mg/kg/min per pmol/l, insulin sensitivity of utilisation
    Km0: float = 225.59  # mg/kg
    p2U: float = 0.0331  # 1/min, rate of insulin action on peripheral utilisation
    # Insulin secretion
    K: float = 2.30  # pmol/kg per mg/dl, beta-cell responsivity to dG/dt
    alpha: float = 0.050  # 1/min, delay between glucose and secretion
    beta: float = 0.11  # pmol/kg/min per mg/dl, beta-cell responsivity to glucose
    gamma: float = 0.5  # 1/min, transfer rate portal vein -> liver
    # Renal excretion
    ke1: float = 0.0005  # 1/min, glomerular filtration rate
    ke2: float = 339.0  # mg/kg, renal threshold of glucose


NORMAL_PARAMS = UVAPadovaParams()

# Basal state of the average normal subject reported with Table 1.
NORMAL_BASAL_GLUCOSE = 91.76  # mg/dl
NORMAL_BASAL_INSULIN = 25.49  # pmol/l


@dataclass(frozen=True)
class Phenotype:
    key: str
    label: str
    description: str
    insulin_sensitivity: float  # multiplies Vmx and kp3
    beta_cell_function: float  # multiplies K and beta
    basal_glucose: float  # mg/dl
    basal_insulin: float  # pmol/l
    published: bool


PHENOTYPES: dict[str, Phenotype] = {
    "normal": Phenotype(
        key="normal",
        label="Healthy adult (published normal parameter set)",
        description="Average normal subject, Dalla Man et al. 2007, Table 1.",
        insulin_sensitivity=1.0,
        beta_cell_function=1.0,
        basal_glucose=NORMAL_BASAL_GLUCOSE,
        basal_insulin=NORMAL_BASAL_INSULIN,
        published=True,
    ),
    "insulin_resistant": Phenotype(
        key="insulin_resistant",
        label="Insulin resistance with compensation (illustrative)",
        description=(
            "Insulin action on utilisation and on the liver halved; beta-cell "
            "responsivity increased 1.5x and basal insulin doubled to model "
            "compensatory hyperinsulinaemia. Illustrative perturbation of the "
            "normal set, not a published parameter set."
        ),
        insulin_sensitivity=0.5,
        beta_cell_function=1.5,
        basal_glucose=98.0,
        basal_insulin=50.0,
        published=False,
    ),
    "type2": Phenotype(
        key="type2",
        label="Type 2 diabetes phenotype (illustrative)",
        description=(
            "Insulin action reduced to 30% and beta-cell responsivity reduced "
            "to 40% of normal, with fasting hyperglycaemia. Illustrative "
            "perturbation of the normal set, not the published T2D set."
        ),
        insulin_sensitivity=0.3,
        beta_cell_function=0.4,
        basal_glucose=150.0,
        basal_insulin=45.0,
        published=False,
    ),
}


def params_for(
    phenotype: str,
    insulin_sensitivity: float | None = None,
    beta_cell_function: float | None = None,
) -> tuple[UVAPadovaParams, Phenotype]:
    """Return the parameter set for a phenotype, optionally overriding scales."""
    if phenotype not in PHENOTYPES:
        raise ValueError(f"Unknown phenotype {phenotype!r}; choose from {sorted(PHENOTYPES)}")
    ph = PHENOTYPES[phenotype]
    si = ph.insulin_sensitivity if insulin_sensitivity is None else insulin_sensitivity
    bf = ph.beta_cell_function if beta_cell_function is None else beta_cell_function
    if si <= 0 or bf <= 0:
        raise ValueError("Scaling factors must be strictly positive")
    p = replace(
        NORMAL_PARAMS,
        Vmx=NORMAL_PARAMS.Vmx * si,
        kp3=NORMAL_PARAMS.kp3 * si,
        K=NORMAL_PARAMS.K * bf,
        beta=NORMAL_PARAMS.beta * bf,
    )
    return p, ph


# --------------------------------------------------------------------------
# Basal steady state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BasalState:
    Gb: float  # mg/dl
    Ib: float  # pmol/l
    Gpb: float  # mg/kg
    Gtb: float  # mg/kg
    Ipb: float  # pmol/kg
    Ilb: float  # pmol/kg
    Sb: float  # pmol/kg/min
    Ipob: float  # pmol/kg
    EGPb: float  # mg/kg/min
    Uidb: float  # mg/kg/min
    Eb: float  # mg/kg/min
    kp1: float  # mg/kg/min
    m6: float  # dimensionless
    m3b: float  # 1/min


def basal_state(p: UVAPadovaParams, Gb: float, Ib: float) -> BasalState:
    """Solve the steady-state constraints of the model for given Gb, Ib."""
    if not (40.0 <= Gb <= 400.0):
        raise ValueError("Basal glucose must be within 40-400 mg/dl")
    if not (2.0 <= Ib <= 500.0):
        raise ValueError("Basal insulin must be within 2-500 pmol/l")

    Gpb = Gb * p.VG
    # dGt/dt = 0  ->  k1*Gpb - k2*Gt - Vm0*Gt/(Km0+Gt) = 0 ; unique positive root
    def g(gt: float) -> float:
        return p.k1 * Gpb - p.k2 * gt - p.Vm0 * gt / (p.Km0 + gt)

    Gtb = brentq(g, 0.0, p.k1 * Gpb / p.k2)
    Uidb = p.Vm0 * Gtb / (p.Km0 + Gtb)
    Eb = p.ke1 * (Gpb - p.ke2) if Gpb > p.ke2 else 0.0
    # dGp/dt = 0 -> EGPb = Fcns + Eb + k1*Gpb - k2*Gtb  (= Fcns + Uidb + Eb)
    EGPb = p.Fcns + Eb + p.k1 * Gpb - p.k2 * Gtb

    m3b = p.HEb * p.m1 / (1.0 - p.HEb)
    Ipb = Ib * p.VI
    Ilb = (p.m2 + p.m4) * Ipb / p.m1  # dIp/dt = 0
    Sb = (p.m1 + m3b) * Ilb - p.m2 * Ipb  # dIl/dt = 0
    Ipob = Sb / p.gamma
    m6 = p.HEb + p.m5 * Sb
    kp1 = EGPb + p.kp2 * Gpb + p.kp3 * Ib + p.kp4 * Ipob
    return BasalState(Gb, Ib, Gpb, Gtb, Ipb, Ilb, Sb, Ipob, EGPb, Uidb, Eb, kp1, m6, m3b)


# --------------------------------------------------------------------------
# Model equations
# --------------------------------------------------------------------------

# State vector indices
GP, GT, IL, IP, I1, ID, X, QSTO1, QSTO2, QGUT, IPO, Y = range(12)
STATE_NAMES = ["Gp", "Gt", "Il", "Ip", "I1", "Id", "X", "Qsto1", "Qsto2", "Qgut", "Ipo", "Y"]


@dataclass
class _Ctx:
    p: UVAPadovaParams
    bs: BasalState
    BW: float
    last_dose: float = 0.0  # mg, amount of the most recent meal (the "D" of kempt)


def _kempt(p: UVAPadovaParams, qsto: float, D: float) -> float:
    if D <= 0.0:
        return p.kmax
    a = 5.0 / (2.0 * D * (1.0 - p.b))
    c = 5.0 / (2.0 * D * p.d)
    return p.kmin + (p.kmax - p.kmin) / 2.0 * (
        np.tanh(a * (qsto - p.b * D)) - np.tanh(c * (qsto - p.d * D)) + 2.0
    )


def _fluxes(x: np.ndarray, ctx: _Ctx) -> dict[str, float]:
    """Algebraic fluxes of the model at state x."""
    p, bs = ctx.p, ctx.bs
    Gp, Gt = x[GP], x[GT]
    I = x[IP] / p.VI
    qsto = x[QSTO1] + x[QSTO2]
    kempt = _kempt(p, qsto, ctx.last_dose)
    Ra = p.f * p.kabs * x[QGUT] / ctx.BW
    EGP = max(0.0, bs.kp1 - p.kp2 * Gp - p.kp3 * x[ID] - p.kp4 * x[IPO])
    Uii = p.Fcns
    Uid = (p.Vm0 + p.Vmx * x[X]) * Gt / (p.Km0 + Gt)
    E = p.ke1 * (Gp - p.ke2) if Gp > p.ke2 else 0.0
    dGp = EGP + Ra - Uii - E - p.k1 * Gp + p.k2 * Gt
    dG = dGp / p.VG
    G = Gp / p.VG
    # Beta-cell secretion (portal)
    Spo = x[Y] + (p.K * dG if dG > 0.0 else 0.0) + bs.Sb
    Spo = max(Spo, 0.0)
    S = p.gamma * x[IPO]
    # Hepatic extraction, constrained to its physical range [0, 1)
    HE = min(max(-p.m5 * S + bs.m6, 0.0), 0.99)
    m3 = HE * p.m1 / (1.0 - HE)
    return dict(
        G=G, I=I, Ra=Ra, EGP=EGP, Uii=Uii, Uid=Uid, E=E, dGp=dGp, dG=dG,
        Spo=Spo, S=S, HE=HE, m3=m3, kempt=kempt,
    )


def _rhs(_t: float, x: np.ndarray, ctx: _Ctx) -> np.ndarray:
    p, bs = ctx.p, ctx.bs
    fl = _fluxes(x, ctx)
    dx = np.zeros(12)
    dx[GP] = fl["dGp"]
    dx[GT] = -fl["Uid"] + p.k1 * x[GP] - p.k2 * x[GT]
    dx[IL] = -(p.m1 + fl["m3"]) * x[IL] + p.m2 * x[IP] + fl["S"]
    dx[IP] = -(p.m2 + p.m4) * x[IP] + p.m1 * x[IL]
    dx[I1] = -p.ki * (x[I1] - fl["I"])
    dx[ID] = -p.ki * (x[ID] - x[I1])
    dx[X] = -p.p2U * x[X] + p.p2U * (fl["I"] - bs.Ib)
    dx[QSTO1] = -p.kgri * x[QSTO1]
    dx[QSTO2] = -fl["kempt"] * x[QSTO2] + p.kgri * x[QSTO1]
    dx[QGUT] = -p.kabs * x[QGUT] + fl["kempt"] * x[QSTO2]
    dx[IPO] = -p.gamma * x[IPO] + fl["Spo"]
    G = fl["G"]
    if p.beta * (G - bs.Gb) >= -bs.Sb:
        dx[Y] = -p.alpha * (x[Y] - p.beta * (G - bs.Gb))
    else:
        dx[Y] = -p.alpha * x[Y] - p.alpha * bs.Sb
    return dx


def initial_state(p: UVAPadovaParams, bs: BasalState) -> np.ndarray:
    x0 = np.zeros(12)
    x0[GP], x0[GT] = bs.Gpb, bs.Gtb
    x0[IL], x0[IP] = bs.Ilb, bs.Ipb
    x0[I1] = x0[ID] = bs.Ib
    x0[IPO] = bs.Ipob
    return x0


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Meal:
    time_min: float
    carbs_g: float


@dataclass
class MealSimulation:
    t: np.ndarray
    G: np.ndarray
    I: np.ndarray
    Ra: np.ndarray
    EGP: np.ndarray
    Uid: np.ndarray
    S: np.ndarray
    E: np.ndarray
    basal: BasalState
    params: UVAPadovaParams
    phenotype: Phenotype
    meals: list[Meal] = field(default_factory=list)
    body_weight: float = 78.0


def simulate_meals(
    meals: Sequence[Meal],
    duration_min: float = 420.0,
    body_weight: float = 78.0,
    phenotype: str = "normal",
    basal_glucose: float | None = None,
    basal_insulin: float | None = None,
    insulin_sensitivity: float | None = None,
    beta_cell_function: float | None = None,
    dt_min: float = 1.0,
) -> MealSimulation:
    """Simulate plasma glucose and insulin for a meal schedule.

    Meals enter the stomach as impulses of glucose (carbohydrate) mass, as in
    the original model. The emptying-rate function uses the amount of the most
    recent meal, the standard convention for multi-meal simulation.
    """
    if not (30.0 <= body_weight <= 250.0):
        raise ValueError("Body weight must be within 30-250 kg")
    if not (10.0 <= duration_min <= 3 * 24 * 60):
        raise ValueError("Duration must be within 10 min and 3 days")
    if not (0.1 <= dt_min <= 15.0):
        raise ValueError("Output step must be within 0.1-15 min")
    for m in meals:
        if not (0.0 <= m.time_min < duration_min):
            raise ValueError("Meal times must lie inside the simulated window")
        if not (0.0 <= m.carbs_g <= 400.0):
            raise ValueError("Meal carbohydrate must be within 0-400 g")

    p, ph = params_for(phenotype, insulin_sensitivity, beta_cell_function)
    Gb = ph.basal_glucose if basal_glucose is None else basal_glucose
    Ib = ph.basal_insulin if basal_insulin is None else basal_insulin
    bs = basal_state(p, Gb, Ib)
    ctx = _Ctx(p=p, bs=bs, BW=body_weight)

    t_out = np.arange(0.0, duration_min + 1e-9, dt_min)
    x = initial_state(p, bs)
    events = sorted(meals, key=lambda m: m.time_min)
    # Group meals at the same instant
    boundaries = sorted({0.0, *[m.time_min for m in events], duration_min})

    ts: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    n_seg = len(boundaries) - 1
    for i in range(n_seg):
        t0, t1 = boundaries[i], boundaries[i + 1]
        dose = sum(m.carbs_g for m in events if m.time_min == t0) * 1000.0  # g -> mg
        if dose > 0:
            x = x.copy()
            x[QSTO1] += dose
            ctx.last_dose = dose
        sol = solve_ivp(
            _rhs, (t0, t1), x, args=(ctx,), method="LSODA",
            dense_output=True, rtol=1e-7, atol=1e-9, max_step=2.0,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        last = i == n_seg - 1
        mask = (t_out >= t0) & ((t_out <= t1) if last else (t_out < t1))
        if mask.any():
            ts.append(t_out[mask])
            xs.append(sol.sol(t_out[mask]).T)
        x = sol.y[:, -1]

    t = np.concatenate(ts)
    X_all = np.vstack(xs)
    # Recompute algebraic outputs along the trajectory
    fl = _trajectory_fluxes(X_all, t, events, ctx)
    return MealSimulation(
        t=t, G=fl["G"], I=fl["I"], Ra=fl["Ra"], EGP=fl["EGP"], Uid=fl["Uid"],
        S=fl["S"], E=fl["E"], basal=bs, params=p, phenotype=ph,
        meals=list(events), body_weight=body_weight,
    )


def _trajectory_fluxes(X_all: np.ndarray, t: np.ndarray, meals: list[Meal], ctx: _Ctx) -> dict[str, np.ndarray]:
    keys = ["G", "I", "Ra", "EGP", "Uid", "S", "E"]
    out = {k: np.empty(len(t)) for k in keys}
    for j, (tj, xj) in enumerate(zip(t, X_all)):
        prior = [m for m in meals if m.time_min <= tj and m.carbs_g > 0]
        ctx.last_dose = (
            sum(m.carbs_g for m in prior if m.time_min == prior[-1].time_min) * 1000.0 if prior else 0.0
        )
        fl = _fluxes(xj, ctx)
        for k in keys:
            out[k][j] = fl[k]
    return out


# --------------------------------------------------------------------------
# Summary metrics
# --------------------------------------------------------------------------


def summarize(sim: MealSimulation) -> dict[str, float | None]:
    t, G, I = sim.t, sim.G, sim.I
    first_meal = sim.meals[0].time_min if sim.meals else 0.0
    k_peak = int(np.argmax(G))
    two_h = first_meal + 120.0
    g2h = float(np.interp(two_h, t, G)) if two_h <= t[-1] else None
    in_range = (G >= 70.0) & (G <= 180.0)
    return {
        "basal_glucose_mg_dl": float(sim.basal.Gb),
        "basal_insulin_pmol_l": float(sim.basal.Ib),
        "peak_glucose_mg_dl": float(G[k_peak]),
        "time_to_peak_min": float(t[k_peak] - first_meal),
        "glucose_2h_mg_dl": g2h,
        "peak_insulin_pmol_l": float(np.max(I)),
        "glucose_iAUC_mg_dl_min": float(np.trapezoid(np.clip(G - sim.basal.Gb, 0, None), t)),
        "insulin_iAUC_pmol_l_min": float(np.trapezoid(np.clip(I - sim.basal.Ib, 0, None), t)),
        "time_in_range_70_180_pct": float(100.0 * np.mean(in_range)),
        "time_above_180_pct": float(100.0 * np.mean(G > 180.0)),
        "time_below_70_pct": float(100.0 * np.mean(G < 70.0)),
        "glucose_appeared_mg_per_kg": float(np.trapezoid(sim.Ra, t)),
        "renal_excretion_mg_per_kg": float(np.trapezoid(sim.E, t)),
    }
