"""Membrane and transport biophysics from first principles.

Physical constants are the exact SI (2019) definitions:
    k_B = 1.380649e-23 J/K, N_A = 6.02214076e23 /mol, e = 1.602176634e-19 C
    R = k_B N_A,  F = e N_A

Laws implemented
----------------
* Nernst equation (equilibrium potential of ion X with valence z):
      E_X = (R T / z F) ln([X]_out / [X]_in)
* Goldman-Hodgkin-Katz (GHK) voltage equation for monovalent K+, Na+, Cl-:
      V_m = (R T / F) ln( (P_K[K]o + P_Na[Na]o + P_Cl[Cl]i) / (P_K[K]i + P_Na[Na]i + P_Cl[Cl]o) )
  (Goldman 1943; Hodgkin & Katz 1949). Divalent ions such as Ca2+ are not
  part of this form of the equation.
* Electrochemical driving force on ion X at membrane potential V: V - E_X.
* Stokes-Einstein diffusion coefficient of a sphere of hydrodynamic radius r
  in a fluid of viscosity eta:  D = k_B T / (6 pi eta r)
* Mean-squared displacement of free diffusion in d dimensions:
      <x^2> = 2 d D t,  so the characteristic time to diffuse a distance L is
      t = L^2 / (2 d D).

Default concentrations are the typical mammalian-cell values tabulated in
Alberts et al., Molecular Biology of the Cell (intracellular K+ 140, Na+ 5-15,
Cl- 5-15, Ca2+ ~1e-4 mM; extracellular K+ 5, Na+ 145, Cl- 110, Ca2+ 1-2 mM).
Default resting permeability ratios P_K : P_Na : P_Cl = 1 : 0.04 : 0.45 are
those reported for the squid giant axon by Hodgkin & Katz (1949); they are
labelled as such and are user-adjustable.
"""

from __future__ import annotations

import math

import numpy as np

K_B = 1.380649e-23  # J/K (exact)
N_A = 6.02214076e23  # 1/mol (exact)
E_CHARGE = 1.602176634e-19  # C (exact)
R_GAS = K_B * N_A  # J/(mol K)
FARADAY = E_CHARGE * N_A  # C/mol

DEFAULT_CONC_MM = {
    "K": {"in": 140.0, "out": 5.0, "z": 1},
    "Na": {"in": 10.0, "out": 145.0, "z": 1},
    "Cl": {"in": 10.0, "out": 110.0, "z": -1},
    "Ca": {"in": 1e-4, "out": 2.0, "z": 2},
}
SQUID_AXON_PERMEABILITY = {"K": 1.0, "Na": 0.04, "Cl": 0.45}


def _kelvin(temperature_c: float) -> float:
    if not (-20.0 <= temperature_c <= 60.0):
        raise ValueError("Temperature must be within -20 to 60 degC")
    return temperature_c + 273.15


def thermal_voltage_mv(temperature_c: float = 37.0) -> float:
    """R T / F in millivolts."""
    return 1000.0 * R_GAS * _kelvin(temperature_c) / FARADAY


def nernst_mv(z: int, c_out: float, c_in: float, temperature_c: float = 37.0) -> float:
    if z == 0:
        raise ValueError("Valence must be non-zero")
    if c_out <= 0 or c_in <= 0:
        raise ValueError("Concentrations must be positive")
    return thermal_voltage_mv(temperature_c) / z * math.log(c_out / c_in)


def ghk_voltage_mv(perm: dict[str, float], conc: dict[str, dict[str, float]], temperature_c: float = 37.0) -> float:
    """GHK voltage for K+, Na+, Cl- (relative permeabilities, concentrations in mM)."""
    for ion in ("K", "Na", "Cl"):
        if perm.get(ion, 0.0) < 0:
            raise ValueError("Permeabilities must be non-negative")
        for side in ("in", "out"):
            if conc[ion][side] <= 0:
                raise ValueError("Concentrations must be positive")
    pk, pna, pcl = perm.get("K", 0.0), perm.get("Na", 0.0), perm.get("Cl", 0.0)
    num = pk * conc["K"]["out"] + pna * conc["Na"]["out"] + pcl * conc["Cl"]["in"]
    den = pk * conc["K"]["in"] + pna * conc["Na"]["in"] + pcl * conc["Cl"]["out"]
    if num <= 0 or den <= 0:
        raise ValueError("At least one permeability must be positive")
    return thermal_voltage_mv(temperature_c) * math.log(num / den)


def membrane_analysis(conc: dict[str, dict[str, float]] | None = None, perm: dict[str, float] | None = None,
                      temperature_c: float = 37.0) -> dict:
    conc = conc or DEFAULT_CONC_MM
    perm = perm or SQUID_AXON_PERMEABILITY
    z = {k: DEFAULT_CONC_MM[k]["z"] for k in DEFAULT_CONC_MM}
    eq = {ion: nernst_mv(z[ion], conc[ion]["out"], conc[ion]["in"], temperature_c) for ion in conc if ion in z}
    vm = ghk_voltage_mv(perm, conc, temperature_c)
    # Sweep: membrane potential as K+ permeability falls (e.g. K_ATP channel closure),
    # holding P_Na and P_Cl fixed.
    fracs = np.linspace(1.0, 0.02, 50)
    sweep = [{"pk_fraction": float(f), "vm_mv": ghk_voltage_mv({**perm, "K": perm["K"] * f}, conc, temperature_c)} for f in fracs]
    return {
        "temperature_c": temperature_c,
        "thermal_voltage_mv": thermal_voltage_mv(temperature_c),
        "equilibrium_potentials_mv": eq,
        "ghk_vm_mv": vm,
        "driving_force_mv": {ion: vm - e for ion, e in eq.items()},
        "pk_sweep": sweep,
        "notes": [
            "GHK voltage equation includes only the monovalent ions K+, Na+ and Cl-.",
            "Default permeability ratios are squid giant axon values (Hodgkin & Katz 1949).",
            "Driving force V_m - E_X: positive means net outward flux of cations (inward for anions).",
        ],
    }


def stokes_einstein(radius_nm: float, temperature_c: float = 37.0, viscosity_mpa_s: float = 0.69) -> float:
    """Diffusion coefficient in m^2/s. Default viscosity: water at 37 degC (~0.69 mPa s)."""
    if radius_nm <= 0 or viscosity_mpa_s <= 0:
        raise ValueError("Radius and viscosity must be positive")
    return K_B * _kelvin(temperature_c) / (6.0 * math.pi * viscosity_mpa_s * 1e-3 * radius_nm * 1e-9)


def diffusion_time_s(distance_um: float, d_m2_s: float, dims: int = 3) -> float:
    if dims not in (1, 2, 3):
        raise ValueError("dims must be 1, 2 or 3")
    if distance_um <= 0 or d_m2_s <= 0:
        raise ValueError("Distance and D must be positive")
    return (distance_um * 1e-6) ** 2 / (2 * dims * d_m2_s)


def diffusion_analysis(radius_nm: float, distance_um: float, temperature_c: float = 37.0,
                       viscosity_mpa_s: float = 0.69, dims: int = 3) -> dict:
    d = stokes_einstein(radius_nm, temperature_c, viscosity_mpa_s)
    t = diffusion_time_s(distance_um, d, dims)
    times = np.linspace(0, 4 * t, 60)
    return {
        "D_m2_s": d,
        "D_um2_s": d * 1e12,
        "time_to_distance_s": t,
        "rms_displacement_um": [{"t_s": float(tt), "rms_um": float(math.sqrt(2 * dims * d * tt) * 1e6)} for tt in times],
        "notes": [
            "Stokes-Einstein assumes a rigid sphere in a continuous Newtonian fluid; cytoplasm is crowded and "
            "effective diffusion of large particles is slower than in water.",
            "Default viscosity 0.69 mPa s is that of water at 37 °C.",
        ],
    }
