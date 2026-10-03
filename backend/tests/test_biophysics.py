"""Membrane biophysics and diffusion against textbook values and exact physical constants."""

import math

import pytest

from app.physiology import biophysics as b


def test_constants_are_exact_si_definitions():
    assert b.R_GAS == pytest.approx(8.314462618, rel=1e-9)
    assert b.FARADAY == pytest.approx(96485.33212, rel=1e-9)


def test_thermal_voltage():
    assert b.thermal_voltage_mv(25.0) == pytest.approx(25.693, abs=1e-3)
    assert b.thermal_voltage_mv(37.0) == pytest.approx(26.727, abs=1e-3)


def test_nernst_textbook_values():
    assert b.nernst_mv(1, 5, 140) == pytest.approx(-89.06, abs=0.05)  # E_K
    assert b.nernst_mv(1, 145, 10) == pytest.approx(71.47, abs=0.05)  # E_Na
    assert b.nernst_mv(2, 2.0, 1e-4) == pytest.approx(132.3, abs=0.1)  # E_Ca, z = 2 halves the slope
    assert b.nernst_mv(-1, 110, 10) == pytest.approx(-64.09, abs=0.05)  # E_Cl
    assert b.nernst_mv(1, 10, 10) == 0.0


def test_ghk_reduces_to_nernst_for_a_single_permeant_ion():
    conc = b.DEFAULT_CONC_MM
    assert b.ghk_voltage_mv({"K": 1, "Na": 0, "Cl": 0}, conc) == pytest.approx(b.nernst_mv(1, 5, 140))
    assert b.ghk_voltage_mv({"K": 0, "Na": 1, "Cl": 0}, conc) == pytest.approx(b.nernst_mv(1, 145, 10))
    assert b.ghk_voltage_mv({"K": 0, "Na": 0, "Cl": 1}, conc) == pytest.approx(b.nernst_mv(-1, 110, 10))


def test_ghk_resting_potential_and_katp_closure_depolarises():
    a = b.membrane_analysis()
    assert -75 < a["ghk_vm_mv"] < -60
    vs = [p["vm_mv"] for p in a["pk_sweep"]]
    assert all(x < y for x, y in zip(vs, vs[1:]))  # monotonic depolarisation as P_K falls
    assert vs[-1] > -50


def test_stokes_einstein_and_msd():
    d = b.stokes_einstein(1.0, 25.0, 0.890)
    expected = 1.380649e-23 * 298.15 / (6 * math.pi * 0.890e-3 * 1e-9)
    assert d == pytest.approx(expected)
    t = b.diffusion_time_s(10, d, 3)
    assert t == pytest.approx((10e-6) ** 2 / (6 * d))
    assert b.diffusion_time_s(10, d, 1) == pytest.approx(3 * t)


@pytest.mark.parametrize("call", [
    lambda: b.nernst_mv(0, 1, 1), lambda: b.nernst_mv(1, -1, 1),
    lambda: b.ghk_voltage_mv({"K": 0, "Na": 0, "Cl": 0}, b.DEFAULT_CONC_MM),
    lambda: b.stokes_einstein(-1), lambda: b.diffusion_time_s(1, 1e-12, 4), lambda: b.thermal_voltage_mv(500),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/biophysics/membrane", json={"K": {"in": 140, "out": 20}})
        assert r.status_code == 200
        assert r.json()["equilibrium_potentials_mv"]["K"] == pytest.approx(b.nernst_mv(1, 20, 140))
        assert c.post("/api/biophysics/membrane", json={"K": {"in": -1, "out": 5}}).status_code == 422
        d = c.post("/api/biophysics/diffusion", json={"radius_nm": 2, "distance_um": 10}).json()
        assert d["D_m2_s"] == pytest.approx(b.stokes_einstein(2.0))
