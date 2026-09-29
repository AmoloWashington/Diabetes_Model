"""Scientific validation of the physiological engines against their publications."""

import math

import numpy as np
import pytest

from app.physiology import beta_cell, indices, minimal_model, uva_padova as up
from app.physiology.uva_padova import _Ctx, _rhs


# ------------------------------------------------------------ Dalla Man 2007

@pytest.fixture(scope="module")
def normal_basal():
    return up.basal_state(up.NORMAL_PARAMS, up.NORMAL_BASAL_GLUCOSE, up.NORMAL_BASAL_INSULIN)


def test_derived_basal_parameters_match_published_table(normal_basal):
    # kp1 = 2.70 mg/kg/min and m6 = 0.6471 are reported in Table 1 of the paper;
    # here they are derived independently from the steady-state constraints.
    assert normal_basal.kp1 == pytest.approx(2.70, abs=0.005)
    assert normal_basal.m6 == pytest.approx(0.6471, abs=0.001)
    assert normal_basal.EGPb == pytest.approx(1.92, abs=0.01)


def test_model_starts_exactly_at_equilibrium(normal_basal):
    ctx = _Ctx(up.NORMAL_PARAMS, normal_basal, 78.0)
    dx = _rhs(0.0, up.initial_state(up.NORMAL_PARAMS, normal_basal), ctx)
    assert np.max(np.abs(dx)) < 1e-12


@pytest.mark.parametrize("phenotype", list(up.PHENOTYPES))
def test_no_meal_means_no_drift(phenotype):
    sim = up.simulate_meals([], duration_min=600, phenotype=phenotype)
    assert np.ptp(sim.G) < 1e-6
    assert np.ptp(sim.I) < 1e-6


def test_normal_ogtt_like_meal_is_physiological():
    sim = up.simulate_meals([up.Meal(0, 75)], duration_min=480)
    s = up.summarize(sim)
    assert 130 < s["peak_glucose_mg_dl"] < 200
    assert 30 <= s["time_to_peak_min"] <= 90
    assert s["glucose_2h_mg_dl"] < 140  # normal glucose tolerance
    assert 100 < s["peak_insulin_pmol_l"] < 600
    assert abs(sim.G[-1] - sim.basal.Gb) < 5  # returns to baseline
    assert s["renal_excretion_mg_per_kg"] == 0.0  # below renal threshold


def test_glucose_mass_balance():
    bw, carbs = 78.0, 75.0
    sim = up.simulate_meals([up.Meal(0, carbs)], duration_min=900, body_weight=bw)
    expected = up.NORMAL_PARAMS.f * carbs * 1000 / bw
    assert up.summarize(sim)["glucose_appeared_mg_per_kg"] == pytest.approx(expected, rel=0.005)


def test_type2_phenotype_is_diabetic_and_triggers_renal_excretion():
    s = up.summarize(up.simulate_meals([up.Meal(0, 75)], duration_min=480, phenotype="type2"))
    assert s["glucose_2h_mg_dl"] >= 200
    assert s["renal_excretion_mg_per_kg"] > 0


def test_insulin_resistance_is_compensated_by_hyperinsulinaemia():
    n = up.summarize(up.simulate_meals([up.Meal(0, 75)], duration_min=480))
    ir = up.summarize(up.simulate_meals([up.Meal(0, 75)], duration_min=480, phenotype="insulin_resistant"))
    assert ir["glucose_2h_mg_dl"] < 140
    assert ir["insulin_iAUC_pmol_l_min"] > 1.5 * n["insulin_iAUC_pmol_l_min"]


def test_two_meals_produce_two_peaks():
    sim = up.simulate_meals([up.Meal(0, 60), up.Meal(300, 60)], duration_min=600)
    first, second = sim.G[sim.t < 300].max(), sim.G[sim.t >= 300].max()
    assert first > 120 and second > 120


@pytest.mark.parametrize("kw", [
    dict(meals=[up.Meal(500, 50)], duration_min=400),
    dict(meals=[up.Meal(0, 1000)]),
    dict(meals=[], body_weight=5),
    dict(meals=[], phenotype="unknown"),
])
def test_invalid_inputs_are_rejected(kw):
    with pytest.raises(ValueError):
        up.simulate_meals(**kw)


# ------------------------------------------------------------ Topp 2000

def test_published_fixed_points():
    fps = {fp["label"]: fp for fp in beta_cell.fixed_points()}
    phys = fps["physiological"]
    assert phys["glucose_mg_dl"] == pytest.approx(100.0)
    assert phys["insulin_uU_ml"] == pytest.approx(10.0)
    assert phys["beta_cell_mass_mg"] == pytest.approx(300.0)
    assert phys["stability"] == "stable"
    assert fps["saddle (threshold)"]["glucose_mg_dl"] == pytest.approx(250.0)
    assert fps["saddle (threshold)"]["stability"] == "saddle"
    path = fps["pathological (beta-cell loss)"]
    assert path["glucose_mg_dl"] == pytest.approx(600.0)
    assert path["stability"] == "stable"


def test_fixed_points_are_true_equilibria():
    p = beta_cell.TOPP_PARAMS
    for fp in beta_cell.fixed_points():
        y = np.array([fp["glucose_mg_dl"], fp["insulin_uU_ml"], fp["beta_cell_mass_mg"]])
        assert np.allclose(beta_cell.rhs(0, y, p, p.SI), 0, atol=1e-8)


def test_compensation_law_beta_star_inversely_proportional_to_si():
    b1 = beta_cell.fixed_points(si=0.72)[0]["beta_cell_mass_mg"]
    b2 = beta_cell.fixed_points(si=0.36)[0]["beta_cell_mass_mg"]
    assert b2 == pytest.approx(2 * b1)


def test_rate_of_insulin_resistance_decides_outcome():
    slow = beta_cell.simulate(years=5, si_final_fraction=0.1, si_decline_years=0.5)
    fast = beta_cell.simulate(years=5, si_final_fraction=0.1, si_decline_years=0.05)
    assert not slow["diabetic_at_end"] and slow["glucose_mg_dl"][-1] == pytest.approx(100, abs=1)
    assert fast["diabetic_at_end"] and fast["beta_cell_mass_mg"][-1] < 1


# ------------------------------------------------------------ Bergman minimal model

def test_minimal_model_recovers_parameters_from_noise_free_data():
    truth = minimal_model.IVGTTParams()
    sim = minimal_model.simulate_ivgtt(truth, dt=0.5)
    t = np.array(sim["t_min"])
    ts = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 22, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180.0])
    # dense insulin input so the forcing function matches the simulation
    tt = np.arange(0, 181, 1.0)
    g = np.interp(tt, t, sim["glucose_mg_dl"])
    i = np.interp(tt, t, sim["insulin_uU_ml"])
    keep = np.isin(tt, ts) | (tt % 2 == 0)
    est = minimal_model.estimate_insulin_sensitivity(tt[keep].tolist(), g[keep].tolist(), i[keep].tolist(), Gb=truth.Gb, Ib=truth.Ib)
    assert est["converged"]
    assert est["SI_per_min_per_uU_ml"] == pytest.approx(truth.SI, rel=0.01)
    assert est["SG_per_min"] == pytest.approx(truth.SG, rel=0.02)
    assert est["p2_per_min"] == pytest.approx(truth.p2, rel=0.02)


def test_minimal_model_rejects_bad_data():
    with pytest.raises(ValueError):
        minimal_model.estimate_insulin_sensitivity([0, 1, 2], [90, 80, 70], [10, 10, 10])
    with pytest.raises(ValueError):
        minimal_model.estimate_insulin_sensitivity(list(range(10))[::-1], [90] * 10, [10] * 10)


def test_ivgtt_initial_glucose_from_dose_and_volume():
    sim = minimal_model.simulate_ivgtt(minimal_model.IVGTTParams(Gb=90, dose_g_per_kg=0.3))
    assert sim["G0_mg_dl"] == pytest.approx(90 + 300 / 1.88)
    assert sim["Kg_pct_per_min"] > 0


# ------------------------------------------------------------ indices

def test_homa1_matches_formula():
    r = indices.homa1(90.0, 10.0)
    g = 90.0 / 18.016
    assert r["homa_ir"] == pytest.approx(10.0 * g / 22.5)
    assert r["homa_beta_pct"] == pytest.approx(20 * 10.0 / (g - 3.5))
    assert indices.homa1(60.0, 10.0)["homa_beta_pct"] is None  # FPG <= 3.5 mmol/l


def test_quicki_and_tyg():
    assert indices.quicki(90, 10) == pytest.approx(1 / (1 + math.log10(90)))
    assert indices.tyg_index(150, 90) == pytest.approx(math.log(150 * 90 / 2))


def test_eag_adag_reference_values():
    # ADAG: HbA1c 7% corresponds to eAG 154 mg/dl (8.6 mmol/l)
    r = indices.eag_from_a1c(7.0)
    assert r["eag_mg_dl"] == pytest.approx(154.2)
    assert r["eag_mmol_l"] == pytest.approx(8.56, abs=0.01)


@pytest.mark.parametrize("fpg,cat", [(99.9, "normal"), (100, "prediabetes (IFG)"), (125.9, "prediabetes (IFG)"), (126, "diabetes")])
def test_ada_fasting_thresholds(fpg, cat):
    assert indices.ada_classification(fpg_mgdl=fpg)["results"][0]["category"] == cat


def test_ada_requires_confirmation_for_single_abnormal_test():
    one = indices.ada_classification(fpg_mgdl=130, a1c_pct=6.0)
    assert "confirmation" in one["summary"]
    two = indices.ada_classification(fpg_mgdl=130, a1c_pct=6.6)
    assert two["summary"] == "Meets ADA criteria for diabetes."
    rnd = indices.ada_classification(random_pg_mgdl=250, classic_symptoms=True)
    assert rnd["summary"] == "Meets ADA criteria for diabetes."
    assert indices.ada_classification(random_pg_mgdl=250)["results"][0]["category"].startswith("abnormal")


def test_bmi_categories():
    assert indices.bmi(70, 175)["who_category"] == "Normal weight"
    assert indices.bmi(95, 175)["who_category"] == "Obesity"
    with pytest.raises(ValueError):
        indices.bmi(70, 10)
