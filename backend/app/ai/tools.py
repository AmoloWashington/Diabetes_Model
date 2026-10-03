"""Tools that let the AI assistant call the verified computational engines.

Every number the assistant reports must come from one of these tools, so the
language model interprets computations instead of inventing values. Inputs
are validated with the same Pydantic models as the public REST API.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
from pydantic import ValidationError

from .. import services
from ..schemas import (
    BetaCellRequest, DiffusionRequest, IndicesRequest, IVGTTRequest, MealSimRequest, MembraneRequest, RiskRequest,
)

_NUM = {"type": "number"}


def _downsample(d: dict, time_key: str, keys: list[str], step: int) -> dict:
    t = d[time_key]
    idx = list(range(0, len(t), step))
    if idx[-1] != len(t) - 1:
        idx.append(len(t) - 1)
    return {k: [round(float(d[k][i]), 3) for i in idx] for k in [time_key, *keys]}


def _meal(args: dict) -> dict:
    req = MealSimRequest.model_validate(args)
    out = services.meal_simulation(req)
    series = _downsample(out["series"], "t_min", ["glucose_mg_dl", "insulin_pmol_l"], step=max(1, int(15 / req.dt_min)))
    return {"summary": out["summary"], "series_every_15_min": series, "phenotype": out["phenotype"], "model": out["model"]}


def _beta(args: dict) -> dict:
    req = BetaCellRequest.model_validate(args)
    out = services.beta_cell_simulation(req)
    series = _downsample(out, "t_years", ["glucose_mg_dl", "insulin_uU_ml", "beta_cell_mass_mg", "si"], step=max(1, len(out["t_years"]) // 20))
    return {"outcome": out["outcome"], "final_fixed_points": out["final_fixed_points"], "series": series, "model": out["model"]}


def _ivgtt(args: dict) -> dict:
    req = IVGTTRequest.model_validate(args)
    out = services.ivgtt_simulation(req)
    series = _downsample(out, "t_min", ["glucose_mg_dl", "insulin_uU_ml"], step=10)
    return {"G0_mg_dl": out["G0_mg_dl"], "Kg_pct_per_min": out["Kg_pct_per_min"], "series": series, "model": out["model"]}


def _indices(args: dict) -> dict:
    return services.clinical_indices(IndicesRequest.model_validate(args))


def _risk(args: dict) -> dict:
    return services.symptom_risk(RiskRequest.model_validate(args))


def _model_card(_args: dict) -> dict:
    rep = services.risk_model_card()
    ev = rep["evaluation"]
    return {
        "dataset": rep["dataset"],
        "evaluation_scheme": ev["scheme"],
        "models": {k: {"pooled": v["pooled"], "ci95": v["ci95"]} for k, v in ev["models"].items()},
        "naive_leaky_cv": ev["naive_leaky_cv"],
        "leakage_note": ev["leakage_note"],
        "top_odds_ratios": rep["explanations"]["odds_ratios"][:8],
        "limitations": rep["limitations"],
    }


def _references(_args: dict) -> dict:
    return {"references": services.REFERENCES}


def _membrane(args: dict) -> dict:
    out = services.membrane(MembraneRequest.model_validate(args))
    out["pk_sweep"] = out["pk_sweep"][::7]
    return out


def _diffusion(args: dict) -> dict:
    out = services.diffusion(DiffusionRequest.model_validate(args))
    out["rms_displacement_um"] = out["rms_displacement_um"][::10]
    return out


def _rna_seq(args: dict) -> str:
    from ..rna import sequences
    from ..rna.store import get_store

    if args.get("sequence_id") is not None:
        row = get_store().get_sequence(int(args["sequence_id"]))
        if not row:
            raise ValueError("Sequence not found")
        return row["sequence"]
    if not args.get("sequence"):
        raise ValueError("Provide 'sequence' or 'sequence_id'")
    seq, _ = sequences.normalize(str(args["sequence"]))
    return seq


def _fold_rna(args: dict) -> dict:
    from ..rna import folding

    r = folding.fold(_rna_seq(args), float(args.get("temperature_c", 37.0)))
    r.pop("pair_probabilities", None)  # large; the summary statistics are enough for interpretation
    return r


def _predict_rna(args: dict) -> dict:
    from ..rna.api import predict_structure

    method = args.get("method", "auto")
    if method not in ("auto", "template", "denovo"):
        raise ValueError("method must be auto, template or denovo")
    r = predict_structure(_rna_seq(args), method)
    conf = r.pop("confidence")
    r.pop("coords")
    r["per_residue_confidence_rounded"] = [round(c, 2) for c in conf]
    return r


def _list_rna(args: dict) -> dict:
    from ..rna.store import get_store

    items, total = get_store().list_sequences(str(args.get("query", ""))[:100], 50, 0)
    return {"total": total, "items": [{k: it[k] for k in ("id", "name", "length", "source")} for it in items]}


def _datasets(_args: dict) -> dict:
    from ..rna import datasets

    return {"datasets": datasets.list_datasets(), "templates_available": len(datasets.template_library()),
            "kaggle_configured": datasets.kaggle_configured()}


_PATIENT_PROPS = {
    "age": {"type": "number", "description": "Age in years"},
    "male": {"type": "boolean"},
    **{k: {"type": "boolean"} for k in [
        "polyuria", "polydipsia", "sudden_weight_loss", "weakness", "polyphagia",
        "genital_thrush", "visual_blurring", "itching", "irritability",
        "delayed_healing", "partial_paresis", "muscle_stiffness", "alopecia", "obesity",
    ]},
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "simulate_meal",
        "description": (
            "Simulate plasma glucose and insulin after meals with the Dalla Man-Rizza-Cobelli 2007 "
            "meal model (UVA/Padova core). Phenotypes: 'normal' (published parameters), "
            "'insulin_resistant' and 'type2' (illustrative parameter scalings). Returns summary metrics "
            "(peak, 2-h glucose, iAUC, time in range) and a 15-min series."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "meals": {"type": "array", "items": {"type": "object", "properties": {
                    "time_min": _NUM, "carbs_g": _NUM}, "required": ["time_min", "carbs_g"]},
                    "description": "Meals as time (min) and carbohydrate (g)."},
                "duration_min": {"type": "number", "description": "30-4320 min"},
                "body_weight_kg": _NUM,
                "phenotype": {"type": "string", "enum": ["normal", "insulin_resistant", "type2"]},
                "basal_glucose_mg_dl": _NUM,
                "basal_insulin_pmol_l": _NUM,
                "insulin_sensitivity_scale": {"type": "number", "description": "Multiplier on insulin action (0.05-3)"},
                "beta_cell_function_scale": {"type": "number", "description": "Multiplier on beta-cell responsivity (0.05-3)"},
            },
            "required": ["meals"],
        },
    },
    {
        "name": "simulate_beta_cell_progression",
        "description": (
            "Simulate years of beta-cell mass, insulin and glucose with the Topp et al. 2000 betaIG model "
            "while insulin sensitivity declines. Returns the outcome, the fixed points with eigenvalues "
            "at the final insulin sensitivity, and a coarse time series."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "years": {"type": "number", "description": "0.5-40"},
                "si_final_fraction": {"type": "number", "description": "Final SI as a fraction of normal (0.02-2)"},
                "si_decline_years": {"type": "number", "description": "Years over which SI declines linearly"},
                "sigma_scale": {"type": "number", "description": "Multiplier on secretory capacity per beta-cell mass"},
                "d0_scale": {"type": "number", "description": "Multiplier on basal beta-cell death rate"},
            },
            "required": [],
        },
    },
    {
        "name": "simulate_ivgtt",
        "description": "Simulate an intravenous glucose tolerance test with the Bergman minimal model.",
        "input_schema": {
            "type": "object",
            "properties": {k: _NUM for k in ["Gb", "Ib", "SG", "SI", "p2", "n", "gamma", "first_phase_uU_ml", "dose_g_per_kg", "duration_min"]},
            "required": [],
        },
    },
    {
        "name": "compute_clinical_indices",
        "description": (
            "Compute HOMA1-IR, HOMA1-%B, QUICKI, eAG, TyG and BMI, and classify tests against ADA "
            "diagnostic thresholds. Supply only the values that are known."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fasting_glucose_mg_dl": _NUM, "fasting_insulin_uU_ml": _NUM, "hba1c_pct": _NUM,
                "ogtt_2h_mg_dl": _NUM, "random_glucose_mg_dl": _NUM, "classic_symptoms": {"type": "boolean"},
                "triglycerides_mg_dl": _NUM, "weight_kg": _NUM, "height_cm": _NUM,
            },
            "required": [],
        },
    },
    {
        "name": "predict_symptom_risk",
        "description": (
            "Estimate the probability of a diabetes-positive label from age, sex and 14 symptoms with the "
            "leak-free validated ensemble trained on the UCI early-stage dataset. Returns probability, a "
            "95% bootstrap interval, exact log-odds contributions, warnings, and optionally a probability "
            "re-expressed for a target prevalence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient": {"type": "object", "properties": _PATIENT_PROPS, "required": list(_PATIENT_PROPS)},
                "target_prevalence": {"type": "number", "description": "Optional prevalence (0.001-0.999)"},
            },
            "required": ["patient"],
        },
    },
    {
        "name": "get_risk_model_card",
        "description": "Return dataset provenance, leak-free validation metrics with CIs, odds ratios and limitations of the symptom model.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "membrane_biophysics",
        "description": (
            "Compute Nernst equilibrium potentials (K+, Na+, Cl-, Ca2+), the Goldman-Hodgkin-Katz membrane potential "
            "from relative permeabilities, driving forces, and V_m as K+ permeability falls (e.g. K_ATP closure). "
            "Concentrations in mM; defaults are typical mammalian values (Alberts) and squid-axon permeability ratios."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "temperature_c": _NUM, "p_K": _NUM, "p_Na": _NUM, "p_Cl": _NUM,
                "K": {"type": "object", "properties": {"in": {"type": "number"}, "out": {"type": "number"}}, "required": ["in", "out"]}, "Na": {"type": "object", "properties": {"in": {"type": "number"}, "out": {"type": "number"}}, "required": ["in", "out"]}, "Cl": {"type": "object", "properties": {"in": {"type": "number"}, "out": {"type": "number"}}, "required": ["in", "out"]}, "Ca": {"type": "object", "properties": {"in": {"type": "number"}, "out": {"type": "number"}}, "required": ["in", "out"]},
            },
            "required": [],
        },
    },
    {
        "name": "diffusion_time",
        "description": (
            "Stokes-Einstein diffusion coefficient for a sphere of hydrodynamic radius (nm) and the characteristic "
            "time t = L^2/(2dD) to diffuse a distance L (um) in d dimensions. Default viscosity: water at 37 C."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"radius_nm": _NUM, "distance_um": _NUM, "temperature_c": _NUM, "viscosity_mpa_s": _NUM,
                           "dims": {"type": "integer"}},
            "required": ["radius_nm", "distance_um"],
        },
    },
    {
        "name": "fold_rna",
        "description": (
            "Predict RNA secondary structure with ViennaRNA (Turner 2004 energies): MFE structure and energy, "
            "ensemble free energy, centroid structure and per-nucleotide confidence from base-pair probabilities. "
            "Give a sequence (ACGU; T is converted) or the id of a stored sequence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sequence": {"type": "string"}, "sequence_id": {"type": "integer"},
                           "temperature_c": {"type": "number"}},
            "required": [],
        },
    },
    {
        "name": "predict_rna_3d",
        "description": (
            "Predict an RNA 3D structure (C1' coarse-grained). 'template' uses loaded structure datasets; "
            "'denovo' embeds the ViennaRNA secondary structure with A-form restraints (low tertiary accuracy); "
            "'auto' tries template first. Returns method, template identity/coverage and per-residue confidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sequence": {"type": "string"}, "sequence_id": {"type": "integer"},
                           "method": {"type": "string", "enum": ["auto", "template", "denovo"]}},
            "required": [],
        },
    },
    {
        "name": "list_rna_sequences",
        "description": "List RNA sequences stored in this workspace (id, name, length, source). Optional name filter.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []},
    },
    {
        "name": "list_datasets",
        "description": "List loaded datasets (Kaggle or uploaded), the number of 3D templates available and whether Kaggle is configured.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_references",
        "description": "Return the bibliographic references of every model and formula implemented in this application.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

_DISPATCH: dict[str, Callable[[dict], dict]] = {
    "simulate_meal": _meal,
    "simulate_beta_cell_progression": _beta,
    "simulate_ivgtt": _ivgtt,
    "compute_clinical_indices": _indices,
    "predict_symptom_risk": _risk,
    "get_risk_model_card": _model_card,
    "get_references": _references,
    "membrane_biophysics": _membrane,
    "diffusion_time": _diffusion,
    "fold_rna": _fold_rna,
    "predict_rna_3d": _predict_rna,
    "list_rna_sequences": _list_rna,
    "list_datasets": _datasets,
}


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def run_tool(name: str, args: Any) -> tuple[str, bool]:
    """Execute a tool. Returns (JSON string result, is_error)."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool {name!r}"}), True
    if not isinstance(args, dict):
        return json.dumps({"error": "Tool input must be a JSON object"}), True
    try:
        return json.dumps(fn(args), cls=_Enc), False
    except ValidationError as e:
        errs = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]
        return json.dumps({"error": "Invalid input", "details": errs}), True
    except (ValueError, RuntimeError) as e:
        return json.dumps({"error": str(e)}), True
