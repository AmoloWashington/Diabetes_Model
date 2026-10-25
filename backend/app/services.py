"""Service layer shared by the REST API and the AI assistant's tools."""

from __future__ import annotations

from .molecular import crispr, dna
from .physiology import beta_cell, biophysics, cgm, indices, minimal_model, uva_padova
from .schemas import (
    BetaCellRequest, CgmAnalyzeRequest, CrisprRequest, DiffusionRequest, GlucoseForecastRequest, IndicesRequest,
    IVGTTFitRequest, IVGTTRequest, MealSimRequest, MembraneRequest, RiskRequest,
)

REFERENCES = [
    {"id": "dallaman2007", "used_for": "Meal simulation model (glucose-insulin)",
     "citation": "Dalla Man C, Rizza RA, Cobelli C. Meal simulation model of the glucose-insulin system. "
                 "IEEE Trans Biomed Eng. 2007;54(10):1740-1749.", "doi": "10.1109/TBME.2007.893506"},
    {"id": "kovatchev2009", "used_for": "FDA acceptance of the UVA/Padova simulator for pre-clinical testing",
     "citation": "Kovatchev BP, Breton M, Dalla Man C, Cobelli C. In silico preclinical trials: a proof of concept "
                 "in closed-loop control of type 1 diabetes. J Diabetes Sci Technol. 2009;3(1):44-55."},
    {"id": "bergman1979", "used_for": "Glucose minimal model, insulin sensitivity SI",
     "citation": "Bergman RN, Ider YZ, Bowden CR, Cobelli C. Quantitative estimation of insulin sensitivity. "
                 "Am J Physiol. 1979;236(6):E667-E677."},
    {"id": "toffolo1980", "used_for": "Insulin minimal model, second-phase secretion term",
     "citation": "Toffolo G, Bergman RN, Finegood DT, Bowden CR, Cobelli C. Quantitative estimation of beta cell "
                 "sensitivity to glucose in the intact organism: a minimal model of insulin kinetics in the dog. "
                 "Diabetes. 1980;29(12):979-990."},
    {"id": "topp2000", "used_for": "betaIG model of beta-cell mass, insulin and glucose",
     "citation": "Topp B, Promislow K, deVries G, Miura RM, Finegood DT. A model of beta-cell mass, insulin, and "
                 "glucose kinetics: pathways to diabetes. J Theor Biol. 2000;206(4):605-619.",
     "doi": "10.1006/jtbi.2000.2150"},
    {"id": "matthews1985", "used_for": "HOMA1-IR and HOMA1-%B",
     "citation": "Matthews DR, Hosker JP, Rudenski AS, Naylor BA, Treacher DF, Turner RC. Homeostasis model "
                 "assessment: insulin resistance and beta-cell function from fasting plasma glucose and insulin "
                 "concentrations in man. Diabetologia. 1985;28(7):412-419."},
    {"id": "katz2000", "used_for": "QUICKI",
     "citation": "Katz A, Nambi SS, Mather K, et al. Quantitative insulin sensitivity check index: a simple, "
                 "accurate method for assessing insulin sensitivity in humans. J Clin Endocrinol Metab. "
                 "2000;85(7):2402-2410."},
    {"id": "nathan2008", "used_for": "Estimated average glucose from HbA1c",
     "citation": "Nathan DM, Kuenen J, Borg R, Zheng H, Schoenfeld D, Heine RJ; ADAG Study Group. Translating the "
                 "A1C assay into estimated average glucose values. Diabetes Care. 2008;31(8):1473-1478."},
    {"id": "simental2008", "used_for": "TyG index",
     "citation": "Simental-Mendia LE, Rodriguez-Moran M, Guerrero-Romero F. The product of fasting glucose and "
                 "triglycerides as surrogate for identifying insulin resistance in apparently healthy subjects. "
                 "Metab Syndr Relat Disord. 2008;6(4):299-304."},
    {"id": "ada_soc", "used_for": "Diagnostic thresholds",
     "citation": "American Diabetes Association Professional Practice Committee. 2. Diagnosis and Classification "
                 "of Diabetes: Standards of Care in Diabetes. Diabetes Care (annual supplement 1)."},
    {"id": "islam2020", "used_for": "Symptom dataset (UCI #529)",
     "citation": "Islam MMF, Ferdousi R, Rahman S, Bushra HY. Likelihood prediction of diabetes at early stage "
                 "using data mining techniques. In: Computer Vision and Machine Intelligence in Medical Image "
                 "Analysis. Adv Intell Syst Comput, vol 992. Springer; 2020:113-125.",
     "doi": "10.1007/978-981-13-8798-2_12"},
    {"id": "rorsman2018", "used_for": "Beta-cell stimulus-secretion coupling (animation)",
     "citation": "Rorsman P, Ashcroft FM. Pancreatic beta-cell electrical activity and insulin secretion: of mice "
                 "and men. Physiol Rev. 2018;98(1):117-214."},
    {"id": "battelino2019", "used_for": "CGM metrics, glucose ranges and targets (time in range)",
     "citation": "Battelino T, et al. Clinical targets for continuous glucose monitoring data interpretation: "
                 "recommendations from the international consensus on time in range. Diabetes Care. 2019;42(8):1593-1603."},
    {"id": "bergenstal2018", "used_for": "Glucose management indicator (GMI) from mean CGM glucose",
     "citation": "Bergenstal RM, et al. Glucose Management Indicator (GMI): a new term for estimating A1C from "
                 "continuous glucose monitoring. Diabetes Care. 2018;41(11):2275-2280."},
    {"id": "gloyn2004", "used_for": "KCNJ11 (Kir6.2) activating mutations cause permanent neonatal diabetes",
     "citation": "Gloyn AL, et al. Activating mutations in the gene encoding the ATP-sensitive potassium-channel "
                 "subunit Kir6.2 and permanent neonatal diabetes. N Engl J Med. 2004;350(18):1838-1849."},
    {"id": "pearson2006", "used_for": "Switching KCNJ11 neonatal diabetes from insulin to sulfonylureas",
     "citation": "Pearson ER, et al. Switching from insulin to oral sulfonylureas in patients with diabetes due "
                 "to Kir6.2 mutations. N Engl J Med. 2006;355(5):467-477."},
    {"id": "pearson2003", "used_for": "HNF1A-MODY sensitivity to sulfonylureas",
     "citation": "Pearson ER, et al. Genetic cause of hyperglycaemia and response to treatment in diabetes. "
                 "Lancet. 2003;362(9392):1275-1281."},
    {"id": "steele2014", "used_for": "GCK-MODY: mild stable hyperglycaemia with low complication prevalence",
     "citation": "Steele AM, et al. Prevalence of vascular complications among patients with glucokinase "
                 "mutations and prolonged, mild hyperglycemia. JAMA. 2014;311(3):279-286."},
    {"id": "hattersley2018", "used_for": "Diagnosis and management of monogenic diabetes (ISPAD guideline)",
     "citation": "Hattersley AT, et al. ISPAD Clinical Practice Consensus Guidelines 2018: the diagnosis and "
                 "management of monogenic diabetes in children and adolescents. Pediatr Diabetes. 2018;19(Suppl 27):47-63."},
    {"id": "jinek2012", "used_for": "CRISPR-Cas9 mechanism and the NGG PAM (guide finder)",
     "citation": "Jinek M, Chylinski K, Fonfara I, Hauer M, Doudna JA, Charpentier E. A programmable "
                 "dual-RNA-guided DNA endonuclease in adaptive bacterial immunity. Science. 2012;337(6096):816-821."},
    {"id": "ran2013", "used_for": "SpCas9 cut site (3 bp upstream of PAM) and guide design (guide finder)",
     "citation": "Ran FA, Hsu PD, Wright J, Agarwala V, Scott DA, Zhang F. Genome engineering using the "
                 "CRISPR-Cas9 system. Nat Protoc. 2013;8(11):2281-2308."},
    {"id": "doench2016", "used_for": "Guide activity and specificity design rules (background for the guide finder)",
     "citation": "Doench JG, et al. Optimized sgRNA design to maximize activity and minimize off-target "
                 "effects of CRISPR-Cas9. Nat Biotechnol. 2016;34(2):184-191."},
    {"id": "shields2012", "used_for": "MODY probability calculator (clinical prediction model)",
     "citation": "Shields BM, et al. The development and validation of a clinical prediction model to determine "
                 "the probability of MODY in patients with young-onset diabetes. Diabetologia. 2012;55(5):1265-1272."},
    {"id": "rorsman2003", "used_for": "Insulin granule numbers and pools (3D cell explorer)",
     "citation": "Rorsman P, Renström E. Insulin granule dynamics in pancreatic beta cells. Diabetologia. "
                 "2003;46(8):1029-1045.", "doi": "10.1007/s00125-003-1153-1"},
    {"id": "saltiel2001", "used_for": "Insulin receptor signalling and GLUT4 translocation (animation)",
     "citation": "Saltiel AR, Kahn CR. Insulin signalling and the regulation of glucose and lipid metabolism. "
                 "Nature. 2001;414(6865):799-806."},
    {"id": "lorenz2011", "used_for": "RNA secondary structure (MFE, partition function, pair probabilities)",
     "citation": "Lorenz R, Bernhart SH, Hoener zu Siederdissen C, Tafer H, Flamm C, Stadler PF, Hofacker IL. "
                 "ViennaRNA Package 2.0. Algorithms Mol Biol. 2011;6:26.", "doi": "10.1186/1748-7188-6-26"},
    {"id": "mccaskill1990", "used_for": "RNA partition function and base-pair probabilities",
     "citation": "McCaskill JS. The equilibrium partition function and base pair binding probabilities for RNA "
                 "secondary structure. Biopolymers. 1990;29(6-7):1105-1119."},
    {"id": "kabsch1976", "used_for": "Optimal superposition (RMSD, TM-score)",
     "citation": "Kabsch W. A solution for the best rotation to relate two sets of vectors. Acta Cryst A. "
                 "1976;32(5):922-923."},
    {"id": "zhang2022", "used_for": "TM-score d0 for RNA structure comparison",
     "citation": "Zhang C, Shine M, Pyle AM, Zhang Y. US-align: universal structure alignments of proteins, "
                 "nucleic acids, and macromolecular complexes. Nat Methods. 2022;19:1109-1115."},
    {"id": "rego2015", "used_for": "3D molecular viewer (3Dmol.js; py3Dmol is its Python wrapper)",
     "citation": "Rego N, Koes D. 3Dmol.js: molecular visualization with WebGL. Bioinformatics. "
                 "2015;31(8):1322-1324.", "doi": "10.1093/bioinformatics/btu829"},
    {"id": "goldman1943", "used_for": "GHK voltage equation",
     "citation": "Goldman DE. Potential, impedance, and rectification in membranes. J Gen Physiol. 1943;27(1):37-60."},
    {"id": "hodgkin1949", "used_for": "GHK voltage equation; squid-axon permeability ratios",
     "citation": "Hodgkin AL, Katz B. The effect of sodium ions on the electrical activity of the giant axon of the "
                 "squid. J Physiol. 1949;108(1):37-77."},
    {"id": "einstein1905", "used_for": "Diffusion (Stokes-Einstein relation, mean-squared displacement)",
     "citation": "Einstein A. Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in "
                 "ruhenden Flüssigkeiten suspendierten Teilchen. Ann Phys. 1905;322(8):549-560."},
    {"id": "alberts", "used_for": "Typical mammalian ion concentrations; cell biology background",
     "citation": "Alberts B, et al. Molecular Biology of the Cell. Garland Science / W. W. Norton (current edition)."},
    {"id": "watson1953", "used_for": "DNA double helix",
     "citation": "Watson JD, Crick FHC. Molecular structure of nucleic acids: a structure for deoxyribose nucleic "
                 "acid. Nature. 1953;171(4356):737-738."},
]


def meal_simulation(req: MealSimRequest) -> dict:
    sim = uva_padova.simulate_meals(
        [uva_padova.Meal(m.time_min, m.carbs_g) for m in req.meals],
        duration_min=req.duration_min,
        body_weight=req.body_weight_kg,
        phenotype=req.phenotype,
        basal_glucose=req.basal_glucose_mg_dl,
        basal_insulin=req.basal_insulin_pmol_l,
        insulin_sensitivity=req.insulin_sensitivity_scale,
        beta_cell_function=req.beta_cell_function_scale,
        dt_min=req.dt_min,
    )
    ph = sim.phenotype
    return {
        "series": {
            "t_min": sim.t.tolist(),
            "glucose_mg_dl": sim.G.tolist(),
            "insulin_pmol_l": sim.I.tolist(),
            "ra_mg_kg_min": sim.Ra.tolist(),
            "egp_mg_kg_min": sim.EGP.tolist(),
            "uid_mg_kg_min": sim.Uid.tolist(),
            "secretion_pmol_kg_min": sim.S.tolist(),
            "renal_mg_kg_min": sim.E.tolist(),
        },
        "summary": uva_padova.summarize(sim),
        "phenotype": {
            "key": ph.key, "label": ph.label, "description": ph.description,
            "published_parameter_set": ph.published,
            "insulin_sensitivity_scale": sim.params.Vmx / uva_padova.NORMAL_PARAMS.Vmx,
            "beta_cell_function_scale": sim.params.K / uva_padova.NORMAL_PARAMS.K,
        },
        "basal": {k: float(v) for k, v in vars(sim.basal).items()},
        "model": "Dalla Man, Rizza, Cobelli 2007 (IEEE TBME 54:1740)",
    }


def beta_cell_simulation(req: BetaCellRequest) -> dict:
    p = beta_cell.with_overrides(
        sigma=beta_cell.TOPP_PARAMS.sigma * req.sigma_scale,
        d0=beta_cell.TOPP_PARAMS.d0 * req.d0_scale,
    )
    out = beta_cell.simulate(
        years=req.years, si_final_fraction=req.si_final_fraction,
        si_decline_years=req.si_decline_years, p=p,
        initial=None if (req.sigma_scale == 1 and req.d0_scale == 1) else _healthy_start(),
    )
    out["model"] = "Topp et al. 2000 (J Theor Biol 206:605)"
    return out


def _healthy_start() -> tuple[float, float, float]:
    fp = beta_cell.fixed_points()[0]
    return fp["glucose_mg_dl"], fp["insulin_uU_ml"], fp["beta_cell_mass_mg"]


def beta_cell_fixed_points(si_fraction: float) -> dict:
    si = beta_cell.TOPP_PARAMS.SI * si_fraction
    return {"si": si, "fixed_points": beta_cell.fixed_points(si=si)}


def ivgtt_simulation(req: IVGTTRequest) -> dict:
    p = minimal_model.IVGTTParams(**req.model_dump(exclude={"duration_min"}))
    out = minimal_model.simulate_ivgtt(p, duration_min=req.duration_min)
    out["model"] = "Bergman et al. 1979 glucose minimal model; Toffolo et al. 1980 secretion term"
    return out


def ivgtt_fit(req: IVGTTFitRequest) -> dict:
    return minimal_model.estimate_insulin_sensitivity(
        req.t_min, req.glucose_mg_dl, req.insulin_uU_ml, Gb=req.Gb, Ib=req.Ib,
        exclude_before_min=req.exclude_before_min,
    )


def clinical_indices(req: IndicesRequest) -> dict:
    out: dict = {}
    g, i = req.fasting_glucose_mg_dl, req.fasting_insulin_uU_ml
    if g is not None and i is not None:
        out["homa1"] = indices.homa1(g, i)
        out["quicki"] = indices.quicki(g, i)
    if req.hba1c_pct is not None:
        out["eag"] = indices.eag_from_a1c(req.hba1c_pct)
    if req.triglycerides_mg_dl is not None and g is not None:
        out["tyg_index"] = indices.tyg_index(req.triglycerides_mg_dl, g)
    if req.weight_kg is not None and req.height_cm is not None:
        out["bmi"] = indices.bmi(req.weight_kg, req.height_cm)
    out["ada"] = indices.ada_classification(
        fpg_mgdl=g, ogtt_2h_mgdl=req.ogtt_2h_mg_dl, a1c_pct=req.hba1c_pct,
        random_pg_mgdl=req.random_glucose_mg_dl, classic_symptoms=req.classic_symptoms,
    )
    if g is not None:
        out["fasting_glucose_mmol_l"] = indices.mgdl_to_mmoll(g)
    return out


def symptom_risk(req: RiskRequest) -> dict:
    from .ml import risk_model  # scikit-learn is slow to import; load it on first use

    return risk_model.predict(risk_model.get_bundle(), req.patient.model_dump(), req.target_prevalence)


def risk_model_card() -> dict:
    from .ml import risk_model

    return risk_model.get_bundle().report


def dna_analysis(seq: str) -> dict:
    return dna.analyze(seq)


def crispr_guides(req: "CrisprRequest") -> dict:
    return crispr.find_guides(req.sequence, req.gc_low, req.gc_high)


def glucose_forecast_benchmark() -> dict:
    from .ml import glucose_forecast
    return glucose_forecast.get_benchmark()


def glucose_forecast_series(req: "GlucoseForecastRequest") -> dict:
    import numpy as np
    from .ml import glucose_forecast
    from .physiology import cgm as cgm_mod
    if req.csv is not None:
        minutes, g, meta = cgm_mod.parse_csv(req.csv)
        used = cgm_mod._detect_unit(g, req.unit)
        g = g * cgm_mod.MGDL_PER_MMOLL if used == "mmol/l" else g
    else:
        minutes = np.array([p.minute for p in req.points], float)
        g = np.array([p.g for p in req.points], float)
        used = cgm_mod._detect_unit(g, req.unit)
        g = g * cgm_mod.MGDL_PER_MMOLL if used == "mmol/l" else g
    out = glucose_forecast.evaluate_series(minutes, g)
    out["unit_detected"] = used
    return out


def membrane(req: MembraneRequest) -> dict:
    conc = {k: dict(v) for k, v in biophysics.DEFAULT_CONC_MM.items()}
    for ion in ("K", "Na", "Cl", "Ca"):
        v = getattr(req, ion)
        if v is not None:
            conc[ion] = {**conc[ion], "in": v.inside, "out": v.outside}
    out = biophysics.membrane_analysis(conc, {"K": req.p_K, "Na": req.p_Na, "Cl": req.p_Cl}, req.temperature_c)
    out["concentrations_mm"] = {k: {"in": v["in"], "out": v["out"]} for k, v in conc.items()}
    out["permeabilities"] = {"K": req.p_K, "Na": req.p_Na, "Cl": req.p_Cl}
    return out


def diffusion(req: DiffusionRequest) -> dict:
    return biophysics.diffusion_analysis(req.radius_nm, req.distance_um, req.temperature_c, req.viscosity_mpa_s, req.dims)


def cgm_analyze(req: CgmAnalyzeRequest) -> dict:
    meta = {}
    if req.csv is not None:
        minutes, g, meta = cgm.parse_csv(req.csv)
        start_tod = meta.pop("start_minute_of_day")
    else:
        minutes = [p.minute for p in req.points]
        g = [p.g for p in req.points]
        start_tod = req.start_minute_of_day
    out = cgm.analyze(minutes, g, req.unit, start_tod)
    out["source"] = meta or {"points": len(g)}
    return out


def cgm_sample(phenotype: str, days: int) -> dict:
    t, g = cgm.synthetic_sample(phenotype, days)
    out = cgm.analyze(t, g, "mg/dl", 0.0)
    out["source"] = {"synthetic": True, "phenotype": phenotype,
                     "description": "Synthetic trace from the Dalla Man meal model (3 randomised meals/day, 5-min "
                                    "samples, 5% noise). Demonstration data, not from a person."}
    return out
