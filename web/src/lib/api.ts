// Typed API client for the GlucoLab backend.

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

type Detail = string | { loc?: (string | number)[]; msg: string }[];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Cannot reach the GlucoLab server. Is the backend running on port 8000?", 0);
  }
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON */
  }
  if (!res.ok) {
    const d = (data as { detail?: Detail } | null)?.detail;
    let msg = `Request failed (${res.status})`;
    if (typeof d === "string") msg = d;
    else if (Array.isArray(d)) msg = d.map((e) => `${(e.loc ?? []).slice(1).join(".") || "input"}: ${e.msg}`).join("; ");
    throw new ApiError(msg, res.status);
  }
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  upload: <T>(path: string, form: FormData) => request<T>(path, { method: "POST", body: form }),
};

// ------------------------------------------------------------------ types

export interface Health { status: string; risk_model_ready: boolean; ai_configured: boolean; ai_model: string | null }

export interface MealSeries {
  t_min: number[]; glucose_mg_dl: number[]; insulin_pmol_l: number[]; ra_mg_kg_min: number[];
  egp_mg_kg_min: number[]; uid_mg_kg_min: number[]; secretion_pmol_kg_min: number[]; renal_mg_kg_min: number[];
}
export interface MealSummary {
  basal_glucose_mg_dl: number; basal_insulin_pmol_l: number; peak_glucose_mg_dl: number; time_to_peak_min: number;
  glucose_2h_mg_dl: number | null; peak_insulin_pmol_l: number; glucose_iAUC_mg_dl_min: number;
  insulin_iAUC_pmol_l_min: number; time_in_range_70_180_pct: number; time_above_180_pct: number;
  time_below_70_pct: number; renal_excretion_mg_per_kg: number;
}
export interface MealResult {
  series: MealSeries; summary: MealSummary;
  phenotype: { key: string; label: string; description: string; published_parameter_set: boolean; insulin_sensitivity_scale: number; beta_cell_function_scale: number };
  basal: Record<string, number>; model: string;
}

export interface FixedPoint {
  label: string; glucose_mg_dl: number; insulin_uU_ml: number; beta_cell_mass_mg: number;
  eigenvalues_per_day: { real: number; imag: number }[]; stability: string;
}
export interface BetaCellResult {
  t_years: number[]; glucose_mg_dl: number[]; insulin_uU_ml: number[]; beta_cell_mass_mg: number[]; si: number[];
  final_fixed_points: FixedPoint[]; outcome: string; diabetic_at_end: boolean; params: Record<string, number>;
}

export interface RnaSequence {
  id: number; name: string; description: string; sequence: string; length: number; source: string;
  tags: string[]; sha256: string; created_at: number; gc_percent?: number | null;
}
export interface StructureSummary {
  id: number; sequence_id: number | null; name: string; kind: string; method: string;
  mean_confidence: number | null; meta: Record<string, unknown>; created_at: number;
}
export interface StructureFull extends StructureSummary { pdb: string; confidence: number[] }

export interface FoldResult {
  sequence: string; mfe_structure: string; mfe_kcal_mol: number; pairs: [number, number][]; n_pairs: number; n_stems: number;
  ensemble_free_energy_kcal_mol?: number; mfe_frequency_in_ensemble?: number; centroid_structure?: string;
  confidence?: number[]; mean_confidence?: number; pair_probabilities?: [number, number, number][];
  method: string; energy_model: string; temperature_c: number;
}

export interface PredictResult {
  pdb: string; confidence: number[]; sequence: string; method: string; mean_confidence: number;
  template_id?: string; identity?: number; coverage?: number; secondary_structure?: string;
  mfe_kcal_mol?: number; restraint_rmse_A?: number; caveat: string; structure_id?: number;
}

export interface DatasetMeta {
  slug: string; title: string; source: string; created_at: number;
  files: { name: string; bytes: number }[]; kaggle_handle?: string;
}
export interface ColumnInfo { name: string; dtype: string; missing: number; unique: number; min?: number; max?: number; mean?: number; std?: number }
export interface DatasetDetail extends Omit<DatasetMeta, "files"> {
  files: { name: string; bytes: number; type: string; rows?: number; columns?: ColumnInfo[]; kind?: string; error?: string }[];
  n_templates: number;
}
