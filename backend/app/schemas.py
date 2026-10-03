"""Validated request models. Bounds reject non-physiological input early."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Phenotype = Literal["normal", "insulin_resistant", "type2"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MealIn(_Strict):
    time_min: float = Field(ge=0, le=4320, description="Minutes after simulation start")
    carbs_g: float = Field(ge=0, le=400, description="Carbohydrate (glucose-equivalent), g")


class MealSimRequest(_Strict):
    meals: list[MealIn] = Field(default_factory=lambda: [MealIn(time_min=0, carbs_g=75)], max_length=12)
    duration_min: float = Field(420, ge=30, le=4320)
    body_weight_kg: float = Field(78, ge=30, le=250)
    phenotype: Phenotype = "normal"
    basal_glucose_mg_dl: float | None = Field(None, ge=60, le=300)
    basal_insulin_pmol_l: float | None = Field(None, ge=5, le=300)
    insulin_sensitivity_scale: float | None = Field(None, ge=0.05, le=3)
    beta_cell_function_scale: float | None = Field(None, ge=0.05, le=3)
    dt_min: float = Field(1.0, ge=0.5, le=10)

    @model_validator(mode="after")
    def _meals_in_window(self):
        for m in self.meals:
            if m.time_min >= self.duration_min:
                raise ValueError("Each meal must occur before the end of the simulation")
        return self


class BetaCellRequest(_Strict):
    years: float = Field(10, ge=0.5, le=40)
    si_final_fraction: float = Field(0.3, ge=0.02, le=2)
    si_decline_years: float | None = Field(None, ge=0, le=40, description="Defaults to min(5, years)")
    sigma_scale: float = Field(1.0, ge=0.1, le=3, description="Scales maximal secretion per beta-cell mass")
    d0_scale: float = Field(1.0, ge=0.2, le=5, description="Scales basal beta-cell death rate")

    @model_validator(mode="after")
    def _decline_within(self):
        if self.si_decline_years is None:
            self.si_decline_years = min(5.0, self.years)
        if self.si_decline_years > self.years:
            raise ValueError("si_decline_years cannot exceed years")
        return self


class IVGTTRequest(_Strict):
    Gb: float = Field(90, ge=60, le=200)
    Ib: float = Field(10, ge=1, le=60)
    SG: float = Field(0.025, ge=0.001, le=0.1)
    SI: float = Field(5e-4, ge=1e-6, le=5e-3)
    p2: float = Field(0.025, ge=0.001, le=0.2)
    n: float = Field(0.14, ge=0.01, le=1)
    gamma: float = Field(0.004, ge=0, le=0.05)
    first_phase_uU_ml: float = Field(80, ge=0, le=500)
    dose_g_per_kg: float = Field(0.3, ge=0.05, le=0.5)
    duration_min: float = Field(180, ge=30, le=300)


class IVGTTFitRequest(_Strict):
    t_min: list[float] = Field(min_length=8, max_length=60)
    glucose_mg_dl: list[float] = Field(min_length=8, max_length=60)
    insulin_uU_ml: list[float] = Field(min_length=8, max_length=60)
    Gb: float | None = Field(None, ge=40, le=300)
    Ib: float | None = Field(None, ge=0.5, le=100)
    exclude_before_min: float = Field(8, ge=0, le=30)


class IndicesRequest(_Strict):
    fasting_glucose_mg_dl: float | None = Field(None, ge=20, le=600)
    fasting_insulin_uU_ml: float | None = Field(None, ge=0.5, le=300)
    hba1c_pct: float | None = Field(None, ge=3, le=20)
    ogtt_2h_mg_dl: float | None = Field(None, ge=20, le=800)
    random_glucose_mg_dl: float | None = Field(None, ge=20, le=1000)
    classic_symptoms: bool = False
    triglycerides_mg_dl: float | None = Field(None, ge=10, le=5000)
    weight_kg: float | None = Field(None, ge=2, le=400)
    height_cm: float | None = Field(None, ge=40, le=260)


class PatientSymptoms(_Strict):
    age: float = Field(ge=1, le=120)
    male: bool
    polyuria: bool
    polydipsia: bool
    sudden_weight_loss: bool
    weakness: bool
    polyphagia: bool
    genital_thrush: bool
    visual_blurring: bool
    itching: bool
    irritability: bool
    delayed_healing: bool
    partial_paresis: bool
    muscle_stiffness: bool
    alopecia: bool
    obesity: bool


class RiskRequest(_Strict):
    patient: PatientSymptoms
    target_prevalence: float | None = Field(None, ge=0.001, le=0.999)


class DNARequest(_Strict):
    sequence: str = Field(min_length=1, max_length=5000)


class ChatMessage(_Strict):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class PageContext(_Strict):
    page: str = Field(max_length=80)
    summary: str = Field(max_length=6000)


class ChatRequest(_Strict):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    context: PageContext | None = None

    @model_validator(mode="after")
    def _alternation(self):
        if self.messages[-1].role != "user":
            raise ValueError("The last message must come from the user")
        if self.messages[0].role != "user":
            raise ValueError("The conversation must start with a user message")
        for a, b in zip(self.messages, self.messages[1:]):
            if a.role == b.role:
                raise ValueError("Messages must alternate between user and assistant")
        return self


class IonConc(_Strict):
    inside: float = Field(gt=0, le=1000, alias="in")
    outside: float = Field(gt=0, le=1000, alias="out")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MembraneRequest(_Strict):
    temperature_c: float = Field(37.0, ge=-20, le=60)
    K: IonConc | None = None
    Na: IonConc | None = None
    Cl: IonConc | None = None
    Ca: IonConc | None = None
    p_K: float = Field(1.0, ge=0, le=100)
    p_Na: float = Field(0.04, ge=0, le=100)
    p_Cl: float = Field(0.45, ge=0, le=100)


class DiffusionRequest(_Strict):
    radius_nm: float = Field(gt=0, le=10000)
    distance_um: float = Field(gt=0, le=100000)
    temperature_c: float = Field(37.0, ge=-20, le=60)
    viscosity_mpa_s: float = Field(0.69, gt=0, le=1e6)
    dims: int = Field(3, ge=1, le=3)
