# GlucoLab: computational physiology of glucose regulation

GlucoLab combines **peer-reviewed mathematical models** of the glucose–insulin system, **model-driven
cell and DNA animations**, a **leak-free validated machine-learning risk model**, and a
**Claude-powered AI research assistant** whose numbers come only from those verified engines.

> Research and education software. Not a medical device and not a substitute for diagnosis by a clinician.

## What is inside

| Area | Implementation | Source |
|---|---|---|
| Meal response | 12-state nonlinear ODE model: gastric emptying, gut absorption, hepatic glucose production, insulin-dependent/independent utilisation, renal excretion, β-cell secretion, hepatic insulin extraction | Dalla Man, Rizza, Cobelli, *IEEE TBME* 54:1740 (2007), the core of the UVA/Padova simulator accepted by the FDA for pre-clinical testing |
| β-cell mass over years | βIG slow–fast dynamical system with fixed-point and eigenvalue (linear stability) analysis | Topp et al., *J Theor Biol* 206:605 (2000) |
| Insulin sensitivity | Bergman minimal model: IVGTT simulation, and nonlinear least-squares estimation of S<sub>I</sub>, S<sub>G</sub>, p₂ with CV% | Bergman et al., *Am J Physiol* 236:E667 (1979); Toffolo et al., *Diabetes* 29:979 (1980) |
| Clinical indices | HOMA1-IR/%B, QUICKI, eAG, TyG, BMI, ADA diagnostic thresholds | Matthews 1985; Katz 2000; Nathan 2008; Simental-Mendía 2008; ADA Standards of Care |
| Symptom risk ML | Logistic regression + random forest ensemble; grouped CV; bootstrap CIs; calibration; exact log-odds explanations; per-patient uncertainty; Bayes prior-shift | UCI dataset #529, Islam et al. (2020) |
| Molecular lab | Standard genetic code (NCBI table 1), 3-frame translation, ORFs, reverse complement, T<sub>m</sub>, 3D B-DNA helix | Watson & Crick 1953 |
| Cell Theatre | β-cell stimulus–secretion coupling; insulin receptor → IRS-1 → PI3K → PIP₃ → Akt → AS160 → GLUT4; INS gene → preproinsulin → proinsulin → insulin + C-peptide, all exportable as WebM video | Rorsman & Ashcroft, *Physiol Rev* 2018; Saltiel & Kahn, *Nature* 2001 |
| AI assistant | Claude (default `claude-opus-5`) in a tool-use loop over the engines above, with every tool call shown to the user | Anthropic API |

Full references are in the app (**References** tab) and at `GET /api/references`.

## Scientific validation (enforced by the test suite)

The tests in `backend/tests/` check the implementations against published results, not just that they run:

- **Dalla Man 2007.** The parameters the paper reports as derived, kp1 = 2.70 mg/kg/min and m6 = 0.6471,
  are *recomputed independently* from steady-state constraints and match (2.698 and 0.6469). The model
  starts at an exact equilibrium (|dx/dt| < 1e-12) and conserves mass (appeared glucose = f·D/BW within 0.5%).
  A 75 g meal in the normal subject peaks at 155 mg/dl at 66 min, with a 2-h glucose of 132 mg/dl (normal tolerance).
- **Topp 2000.** Fixed points G = 100 mg/dl, I = 10 µU/ml, β = 300 mg (stable), the saddle at G = 250 mg/dl,
  and the pathological state at 600 mg/dl are reproduced. The compensation law (β\* ∝ 1/S<sub>I</sub>) holds,
  and a rapid fall in S<sub>I</sub> leads to β-cell collapse while a slow fall is compensated.
- **Minimal model.** The estimator recovers S<sub>I</sub>, S<sub>G</sub> and p₂ within 1–2% from noise-free data.
- **Indices.** Formula-exact values and ADA threshold boundaries (e.g. FPG 125.9 → prediabetes, 126 → diabetes; HbA1c 7% → eAG 154 mg/dl).
- **Molecular.** The codon table has 64 codons and three stops. The demo sequence translates to the true insulin B chain,
  and the cysteine positions (A6, A7, A11, A20; B7, B19) give the A6–A11, A7–B7 and A20–B19 disulfides drawn in the animation.

## Key finding about the original model: data leakage

**269 of the dataset's 520 rows are exact duplicates** (only 251 unique records). The previous app used a
random train/test split, which puts copies of the same record on both sides and inflates accuracy.
GlucoLab evaluates with `StratifiedGroupKFold`, grouping identical records (5×5 repeated), and reports:

| Evaluation | Accuracy | ROC AUC |
|---|---|---|
| Naive random K-fold (leaky) | 96.9% | 0.997 |
| **Grouped, leak-free (ensemble)** | **90.8%** | **0.972** (95% CI 0.95–0.99) |

Other changes from the previous version:

- The old "Mild Diabetes (Borderline)" stage derived from classifier probability has been removed. A probability is not a disease stage.
- The model trains once and is cached. It no longer retrains on every page visit.
- The committed virtualenv and the unversioned `.pkl` are no longer used.

The original Streamlit code is preserved on the `master` branch.

## Running

```bash
pip install -r requirements.txt
cd backend
uvicorn app.main:app --port 8000
# open http://localhost:8000
```

The first start trains the risk model (about 30 s, in the background). The model is then cached in `backend/.model_cache/`.

### Enabling the AI assistant

Create a git-ignored `.env` in the repository root (see `.env.example`) or export the variables:

```bash
ANTHROPIC_API_KEY=sk-ant-...        # never commit this
CLAUDE_MODEL=claude-opus-5          # default
CLAUDE_EFFORT=high                  # low | medium | high | xhigh | max
CLAUDE_FALLBACKS=true               # server-side refusal fallback ("default" routing)
AI_RATE_LIMIT_PER_MIN=12            # per client IP
```

Everything except the chat works without a key. The assistant uses adaptive thinking and a manual tool-use
loop (at most 8 rounds). Every tool input is validated with the same Pydantic schemas as the REST API, and invalid
inputs come back to the model as tool errors instead of crashing the request.

### Docker

```bash
docker build -t glucolab .
docker run -p 8000:8000 --env-file .env glucolab
```

### Tests

```bash
pip install -r requirements-dev.txt
cd backend && python -m pytest       # 55 tests; no network needed (the AI loop uses a fake client)
```

CI (`.github/workflows/ci.yml`) runs the suite and a syntax check of the frontend on every push.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model readiness, AI configuration |
| GET | `/api/physiology/phenotypes` | Available meal-model phenotypes |
| POST | `/api/physiology/meal` | Meal simulation (Dalla Man 2007) |
| POST | `/api/physiology/beta-cell` | Long-term β-cell simulation (Topp 2000) |
| GET | `/api/physiology/beta-cell/fixed-points?si_fraction=` | Fixed points and eigenvalues |
| POST | `/api/physiology/ivgtt` · `/api/physiology/ivgtt/fit` | Minimal-model simulation and S<sub>I</sub> estimation |
| POST | `/api/clinical/indices` | HOMA, QUICKI, eAG, TyG, BMI, ADA classification |
| POST | `/api/risk/predict` · GET `/api/risk/model-card` | Risk prediction and full validation report |
| GET | `/api/dna/demo` · POST `/api/dna/analyze` | Sequence analysis |
| GET | `/api/ai/status` · POST `/api/ai/chat` | AI assistant |

Interactive OpenAPI docs are at `/docs`.

## Honest limitations

- Only the "Healthy adult" meal phenotype uses a published parameter set. The insulin-resistant and
  type 2 phenotypes are transparent scalings of it and are labelled as such everywhere.
- Minimal-model defaults are illustrative, typical-order values. Parameter estimates are meaningful only with a frequently sampled IVGTT.
- The symptom model comes from a single hospital-based population (61.5% positive). It is not a population screening test.
- Animations are schematic. Their *rates* come from the models, their *sequence of events* from the cited reviews, and their geometry is illustrative.
- The AI assistant can still make mistakes in interpretation. The tool-call trace under each answer exists so that you can check it.

## Project layout

```
backend/app/
  physiology/  uva_padova.py · beta_cell.py · minimal_model.py · indices.py
  ml/          risk_model.py
  molecular/   dna.py
  ai/          assistant.py (agent loop) · tools.py (engine tools)
  services.py · schemas.py · config.py · main.py (FastAPI)
backend/tests/ scientific, API and AI-loop tests
frontend/      index.html · css/ · js/ (app, cells, dna3d, util) · vendor/ (three.js, Chart.js; MIT)
data/          diabetes_symptoms_data.csv (UCI #529)
```
