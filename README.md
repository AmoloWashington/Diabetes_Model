# GlucoLab

**A research workbench for glucose physiology and RNA structure.** GlucoLab implements peer-reviewed
mathematical models, validates each one against its publication in an automated test suite, animates the cell
biology they describe, and adds a Claude-powered research assistant that reports only numbers it has computed
with those engines.

> Research and education software. Not a medical device and not a substitute for diagnosis by a clinician.

---

## Run it on your computer

You need **Python 3.11–3.13** ([python.org](https://www.python.org/downloads/)), **Node.js 20+**
([nodejs.org](https://nodejs.org)) and **Git**.

### Windows (PowerShell)

```powershell
git clone https://github.com/AmoloWashington/Diabetes_Model.git
cd Diabetes_Model
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1    # one time: installs Python + web dependencies, builds the UI
powershell -ExecutionPolicy Bypass -File scripts\run.ps1      # every time
```

### macOS / Linux

```bash
git clone https://github.com/AmoloWashington/Diabetes_Model.git
cd Diabetes_Model
scripts/setup.sh     # one time
scripts/run.sh       # every time
```

Then open **http://localhost:8000**.

The first start trains the diabetes risk model in the background (about 30 s); after that it loads from cache.

### Optional: API keys

`setup` creates a git-ignored `.env` from `.env.example`. Add keys there and restart:

| Variable | Enables |
|---|---|
| `ANTHROPIC_API_KEY` | AI research assistant (default model `claude-opus-5`, set with `CLAUDE_MODEL`) |
| `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`) | Importing datasets from Kaggle. Create the token at kaggle.com → Settings → API. For competitions, accept the rules on kaggle.com first. |

Every other feature works offline, without keys. **Never commit `.env`.**

### Other ways to run

- **Development with hot reload:** `scripts/dev.sh` (or `scripts\dev.ps1`). The UI runs on http://localhost:5173 and proxies to the API on :8000.
- **Docker** (also the fallback for Intel Macs, which have no ViennaRNA wheel):
  ```bash
  docker build -t glucolab .
  docker run -p 8000:8000 --env-file .env -v glucolab-data:/data glucolab
  ```
- **Deploy online (Render, free tier):** sign in at render.com, choose *New → Blueprint*, and pick this repository. `render.yaml` builds the Dockerfile as one web service and gives you a public `https://….onrender.com` URL. If you want the AI assistant, add `ANTHROPIC_API_KEY` in the Render dashboard, and note that every visitor's questions are billed to that key (requests are rate-limited per minute). On the free tier the service sleeps when idle, and stored sequences and datasets are lost on restart. Attach a disk at `/data` to keep them. The same Dockerfile also runs on Railway, Fly.io and Google Cloud Run, which supply `$PORT`.
- **Custom domain (for example glucolab.org):**
  1. Buy the domain from any registrar (Cloudflare, Namecheap, Porkbun and others).
  2. In Render: your service → **Settings → Custom Domains → Add**, and enter both `glucolab.org` and `www.glucolab.org`.
  3. Render shows the DNS records to create (a CNAME for `www`, and an A or ALIAS record for the bare domain). Add exactly those at your registrar. Render issues the HTTPS certificate automatically once DNS has propagated, which can take minutes to a few hours.
  4. Set the `SITE_URL` environment variable to `https://glucolab.org` and redeploy, so link previews on WhatsApp and LinkedIn point to the new address. If previews still show the old address, change the default in `web/vite.config.ts`.
- **Feedback:** the **Give feedback** page stores submissions on the server. To keep them on Render's free tier (which loses files on restart), set `FEEDBACK_GITHUB_TOKEN` (a fine-grained GitHub token with *Issues: read and write* on this repository only) and `FEEDBACK_GITHUB_REPO`; each submission then becomes a GitHub issue. See [docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md) for who to ask and what to send them.
- **Manual:**
  1. `pip install -r requirements.txt`
  2. `cd web && npm ci && npm run build`
  3. `cd ../backend && uvicorn app.main:app`

---

## What's inside

| Area | Module | Scientific basis |
|---|---|---|
| **Physiology** | Meal simulation | Dalla Man, Rizza & Cobelli, *IEEE TBME* 2007: the 12-state core of the FDA-accepted UVA/Padova simulator |
| | β-cell dynamics | Topp et al., *J Theor Biol* 2000: slow–fast dynamics with fixed points and eigenvalue stability |
| | Insulin sensitivity | Bergman minimal model (1979): IVGTT simulation and S<sub>I</sub>/S<sub>G</sub>/p₂ estimation with CV% |
| | Clinical indices | HOMA1, QUICKI, eAG, TyG, BMI, ADA diagnostic thresholds |
| | Glucose data (CGM) | Upload a continuous glucose monitor CSV (generic, or exports with metadata lines such as LibreView): time in ranges, CV, GMI (Bergenstal 2018) and the ambulatory glucose profile, checked against the international consensus targets (Battelino 2019); mg/dl or mmol/l; synthetic 14-day samples from the meal model |
| | Glucose forecasting (benchmark) | Predicts glucose 30 and 60 min ahead from the last hour of CGM and scores persistence, linear and ridge-regression models with a leak-free temporal split (RMSE, MAE, skill vs persistence). On a synthetic cohort ridge beats the persistence baseline by ~17% (30 min) and ~25% (60 min). In-silico demonstration, not OhioT1DM; upload your own CGM to benchmark on real data |
| | Membrane biophysics | Nernst and Goldman–Hodgkin–Katz potentials with exact SI constants (RT/F = 25.69 mV at 25 °C), V<sub>m</sub> vs K⁺ permeability (the physics of K<sub>ATP</sub> closure), Stokes–Einstein diffusion and ⟨x²⟩ = 2dDt |
| **Cell & molecular** | 3D cell explorer | Interactive three.js β-cell: nucleus (envelope, pores, nucleolus, chromatin fibres), rough ER with ribosomes, Golgi, mitochondria with cristae, lysosomes, centrosome, microtubules, primary cilium and insulin granules. Labels appear as you zoom closer (whole cell → organelles → sub-structures); clicking a part shows its role and source. Illustrated, fluorescence-style and EM-style rendering, all labelled as rendered models, not micrographs. Granule exocytosis rate follows the meal model's static secretion S = S<sub>b</sub> + β(G − G<sub>b</sub>). |
| | Cell theatre | Model-driven animations: β-cell stimulus–secretion coupling, insulin → GLUT4 signalling, INS gene → insulin. Export to WebM video. |
| | Monogenic diabetes | Gene → channel → cell: KCNJ11/K<sub>ATP</sub> scenarios computed with the GHK engine (normal, activating mutation, sulfonylurea), a GCK-MODY meal-model illustration, and a sourced table of the main genes (KCNJ11, ABCC8, GCK, HNF1A, HNF4A, INS). Educational, not diagnostic |
| | CRISPR guide finder | Educational SpCas9 (NGG PAM) guide-RNA finder: scans both strands for 20-nt protospacers, marks the cut site 3 bp upstream of the PAM, and flags GC content, poly-T runs and non-uniqueness within the pasted sequence. Not a genome-wide off-target search and not for planning real editing (Jinek 2012; Ran 2013) |
| | DNA lab | 3D B-DNA built from any sequence (phosphates, deoxyribose, purine/pyrimidine slabs, 2 or 3 hydrogen bonds per A–T/G–C pair, major and minor grooves) with zoom labels, 3-frame translation (NCBI table 1), ORFs, T<sub>m</sub> |
| **RNA structure** | Sequences | Upload/paste FASTA, validation (ACGU, T→U, IUPAC flagged), SQLite storage with SHA-256 de-duplication, ViennaRNA folding (MFE, ensemble, pair probabilities, arc diagram) |
| | 3D prediction | Template-based modelling from loaded structure datasets; coarse-grained de novo fallback from ViennaRNA structure + A-form restraints; per-residue confidence; PDB export; leave-one-out TM-score benchmark |
| | Structure viewer | 3Dmol.js (the engine behind py3Dmol): predicted, uploaded and RCSB PDB structures; cartoon, stick, sphere and molecular-surface styles; residue labels on hover; helix and hairpin-loop labels on predicted models; colour by confidence/nucleotide/chain; TM-score and RMSD comparison; py3Dmol notebook snippet |
| **Data & models** | Dataset explorer | Kaggle import via the official `kagglehub` client (e.g. Stanford RNA 3D Folding), file upload, schema and row browser, RNA analytics, template library |
| | Diabetes risk model | Symptom classifier with leak-free grouped cross-validation, bootstrap CIs, calibration, exact explanations |
| **AI** | Research assistant | Claude in a tool-use loop over all engines above, including the biophysics calculators. On the CGM page, "Explain my report" sends the report's summary numbers (never the individual readings) so the assistant can explain it in plain words, against the consensus targets and without dosing advice. It replies in the user's language, including Swahili. Answers stream live with a reasoning summary; each tool call and its output appear as it runs. "Ask AI about this" on each page sends that page's current results as context. Conversations are saved in the browser, and equations render with KaTeX. |
| **Access** | Install, language, feedback | Installable on phones and desktops (web app manifest and a service worker that never caches calculations); English and a **draft** Swahili interface (menu, CGM page, assistant, feedback; awaiting review by a Swahili-speaking health professional); a feedback page for reviewers |
| **Methods** | Model & methods panels | Every model page ends with the governing equations (KaTeX), numerical method, code path and primary reference, mirrored from the backend implementation. |

### How the RNA 3D confidence works (read this before presenting predictions)

- **Template-based models:** confidence is a heuristic of alignment quality (sequence identity, whether each residue matched or was gap-filled). The *Calibration benchmark* on the prediction page runs a leave-one-out test on your loaded structures, so you can see how confidence relates to TM-score before trusting it.
- **Coarse-grained de novo models:** confidence is the ViennaRNA ensemble probability of each nucleotide's secondary-structure state. It measures secondary-structure certainty only. Tertiary packing in these models is low-accuracy, and the app says so on every result.
- Coordinates are one bead per nucleotide (the C1′ atom), the representation used by the Stanford RNA 3D Folding competition. The B-factor column of exported PDBs holds confidence × 100.

## Validation (enforced by 124 automated tests)

| Model | Check | Published | GlucoLab |
|---|---|---|---|
| Dalla Man 2007 | k<sub>p1</sub> derived from steady state | 2.70 | 2.698 |
| Dalla Man 2007 | m6 (hepatic extraction) | 0.6471 | 0.6469 |
| Topp 2000 | Physiological fixed point G, I, β | 100, 10, 300 | 100, 10, 300 |
| Topp 2000 | Saddle glucose | 250 mg/dl | 250 mg/dl |
| ADAG 2008 | eAG at HbA1c 7% | 154 mg/dl | 154.2 mg/dl |
| Minimal model | S<sub>I</sub> recovery from noise-free data | 5.00e-4 | 5.00e-4 |
| RNA 3D | TM-score of a rotated copy / random coil | 1 / ≈0 | 1.000 / <0.2 |
| ViennaRNA | GGGAAAUCCCGCGCAAAGCGC MFE | `(((....)))((((...))))` | identical |
| Nernst | E<sub>K</sub>, K⁺ 140/5 mM, 37 °C | −89.1 mV (RT/F·ln 28) | −89.1 mV |
| GHK | Reduces to Nernst for a single permeant ion | exact | exact |

Also tested:
- mass conservation, and exact equilibrium at basal;
- de novo helix geometry: paired C1′–C1′ distance 10.5 Å, RMSD 0.004 Å to an ideal right-handed A-form stem;
- PDB round-trip;
- FASTA validation;
- Kaggle import with a mocked downloader;
- path-traversal rejection;
- the AI loop and its streaming endpoint, with a fake client.

### Data-leakage finding in the original model

269 of the 520 records in the public symptom dataset are exact duplicates. A random train/test split puts copies
on both sides and inflates accuracy from **90.8% (leak-free) to 96.9%**. GlucoLab evaluates with grouped
cross-validation and reports both numbers.

## Architecture

```
web/        React 19 + TypeScript + Vite + Tailwind CSS 4 · TanStack Query · Recharts · three.js · 3Dmol.js
backend/    FastAPI · NumPy/SciPy · scikit-learn · ViennaRNA · Biopython · kagglehub · Anthropic SDK · SQLite
  app/physiology/   meal model, β-cell model, minimal model, clinical indices, membrane biophysics, CGM metrics
  app/rna/          sequences, folding, 3D prediction & comparison, datasets/Kaggle, storage, REST API
  app/ml/           leak-free risk model
  app/ai/           Claude agent loop and tools
  tests/            124 tests
data/       UCI early-stage diabetes dataset (#529)
scripts/    setup / run / dev for Windows and macOS/Linux
```

Interactive API documentation: http://localhost:8000/docs (set `ENABLE_API_DOCS=false` to turn it off; the Render blueprint does this for the public site)

```bash
cd backend && python -m pytest            # backend tests
cd web && npm run typecheck && npm run build
```

## Honest limitations

- Only the healthy-adult meal phenotype uses a published parameter set. The insulin-resistant and type 2 phenotypes are labelled scalings of it.
- RNA 3D prediction is template-based or coarse-grained. It is not a substitute for experimental structures or for deep-learning structure predictors, and the confidence scores are explained above.
- The symptom model comes from one hospital population (61.5% positive). It is not a screening test.
- The 3D cell and the animations are schematic in geometry and counts (a real β-cell holds about 10,000 granules; a subset is drawn). The fluorescence and EM modes are rendering styles, not microscopy data.
- Animations are schematic in geometry. Their rates come from the models, and their sequence of events from the cited reviews.
- RCSB PDB and Kaggle downloads need internet access, and Kaggle needs your own token.

The original Streamlit prototype is preserved on the `master` branch.
