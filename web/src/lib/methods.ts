// "Model & methods" content for every engine. Each equation mirrors the backend
// implementation (backend/app/...). Displayed with KaTeX.

export interface MethodsSection { title: string; eqs?: string[]; text?: string }
export interface Methods { title: string; source: string; sections: MethodsSection[]; numerics: string; code: string }

export const METHODS: Record<string, Methods> = {
  meal: {
    title: "Dalla Man–Rizza–Cobelli meal model",
    source: "Dalla Man C, Rizza RA, Cobelli C. IEEE Trans Biomed Eng 2007;54:1740–1749",
    code: "backend/app/physiology/uva_padova.py",
    sections: [
      {
        title: "Glucose kinetics (two compartments)",
        eqs: [
          String.raw`\dot G_p = EGP + R_a - U_{ii} - E - k_1 G_p + k_2 G_t,\qquad \dot G_t = -U_{id} + k_1 G_p - k_2 G_t,\qquad G = G_p / V_G`,
        ],
      },
      {
        title: "Insulin kinetics and hepatic extraction",
        eqs: [
          String.raw`\dot I_l = -(m_1 + m_3) I_l + m_2 I_p + S,\qquad \dot I_p = -(m_2 + m_4) I_p + m_1 I_l,\qquad I = I_p / V_I`,
          String.raw`HE = -m_5 S + m_6,\qquad m_3 = \frac{HE\, m_1}{1 - HE}`,
        ],
      },
      {
        title: "Gastro-intestinal absorption",
        eqs: [
          String.raw`\dot Q_{sto1} = -k_{gri} Q_{sto1} + D\,\delta(t),\quad \dot Q_{sto2} = -k_{empt}(Q_{sto}) Q_{sto2} + k_{gri} Q_{sto1},\quad \dot Q_{gut} = -k_{abs} Q_{gut} + k_{empt} Q_{sto2}`,
          String.raw`k_{empt} = k_{min} + \tfrac{k_{max}-k_{min}}{2}\Big[\tanh\big(\alpha(Q_{sto}-bD)\big) - \tanh\big(\beta(Q_{sto}-dD)\big) + 2\Big],\quad \alpha = \tfrac{5}{2D(1-b)},\ \beta = \tfrac{5}{2Dd}`,
          String.raw`R_a = \frac{f\,k_{abs}\,Q_{gut}}{BW}`,
        ],
      },
      {
        title: "Production, utilisation and renal excretion",
        eqs: [
          String.raw`EGP = \max\!\big(0,\ k_{p1} - k_{p2} G_p - k_{p3} I_d - k_{p4} I_{po}\big),\quad \dot I_1 = -k_i (I_1 - I),\ \dot I_d = -k_i (I_d - I_1)`,
          String.raw`U_{id} = \frac{(V_{m0} + V_{mx} X)\,G_t}{K_{m0} + G_t},\qquad \dot X = -p_{2U} X + p_{2U}(I - I_b),\qquad U_{ii} = F_{cns}`,
          String.raw`E = k_{e1}\,(G_p - k_{e2})\ \text{ if } G_p > k_{e2},\ \text{else } 0`,
        ],
      },
      {
        title: "β-cell secretion",
        eqs: [
          String.raw`S = \gamma I_{po},\qquad \dot I_{po} = -\gamma I_{po} + S_{po},\qquad S_{po} = Y + K\,\max(\dot G, 0) + S_b`,
          String.raw`\dot Y = -\alpha\big[Y - \beta(G - G_b)\big]\ \text{ if } \beta(G - G_b) \ge -S_b,\ \text{ else } \dot Y = -\alpha Y - \alpha S_b`,
        ],
      },
      {
        title: "Basal steady state",
        text: "k_p1, S_b, I_lb, I_pb, G_tb, EGP_b and m6 are solved from the steady-state constraints for the chosen basal glucose and insulin, so every simulation starts exactly at equilibrium. For the normal subject this reproduces the published k_p1 = 2.70 and m6 = 0.6471.",
      },
    ],
    numerics: "12 coupled nonlinear ODEs integrated with LSODA (automatic stiff/non-stiff switching), rtol 1e-7, atol 1e-9, maximum step 2 min. Meals are impulses added to Q_sto1 between integration segments.",
  },
  betacell: {
    title: "βIG model of β-cell mass, insulin and glucose",
    source: "Topp B, Promislow K, deVries G, Miura RM, Finegood DT. J Theor Biol 2000;206:605–619",
    code: "backend/app/physiology/beta_cell.py",
    sections: [
      {
        title: "Dynamics (time in days)",
        eqs: [
          String.raw`\dot G = R_0 - (E_{G0} + S_I I)\,G,\qquad \dot I = \frac{\beta\,\sigma\,G^2}{\alpha + G^2} - kI,\qquad \dot\beta = (-d_0 + r_1 G - r_2 G^2)\,\beta`,
        ],
      },
      {
        title: "Fixed points and stability",
        eqs: [
          String.raw`r_2 G^{*2} - r_1 G^* + d_0 = 0 \Rightarrow G^* \in \{100,\ 250\}\ \text{mg/dl},\qquad I^* = \frac{R_0/G^* - E_{G0}}{S_I},\qquad \beta^* = \frac{k I^* (\alpha + G^{*2})}{\sigma G^{*2}}`,
          String.raw`J = \frac{\partial(\dot G, \dot I, \dot\beta)}{\partial(G, I, \beta)}\Big|_{x^*},\qquad \text{stable} \iff \operatorname{Re}\lambda_i(J) < 0\ \ \forall i`,
        ],
        text: "Because G* does not depend on S_I, β* ∝ 1/S_I: β-cell mass compensates for insulin resistance. A fast fall in S_I can carry glucose past the saddle at 250 mg/dl before β-cell mass adapts, after which the trajectory goes to the pathological state (β = 0, G = R₀/E_G0 = 600 mg/dl).",
      },
    ],
    numerics: "Stiff slow–fast system (fast G, I; slow β) integrated with Radau IIA using the analytic Jacobian, rtol = atol = 1e-8. Eigenvalues from numpy.linalg.eigvals.",
  },
  minimal: {
    title: "Bergman minimal model",
    source: "Bergman RN, Ider YZ, Bowden CR, Cobelli C. Am J Physiol 1979;236:E667–E677",
    code: "backend/app/physiology/minimal_model.py",
    sections: [
      {
        title: "Glucose minimal model",
        eqs: [
          String.raw`\dot G = -(S_G + X)\,G + S_G G_b,\quad G(0) = G_0,\qquad \dot X = -p_2 X + p_3\,(I(t) - I_b),\quad X(0) = 0,\qquad S_I = p_3 / p_2`,
        ],
      },
      {
        title: "Insulin used for forward simulation",
        eqs: [String.raw`\dot I = -n\,(I - I_b) + \gamma\,[G - h]^+\,t,\qquad G_0 = G_b + \frac{\text{dose}}{V_G}`],
      },
      {
        title: "Parameter estimation",
        eqs: [
          String.raw`\hat\theta = \arg\min_{\theta}\ \sum_k \left(\frac{G(t_k;\theta) - G_k}{\sigma_k}\right)^2,\quad \theta = \log(S_G, S_I, p_2, G_0),\quad \sigma_k = \max(0.02\,G_k,\ 1\ \text{mg/dl})`,
          String.raw`\operatorname{Cov}(\hat\theta) \approx s^2 (J^\top J)^{-1},\qquad \mathrm{CV}(\theta_j) \approx 100\,\sqrt{\operatorname{Cov}_{jj}}\ \%`,
        ],
        text: "Measured insulin is linearly interpolated as the forcing function; samples before 8 min are excluded (incomplete mixing).",
      },
    ],
    numerics: "Trust-region reflective least squares from four starting points; LSODA with rtol 1e-10 inside the objective.",
  },
  clinical: {
    title: "Clinical indices",
    source: "Matthews 1985; Katz 2000; Nathan (ADAG) 2008; Simental-Mendía 2008; ADA Standards of Care",
    code: "backend/app/physiology/indices.py",
    sections: [
      {
        title: "Formulas",
        eqs: [
          String.raw`\text{HOMA1-IR} = \frac{I_0\,[\mu U/ml] \times G_0\,[mmol/l]}{22.5},\qquad \text{HOMA1-\%B} = \frac{20\, I_0}{G_0 - 3.5}`,
          String.raw`\text{QUICKI} = \frac{1}{\log_{10} I_0 + \log_{10} G_0\,[mg/dl]},\qquad \text{eAG}\,[mg/dl] = 28.7 \times \text{HbA1c} - 46.7`,
          String.raw`\text{TyG} = \ln\!\left(\frac{TG\,[mg/dl] \times G_0\,[mg/dl]}{2}\right),\qquad 1\ \text{mmol/l glucose} = 18.016\ \text{mg/dl}`,
        ],
      },
    ],
    numerics: "Closed-form expressions; input ranges validated.",
  },
  rnafold: {
    title: "RNA secondary structure (ViennaRNA)",
    source: "Lorenz R et al. Algorithms Mol Biol 2011;6:26 · McCaskill JS. Biopolymers 1990;29:1105–1119",
    code: "backend/app/rna/folding.py",
    sections: [
      {
        title: "Minimum free energy and the Boltzmann ensemble",
        eqs: [
          String.raw`s^{MFE} = \arg\min_s \Delta G(s),\qquad Z = \sum_{s} e^{-\Delta G(s)/RT},\qquad P(s) = \frac{e^{-\Delta G(s)/RT}}{Z},\qquad G_{ens} = -RT \ln Z`,
          String.raw`p_{ij} = \sum_{s \ni (i,j)} P(s),\qquad \text{conf}_i = \begin{cases} p_{i,j} & i \text{ paired with } j \text{ in } s^{MFE} \\ 1 - \sum_j p_{ij} & i \text{ unpaired} \end{cases}`,
        ],
        text: "ΔG(s) is the nearest-neighbour free energy with Turner 2004 parameters.",
      },
    ],
    numerics: "Zuker-type dynamic programming for the MFE and McCaskill's algorithm for Z and p_ij; both O(N³) time.",
  },
  rna3d: {
    title: "RNA 3D modelling and comparison",
    source: "Kabsch W. Acta Cryst A 1976;32:922–923 · Zhang C et al. (US-align) Nat Methods 2022;19:1109–1115",
    code: "backend/app/rna/structure3d.py",
    sections: [
      {
        title: "Coarse-grained de novo embedding (one bead per nucleotide, C1′)",
        eqs: [
          String.raw`\mathcal{E}(\mathbf{x}) = \sum_{(i,j)\in\mathcal{R}} w_{ij}\big(\lVert \mathbf{x}_i - \mathbf{x}_j\rVert - d_{ij}\big)^2 + \sum_{|i-j|\ge 2} \max\!\big(0,\ d_{min} - \lVert \mathbf{x}_i - \mathbf{x}_j\rVert\big)^2`,
          String.raw`\text{A-form stem: } \mathbf{x}_k = \big(r\cos k\theta,\ r\sin k\theta,\ k h\big),\quad \theta = 32.7^\circ,\ h = 2.81\,\text{\AA},\quad d^{C1'\!-C1'}_{pair} = 10.5\,\text{\AA}`,
        ],
        text: "Restraints R: consecutive C1′ distances, Watson–Crick pair distances and all intra-stem distances of the ideal helix. The helix radius r = 8.7 Å and the consecutive distance 5.9 Å are approximations.",
      },
      {
        title: "Superposition and TM-score",
        eqs: [
          String.raw`(R^*, \mathbf{t}^*) = \arg\min_{R \in SO(3),\,\mathbf{t}} \sum_i \lVert R\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \rVert^2 \quad (\text{SVD of } H = \textstyle\sum_i \tilde{\mathbf{p}}_i \tilde{\mathbf{q}}_i^\top)`,
          String.raw`\text{TM} = \max_{R,\mathbf{t}} \frac{1}{L_{ref}} \sum_i \frac{1}{1 + (d_i/d_0)^2},\qquad d_0 = 0.6\sqrt{L_{ref} - 0.5} - 2.5\ \ (L_{ref} \ge 30)`,
        ],
      },
    ],
    numerics: "L-BFGS-B with analytic gradients for the embedding; TM-score superpositions seeded from fragments and refined iteratively.",
  },
  risk: {
    title: "Symptom risk model",
    source: "Data: Islam MMF et al. 2020 (UCI #529)",
    code: "backend/app/ml/risk_model.py",
    sections: [
      {
        title: "Model and explanation",
        eqs: [
          String.raw`p = \tfrac12\big[\sigma(\beta_0 + \boldsymbol\beta^\top \mathbf{x}) + p_{RF}(\mathbf{x})\big],\qquad \sigma(z) = \frac{1}{1 + e^{-z}}`,
          String.raw`\operatorname{logit} p_{LR}(\mathbf{x}) = \operatorname{logit} p_{LR}(\mathbf{x}_{ref}) + \sum_j \beta_j (x_j - x_{ref,j})`,
        ],
      },
      {
        title: "Evaluation and prevalence adjustment",
        eqs: [
          String.raw`\text{Brier} = \frac{1}{N}\sum_n (p_n - y_n)^2,\qquad \text{AUC} = P(p_{pos} > p_{neg})`,
          String.raw`\frac{p'}{1-p'} = \frac{p}{1-p}\cdot\frac{\pi'/(1-\pi')}{\pi/(1-\pi)}`,
        ],
        text: "Cross-validation folds group identical records, so duplicates never span training and test folds.",
      },
    ],
    numerics: "5×5 repeated StratifiedGroupKFold; 500 group-bootstrap resamples for confidence intervals.",
  },
  membrane: {
    title: "Membrane potential",
    source: "Nernst 1888 · Goldman 1943 · Hodgkin & Katz 1949",
    code: "backend/app/physiology/biophysics.py",
    sections: [
      {
        title: "Equilibrium and resting potentials",
        eqs: [
          String.raw`E_X = \frac{RT}{zF}\ln\frac{[X]_{out}}{[X]_{in}},\qquad V_m = \frac{RT}{F}\ln\frac{P_K[K^+]_o + P_{Na}[Na^+]_o + P_{Cl}[Cl^-]_i}{P_K[K^+]_i + P_{Na}[Na^+]_i + P_{Cl}[Cl^-]_o}`,
          String.raw`R = k_B N_A = 8.314462618\ \tfrac{J}{mol\,K},\qquad F = e N_A = 96485.33212\ \tfrac{C}{mol},\qquad \text{driving force} = V_m - E_X`,
        ],
      },
    ],
    numerics: "Closed form. The P_K sweep holds P_Na and P_Cl fixed.",
  },
  diffusion: {
    title: "Diffusion",
    source: "Einstein A. Ann Phys 1905;322:549–560",
    code: "backend/app/physiology/biophysics.py",
    sections: [
      {
        title: "Stokes–Einstein and mean-squared displacement",
        eqs: [String.raw`D = \frac{k_B T}{6\pi\eta r},\qquad \langle x^2 \rangle = 2 d D t \ \Rightarrow\ t_L = \frac{L^2}{2dD}`],
      },
    ],
    numerics: "Closed form.",
  },
};
