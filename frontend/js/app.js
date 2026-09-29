import { api, toast, fmt, pct, esc, markdown, makeRecorder } from "./util.js";
import { CellTheatre } from "./cells.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

// ------------------------------------------------------------------ charts
const charts = {};
function chartDefaults() {
  const C = window.Chart;
  C.defaults.color = "#93a2b8";
  C.defaults.borderColor = "rgba(148,163,184,0.12)";
  C.defaults.font.family = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  C.defaults.animation.duration = 700;
  C.defaults.plugins.legend.labels.boxWidth = 12;
  C.defaults.maintainAspectRatio = false;
}
function lineChart(id, datasets, { xTitle, yTitle, annotations = [], xType = "linear", yMin } = {}) {
  if (charts[id]) charts[id].destroy();
  const bands = {
    id: "bands",
    beforeDatasetsDraw(chart) {
      const { ctx, chartArea: a, scales: { x, y } } = chart;
      for (const an of annotations) {
        ctx.save();
        if (an.type === "band") {
          ctx.fillStyle = an.color;
          const y0 = y.getPixelForValue(an.from), y1 = y.getPixelForValue(an.to);
          ctx.fillRect(a.left, Math.min(y0, y1), a.right - a.left, Math.abs(y1 - y0));
        } else if (an.type === "hline") {
          const yy = y.getPixelForValue(an.value);
          if (yy >= a.top && yy <= a.bottom) {
            ctx.strokeStyle = an.color; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(a.left, yy); ctx.lineTo(a.right, yy); ctx.stroke();
            ctx.fillStyle = an.color; ctx.font = "11px system-ui"; ctx.fillText(an.label || "", a.left + 6, yy - 4);
          }
        } else if (an.type === "vline") {
          const xx = x.getPixelForValue(an.value);
          if (xx >= a.left && xx <= a.right) {
            ctx.strokeStyle = an.color; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(xx, a.top); ctx.lineTo(xx, a.bottom); ctx.stroke();
            ctx.fillStyle = an.color; ctx.font = "11px system-ui"; ctx.fillText(an.label || "", xx + 4, a.top + 12);
          }
        }
        ctx.restore();
      }
    },
  };
  charts[id] = new window.Chart($("#" + id), {
    type: "line",
    data: { datasets },
    options: {
      parsing: false, normalized: true, interaction: { mode: "index", intersect: false },
      elements: { point: { radius: 0, hoverRadius: 3 }, line: { tension: 0.25, borderWidth: 2.2 } },
      scales: {
        x: { type: xType, title: { display: !!xTitle, text: xTitle } },
        y: { title: { display: !!yTitle, text: yTitle }, min: yMin },
      },
      plugins: { decimation: { enabled: true, algorithm: "lttb", samples: 300 } },
    },
    plugins: [bands],
  });
  return charts[id];
}
const xy = (xs, ys) => xs.map((x, i) => ({ x, y: ys[i] }));
const PHENO_COLORS = { normal: "#34d399", insulin_resistant: "#fbbf24", type2: "#f87171" };

// ------------------------------------------------------------------ tabs
function initTabs() {
  const show = () => {
    const id = (location.hash || "#overview").slice(1).split("?")[0];
    const target = $("#" + id) && $("#" + id).classList.contains("tab-panel") ? id : "overview";
    $$(".tab-panel").forEach((p) => (p.hidden = p.id !== target));
    $$(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === target));
    $(".tabs").classList.remove("open");
    window.scrollTo({ top: 0, behavior: "instant" });
    onTab[target]?.();
    window.dispatchEvent(new Event("resize"));
  };
  window.addEventListener("hashchange", show);
  $(".nav-toggle").addEventListener("click", () => $(".tabs").classList.toggle("open"));
  show();
}
const onTab = {};
const once = (fn) => { let done = false; return () => { if (!done) { done = true; fn(); } }; };

// ------------------------------------------------------------------ status
async function initStatus() {
  const pill = $("#status-pill");
  try {
    const h = await api("/api/health");
    pill.textContent = h.ai_configured ? `online · AI: ${h.ai_model}` : "online · AI off";
    pill.className = "pill ok";
    return h;
  } catch {
    pill.textContent = "offline";
    pill.className = "pill bad";
    return null;
  }
}

// ------------------------------------------------------------------ DNA (3D)
let DNAHelixClass = null;
async function loadDNA() {
  if (!DNAHelixClass) {
    try { DNAHelixClass = (await import("./dna3d.js")).DNAHelix; }
    catch (e) { console.error(e); return null; }
  }
  return DNAHelixClass;
}

async function initHero() {
  const H = await loadDNA();
  if (!H) { $("#hero-dna").innerHTML = "<p class='hint'>WebGL unavailable.</p>"; return; }
  try {
    const demo = await api("/api/dna/demo");
    const helix = new H($("#hero-dna"), { interactive: false, maxBp: 42 });
    helix.setSequence(demo.sequence);
  } catch (e) { console.warn(e); }
}

const initDnaLab = once(async () => {
  const H = await loadDNA();
  const demo = await api("/api/dna/demo").catch(() => null);
  const helix = H ? new H($("#dna-3d"), { interactive: true, autoRotate: true, maxBp: 150 }) : null;
  const run = async () => {
    const seq = $("#dna-seq").value;
    try {
      const r = await api("/api/dna/analyze", { sequence: seq });
      helix?.setSequence(seq);
      const rc = r.reverse_complement;
      $("#dna-results").innerHTML = `<dl>
        <dt>Length</dt><dd>${r.length_nt} nt · ${fmt(r.helix_geometry.turns, 1)} turns · ${fmt(r.helix_geometry.length_nm, 1)} nm</dd>
        <dt>GC content</dt><dd>${fmt(r.gc_percent, 1)}%</dd>
        <dt>T<sub>m</sub></dt><dd>${fmt(r.melting_temperature.tm_c, 1)} °C <span class="hint">(${esc(r.melting_temperature.method)}; no salt correction)</span></dd>
        <dt>Composition</dt><dd>A ${r.counts.A} · C ${r.counts.C} · G ${r.counts.G} · T ${r.counts.T}${r.counts.N ? " · N " + r.counts.N : ""}</dd>
        <dt>Longest ORF</dt><dd>${r.orfs[0] ? `frame ${r.orfs[0].frame}, nt ${r.orfs[0].start_nt}–${r.orfs[0].end_nt}, ${r.orfs[0].length_aa} aa` : "none ≥ 5 aa"}</dd>
        <dt>Insulin B chain?</dt><dd>${r.contains_insulin_b_chain ? "yes, encoded in this sequence" : "no"}</dd>
        <dt>Reverse complement</dt><dd>${esc(rc.length > 120 ? rc.slice(0, 120) + "…" : rc)}</dd></dl>`;
      $("#dna-translation").innerHTML = Object.entries(r.translations).map(([k, v]) => {
        const html = esc(v).replace(/(M[^*]{4,}\*)/g, "<span class='orf'>$1</span>").replace(/\*/g, "<span class='stop'>*</span>");
        return `<b>${k.replace("_", " ")}</b>\n${html}`;
      }).join("\n\n");
    } catch (e) { toast(e.message, true); }
  };
  $("#dna-run").addEventListener("click", run);
  $("#dna-demo").addEventListener("click", () => { if (demo) { $("#dna-seq").value = demo.sequence; run(); } });
  $("#dna-unzip").addEventListener("click", () => helix?.startBubble());
  if (helix) makeRecorder(() => helix.canvas, $("#dna-rec"), "glucolab-dna");
  if (demo) { $("#dna-seq").value = demo.sequence; $("#dna-note").textContent = demo.note; run(); }
});

// ------------------------------------------------------------------ Meal Lab
let lastMeal = null;
const theatreDrive = { series: null, playing: false, t: 0 };

function mealRequest(form, phenotype) {
  const f = new FormData(form);
  const meals = [];
  const c1 = +f.get("carbs1"), c2 = +f.get("carbs2");
  if (c1 > 0) meals.push({ time_min: +f.get("time1"), carbs_g: c1 });
  if (c2 > 0) meals.push({ time_min: +f.get("time2"), carbs_g: c2 });
  const req = {
    meals, phenotype: phenotype || f.get("phenotype"),
    body_weight_kg: +f.get("bw"), duration_min: +f.get("duration"),
  };
  if (f.get("override")) {
    req.insulin_sensitivity_scale = +f.get("si");
    req.beta_cell_function_scale = +f.get("bf");
  }
  return req;
}

function mealKpis(s) {
  const g2 = s.glucose_2h_mg_dl;
  const cls = (v, a, b) => (v === null ? "" : v < a ? "good" : v < b ? "warn" : "bad");
  const k = [
    ["Peak glucose", fmt(s.peak_glucose_mg_dl, 0) + " mg/dl", cls(s.peak_glucose_mg_dl, 180, 250)],
    ["2-h glucose", g2 === null ? "–" : fmt(g2, 0) + " mg/dl", cls(g2, 140, 200)],
    ["Time to peak", fmt(s.time_to_peak_min, 0) + " min", ""],
    ["Peak insulin", fmt(s.peak_insulin_pmol_l, 0) + " pmol/l", ""],
    ["Time in range 70–180", fmt(s.time_in_range_70_180_pct, 0) + "%", s.time_in_range_70_180_pct > 95 ? "good" : s.time_in_range_70_180_pct > 70 ? "warn" : "bad"],
    ["Glucose iAUC", fmt(s.glucose_iAUC_mg_dl_min / 1000, 1) + "k mg/dl·min", ""],
  ];
  $("#meal-kpis").innerHTML = k.map(([l, v, c]) => `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function drawMeal(results) {
  const annotations = [
    { type: "band", from: 70, to: 180, color: "rgba(52, 211, 153, 0.06)" },
    { type: "hline", value: 140, color: "rgba(251,191,36,.7)", label: "140 (2-h IGT threshold)" },
    { type: "hline", value: 200, color: "rgba(248,113,113,.7)", label: "200 (2-h diabetes threshold)" },
  ];
  const first = results[0].req.meals[0];
  if (first) annotations.push({ type: "vline", value: first.time_min + 120, color: "rgba(148,163,184,.6)", label: "2 h" });
  lineChart("chart-glucose", results.map((r) => ({
    label: r.data.phenotype.label, data: xy(r.data.series.t_min, r.data.series.glucose_mg_dl),
    borderColor: PHENO_COLORS[r.data.phenotype.key], backgroundColor: PHENO_COLORS[r.data.phenotype.key],
  })), { xTitle: "min", annotations, yMin: 40 });
  lineChart("chart-insulin", results.map((r) => ({
    label: r.data.phenotype.label, data: xy(r.data.series.t_min, r.data.series.insulin_pmol_l),
    borderColor: PHENO_COLORS[r.data.phenotype.key],
  })), { xTitle: "min", yMin: 0 });
  const d = results[0].data.series;
  lineChart("chart-fluxes", [
    { label: "Ra (gut appearance)", data: xy(d.t_min, d.ra_mg_kg_min), borderColor: "#fbbf24" },
    { label: "EGP (liver)", data: xy(d.t_min, d.egp_mg_kg_min), borderColor: "#a78bfa" },
    { label: "U_id (insulin-dependent use)", data: xy(d.t_min, d.uid_mg_kg_min), borderColor: "#22d3ee" },
    { label: "Renal excretion", data: xy(d.t_min, d.renal_mg_kg_min), borderColor: "#f87171" },
  ], { xTitle: "min", yMin: 0 });
}

function initMeal() {
  const form = $("#meal-form");
  $$("input[type=range]", form).forEach((r) => {
    const out = $(`output[data-for=${r.name}]`, form);
    r.addEventListener("input", () => { out.textContent = (+r.value).toFixed(2); form.override.checked = true; });
  });
  const run = async (phenotypes) => {
    const btns = $$("button", form); btns.forEach((b) => (b.disabled = true));
    try {
      const results = [];
      for (const ph of phenotypes) {
        const req = mealRequest(form, ph);
        results.push({ req, data: await api("/api/physiology/meal", req) });
      }
      lastMeal = results[0];
      mealKpis(results[0].data.summary);
      drawMeal(results);
      const ph = results[0].data.phenotype;
      $("#meal-note").textContent = `${ph.label}. ${ph.description} Basal: G = ${fmt(results[0].data.basal.Gb, 1)} mg/dl, I = ${fmt(results[0].data.basal.Ib, 1)} pmol/l, kp1 = ${fmt(results[0].data.basal.kp1, 3)} mg/kg/min (derived from steady state).`;
      loadTheatreSeries(results[0].data);
    } catch (e) { toast(e.message, true); }
    finally { btns.forEach((b) => (b.disabled = false)); }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); run([null]); });
  $("#meal-compare").addEventListener("click", () => run(["normal", "insulin_resistant", "type2"]));
  onTab.meal = once(() => run([null]));
}

// ------------------------------------------------------------------ Cell Theatre
let theatre = null;
function loadTheatreSeries(data) {
  const s = data.series;
  const Sb = data.basal.Sb;
  theatreDrive.series = {
    t: s.t_min, g: s.glucose_mg_dl, i: s.insulin_pmol_l,
    fold: s.secretion_pmol_kg_min.map((v) => v / Sb),
    ir: Math.max(0, 1 - data.phenotype.insulin_sensitivity_scale),
  };
  const scrub = $("#t-scrub");
  scrub.max = s.t_min[s.t_min.length - 1];
  $("#drive-sim").disabled = false;
}
function applyTheatreDrive() {
  if (!theatre) return;
  const sim = $("#drive-sim").checked && theatreDrive.series;
  if (sim) {
    const S = theatreDrive.series, t = +$("#t-scrub").value;
    const idx = Math.min(S.t.length - 1, Math.max(0, S.t.findIndex((x) => x >= t)));
    theatre.setDrive({ glucose: S.g[idx], insulin: S.i[idx], secretion: Math.max(0, S.fold[idx]), ir: +$("#ir-manual").value / 100 });
    $("#t-out").textContent = `${t.toFixed(0)} min`;
    $("#g-out").textContent = `${S.g[idx].toFixed(0)} mg/dl (sim)`;
  } else {
    // Manual: static secretion of the Dalla Man model, S_po = S_b + beta (G - G_b), with published normal values;
    // insulin approximated as proportional to secretion at steady state.
    const G = +$("#g-manual").value, Gb = 91.76, beta = 0.11, Sb = 1.5434;
    const fold = Math.max(0, (Sb + beta * (G - Gb)) / Sb);
    theatre.setDrive({ glucose: G, secretion: fold, insulin: 25.49 * fold, ir: +$("#ir-manual").value / 100 });
    $("#g-out").textContent = `${G} mg/dl`;
  }
  $("#ir-out").textContent = `${$("#ir-manual").value}%`;
}
const initTheatre = once(() => {
  theatre = new CellTheatre($("#cell-canvas"), (lines) => {
    $("#cell-readout").innerHTML = lines.map((l) => `<span>${esc(l)}</span>`).join("");
  });
  $("#cell-legend").innerHTML = theatre.setScene("beta");
  $$("#scene-select button").forEach((b) => b.addEventListener("click", () => {
    $$("#scene-select button").forEach((x) => x.classList.toggle("active", x === b));
    $("#cell-legend").innerHTML = theatre.setScene(b.dataset.scene);
  }));
  $("#drive-sim").addEventListener("change", (e) => {
    const on = e.target.checked;
    if (on && !theatreDrive.series) { toast("Run a Meal Lab simulation first."); e.target.checked = false; return; }
    $("#t-scrub").disabled = !on; $("#t-play").disabled = !on; $("#g-manual").disabled = on;
    if (on && theatreDrive.series.ir > 0) $("#ir-manual").value = Math.round(100 * theatreDrive.series.ir / 5) * 5;
    applyTheatreDrive();
  });
  ["#t-scrub", "#g-manual", "#ir-manual"].forEach((s) => $(s).addEventListener("input", applyTheatreDrive));
  let playTimer = null;
  $("#t-play").addEventListener("click", () => {
    if (playTimer) { clearInterval(playTimer); playTimer = null; $("#t-play").textContent = "▶ Play"; return; }
    $("#t-play").textContent = "❚❚ Pause";
    playTimer = setInterval(() => {
      const s = $("#t-scrub");
      let v = +s.value + 1;
      if (v > +s.max) v = 0;
      s.value = v; applyTheatreDrive();
    }, 60); // 1 simulated minute per 60 ms
  });
  $("#cell-canvas").addEventListener("click", () => theatre.click());
  makeRecorder(() => $("#cell-canvas"), $("#rec-btn"), "glucolab-cell");
  applyTheatreDrive();
});

// ------------------------------------------------------------------ β-cell dynamics (Topp)
function initBetaCell() {
  const form = $("#bc-form");
  const sync = () => $$("input[type=range]", form).forEach((r) => ($(`output[data-for=${r.name}]`, form).textContent = (+r.value).toFixed(2)));
  $$("input[type=range]", form).forEach((r) => r.addEventListener("input", sync));
  sync();
  const run = async () => {
    const f = new FormData(form);
    const years = +f.get("years");
    const req = {
      years, si_final_fraction: +f.get("si_final_fraction"),
      si_decline_years: Math.min(+f.get("si_decline_years"), years),
      sigma_scale: +f.get("sigma_scale"), d0_scale: +f.get("d0_scale"),
    };
    try {
      const r = await api("/api/physiology/beta-cell", req);
      lineChart("chart-bc-g", [{ label: "Glucose", data: xy(r.t_years, r.glucose_mg_dl), borderColor: "#fbbf24" }],
        { xTitle: "years", annotations: [{ type: "hline", value: 126, color: "rgba(248,113,113,.7)", label: "126 fasting diabetes threshold" }, { type: "hline", value: 250, color: "rgba(167,139,250,.6)", label: "250 saddle" }], yMin: 0 });
      const c = lineChart("chart-bc-b", [
        { label: "β-cell mass (mg)", data: xy(r.t_years, r.beta_cell_mass_mg), borderColor: "#34d399", yAxisID: "y" },
        { label: "S_I (ml/µU/day)", data: xy(r.t_years, r.si), borderColor: "#22d3ee", yAxisID: "y2" },
      ], { xTitle: "years", yMin: 0 });
      c.options.scales.y2 = { position: "right", min: 0, grid: { drawOnChartArea: false } };
      c.update();
      const fail = r.diabetic_at_end;
      $("#bc-outcome").className = "callout " + (fail ? "bad" : "good");
      $("#bc-outcome").textContent = r.outcome + (fail ? " Glucose crossed the saddle before β-cell mass could compensate." : "");
      $("#bc-fp").innerHTML = `<table><thead><tr><th>Fixed point</th><th>G (mg/dl)</th><th>I (µU/ml)</th><th>β (mg)</th><th>Eigenvalues (day⁻¹)</th><th>Stability</th></tr></thead><tbody>${
        r.final_fixed_points.map((p) => `<tr><td>${esc(p.label)}</td><td class="num">${fmt(p.glucose_mg_dl, 1)}</td><td class="num">${fmt(p.insulin_uU_ml, 2)}</td><td class="num">${fmt(p.beta_cell_mass_mg, 1)}</td><td class="num">${p.eigenvalues_per_day.map((e) => fmt(e.real, 4) + (Math.abs(e.imag) > 1e-9 ? (e.imag > 0 ? "+" : "−") + fmt(Math.abs(e.imag), 3) + "i" : "")).join(", ")}</td><td><span class="tag ${p.stability}">${p.stability}</span></td></tr>`).join("")
      }</tbody></table>`;
      // Closed-form compensation law from the published parameters
      const P = r.params;
      const Gp = (P.r1 - Math.sqrt(P.r1 ** 2 - 4 * P.r2 * P.d0)) / (2 * P.r2);
      const pts = [];
      for (let f = 0.05; f <= 1.5001; f += 0.01) {
        const si = 0.72 * f, I = (P.R0 / Gp - P.EG0) / si;
        pts.push({ x: f, y: (P.k * I * (P.alpha + Gp ** 2)) / (P.sigma * Gp ** 2) });
      }
      lineChart("chart-bc-law", [{ label: `β* at G* = ${fmt(Gp, 0)} mg/dl`, data: pts, borderColor: "#a78bfa" }], { xTitle: "S_I / S_I,normal", yTitle: "β* (mg)" });
    } catch (e) { toast(e.message, true); }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("#bc-fast").addEventListener("click", () => {
    form.years.value = 5; form.si_final_fraction.value = 0.1; form.si_decline_years.value = 0.05; form.sigma_scale.value = 1; form.d0_scale.value = 1;
    sync(); run();
  });
  onTab.betacell = once(run);
}

// ------------------------------------------------------------------ IVGTT
function initIVGTT() {
  const form = $("#iv-form");
  let lastSim = null;
  const sim = async () => {
    const f = new FormData(form);
    const req = { SI: +f.get("SI") * 1e-4, SG: +f.get("SG"), p2: +f.get("p2"), dose_g_per_kg: +f.get("dose_g_per_kg"), Gb: +f.get("Gb"), Ib: +f.get("Ib") };
    try {
      const r = await api("/api/physiology/ivgtt", req);
      lastSim = r;
      const c = lineChart("chart-iv", [
        { label: "Glucose (mg/dl)", data: xy(r.t_min, r.glucose_mg_dl), borderColor: "#fbbf24", yAxisID: "y" },
        { label: "Insulin (µU/ml)", data: xy(r.t_min, r.insulin_uU_ml), borderColor: "#22d3ee", yAxisID: "y2" },
      ], { xTitle: "min" });
      c.options.scales.y2 = { position: "right", grid: { drawOnChartArea: false } }; c.update();
      $("#iv-kg").textContent = `G₀ = ${fmt(r.G0_mg_dl, 0)} mg/dl · K_G (10–40 min) = ${fmt(r.Kg_pct_per_min, 2)} %/min`;
    } catch (e) { toast(e.message, true); }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); sim(); });
  $("#fit-example").addEventListener("click", async () => {
    if (!lastSim) await sim();
    const ts = [0, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180];
    const interp = (arr, t) => { const i = lastSim.t_min.findIndex((x) => x >= t); return arr[i < 0 ? arr.length - 1 : i]; };
    const gauss = () => Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());
    const Gb = +form.Gb.value, Ib = +form.Ib.value;
    const rows = ts.map((t) => t === 0 ? `0, ${Gb}, ${Ib}` : `${t}, ${(interp(lastSim.glucose_mg_dl, t) * (1 + 0.015 * gauss())).toFixed(1)}, ${interp(lastSim.insulin_uU_ml, t).toFixed(1)}`);
    $("#fit-data").value = "# SYNTHETIC data: simulated IVGTT with 1.5% glucose noise\n# t_min, glucose_mg_dl, insulin_uU_ml (t=0 row = basal)\n" + rows.join("\n");
    toast(`Synthetic data generated with true S_I = ${form.SI.value}×10⁻⁴`);
  });
  $("#fit-run").addEventListener("click", async () => {
    const rows = $("#fit-data").value.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#")).map((l) => l.split(/[,;\s\t]+/).map(Number));
    if (rows.some((r) => r.length < 3 || r.some((v) => !Number.isFinite(v)))) { toast("Each line needs three numbers: t, glucose, insulin.", true); return; }
    const btn = $("#fit-run"); btn.disabled = true; btn.textContent = "Fitting…";
    try {
      const r = await api("/api/physiology/ivgtt/fit", { t_min: rows.map((x) => x[0]), glucose_mg_dl: rows.map((x) => x[1]), insulin_uU_ml: rows.map((x) => x[2]) });
      const cv = (k) => (r.cv_percent[k] == null ? "–" : fmt(r.cv_percent[k], 1) + "%");
      $("#fit-results").innerHTML = `<dl>
        <dt>S<sub>I</sub></dt><dd>${fmt(r.SI_per_min_per_uU_ml * 1e4, 3)} ×10⁻⁴ min⁻¹ per µU/ml (CV ${cv("SI")})</dd>
        <dt>S<sub>G</sub></dt><dd>${fmt(r.SG_per_min, 4)} min⁻¹ (CV ${cv("SG")})</dd>
        <dt>p₂</dt><dd>${fmt(r.p2_per_min, 4)} min⁻¹ (CV ${cv("p2")})</dd>
        <dt>G₀</dt><dd>${fmt(r.G0_mg_dl, 1)} mg/dl</dd>
        <dt>RMSE</dt><dd>${fmt(r.rmse_mg_dl, 2)} mg/dl · ${r.converged ? "converged" : "NOT converged"}</dd></dl>
        <p class="hint">${esc(r.note)}</p>`;
      const c = lineChart("chart-fit", [
        { label: "Measured glucose", data: rows.filter((x) => x[0] >= 0).map((x) => ({ x: x[0], y: x[1] })), borderColor: "#fbbf24", backgroundColor: "#fbbf24", showLine: false, pointRadius: 4 },
        { label: "Minimal-model fit", data: xy(r.fitted_t_min, r.fitted_glucose_mg_dl), borderColor: "#a78bfa" },
      ], { xTitle: "min" });
      c.data.datasets[0].pointRadius = 4; c.update();
    } catch (e) { toast(e.message, true); }
    finally { btn.disabled = false; btn.textContent = "Fit minimal model"; }
  });
  onTab.ivgtt = once(sim);
}

// ------------------------------------------------------------------ Clinical
function initClinical() {
  const form = $("#clin-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const req = {};
    for (const el of form.elements) {
      if (!el.name) continue;
      if (el.type === "checkbox") req[el.name] = el.checked;
      else if (el.value !== "") req[el.name] = +el.value;
    }
    try {
      const r = await api("/api/clinical/indices", req);
      const cards = [];
      if (r.homa1) cards.push(`<div class="kpi"><div class="v">${fmt(r.homa1.homa_ir, 2)}</div><div class="l">HOMA1-IR</div></div>`,
        `<div class="kpi"><div class="v">${r.homa1.homa_beta_pct == null ? "undefined" : fmt(r.homa1.homa_beta_pct, 0) + "%"}</div><div class="l">HOMA1-%B</div></div>`,
        `<div class="kpi"><div class="v">${fmt(r.quicki, 3)}</div><div class="l">QUICKI</div></div>`);
      if (r.eag) cards.push(`<div class="kpi"><div class="v">${fmt(r.eag.eag_mg_dl, 0)}</div><div class="l">eAG mg/dl (${fmt(r.eag.eag_mmol_l, 1)} mmol/l)</div></div>`);
      if (r.tyg_index != null) cards.push(`<div class="kpi"><div class="v">${fmt(r.tyg_index, 2)}</div><div class="l">TyG index</div></div>`);
      if (r.bmi) cards.push(`<div class="kpi"><div class="v">${fmt(r.bmi.bmi, 1)}</div><div class="l">BMI: ${esc(r.bmi.who_category)}</div></div>`);
      const ada = r.ada;
      const cat = (c) => (c.startsWith("diabetes") ? "diabetes" : c.includes("prediabetes") ? "prediabetes" : c === "normal" ? "normal" : "");
      $("#clin-results").innerHTML = `
        ${cards.length ? `<div class="kpis">${cards.join("")}</div>` : ""}
        <div class="card"><h4>ADA diagnostic classification</h4>
          ${ada.results.length ? `<div class="table-wrap"><table><thead><tr><th>Test</th><th>Value</th><th>Category</th><th>Thresholds</th></tr></thead><tbody>${ada.results.map((x) => `<tr><td>${esc(x.test)}</td><td class="num">${fmt(x.value, 1)} ${x.unit}</td><td><span class="tag ${cat(x.category)}">${esc(x.category)}</span></td><td class="hint">${esc(x.thresholds)}</td></tr>`).join("")}</tbody></table></div>` : ""}
          <div class="callout" style="margin-top:.8rem">${esc(ada.summary)}</div>
          <p class="hint" style="margin-top:.6rem">${esc(ada.disclaimer)} ${r.homa1 ? esc(r.homa1.note) : ""} ${r.bmi ? esc(r.bmi.note) : ""}</p>
        </div>`;
    } catch (err) { toast(err.message, true); }
  });
}

// ------------------------------------------------------------------ Risk ML
const SYMPTOMS = [
  ["polyuria", "Polyuria"], ["polydipsia", "Polydipsia"], ["sudden_weight_loss", "Sudden weight loss"],
  ["weakness", "Weakness"], ["polyphagia", "Polyphagia"], ["genital_thrush", "Genital thrush"],
  ["visual_blurring", "Visual blurring"], ["itching", "Itching"], ["irritability", "Irritability"],
  ["delayed_healing", "Delayed healing"], ["partial_paresis", "Partial paresis"], ["muscle_stiffness", "Muscle stiffness"],
  ["alopecia", "Alopecia"], ["obesity", "Obesity"],
];
function gaugeSVG(p) {
  const r = 52, c = 2 * Math.PI * r, col = p < 0.2 ? "#34d399" : p < 0.5 ? "#fbbf24" : p < 0.8 ? "#fb923c" : "#f87171";
  return `<svg width="130" height="130" viewBox="0 0 130 130"><circle cx="65" cy="65" r="${r}" stroke="rgba(148,163,184,.15)" stroke-width="12" fill="none"/>
    <circle cx="65" cy="65" r="${r}" stroke="${col}" stroke-width="12" fill="none" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - p)}" transform="rotate(-90 65 65)" style="transition:stroke-dashoffset 1s ease"/>
    <text x="65" y="72" text-anchor="middle" fill="#e6edf7" font-size="22" font-weight="700">${Math.round(p * 100)}%</text></svg>`;
}
function initRisk() {
  const chips = $("#symptom-chips");
  SYMPTOMS.forEach(([k, l]) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.dataset.key = k; b.textContent = l; b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") === "true" ? "false" : "true"));
    chips.appendChild(b);
  });
  const form = $("#risk-form");
  const prev = form.prev, prevOut = $("output[data-for=prev]", form);
  prev.addEventListener("input", () => (prevOut.textContent = +prev.value ? prev.value + "%" : "off"));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const patient = { age: +form.age.value, male: form.male.value === "true" };
    $$(".chip", chips).forEach((c) => (patient[c.dataset.key] = c.getAttribute("aria-pressed") === "true"));
    const body = { patient };
    if (+prev.value) body.target_prevalence = +prev.value / 100;
    $("#risk-result").innerHTML = `<div class="skeleton"></div>`;
    try {
      const r = await api("/api/risk/predict", body);
      const adj = r.prevalence_adjusted;
      $("#risk-result").innerHTML = `<div class="gauge">${gaugeSVG(r.probability)}
        <div><div class="big">${pct(r.probability, 1)}</div>
        <div class="hint">Ensemble probability of a diabetes-positive label in the training population (prevalence ${pct(r.train_prevalence, 1)}).</div>
        <div class="hint">Logistic ${pct(r.probability_logistic)} (95% bootstrap interval ${pct(r.logistic_95ci[0])}–${pct(r.logistic_95ci[1])}) · Random forest ${pct(r.probability_random_forest)}</div>
        ${adj ? `<div class="callout" style="margin-top:.5rem">At a prevalence of ${pct(adj.target_prevalence, 0)}: <b>${pct(adj.probability, 1)}</b>. <span class="hint">${esc(adj.assumption)}</span></div>` : ""}
        </div></div>
        ${r.warnings.map((w) => `<div class="callout warn" style="margin-top:.6rem">${esc(w)}</div>`).join("")}
        <p class="hint" style="margin-top:.6rem">${esc(r.disclaimer)}</p>`;
      const cs = r.explanation.contributions.slice(0, 12);
      if (charts["chart-contrib"]) charts["chart-contrib"].destroy();
      charts["chart-contrib"] = new window.Chart($("#chart-contrib"), {
        type: "bar",
        data: { labels: cs.map((c) => c.label), datasets: [{ data: cs.map((c) => c.log_odds), backgroundColor: cs.map((c) => (c.log_odds > 0 ? "rgba(248,113,113,.75)" : "rgba(52,211,153,.75)")), borderRadius: 4 }] },
        options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "log-odds vs reference patient (+ raises risk)" } } } },
      });
    } catch (err) { $("#risk-result").innerHTML = ""; toast(err.message, true); }
  });
  onTab.risk = once(loadModelCard);
}

async function loadModelCard() {
  const el = $("#risk-card");
  el.innerHTML = `<div class="card"><div class="skeleton"></div><p class="hint">Training with repeated grouped cross-validation (first run only, about 30 s)…</p></div>`;
  try {
    const r = await api("/api/risk/model-card");
    const ev = r.evaluation, ds = r.dataset;
    const names = { logistic: "Logistic regression", random_forest: "Random forest", ensemble: "Ensemble (primary)" };
    const row = (k, m) => `<tr><td>${names[k] || k}</td>${["auc", "brier", "accuracy", "sensitivity", "specificity"].map((x) =>
      `<td class="num">${fmt(m.pooled[x], 3)}<br><span class="hint">[${fmt(m.ci95[x][0], 3)}, ${fmt(m.ci95[x][1], 3)}]</span></td>`).join("")}</tr>`;
    el.innerHTML = `
      <div class="metric-grid">
        <div class="kpi"><div class="v">${ds.rows}</div><div class="l">records (${ds.unique_rows} unique)</div></div>
        <div class="kpi warn"><div class="v">${ds.duplicate_rows}</div><div class="l">exact duplicate rows</div></div>
        <div class="kpi"><div class="v">${fmt(ev.models.ensemble.pooled.auc, 3)}</div><div class="l">leak-free ROC AUC (ensemble)</div></div>
        <div class="kpi bad"><div class="v">${fmt(ev.naive_leaky_cv.accuracy * 100, 1)}% → ${fmt(ev.models.ensemble.pooled.accuracy * 100, 1)}%</div><div class="l">accuracy: naive (leaky) → grouped</div></div>
      </div>
      <div class="card"><h4>Validation: ${esc(ev.scheme)}</h4>
        <div class="table-wrap"><table><thead><tr><th>Model</th><th>ROC AUC</th><th>Brier ↓</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th></tr></thead>
        <tbody>${Object.entries(ev.models).map(([k, m]) => row(k, m)).join("")}
        <tr><td>Naive random K-fold RF (leaky, for comparison)</td>${["auc", "brier", "accuracy", "sensitivity", "specificity"].map((x) => `<td class="num" style="color:#f87171">${fmt(ev.naive_leaky_cv[x], 3)}</td>`).join("")}</tr>
        </tbody></table></div>
        <p class="hint">95% CIs from 500 bootstrap resamples of duplicate groups. ${esc(ev.leakage_note)}</p></div>
      <div class="grid-2">
        <div class="card chart-card"><h4>ROC curve (pooled out-of-fold)</h4><div class="chart-box"><canvas id="chart-roc"></canvas></div></div>
        <div class="card chart-card"><h4>Calibration (reliability)</h4><div class="chart-box"><canvas id="chart-cal"></canvas></div></div>
      </div>
      <div class="grid-2">
        <div class="card"><h4>Adjusted odds ratios (logistic)</h4><div class="table-wrap"><table><thead><tr><th>Feature</th><th>OR</th><th>per</th></tr></thead><tbody>${
          r.explanations.odds_ratios.map((o) => `<tr><td>${esc(o.label)}</td><td class="num">${fmt(o.odds_ratio, 2)}</td><td class="hint">${esc(o.per)}</td></tr>`).join("")}</tbody></table></div></div>
        <div class="card"><h4>Held-out permutation importance (ΔAUC)</h4><div class="table-wrap"><table><thead><tr><th>Feature</th><th>ΔAUC</th><th>SD</th></tr></thead><tbody>${
          r.explanations.permutation_importance.map((o) => `<tr><td>${esc(o.label)}</td><td class="num">${fmt(o.auc_drop, 4)}</td><td class="num">${fmt(o.sd_across_folds, 4)}</td></tr>`).join("")}</tbody></table></div></div>
      </div>
      <div class="card"><h4>Data & limitations</h4><p class="hint">${esc(ds.name)}: ${esc(ds.source)} Positive ${ds.positive}, negative ${ds.negative}; ages ${ds.age_range[0]}–${ds.age_range[1]}. SHA-256 prefix ${esc(ds.sha256_16)}.</p>
        <ul>${r.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`;
    lineChart("chart-roc", [
      { label: `Ensemble (AUC ${fmt(ev.models.ensemble.pooled.auc, 3)})`, data: xy(ev.roc_curve.fpr, ev.roc_curve.tpr), borderColor: "#22d3ee", stepped: true, tension: 0 },
      { label: "Chance", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "rgba(148,163,184,.5)", borderDash: [5, 5] },
    ], { xTitle: "False positive rate", yTitle: "True positive rate" });
    lineChart("chart-cal", [
      { label: "Observed vs predicted", data: ev.calibration.map((b) => ({ x: b.mean_pred, y: b.observed })), borderColor: "#a78bfa", backgroundColor: "#a78bfa", pointRadius: 5, tension: 0 },
      { label: "Perfect calibration", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: "rgba(148,163,184,.5)", borderDash: [5, 5] },
    ], { xTitle: "Mean predicted probability", yTitle: "Observed frequency" });
    charts["chart-cal"].data.datasets[0].pointRadius = 5; charts["chart-cal"].update();
  } catch (e) { el.innerHTML = `<div class="callout bad">${esc(e.message)}</div>`; }
}

// ------------------------------------------------------------------ AI assistant
const SUGGESTIONS = [
  "Simulate a 75 g meal in the healthy and type 2 phenotypes and explain the mechanistic differences.",
  "Why does a rapid fall in insulin sensitivity cause β-cell failure in the Topp model, but a slow one does not? Show the fixed points.",
  "Compute HOMA-IR and QUICKI for fasting glucose 105 mg/dl and insulin 14 µU/ml and interpret them.",
  "How reliable is the symptom risk model, and what does the duplicate-record problem do to naive accuracy?",
];
function initAssistant(health) {
  const log = $("#chat-log"), input = $("#chat-input"), form = $("#chat-form");
  const history = [];
  if (health && health.ai_configured) $("#ai-model").textContent = `model: ${health.ai_model}`;
  else if (health) {
    const b = $("#ai-banner"); b.hidden = false;
    b.innerHTML = "The AI assistant is not configured on this server. Set <code>ANTHROPIC_API_KEY</code> in the environment (or in a git-ignored <code>.env</code> file) and restart. All other features work without it.";
  }
  SUGGESTIONS.forEach((s) => {
    const b = document.createElement("button"); b.className = "btn small ghost"; b.type = "button"; b.textContent = s;
    b.addEventListener("click", () => { input.value = s; form.requestSubmit(); });
    $("#chat-suggestions").appendChild(b);
  });
  const add = (role, html, cls = "") => {
    const d = document.createElement("div"); d.className = `msg ${role} ${cls}`; d.innerHTML = html; log.appendChild(d);
    log.scrollTop = log.scrollHeight; return d;
  };
  add("assistant", markdown("Hello. I can run the meal model, the β-cell mass model, the minimal model, clinical indices and the validated risk model for you, then explain the results. What would you like to explore?"));
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    history.push({ role: "user", content: text });
    add("user", esc(text));
    const pending = add("assistant", `<span class="typing"><span></span><span></span><span></span></span> <span class="hint">thinking and running models…</span>`);
    $("#chat-send").disabled = true;
    try {
      const r = await api("/api/ai/chat", { messages: history.slice(-40) });
      history.push({ role: "assistant", content: r.text });
      const tools = r.tool_calls.length
        ? `<details><summary>${r.tool_calls.length} tool call${r.tool_calls.length > 1 ? "s" : ""}: ${r.tool_calls.map((t) => `<span class="tool-chip ${t.is_error ? "err" : ""}">${esc(t.name)}</span>`).join("")}</summary>${
            r.tool_calls.map((t) => `<p><b>${esc(t.name)}</b> input:</p><pre>${esc(JSON.stringify(t.input, null, 2))}</pre><p>output (preview):</p><pre>${esc(t.output_preview)}</pre>`).join("")}</details>`
        : "";
      pending.innerHTML = markdown(r.text) + tools + `<div class="meta">${esc(r.model)} · ${r.usage.input_tokens + r.usage.output_tokens} tokens</div>`;
    } catch (err) {
      history.pop();
      pending.classList.add("error");
      pending.textContent = err.message;
    } finally {
      $("#chat-send").disabled = false;
      log.scrollTop = log.scrollHeight;
    }
  });
}

// ------------------------------------------------------------------ references
async function initRefs() {
  try {
    const r = await api("/api/references");
    $("#ref-list").innerHTML = r.references.map((x) => `<li><span class="used">${esc(x.used_for)}</span>${esc(x.citation)}${x.doi ? ` <a href="https://doi.org/${encodeURI(x.doi)}" target="_blank" rel="noopener noreferrer">doi:${esc(x.doi)}</a>` : ""}</li>`).join("");
  } catch (e) { $("#ref-list").innerHTML = `<li>${esc(e.message)}</li>`; }
}

// ------------------------------------------------------------------ boot
window.addEventListener("DOMContentLoaded", async () => {
  if (!window.Chart) await new Promise((r) => window.addEventListener("load", r, { once: true }));
  chartDefaults();
  initMeal(); initBetaCell(); initIVGTT(); initClinical(); initRisk();
  onTab.cells = initTheatre;
  onTab.dna = initDnaLab;
  onTab.refs = once(initRefs);
  $("#to-theatre").addEventListener("click", () => {
    setTimeout(() => { if (theatreDrive.series) { $("#drive-sim").checked = true; $("#drive-sim").dispatchEvent(new Event("change")); } }, 50);
  });
  initTabs();
  const health = await initStatus();
  initAssistant(health);
  initHero();
});
