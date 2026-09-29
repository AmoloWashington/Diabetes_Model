// Model-driven cell animations on a 2D canvas.
//
// Drive inputs (set from the Meal Lab simulation or from manual sliders):
//   glucose      plasma glucose, mg/dl
//   secretion    beta-cell secretion as a fold change over basal
//   insulin      plasma insulin, pmol/l
//   ir           insulin resistance, 0..0.95 (fraction of post-receptor signal lost)
//
// Sequences of molecular events follow Rorsman & Ashcroft (Physiol Rev 2018)
// for stimulus-secretion coupling and Saltiel & Kahn (Nature 2001) for insulin
// signalling. Geometry and particle counts are schematic.

const W = 1200, H = 680;
const TAU = Math.PI * 2;
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;

export const INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT";
export const INSULIN_A = "GIVEQCCTSICSLYQLENYCN";

const C = {
  glucose: "#fbbf24", g6p: "#fb923c", atp: "#4ade80", k: "#c084fc", ca: "#22d3ee",
  insulin: "#60a5fa", membrane: "#5eead4", cyto: "rgba(20, 45, 70, 0.55)", text: "#e6edf7", muted: "#93a2b8",
};

function hexagon(ctx, x, y, r, fill) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * TAU + Math.PI / 6;
    ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a));
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

function label(ctx, text, x, y, { color = C.text, size = 13, align = "center", bg = true } = {}) {
  ctx.font = `600 ${size}px system-ui, sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  if (bg) {
    const w = ctx.measureText(text).width + 10;
    const x0 = align === "center" ? x - w / 2 : align === "right" ? x - w : x - 5;
    ctx.fillStyle = "rgba(5, 9, 16, 0.72)";
    ctx.beginPath();
    ctx.roundRect(x0, y - size * 0.8, w, size * 1.6, 5);
    ctx.fill();
  }
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function glow(ctx, x, y, r, color, alpha = 0.35) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, color.replace(")", `, ${alpha})`).replace("rgb", "rgba"));
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
}

// ============================================================ beta cell

class BetaCellScene {
  constructor() {
    this.name = "beta";
    this.cell = { x: 250, y: 70, w: 740, h: 540 };
    this.gluts = [190, 340, 490].map((y) => ({ x: this.cell.x, y }));
    this.katp = [430, 560, 690].map((x) => ({ x, y: this.cell.y, open: 1 }));
    this.vgcc = [420, 560, 700].map((x) => ({ x, y: this.cell.y + this.cell.h, open: 0 }));
    this.mito = [{ x: 520, y: 300, rx: 70, ry: 30, rot: -0.3 }, { x: 610, y: 430, rx: 60, ry: 26, rot: 0.4 }];
    this.gck = { x: 360, y: 330 };
    this.nucleus = { x: 840, y: 190, r: 70 };
    this.golgi = { x: 820, y: 380 };
    this.particles = [];
    this.granules = [];
    for (let i = 0; i < 38; i++) this.granules.push(this._newGranule(true));
    this.fusions = [];
    this.vm = -70; this.vmTrace = new Array(260).fill(-70);
    this.atpRatio = 0.2; this.katpOpen = 1; this.t = 0;
    this.spawn = { glucose: 0, k: 0, ca: 0, exo: 0, bg: 0 };
  }

  _newGranule(scatter) {
    const c = this.cell;
    return {
      x: scatter ? rand(c.x + 180, c.x + c.w - 60) : this.golgi.x + rand(-20, 20),
      y: scatter ? rand(c.y + 60, c.y + c.h - 60) : this.golgi.y + rand(-15, 15),
      vx: rand(-6, 6), vy: rand(-6, 6), r: rand(7, 10), state: "reserve", target: null,
    };
  }

  update(dt, d) {
    this.t += dt;
    const G = d.glucose;
    // Visual metabolic flux: glucokinase kinetics, S0.5 ~ 8 mM (~144 mg/dl), Hill ~1.7
    const flux = G ** 1.7 / (G ** 1.7 + 144 ** 1.7);
    this.atpRatio = lerp(this.atpRatio, flux, dt * 1.5);
    // K_ATP closes as ATP/ADP rises (schematic dose-response)
    this.katpOpen = 1 / (1 + (this.atpRatio / 0.35) ** 4);
    this.katp.forEach((ch) => (ch.open = this.katpOpen));
    // Schematic membrane potential: depolarises as K_ATP closes; bursts above threshold
    const plateau = -70 + 32 * (1 - this.katpOpen);
    const active = plateau > -52;
    let vm = plateau;
    if (active) {
      const burstPhase = (this.t * (0.6 + 0.8 * (1 - this.katpOpen))) % 1;
      const inBurst = burstPhase < 0.25 + 0.6 * (1 - this.katpOpen);
      if (inBurst) vm = -45 + 30 * Math.max(0, Math.sin(this.t * 38)) ** 6;
    }
    this.vm = lerp(this.vm, vm, clamp(dt * 25, 0, 1));
    this.vmTrace.push(this.vm); this.vmTrace.shift();
    const caOpen = clamp((this.vm + 50) / 25, 0, 1);
    this.vgcc.forEach((ch) => (ch.open = lerp(ch.open, caOpen, dt * 10)));

    // --- spawn particles
    const c = this.cell;
    this.spawn.bg += dt * (4 + G / 12);
    while (this.spawn.bg > 1) { // extracellular glucose flowing in the capillary
      this.spawn.bg -= 1;
      this.particles.push({ kind: "gluOut", x: rand(40, 190), y: -10, vx: rand(-4, 4), vy: rand(50, 80) });
    }
    this.spawn.glucose += dt * flux * 7;
    while (this.spawn.glucose > 1) {
      this.spawn.glucose -= 1;
      const g = this.gluts[Math.floor(Math.random() * 3)];
      this.particles.push({ kind: "glucose", x: g.x - 30, y: g.y + rand(-6, 6), phase: 0 });
    }
    this.spawn.k += dt * this.katpOpen * 6;
    while (this.spawn.k > 1) {
      this.spawn.k -= 1;
      const ch = this.katp[Math.floor(Math.random() * 3)];
      this.particles.push({ kind: "k", x: ch.x + rand(-4, 4), y: ch.y + 20, vx: rand(-8, 8), vy: -rand(40, 70), life: 2 });
    }
    this.spawn.ca += dt * caOpen * 14;
    while (this.spawn.ca > 1) {
      this.spawn.ca -= 1;
      const ch = this.vgcc[Math.floor(Math.random() * 3)];
      this.particles.push({ kind: "ca", x: ch.x + rand(-5, 5), y: ch.y + 30, vx: rand(-30, 30), vy: -rand(60, 110), life: 1.6 });
    }
    // Exocytosis rate follows the (model) secretion fold change
    this.spawn.exo += dt * 0.35 * d.secretion;
    while (this.spawn.exo > 1) {
      this.spawn.exo -= 1;
      const g = this.granules.find((q) => q.state === "reserve");
      if (g) { g.state = "docking"; g.target = { x: c.x + c.w, y: rand(c.y + 110, c.y + c.h - 110) }; }
    }

    // --- update particles
    for (const p of this.particles) {
      if (p.kind === "gluOut") { p.x += p.vx * dt; p.y += p.vy * dt; if (p.y > H + 10) p.dead = true; }
      else if (p.kind === "glucose") {
        if (p.phase === 0) { p.x += 90 * dt; if (p.x > c.x + 25) p.phase = 1; }
        else if (p.phase === 1) { // to glucokinase
          const dx = this.gck.x - p.x, dy = this.gck.y - p.y, dist = Math.hypot(dx, dy);
          if (dist < 12) { p.phase = 2; p.kind = "g6p"; p.m = this.mito[Math.floor(Math.random() * 2)]; }
          else { p.x += (dx / dist) * 110 * dt; p.y += (dy / dist) * 110 * dt; }
        }
      } else if (p.kind === "g6p") {
        const dx = p.m.x - p.x, dy = p.m.y - p.y, dist = Math.hypot(dx, dy);
        if (dist < 20) {
          p.dead = true;
          for (let i = 0; i < 2; i++) this.particles.push({ kind: "atp", x: p.m.x, y: p.m.y, tx: this.katp[Math.floor(Math.random() * 3)].x, life: 2.4 });
        } else { p.x += (dx / dist) * 95 * dt; p.y += (dy / dist) * 95 * dt; }
      } else if (p.kind === "atp") {
        p.life -= dt;
        const ty = c.y + 30, dx = p.tx - p.x, dy = ty - p.y, dist = Math.hypot(dx, dy);
        if (dist > 5) { p.x += (dx / dist) * 80 * dt; p.y += (dy / dist) * 80 * dt; }
        if (p.life < 0) p.dead = true;
      } else if (p.kind === "k" || p.kind === "ca" || p.kind === "ins") {
        p.x += p.vx * dt; p.y += p.vy * dt; p.life -= dt;
        if (p.kind === "ca") p.vy *= 0.985;
        if (p.life < 0 || p.y < -10) p.dead = true;
      }
    }
    this.particles = this.particles.filter((p) => !p.dead);
    if (this.particles.length > 900) this.particles.splice(0, this.particles.length - 900);

    // --- granules
    for (const g of this.granules) {
      if (g.state === "reserve") {
        g.vx += rand(-20, 20) * dt; g.vy += rand(-20, 20) * dt;
        g.vx *= 0.98; g.vy *= 0.98;
        g.x = clamp(g.x + g.vx * dt, c.x + 160, c.x + c.w - 40);
        g.y = clamp(g.y + g.vy * dt, c.y + 40, c.y + c.h - 40);
        if (Math.hypot(g.x - this.nucleus.x, g.y - this.nucleus.y) < this.nucleus.r + 14) { g.vx += (g.x - this.nucleus.x) * 0.5 * dt; g.vy += (g.y - this.nucleus.y) * 0.5 * dt; }
      } else if (g.state === "docking") {
        const dx = g.target.x - g.r - 4 - g.x, dy = g.target.y - g.y, dist = Math.hypot(dx, dy);
        if (dist < 3) { g.state = "fusing"; g.f = 0; }
        else { g.x += (dx / dist) * 140 * dt; g.y += (dy / dist) * 140 * dt; }
      } else if (g.state === "fusing") {
        g.f += dt * 1.6;
        if (g.f >= 1) {
          g.dead = true;
          this.fusions.push({ x: c.x + c.w, y: g.y, t: 0 });
          for (let i = 0; i < 6; i++) this.particles.push({ kind: "ins", x: c.x + c.w + 8, y: g.y + rand(-6, 6), vx: rand(40, 110), vy: -rand(20, 60), life: 3 });
        }
      }
    }
    const lost = this.granules.filter((g) => g.dead).length;
    this.granules = this.granules.filter((g) => !g.dead);
    for (let i = 0; i < lost; i++) this.granules.push(this._newGranule(false)); // new granules bud from the Golgi
    this.fusions.forEach((f) => (f.t += dt));
    this.fusions = this.fusions.filter((f) => f.t < 0.8);
  }

  draw(ctx, d) {
    const c = this.cell;
    ctx.fillStyle = "#050910"; ctx.fillRect(0, 0, W, H);
    // capillaries
    for (const [x0, x1, lab] of [[20, 210, "Blood: glucose"], [c.x + c.w + 20, W - 20, "Portal blood: insulin"]]) {
      const g = ctx.createLinearGradient(x0, 0, x1, 0);
      g.addColorStop(0, "rgba(127, 29, 29, 0.28)"); g.addColorStop(0.5, "rgba(153, 27, 27, 0.18)"); g.addColorStop(1, "rgba(127, 29, 29, 0.28)");
      ctx.fillStyle = g; ctx.fillRect(x0, 0, x1 - x0, H);
      ctx.strokeStyle = "rgba(248, 113, 113, 0.35)"; ctx.setLineDash([6, 8]); ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, -2, x1 - x0, H + 4); ctx.setLineDash([]);
      label(ctx, lab, (x0 + x1) / 2, H - 22, { color: C.muted, size: 12 });
    }
    // cell body
    ctx.save();
    ctx.beginPath(); ctx.roundRect(c.x, c.y, c.w, c.h, 60);
    const cg = ctx.createRadialGradient(c.x + c.w * 0.5, c.y + c.h * 0.45, 50, c.x + c.w * 0.5, c.y + c.h * 0.5, c.w * 0.7);
    cg.addColorStop(0, "rgba(30, 64, 95, 0.55)"); cg.addColorStop(1, "rgba(12, 30, 52, 0.55)");
    ctx.fillStyle = cg; ctx.fill();
    ctx.lineWidth = 7; ctx.strokeStyle = "rgba(94, 234, 212, 0.25)"; ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = C.membrane; ctx.stroke();
    ctx.restore();
    label(ctx, "Pancreatic β-cell", c.x + 110, c.y + c.h - 24, { size: 14 });

    // nucleus with chromatin
    const n = this.nucleus;
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, TAU);
    ctx.fillStyle = "rgba(76, 29, 149, 0.35)"; ctx.fill(); ctx.strokeStyle = "rgba(167, 139, 250, 0.7)"; ctx.lineWidth = 2; ctx.stroke();
    ctx.strokeStyle = "rgba(196, 181, 253, 0.45)"; ctx.lineWidth = 1.5;
    for (let k = 0; k < 2; k++) {
      ctx.beginPath();
      for (let x = -50; x <= 50; x += 2) ctx.lineTo(n.x + x, n.y + 12 * Math.sin(x / 7 + this.t + k * Math.PI) + (k ? 8 : -8));
      ctx.stroke();
    }
    label(ctx, "Nucleus (INS gene)", n.x, n.y + n.r + 14, { size: 11, color: C.muted });
    // Golgi stacks
    ctx.strokeStyle = "rgba(251, 191, 36, 0.55)"; ctx.lineWidth = 4; ctx.lineCap = "round";
    for (let k = 0; k < 4; k++) { ctx.beginPath(); ctx.arc(this.golgi.x, this.golgi.y + 40 + k * 9, 38 - k * 5, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke(); }
    label(ctx, "Golgi → new granules", this.golgi.x, this.golgi.y + 70, { size: 11, color: C.muted });
    // mitochondria
    for (const m of this.mito) {
      ctx.save(); ctx.translate(m.x, m.y); ctx.rotate(m.rot);
      ctx.beginPath(); ctx.ellipse(0, 0, m.rx, m.ry, 0, 0, TAU);
      ctx.fillStyle = "rgba(22, 101, 52, 0.45)"; ctx.fill(); ctx.strokeStyle = "rgba(74, 222, 128, 0.8)"; ctx.lineWidth = 2; ctx.stroke();
      ctx.strokeStyle = "rgba(74, 222, 128, 0.5)"; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let x = -m.rx + 12; x < m.rx - 12; x += 3) ctx.lineTo(x, (m.ry - 8) * Math.sin(x / 5));
      ctx.stroke(); ctx.restore();
      glow(ctx, m.x, m.y, 80, "rgb(74, 222, 128)", 0.12 + 0.3 * this.atpRatio);
    }
    label(ctx, "Mitochondria: ATP", this.mito[0].x, this.mito[0].y - 48, { size: 11, color: "#86efac" });
    // glucokinase
    glow(ctx, this.gck.x, this.gck.y, 40, "rgb(251, 146, 60)", 0.35);
    ctx.beginPath(); ctx.arc(this.gck.x, this.gck.y, 16, 0.4, TAU - 0.4); ctx.lineTo(this.gck.x, this.gck.y); ctx.closePath();
    ctx.fillStyle = "#f97316"; ctx.fill();
    label(ctx, "Glucokinase", this.gck.x, this.gck.y + 30, { size: 11 });

    // glucose transporters
    for (const g of this.gluts) {
      ctx.fillStyle = "#b45309"; ctx.beginPath(); ctx.roundRect(g.x - 12, g.y - 20, 10, 40, 4); ctx.fill();
      ctx.beginPath(); ctx.roundRect(g.x + 2, g.y - 20, 10, 40, 4); ctx.fill();
    }
    label(ctx, "GLUT1 (human) / GLUT2 (rodent)", this.gluts[0].x + 10, this.gluts[0].y - 40, { size: 11, align: "left" });
    // K_ATP channels (top)
    for (const ch of this.katp) {
      const gap = 4 + 10 * ch.open;
      ctx.fillStyle = "#7c3aed";
      ctx.beginPath(); ctx.roundRect(ch.x - gap - 12, ch.y - 16, 12, 32, 4); ctx.fill();
      ctx.beginPath(); ctx.roundRect(ch.x + gap, ch.y - 16, 12, 32, 4); ctx.fill();
    }
    label(ctx, `K_ATP (Kir6.2/SUR1) ${Math.round(100 * this.katpOpen)}% open`, this.katp[1].x, this.katp[1].y - 30, { size: 11, color: "#d8b4fe" });
    // voltage-gated Ca2+ channels (bottom)
    for (const ch of this.vgcc) {
      const gap = 3 + 11 * ch.open;
      ctx.fillStyle = "#0891b2";
      ctx.beginPath(); ctx.roundRect(ch.x - gap - 12, ch.y - 16, 12, 32, 4); ctx.fill();
      ctx.beginPath(); ctx.roundRect(ch.x + gap, ch.y - 16, 12, 32, 4); ctx.fill();
    }
    label(ctx, "Voltage-gated Ca²⁺ channels", this.vgcc[1].x, this.vgcc[1].y + 32, { size: 11, color: "#67e8f9" });

    // granules
    for (const g of this.granules) {
      const r = g.state === "fusing" ? g.r * (1 - 0.6 * g.f) : g.r;
      ctx.beginPath(); ctx.arc(g.x, g.y, r + 3, 0, TAU); ctx.fillStyle = "rgba(147, 197, 253, 0.18)"; ctx.fill();
      ctx.beginPath(); ctx.arc(g.x, g.y, r, 0, TAU); ctx.strokeStyle = "rgba(147, 197, 253, 0.85)"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.beginPath(); ctx.arc(g.x, g.y, r * 0.55, 0, TAU); ctx.fillStyle = "#1e3a8a"; ctx.fill(); // dense insulin-zinc core
    }
    for (const f of this.fusions) {
      ctx.beginPath(); ctx.arc(f.x, f.y, 6 + 26 * f.t, 0, TAU);
      ctx.strokeStyle = `rgba(96, 165, 250, ${0.8 - f.t})`; ctx.lineWidth = 2; ctx.stroke();
    }
    label(ctx, "Insulin granules (dense core: insulin–Zn²⁺ hexamers)", c.x + c.w - 190, c.y + c.h - 30, { size: 11, color: "#bfdbfe" });

    // particles
    for (const p of this.particles) {
      if (p.kind === "gluOut" || p.kind === "glucose") hexagon(ctx, p.x, p.y, 6, C.glucose);
      else if (p.kind === "g6p") hexagon(ctx, p.x, p.y, 6, C.g6p);
      else if (p.kind === "atp") { ctx.fillStyle = C.atp; ctx.font = "700 10px system-ui"; ctx.textAlign = "center"; ctx.fillText("ATP", p.x, p.y); }
      else if (p.kind === "k") { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, TAU); ctx.fillStyle = C.k; ctx.fill(); }
      else if (p.kind === "ca") { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, TAU); ctx.fillStyle = C.ca; ctx.fill(); }
      else if (p.kind === "ins") {
        ctx.fillStyle = C.insulin;
        for (let k = 0; k < 6; k++) { const a = (k / 6) * TAU; ctx.beginPath(); ctx.arc(p.x + 5 * Math.cos(a), p.y + 5 * Math.sin(a), 2.4, 0, TAU); ctx.fill(); }
      }
    }
    label(ctx, "K⁺ efflux", this.katp[2].x + 70, this.katp[2].y - 30, { size: 11, color: C.k });

    // oscilloscope (schematic membrane potential)
    const sx = c.x + 40, sy = c.y + c.h - 140, sw = 300, sh = 90;
    ctx.fillStyle = "rgba(2, 6, 14, 0.8)"; ctx.strokeStyle = "rgba(148,163,184,.35)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(sx, sy, sw, sh, 8); ctx.fill(); ctx.stroke();
    const vy = (v) => sy + sh - ((v + 75) / 65) * sh;
    ctx.strokeStyle = "rgba(148,163,184,.25)"; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(sx, vy(-50)); ctx.lineTo(sx + sw, vy(-50)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#f472b6"; ctx.lineWidth = 1.6; ctx.beginPath();
    this.vmTrace.forEach((v, i) => ctx.lineTo(sx + (i / (this.vmTrace.length - 1)) * sw, vy(v)));
    ctx.stroke();
    label(ctx, "Membrane potential (schematic)", sx + sw / 2, sy - 10, { size: 10, color: "#f9a8d4" });
  }

  readout(d) {
    return [
      `Glucose ${d.glucose.toFixed(0)} mg/dl (${(d.glucose / 18.016).toFixed(1)} mM)`,
      `Secretion ×${d.secretion.toFixed(2)} basal`,
      `K_ATP open ${Math.round(100 * this.katpOpen)}%`,
      `Vm ≈ ${this.vm.toFixed(0)} mV (schematic)`,
    ];
  }

  legend() {
    return `<b>Stimulus–secretion coupling in the β-cell</b> (Rorsman &amp; Ashcroft, Physiol Rev 2018)
      <ol>
        <li>Glucose enters by facilitated diffusion: mainly <b>GLUT1</b> in human β-cells, <b>GLUT2</b> in rodents.</li>
        <li><b>Glucokinase</b>, the β-cell glucose sensor, phosphorylates glucose to glucose-6-phosphate; glycolysis and mitochondrial oxidation raise the <b>ATP/ADP ratio</b>.</li>
        <li>ATP closes <b>K<sub>ATP</sub> channels</b> (Kir6.2/SUR1). Less K⁺ efflux depolarises the membrane.</li>
        <li>Depolarisation opens <b>voltage-gated Ca²⁺ channels</b>. Ca²⁺ influx triggers exocytosis of insulin granules.</li>
        <li>Granules release insulin (stored as Zn²⁺ hexamers) and C-peptide into the portal circulation.</li>
      </ol>
      Exocytosis rate is set by the model's secretion rate. The membrane-potential trace and K<sub>ATP</sub> dose-response are schematic.`;
  }
}

// ============================================================ muscle cell

class MuscleCellScene {
  constructor() {
    this.name = "muscle";
    this.memY = 200;
    this.receptors = [150, 330, 510, 690, 870, 1050].map((x) => ({ x, bound: 0, p: 0 }));
    this.pathX = 560;
    this.slots = [];
    for (let x = 80; x < W - 60; x += 70) if (!this.receptors.some((r) => Math.abs(r.x - x) < 40)) this.slots.push({ x, glut4: 0 });
    // IRS-1 and PI3K in the cytosol; PIP3 is a membrane lipid, so it sits at the
    // inner leaflet; Akt is recruited there and then acts on AS160 in the cytosol.
    this.path = [
      { id: "IRS-1", x: 560, y: 280 }, { id: "PI3K", x: 560, y: 360 }, { id: "PIP₃", x: 720, y: 228 },
      { id: "Akt", x: 720, y: 330 }, { id: "AS160", x: 720, y: 430 },
    ];
    this.signals = []; this.vesicles = []; this.particles = []; this.glycogen = 0;
    for (let i = 0; i < 14; i++) this.vesicles.push(this._newVesicle());
    this.spawn = { ins: 0, glu: 0, rbc: 0 };
    this.akt = 0; this.t = 0; this.uptake = 0; this.uptakeCount = 0;
  }

  _newVesicle() { return { x: rand(80, W - 80), y: rand(470, 600), state: "storage", target: null, vx: 0, vy: 0 }; }

  update(dt, d) {
    this.t += dt;
    const occupancy = d.insulin / (d.insulin + 200); // schematic receptor occupancy
    this.spawn.ins += dt * (1 + d.insulin / 25);
    while (this.spawn.ins > 1) { this.spawn.ins -= 1; this.particles.push({ kind: "ins", x: -10, y: rand(30, 120), vx: rand(80, 140), vy: 0 }); }
    this.spawn.glu += dt * (3 + d.glucose / 15);
    while (this.spawn.glu > 1) { this.spawn.glu -= 1; this.particles.push({ kind: "glu", x: -10, y: rand(25, 125), vx: rand(70, 130), vy: 0 }); }
    this.spawn.rbc += dt * 1.2;
    while (this.spawn.rbc > 1) { this.spawn.rbc -= 1; this.particles.push({ kind: "rbc", x: -40, y: rand(40, 110), vx: rand(60, 90), vy: 0, rot: rand(0, TAU) }); }

    // Insulin binding to receptors
    for (const r of this.receptors) {
      // Two-state binding: on-rate ~ occupancy, off-rate ~ (1 - occupancy), so the
      // long-run bound fraction equals the occupancy.
      if (!r.bound && Math.random() < dt * 1.2 * occupancy) r.bound = 1;
      else if (r.bound && Math.random() < dt * 1.2 * (1 - occupancy)) r.bound = 0;
      r.p = lerp(r.p, r.bound, dt * 2); // autophosphorylation follows binding
      if (r.p > 0.5 && Math.random() < dt * 1.5 * r.p) {
        // signal propagates only with probability (1 - ir): impaired post-receptor signalling
        this.signals.push({ x: r.x, y: this.memY + 20, stage: 0, ok: Math.random() > d.ir, t: 0 });
      }
    }
    // Signal cascade
    for (const s of this.signals) {
      s.t += dt;
      if (s.t > 0.35) {
        s.t = 0; s.stage += 1;
        if (!s.ok && s.stage >= 1) s.dead = true;
        if (s.stage >= this.path.length) {
          s.dead = true;
          this.akt = Math.min(1, this.akt + 0.12);
          const v = this.vesicles.find((q) => q.state === "storage");
          const slot = this.slots.filter((q) => q.glut4 < 1).sort(() => Math.random() - 0.5)[0];
          if (v && slot) { v.state = "moving"; v.target = slot; }
        }
      }
    }
    this.signals = this.signals.filter((s) => !s.dead);
    this.akt *= Math.exp(-dt * 0.4);

    // GLUT4 vesicles: translocation, fusion, slow endocytosis (recycling)
    for (const v of this.vesicles) {
      if (v.state === "storage") {
        v.vx += rand(-15, 15) * dt; v.vy += rand(-15, 15) * dt; v.vx *= 0.97; v.vy *= 0.97;
        v.x = clamp(v.x + v.vx * dt, 60, W - 60); v.y = clamp(v.y + v.vy * dt, 440, 620);
      } else if (v.state === "moving") {
        const dx = v.target.x - v.x, dy = this.memY + 14 - v.y, dist = Math.hypot(dx, dy);
        if (dist < 4) { v.state = "fused"; v.target.glut4 = 1; v.dead = true; }
        else { v.x += (dx / dist) * 150 * dt; v.y += (dy / dist) * 150 * dt; }
      }
    }
    const fused = this.vesicles.filter((v) => v.dead).length;
    this.vesicles = this.vesicles.filter((v) => !v.dead);
    for (let i = 0; i < fused; i++) { const nv = this._newVesicle(); nv.y = 640; this.vesicles.push(nv); }
    for (const s of this.slots) if (s.glut4 > 0 && Math.random() < dt * 0.05) s.glut4 = 0; // endocytosis

    // Glucose uptake through surface GLUT4
    const surface = this.slots.filter((s) => s.glut4 > 0);
    for (const s of surface) {
      if (Math.random() < dt * (0.6 + d.glucose / 150)) {
        this.particles.push({ kind: "gluIn", x: s.x, y: this.memY - 30, vx: rand(-15, 15), vy: rand(60, 100) });
        this.uptakeCount += 1;
      }
    }
    this.uptake = lerp(this.uptake, surface.length, dt);

    for (const p of this.particles) {
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.kind === "rbc") p.rot += dt;
      if (p.kind === "gluIn" && p.y > 560) { p.dead = true; this.glycogen = Math.min(1, this.glycogen + 0.004); }
      if (p.x > W + 50) p.dead = true;
    }
    this.particles = this.particles.filter((p) => !p.dead);
    this.glycogen *= Math.exp(-dt * 0.01);
  }

  draw(ctx, d) {
    ctx.fillStyle = "#050910"; ctx.fillRect(0, 0, W, H);
    // capillary
    const g = ctx.createLinearGradient(0, 0, 0, 150);
    g.addColorStop(0, "rgba(127, 29, 29, 0.35)"); g.addColorStop(1, "rgba(127, 29, 29, 0.12)");
    ctx.fillStyle = g; ctx.fillRect(0, 10, W, 135);
    ctx.strokeStyle = "rgba(248,113,113,.4)"; ctx.setLineDash([6, 8]);
    ctx.beginPath(); ctx.moveTo(0, 10); ctx.lineTo(W, 10); ctx.moveTo(0, 145); ctx.lineTo(W, 145); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, "Capillary: insulin & glucose", 120, 160, { size: 11, color: C.muted });
    // cytoplasm
    const cg = ctx.createLinearGradient(0, this.memY, 0, H);
    cg.addColorStop(0, "rgba(30, 58, 95, 0.55)"); cg.addColorStop(1, "rgba(15, 30, 55, 0.55)");
    ctx.fillStyle = cg; ctx.fillRect(0, this.memY, W, H - this.memY);
    // sarcomere-like striations (skeletal muscle)
    ctx.strokeStyle = "rgba(148, 163, 184, 0.06)"; ctx.lineWidth = 10;
    for (let x = 30; x < W; x += 60) { ctx.beginPath(); ctx.moveTo(x, this.memY + 20); ctx.lineTo(x, H); ctx.stroke(); }
    // membrane bilayer
    for (const [off, a] of [[-5, 0.9], [5, 0.9]]) {
      ctx.strokeStyle = `rgba(94, 234, 212, ${a})`; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(0, this.memY + off); ctx.lineTo(W, this.memY + off); ctx.stroke();
    }
    label(ctx, "Skeletal muscle cell", W - 110, H - 22, { size: 13 });

    // glycogen rosette
    const gx = 160, gy = 590, gr = 12 + 50 * this.glycogen;
    for (let k = 0; k < 18; k++) {
      const a = (k / 18) * TAU + this.t * 0.05;
      ctx.beginPath(); ctx.arc(gx + gr * 0.8 * Math.cos(a), gy + gr * 0.8 * Math.sin(a), 4 + gr * 0.12, 0, TAU);
      ctx.fillStyle = "rgba(250, 204, 21, 0.55)"; ctx.fill();
    }
    label(ctx, "Glycogen (via hexokinase → G6P)", gx, gy + gr + 18, { size: 11, color: "#fde68a" });

    // pathway column
    ctx.strokeStyle = "rgba(167, 139, 250, 0.35)"; ctx.lineWidth = 2; ctx.setLineDash([4, 5]);
    ctx.beginPath(); this.path.forEach((n) => ctx.lineTo(n.x, n.y)); ctx.stroke(); ctx.setLineDash([]);
    this.path.forEach((n, i) => {
      const act = this.signals.filter((s) => s.ok && s.stage === i).length;
      glow(ctx, n.x, n.y, 36, "rgb(167, 139, 250)", Math.min(0.7, 0.1 + act * 0.12));
      ctx.beginPath(); ctx.arc(n.x, n.y, 17, 0, TAU); ctx.fillStyle = "#4c1d95"; ctx.fill();
      ctx.strokeStyle = "#c4b5fd"; ctx.lineWidth = 1.5; ctx.stroke();
      label(ctx, n.id, n.x + 26, n.y, { size: 12, color: "#ddd6fe", align: "left" });
    });

    // receptors: alpha subunits outside, beta subunits spanning into the cell
    for (const r of this.receptors) {
      ctx.strokeStyle = "#818cf8"; ctx.lineWidth = 6; ctx.lineCap = "round";
      for (const s of [-1, 1]) {
        ctx.beginPath(); ctx.moveTo(r.x + 7 * s, this.memY + 26); ctx.lineTo(r.x + 7 * s, this.memY - 12);
        ctx.quadraticCurveTo(r.x + 18 * s, this.memY - 30, r.x + 12 * s, this.memY - 44); ctx.stroke();
      }
      if (r.bound > 0.3) { // bound insulin
        ctx.fillStyle = C.insulin; ctx.beginPath(); ctx.arc(r.x, this.memY - 40, 7, 0, TAU); ctx.fill();
      }
      if (r.p > 0.4) { // phosphotyrosines
        for (const s of [-1, 1]) { ctx.fillStyle = "#facc15"; ctx.beginPath(); ctx.arc(r.x + 16 * s, this.memY + 20, 6, 0, TAU); ctx.fill(); ctx.fillStyle = "#1c1917"; ctx.font = "700 8px system-ui"; ctx.textAlign = "center"; ctx.fillText("P", r.x + 16 * s, this.memY + 21); }
      }
    }
    label(ctx, "Insulin receptor (α₂β₂)", this.receptors[0].x, this.memY + 48, { size: 11, color: "#c7d2fe" });

    // signals travelling
    for (const s of this.signals) {
      const from = s.stage === 0 ? { x: s.x, y: s.y } : this.path[s.stage - 1];
      const to = this.path[Math.min(s.stage, this.path.length - 1)];
      const t = s.t / 0.35;
      const x = lerp(from.x, to.x, t), y = lerp(from.y, to.y, t);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, TAU); ctx.fillStyle = s.ok ? "#e9d5ff" : "rgba(248,113,113,.8)"; ctx.fill();
    }

    // GLUT4 slots in membrane
    for (const s of this.slots) {
      if (!s.glut4) continue;
      ctx.fillStyle = "#10b981";
      ctx.beginPath(); ctx.roundRect(s.x - 11, this.memY - 14, 8, 28, 3); ctx.fill();
      ctx.beginPath(); ctx.roundRect(s.x + 3, this.memY - 14, 8, 28, 3); ctx.fill();
    }
    // GLUT4 storage vesicles
    for (const v of this.vesicles) {
      ctx.beginPath(); ctx.arc(v.x, v.y, 13, 0, TAU); ctx.strokeStyle = "rgba(52, 211, 153, 0.9)"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = "rgba(16, 185, 129, 0.2)"; ctx.fill();
      ctx.fillStyle = "#10b981"; ctx.fillRect(v.x - 3, v.y - 9, 6, 6);
    }
    label(ctx, "GLUT4 storage vesicles", 1000, 640, { size: 11, color: "#6ee7b7" });

    for (const p of this.particles) {
      if (p.kind === "rbc") {
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot);
        ctx.beginPath(); ctx.ellipse(0, 0, 22, 14, 0, 0, TAU); ctx.fillStyle = "rgba(220, 38, 38, 0.55)"; ctx.fill();
        ctx.beginPath(); ctx.ellipse(0, 0, 10, 6, 0, 0, TAU); ctx.fillStyle = "rgba(127, 29, 29, 0.6)"; ctx.fill();
        ctx.restore();
      } else if (p.kind === "ins") { ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, TAU); ctx.fillStyle = C.insulin; ctx.fill(); }
      else hexagon(ctx, p.x, p.y, 6, p.kind === "gluIn" ? C.g6p : C.glucose);
    }
  }

  readout(d) {
    const surface = this.slots.filter((s) => s.glut4 > 0).length;
    return [
      `Insulin ${d.insulin.toFixed(0)} pmol/l`,
      `Glucose ${d.glucose.toFixed(0)} mg/dl`,
      `Surface GLUT4 ${surface}/${this.slots.length}`,
      `Signal loss (insulin resistance) ${Math.round(100 * d.ir)}%`,
    ];
  }

  legend() {
    return `<b>Insulin signalling to GLUT4 in skeletal muscle</b> (Saltiel &amp; Kahn, Nature 2001)
      <ol>
        <li>Insulin binds the <b>insulin receptor</b>, an α₂β₂ receptor tyrosine kinase, which autophosphorylates its β-subunits.</li>
        <li>The receptor phosphorylates <b>IRS-1</b>, which recruits and activates <b>PI3-kinase</b>, producing <b>PIP₃</b> in the membrane.</li>
        <li>PIP₃ recruits and activates <b>Akt</b>. Akt phosphorylates <b>AS160</b> (TBC1D4), a Rab-GAP, releasing its brake on GLUT4 vesicle traffic.</li>
        <li><b>GLUT4 storage vesicles</b> translocate to and fuse with the plasma membrane, increasing glucose uptake. GLUT4 is recycled by endocytosis.</li>
        <li>Glucose is phosphorylated by hexokinase and stored as <b>glycogen</b>.</li>
      </ol>
      The insulin-resistance slider removes a fraction of post-receptor signals (shown in red). Receptor occupancy is schematic.`;
  }
}

// ============================================================ central dogma

class DogmaScene {
  constructor() {
    this.name = "dogma";
    this.t = 0;
    this.steps = [
      { dur: 7, title: "1 · Transcription", text: "RNA polymerase II transcribes the INS gene (chromosome 11p15.5) into pre-mRNA, which is spliced and exported through a nuclear pore." },
      { dur: 7, title: "2 · Translation into the ER", text: "Ribosomes translate preproinsulin (110 aa). Its 24-aa signal peptide directs the chain into the endoplasmic reticulum, where signal peptidase removes it." },
      { dur: 7, title: "3 · Folding: proinsulin (86 aa)", text: "Proinsulin folds as B chain – C-peptide – A chain, with three disulfide bonds: A6–A11, A7–B7 and A20–B19." },
      { dur: 7, title: "4 · Processing in the granule", text: "Prohormone convertases PC1/3 (B–C junction) and PC2 (C–A junction) cut at dibasic sites; carboxypeptidase E trims the basic residues." },
      { dur: 7, title: "5 · Mature insulin + C-peptide", text: "Insulin (A 21 aa + B 30 aa) and C-peptide are stored in equimolar amounts; insulin crystallises as Zn²⁺ hexamers and is released by exocytosis." },
    ];
    this.total = this.steps.reduce((a, s) => a + s.dur, 0);
    // Cysteine positions computed from the sequences (1-based)
    const cys = (s) => [...s].map((c, i) => (c === "C" ? i + 1 : 0)).filter(Boolean);
    this.cysA = cys(INSULIN_A); // [6, 7, 11, 20]
    this.cysB = cys(INSULIN_B); // [7, 19]
  }

  update(dt) { this.t = (this.t + dt) % this.total; }

  next() {
    const { i } = this._step();
    this.t = this.steps.slice(0, (i + 1) % this.steps.length).reduce((a, s) => a + s.dur, 0) + 0.001;
  }

  _step() {
    let acc = 0;
    for (let i = 0; i < this.steps.length; i++) {
      if (this.t < acc + this.steps[i].dur) return { i, p: (this.t - acc) / this.steps[i].dur };
      acc += this.steps[i].dur;
    }
    return { i: 0, p: 0 };
  }

  _chain(ctx, seq, x0, y0, dx, color, upto = seq.length, labelEvery = 1, dy = 0) {
    const pts = [];
    for (let i = 0; i < Math.min(upto, seq.length); i++) {
      const x = x0 + i * dx, y = y0 + (dy ? dy * Math.sin(i / 2.2) : 0);
      pts.push({ x, y });
      ctx.beginPath(); ctx.arc(x, y, 9, 0, TAU); ctx.fillStyle = color; ctx.fill();
      if (i % labelEvery === 0) { ctx.fillStyle = "#0b1220"; ctx.font = "700 10px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(seq[i], x, y + 0.5); }
    }
    return pts;
  }

  draw(ctx) {
    ctx.fillStyle = "#050910"; ctx.fillRect(0, 0, W, H);
    const { i, p } = this._step();
    const step = this.steps[i];
    // progress bar
    this.steps.forEach((s, k) => {
      const x = 60 + k * 220;
      ctx.fillStyle = k < i ? "#22d3ee" : k === i ? "rgba(34,211,238,.5)" : "rgba(148,163,184,.2)";
      ctx.beginPath(); ctx.roundRect(x, 24, 200, 6, 3); ctx.fill();
      if (k === i) { ctx.fillStyle = "#22d3ee"; ctx.beginPath(); ctx.roundRect(x, 24, 200 * p, 6, 3); ctx.fill(); }
    });
    label(ctx, step.title, W / 2, 62, { size: 20, bg: false });
    ctx.font = "15px system-ui"; ctx.fillStyle = C.muted; ctx.textAlign = "center";
    wrap(ctx, step.text, W / 2, 96, 980, 21);

    if (i === 0) this._transcription(ctx, p);
    else if (i === 1) this._translation(ctx, p);
    else if (i === 2) this._folding(ctx, p);
    else if (i === 3) this._processing(ctx, p);
    else this._mature(ctx, p);
  }

  _transcription(ctx, p) {
    // nucleus
    ctx.beginPath(); ctx.ellipse(600, 400, 520, 210, 0, 0, TAU);
    ctx.fillStyle = "rgba(76, 29, 149, 0.22)"; ctx.fill(); ctx.strokeStyle = "rgba(167,139,250,.6)"; ctx.lineWidth = 2; ctx.stroke();
    label(ctx, "Nucleus", 180, 250, { size: 12, color: "#c4b5fd" });
    const polX = 180 + p * 760;
    for (let k = 0; k < 2; k++) {
      ctx.strokeStyle = k ? "#a78bfa" : "#22d3ee"; ctx.lineWidth = 4; ctx.beginPath();
      for (let x = 120; x <= 1080; x += 4) {
        const open = Math.max(0, 1 - Math.abs(x - polX) / 60);
        ctx.lineTo(x, 400 + (k ? 1 : -1) * (14 + 26 * open) * (0.6 + 0.4 * Math.cos(x / 18)));
      }
      ctx.stroke();
    }
    for (let x = 124; x <= 1080; x += 16) {
      const open = Math.max(0, 1 - Math.abs(x - polX) / 60);
      if (open > 0.2) continue;
      ctx.strokeStyle = ["#34d399", "#f87171", "#fbbf24", "#60a5fa"][Math.floor(x / 16) % 4]; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x, 390); ctx.lineTo(x, 410); ctx.stroke();
    }
    // mRNA
    ctx.strokeStyle = "#fb923c"; ctx.lineWidth = 4; ctx.beginPath();
    for (let x = 180; x <= polX; x += 4) ctx.lineTo(x, 350 - (polX - x) * 0.18 + 8 * Math.sin(x / 14));
    ctx.stroke();
    glow(ctx, polX, 395, 70, "rgb(34, 211, 238)", 0.35);
    ctx.beginPath(); ctx.ellipse(polX, 395, 42, 34, 0, 0, TAU); ctx.fillStyle = "rgba(14, 116, 144, .85)"; ctx.fill();
    label(ctx, "RNA Pol II", polX, 395, { size: 11, bg: false });
    label(ctx, "pre-mRNA", polX - 120, 300, { size: 11, color: "#fdba74" });
    label(ctx, "INS gene", 200, 440, { size: 11, color: C.muted });
  }

  _translation(ctx, p) {
    // ER membrane
    ctx.fillStyle = "rgba(34, 197, 94, 0.08)"; ctx.fillRect(0, 430, W, 250);
    ctx.strokeStyle = "rgba(74, 222, 128, .7)"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(0, 430); ctx.lineTo(W, 430); ctx.stroke();
    label(ctx, "ER lumen", 80, 470, { size: 12, color: "#86efac" });
    label(ctx, "Cytosol", 80, 180, { size: 12, color: C.muted });
    // mRNA
    ctx.strokeStyle = "#fb923c"; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(80, 360); ctx.lineTo(1120, 360); ctx.stroke();
    // ribosome
    const rx = 200 + p * 700;
    ctx.beginPath(); ctx.ellipse(rx, 340, 50, 30, 0, 0, TAU); ctx.fillStyle = "#475569"; ctx.fill();
    ctx.beginPath(); ctx.ellipse(rx, 385, 40, 22, 0, 0, TAU); ctx.fillStyle = "#64748b"; ctx.fill();
    label(ctx, "Ribosome", rx, 300, { size: 11 });
    // nascent chain through the translocon: 24-aa signal peptide (red) then proinsulin
    const n = Math.floor(p * 110);
    const sig = 24;
    for (let k = 0; k < Math.min(n, 60); k++) {
      const aa = n - 1 - k; // newest residue near the ribosome
      const x = rx + 10;
      const y = 430 + k * 12;
      if (y > H - 10) break;
      ctx.beginPath(); ctx.arc(x + 14 * Math.sin(k / 2), y, 6, 0, TAU);
      ctx.fillStyle = aa < sig ? "#ef4444" : "#93c5fd"; ctx.fill();
    }
    ctx.fillStyle = "#16a34a"; ctx.fillRect(rx - 14, 418, 8, 26); ctx.fillRect(rx + 6, 418, 8, 26);
    label(ctx, `${n} / 110 aa`, rx + 110, 470, { size: 12 });
    label(ctx, "signal peptide (24 aa)", rx + 130, 520, { size: 11, color: "#fca5a5" });
    label(ctx, "translocon", rx - 90, 430, { size: 10, color: "#86efac" });
  }

  _proinsulinLayout() {
    // Topologically correct schematic: B chain (N->C) on top; the connecting
    // region runs from B30 around the right side and back underneath to A1;
    // the A chain (N->C) runs left to right so that A7 sits under B7 and A20
    // near B19, making the interchain disulfides vertical.
    const x0 = 200, dx = 26;
    const B = [], A = [];
    for (let k = 0; k < 30; k++) B.push({ x: x0 + k * dx, y: 300 });
    for (let k = 0; k < 21; k++) A.push({ x: x0 + k * dx, y: 470 });
    // dense path for the connecting peptide, then resample 35 evenly spaced beads
    const path = [];
    const xr = x0 + 29 * dx + 30, cy = 440, r = 125;
    for (let a = -Math.PI / 2; a <= Math.PI / 2; a += 0.02) path.push({ x: xr + 40 + r * Math.cos(a), y: cy + r * Math.sin(a) });
    for (let x = xr + 40; x >= x0 - 60; x -= 4) path.push({ x, y: cy + r });
    for (let y = cy + r; y >= 470; y -= 4) path.push({ x: x0 - 60 + 0 * y, y });
    path.push({ x: x0 - 30, y: 470 });
    const Cp = resample(path, 35);
    return { A, B, Cp };
  }

  _drawBeads(ctx, pts, seq, color) {
    pts.forEach((q, k) => {
      ctx.beginPath(); ctx.arc(q.x, q.y, 10, 0, TAU); ctx.fillStyle = color; ctx.fill();
      if (seq) { ctx.fillStyle = "#0b1220"; ctx.font = "700 10px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(seq[k], q.x, q.y + 0.5); }
    });
  }

  _disulfides(ctx, A, B, alpha = 1) {
    const [a6, a7, a11, a20] = this.cysA.map((i) => A[i - 1]);
    const [b7, b19] = this.cysB.map((i) => B[i - 1]);
    ctx.strokeStyle = `rgba(250, 204, 21, ${alpha})`; ctx.lineWidth = 3; ctx.setLineDash([5, 4]);
    for (const [u, v] of [[a7, b7], [a20, b19]]) { ctx.beginPath(); ctx.moveTo(u.x, u.y); ctx.lineTo(v.x, v.y); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(a6.x, a6.y + 10); ctx.quadraticCurveTo((a6.x + a11.x) / 2, a6.y + 50, a11.x, a11.y + 10); ctx.stroke();
    ctx.setLineDash([]);
    label(ctx, "A7–B7", (a7.x + b7.x) / 2 - 34, (a7.y + b7.y) / 2, { size: 11, color: "#fde047" });
    label(ctx, "A20–B19", (a20.x + b19.x) / 2 + 42, (a20.y + b19.y) / 2, { size: 11, color: "#fde047" });
    label(ctx, "A6–A11", (a6.x + a11.x) / 2, a6.y + 56, { size: 11, color: "#fde047" });
  }

  _folding(ctx, p) {
    const { A, B, Cp } = this._proinsulinLayout();
    this._drawBeads(ctx, B, INSULIN_B, "#60a5fa");
    this._drawBeads(ctx, Cp, null, "#94a3b8");
    this._drawBeads(ctx, A, INSULIN_A, "#34d399");
    if (p > 0.35) this._disulfides(ctx, A, B, Math.min(1, (p - 0.35) * 3));
    label(ctx, "B chain (30 aa)", B[0].x + 60, 262, { size: 12, color: "#93c5fd" });
    label(ctx, "A chain (21 aa)", A[20].x + 90, 470, { size: 12, color: "#6ee7b7" });
    label(ctx, "C-peptide region (35 aa incl. RR…KR)", Cp[20].x, Cp[20].y + 28, { size: 12, color: "#cbd5e1" });
  }

  _processing(ctx, p) {
    ctx.beginPath(); ctx.arc(600, 410, 290, 0, TAU);
    ctx.fillStyle = "rgba(59, 130, 246, 0.07)"; ctx.fill(); ctx.strokeStyle = "rgba(147,197,253,.5)"; ctx.lineWidth = 2; ctx.stroke();
    label(ctx, "Immature secretory granule", 600, 140, { size: 12, color: "#bfdbfe" });
    const { A, B, Cp } = this._proinsulinLayout();
    const sep = Math.max(0, (p - 0.4) / 0.6);
    const Cs = Cp.map((q) => ({ x: q.x, y: q.y + 70 * sep }));
    this._drawBeads(ctx, B, INSULIN_B, "#60a5fa");
    this._drawBeads(ctx, A, INSULIN_A, "#34d399");
    // dibasic residues at both ends of the connecting region (removed by CPE)
    this._drawBeads(ctx, Cs.slice(2, 33), null, "#94a3b8");
    if (sep < 0.8) { this._drawBeads(ctx, Cs.slice(0, 2), "RR", "#f472b6"); this._drawBeads(ctx, Cs.slice(33), "KR", "#f472b6"); }
    this._disulfides(ctx, A, B);
    if (p > 0.15) {
      for (const [q, name] of [[Cp[1], "PC1/3"], [Cp[33], "PC2"]]) {
        glow(ctx, q.x, q.y, 40, "rgb(244, 114, 182)", 0.5);
        label(ctx, `✂ ${name}`, q.x + 40, q.y, { size: 12, color: "#f9a8d4" });
      }
    }
    if (sep > 0.6) label(ctx, "C-peptide (31 aa)", Cs[20].x, Cs[20].y + 28, { size: 12, color: "#cbd5e1" });
  }

  _mature(ctx, p) {
    // insulin hexamers with two Zn2+ ions in a dense core
    ctx.beginPath(); ctx.arc(600, 410, 290, 0, TAU);
    ctx.fillStyle = "rgba(59, 130, 246, 0.07)"; ctx.fill(); ctx.strokeStyle = "rgba(147,197,253,.5)"; ctx.lineWidth = 2; ctx.stroke();
    const core = 150;
    ctx.beginPath(); ctx.arc(600, 410, core, 0, TAU); ctx.fillStyle = "rgba(30, 58, 138, 0.55)"; ctx.fill();
    const hexes = [];
    for (let ring = 0; ring < 3; ring++) {
      const count = ring === 0 ? 1 : ring * 6;
      for (let k = 0; k < count; k++) {
        const a = (k / count) * TAU + ring * 0.3 + p * 0.3;
        hexes.push({ x: 600 + ring * 52 * Math.cos(a), y: 410 + ring * 52 * Math.sin(a) });
      }
    }
    for (const h of hexes) {
      for (let k = 0; k < 6; k++) { const a = (k / 6) * TAU; ctx.beginPath(); ctx.arc(h.x + 13 * Math.cos(a), h.y + 13 * Math.sin(a), 7, 0, TAU); ctx.fillStyle = "#60a5fa"; ctx.fill(); }
      for (const s of [-1, 1]) { ctx.beginPath(); ctx.arc(h.x, h.y + 3 * s, 3, 0, TAU); ctx.fillStyle = "#e5e7eb"; ctx.fill(); }
    }
    label(ctx, "Insulin hexamers (2 Zn²⁺ each)", 600, 410 + core + 22, { size: 12, color: "#bfdbfe" });
    // C-peptide in the halo
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * TAU - p;
      const r = 215 + 10 * Math.sin(k + p * 5);
      ctx.strokeStyle = "#94a3b8"; ctx.lineWidth = 4; ctx.lineCap = "round"; ctx.beginPath();
      const x = 600 + r * Math.cos(a), y = 410 + r * Math.sin(a);
      ctx.moveTo(x - 10, y); ctx.quadraticCurveTo(x, y - 10, x + 10, y); ctx.stroke();
    }
    label(ctx, "C-peptide (equimolar with insulin)", 600, 150, { size: 12, color: "#cbd5e1" });
  }

  readout() {
    const { i } = this._step();
    return [`Step ${i + 1} / ${this.steps.length}`, `B chain ${INSULIN_B.length} aa · A chain ${INSULIN_A.length} aa`];
  }

  legend() {
    return `<b>From the INS gene to secreted insulin</b>
      <ol>
        <li>Preproinsulin, 110 aa = 24-aa signal peptide + B chain (30) + connecting region (35: C-peptide 31 with Arg-Arg and Lys-Arg cleavage sites) + A chain (21).</li>
        <li>Three disulfide bonds: A6–A11 (intrachain), A7–B7 and A20–B19 (interchain). Cysteine positions are computed from the sequences shown.</li>
        <li>C-peptide is secreted in equimolar amounts with insulin, so it is used clinically to measure endogenous insulin secretion.</li>
      </ol>
      Chain sequences: human insulin (UniProt P01308). The DNA and ribosome drawings are schematic. <b>Click the animation to skip to the next step.</b>`;
  }
}

function resample(path, n) {
  const d = [0];
  for (let i = 1; i < path.length; i++) d.push(d[i - 1] + Math.hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y));
  const out = [];
  for (let k = 0; k < n; k++) {
    const target = (k / (n - 1)) * d[d.length - 1];
    let j = 1;
    while (j < d.length - 1 && d[j] < target) j++;
    const t = (target - d[j - 1]) / (d[j] - d[j - 1] || 1);
    out.push({ x: lerp(path[j - 1].x, path[j].x, t), y: lerp(path[j - 1].y, path[j].y, t) });
  }
  return out;
}

function wrap(ctx, text, x, y, maxW, lh) {
  const words = text.split(" ");
  let line = "";
  for (const w of words) {
    const test = line ? line + " " + w : w;
    if (ctx.measureText(test).width > maxW && line) { ctx.fillText(line, x, y); line = w; y += lh; }
    else line = test;
  }
  ctx.fillText(line, x, y);
}

// ============================================================ theatre

export class CellTheatre {
  constructor(canvas, onReadout) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.scenes = { beta: new BetaCellScene(), muscle: new MuscleCellScene(), dogma: new DogmaScene() };
    this.scene = this.scenes.beta;
    this.drive = { glucose: 90, secretion: 1, insulin: 25, ir: 0 };
    this.onReadout = onReadout;
    this.last = performance.now();
    this.running = true;
    this._readoutT = 0;
    const loop = (now) => {
      const dt = Math.min((now - this.last) / 1000, 0.05);
      this.last = now;
      if (this.running && !document.hidden) {
        this.scene.update(dt, this.drive);
        this.scene.draw(this.ctx, this.drive);
        this._readoutT += dt;
        if (this._readoutT > 0.2 && this.onReadout) { this._readoutT = 0; this.onReadout(this.scene.readout(this.drive)); }
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  setScene(name) {
    this.scene = this.scenes[name];
    if (!this.scene.warmed && this.scene.name !== "dogma") {
      for (let k = 0; k < 150; k++) this.scene.update(0.04, this.drive); // pre-populate ~6 s
      this.scene.warmed = true;
    }
    return this.scene.legend();
  }
  setDrive(d) { Object.assign(this.drive, d); }
  click() { if (this.scene.next) this.scene.next(); }
}
