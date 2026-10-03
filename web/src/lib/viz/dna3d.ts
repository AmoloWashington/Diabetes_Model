// Molecular-style B-DNA model built base pair by base pair from a sequence.
// Units: nm. Geometry: 10.5 bp per turn, 0.34 nm rise, ~2 nm diameter, right-handed.
// Phosphates, deoxyriboses, base slabs (purines longer than pyrimidines) and
// hydrogen bonds (A–T: 2, G–C: 3). The angular offset between the strands
// (which produces the major and minor grooves) is approximate.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const BP_PER_TURN = 10.5;
const RISE = 0.34;
const R_PHOSPHATE = 1.0;
const R_SUGAR = 0.78;
const GROOVE_OFFSET = (144 / 180) * Math.PI; // narrow side = minor groove
const PAIR: Record<string, string> = { A: "T", T: "A", G: "C", C: "G", N: "N" };
const HBONDS: Record<string, number> = { A: 2, T: 2, G: 3, C: 3, N: 0 };
const BASE_COLOR: Record<string, number> = { A: 0x3e9b74, T: 0xc4553f, G: 0xd29a3c, C: 0x5a7fb0, N: 0x9a9ea9 };
const PURINE = new Set(["A", "G"]);

interface Rung { i: number; halves: THREE.Mesh[]; hb: THREE.Line[]; from: THREE.Vector3[]; to: THREE.Vector3[] }
interface Label { obj: CSS2DObject; minD: number; maxD: number }

export class DNAHelix {
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(38, 1, 0.05, 500);
  private renderer: THREE.WebGLRenderer;
  private labelRenderer: CSS2DRenderer | null = null;
  private controls: OrbitControls | null = null;
  private group = new THREE.Group();
  private rungs: Rung[] = [];
  private labels: Label[] = [];
  private bubble: { pos: number; width: number } | null = null;
  private clock = new THREE.Clock();
  private visible = true;
  private resizeObs: ResizeObserver;
  private interObs: IntersectionObserver;
  private autoRotate: boolean;
  private showLabels: boolean;
  private maxBp: number;
  private height = 0;
  private titleLabel?: CSS2DObject;
  seq = "";

  constructor(private container: HTMLElement, opts: { interactive?: boolean; autoRotate?: boolean; maxBp?: number; labels?: boolean } = {}) {
    const { interactive = true, autoRotate = true, maxBp = 120, labels = interactive } = opts;
    this.autoRotate = autoRotate;
    this.showLabels = labels;
    this.maxBp = maxBp;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);
    if (labels) {
      this.labelRenderer = new CSS2DRenderer();
      Object.assign(this.labelRenderer.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
      container.style.position = container.style.position || "relative";
      container.appendChild(this.labelRenderer.domElement);
    }
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(5, 8, 10); this.scene.add(key);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.35));
    this.scene.add(this.group);
    if (interactive) {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.enablePan = false;
      this.controls.minDistance = 1.5;
      this.controls.maxDistance = 80;
    }
    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.interObs = new IntersectionObserver(([e]) => (this.visible = e.isIntersecting));
    this.interObs.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  private addLabel(text: string, at: THREE.Vector3, minD: number, maxD: number, level: number, parent: THREE.Object3D = this.group) {
    if (!this.showLabels) return;
    const el = document.createElement("div");
    el.className = "cell-label";
    el.dataset.level = String(level);
    el.textContent = text;
    const obj = new CSS2DObject(el);
    obj.position.copy(at);
    parent.add(obj);
    this.labels.push({ obj, minD, maxD });
  }

  setSequence(seq: string) {
    const s = (seq || "").toUpperCase().replace(/U/g, "T").replace(/[^ACGTN]/g, "").slice(0, this.maxBp) || "ATGC";
    this.seq = s;
    this.group.clear();
    this.titleLabel?.removeFromParent();
    this.rungs = [];
    this.labels = [];
    const n = s.length;
    const height = (n - 1) * RISE;
    const y0 = -height / 2;

    const phosMat = new THREE.MeshPhysicalMaterial({ color: 0xe8893a, roughness: 0.35, clearcoat: 0.6 });
    const sugarMat = [
      new THREE.MeshPhysicalMaterial({ color: 0xcfc6b4, roughness: 0.45, clearcoat: 0.3 }),
      new THREE.MeshPhysicalMaterial({ color: 0xb9b0a0, roughness: 0.45, clearcoat: 0.3 }),
    ];
    const tubeMat = [
      new THREE.MeshPhysicalMaterial({ color: 0x2f7d5b, roughness: 0.4, transparent: true, opacity: 0.55 }),
      new THREE.MeshPhysicalMaterial({ color: 0xa8692a, roughness: 0.4, transparent: true, opacity: 0.55 }),
    ];
    const hbMat = new THREE.LineDashedMaterial({ color: 0x7c8a96, dashSize: 0.04, gapSize: 0.03 });
    const phosGeo = new THREE.SphereGeometry(0.15, 20, 14);
    const sugarGeo = new THREE.DodecahedronGeometry(0.12, 0); // five-membered ring evoked by a pentagonal solid
    const backbone: THREE.Vector3[][] = [[], []];

    for (let i = 0; i < n; i++) {
      const th = (2 * Math.PI * i) / BP_PER_TURN;
      const y = y0 + i * RISE;
      const angles = [th, th + GROOVE_OFFSET];
      const sug: THREE.Vector3[] = [];
      for (let k = 0; k < 2; k++) {
        const a = angles[k];
        // phosphate sits half a step offset along the strand
        const pa = a - (k === 0 ? 1 : -1) * Math.PI / BP_PER_TURN;
        const py = y - (k === 0 ? 1 : -1) * RISE / 2;
        const P = new THREE.Vector3(R_PHOSPHATE * Math.cos(pa), py, R_PHOSPHATE * Math.sin(pa));
        const S = new THREE.Vector3(R_SUGAR * Math.cos(a), y, R_SUGAR * Math.sin(a));
        const pm = new THREE.Mesh(phosGeo, phosMat); pm.position.copy(P); this.group.add(pm);
        const sm = new THREE.Mesh(sugarGeo, sugarMat[k]); sm.position.copy(S); this.group.add(sm);
        backbone[k].push(P, S);
        sug.push(S);
      }
      // Base slabs: purine spans ~55% of the C1'-C1' gap, pyrimidine ~45%
      const b = s[i], c = PAIR[b];
      const span = sug[1].clone().sub(sug[0]);
      const split = PURINE.has(b) ? 0.55 : 0.45;
      const mid = sug[0].clone().add(span.clone().multiplyScalar(split));
      const halves: THREE.Mesh[] = [];
      const from: THREE.Vector3[] = [], to: THREE.Vector3[] = [];
      for (const [start, end, base] of [[sug[0], mid, b], [sug[1], mid, c]] as const) {
        const len = start.distanceTo(end) * 0.96;
        const g = new THREE.BoxGeometry(0.22, 0.07, len);
        const mesh = new THREE.Mesh(g, new THREE.MeshPhysicalMaterial({ color: BASE_COLOR[base], roughness: 0.35, clearcoat: 0.5 }));
        mesh.position.copy(start.clone().lerp(end, 0.5));
        mesh.lookAt(end);
        this.group.add(mesh);
        halves.push(mesh);
        from.push(start.clone());
        to.push(end.clone());
      }
      // Hydrogen bonds across the pair
      const hb: THREE.Line[] = [];
      const perp = new THREE.Vector3(0, 1, 0).cross(span).normalize();
      const nb = HBONDS[b];
      for (let h = 0; h < nb; h++) {
        const off = perp.clone().multiplyScalar((h - (nb - 1) / 2) * 0.07);
        const p1 = mid.clone().add(span.clone().normalize().multiplyScalar(-0.07)).add(off);
        const p2 = mid.clone().add(span.clone().normalize().multiplyScalar(0.07)).add(off);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([p1, p2]), hbMat);
        line.computeLineDistances();
        this.group.add(line);
        hb.push(line);
      }
      this.rungs.push({ i, halves, hb, from, to });
      if (i === Math.floor(n / 2) || (n > 30 && i === Math.floor(n / 4))) {
        this.addLabel(`${b}–${c} pair · ${HBONDS[b]} H-bonds`, mid.clone().add(new THREE.Vector3(0, 0.12, 0)), 1.5, 6, 2);
      }
    }
    for (let k = 0; k < 2; k++) {
      if (backbone[k].length < 2) continue;
      const curve = new THREE.CatmullRomCurve3(backbone[k]);
      this.group.add(new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(16, n * 8), 0.05, 8, false), tubeMat[k]));
    }

    // Labels (semantic zoom)
    const midI = Math.floor(n / 2);
    const thM = (2 * Math.PI * midI) / BP_PER_TURN, yM = y0 + midI * RISE;
    const ring = (a: number, r: number, y: number) => new THREE.Vector3(r * Math.cos(a), y, r * Math.sin(a));
    // The title label lives in the scene (not the spinning group) so it stays above the helix centre.
    this.addLabel("B-DNA double helix", new THREE.Vector3(0, 2.2, 0), 12, 999, 0, this.scene);
    this.titleLabel = this.labels[this.labels.length - 1]?.obj;
    this.addLabel("Minor groove", ring(thM + GROOVE_OFFSET / 2, 1.25, yM + 0.6), 3, 18, 1);
    this.addLabel("Major groove", ring(thM + GROOVE_OFFSET + (2 * Math.PI - GROOVE_OFFSET) / 2, 1.3, yM - 0.8), 3, 18, 1);
    this.addLabel("Sugar–phosphate backbone", ring(thM - 0.5, 1.35, yM + 3.2), 3, 16, 1);
    this.addLabel("5′ end of input strand", ring(0, 1.3, y0 - 0.4), 3, 30, 1);
    this.addLabel("3′ end of input strand", ring((2 * Math.PI * (n - 1)) / BP_PER_TURN, 1.3, y0 + height + 0.4), 3, 30, 1);
    const pI = Math.max(0, midI - 2);
    const thP = (2 * Math.PI * pI) / BP_PER_TURN - Math.PI / BP_PER_TURN;
    this.addLabel("Phosphate", ring(thP, 1.25, y0 + pI * RISE - RISE / 2), 1.5, 5, 2);
    this.addLabel("Deoxyribose", ring((2 * Math.PI * pI) / BP_PER_TURN, 0.55, y0 + pI * RISE + 0.12), 1.5, 5, 2);

    // "ZYX" order: the spin (y) is applied about the helix's own axis, then the axis is tilted (z),
    // so auto-rotation turns the molecule about its long axis instead of swinging it toward the camera.
    this.group.rotation.order = "ZYX";
    this.group.rotation.z = Math.PI / 2.15;
    this.height = height;
    const dist = this.fitDistance();
    this.camera.position.set(0, 0, dist);
    this.camera.lookAt(0, 0, 0);
    if (this.controls) { this.controls.target.set(0, 0, 0); this.controls.update(); }
  }

  /** Camera distance at which the whole helix (plus a margin) fits the viewport width and height. */
  private fitDistance() {
    const vHalf = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const hHalf = Math.atan(Math.tan(vHalf) * this.camera.aspect);
    const half = this.height / 2 + 1.5;
    return Math.min(78, Math.max(8, half / Math.tan(hHalf), 2.5 / Math.tan(vHalf)));
  }

  /** A transcription bubble (~12–14 bp in vivo) travelling along the helix. */
  startBubble() { this.bubble = { pos: 0, width: 13 }; }

  private resize() {
    const w = this.container.clientWidth || 400, h = this.container.clientHeight || 400;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.labelRenderer?.setSize(w, h);
  }

  private frame() {
    if (!this.visible) return;
    const dt = Math.min(this.clock.getDelta(), 0.05);
    if (this.autoRotate) this.group.rotation.y += dt * 0.3;
    if (this.bubble) {
      this.bubble.pos += dt * 6;
      if (this.bubble.pos > this.rungs.length + this.bubble.width) this.bubble = null;
    }
    for (const r of this.rungs) {
      const d = this.bubble ? Math.abs(r.i - this.bubble.pos) : Infinity;
      const open = this.bubble ? Math.max(0, 1 - d / (this.bubble.width / 2)) : 0;
      r.halves.forEach((m, k) => {
        const end = r.from[k].clone().lerp(r.to[k], 1 - 0.6 * open);
        m.position.copy(r.from[k].clone().lerp(end, 0.5));
        m.scale.z = 1 - 0.6 * open;
      });
      r.hb.forEach((l) => (l.visible = open < 0.2));
    }
    this.controls?.update();
    if (this.labelRenderer) {
      const d = this.camera.position.distanceTo(this.controls?.target ?? new THREE.Vector3());
      this.labels.forEach((l) => {
        const show = d >= l.minD && d <= l.maxD;
        l.obj.visible = show;
        (l.obj.element as HTMLElement).style.opacity = show ? "1" : "0";
      });
    }
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer?.render(this.scene, this.camera);
  }

  get canvas() { return this.renderer.domElement; }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.resizeObs.disconnect();
    this.interObs.disconnect();
    this.controls?.dispose();
    this.scene.traverse((o) => (o as THREE.Mesh).geometry?.dispose?.());
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer?.domElement.remove();
  }
}
