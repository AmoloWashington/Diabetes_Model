// 3D B-DNA double helix built base-pair by base-pair from a sequence.
// Geometry (schematic, in nm): 10.5 bp per turn, 0.34 nm rise, radius 1.0 nm.
// The angular offset between the two backbones creates the major and minor
// grooves; its value here is approximate and chosen for visual clarity.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BP_PER_TURN = 10.5;
const RISE = 0.34;
const RADIUS = 1.0;
const GROOVE_OFFSET = (144 / 180) * Math.PI;
const COLORS = { A: 0x3e9b74, T: 0xc4553f, G: 0xd29a3c, C: 0x5a7fb0, N: 0x9a9ea9 };
const PAIR = { A: "T", T: "A", G: "C", C: "G", N: "N" };

export class DNAHelix {
  constructor(container, { interactive = true, autoRotate = true, maxBp = 120, dustColor = 0x7a8a82 } = {}) {
    this.container = container;
    this.dustColor = dustColor;
    this.maxBp = maxBp;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 500);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(5, 8, 10); this.scene.add(key);
    const rim = new THREE.PointLight(0x9fd8bf, 25, 60); rim.position.set(-8, -4, 6); this.scene.add(rim);
    const rim2 = new THREE.PointLight(0xf0d3a0, 25, 60); rim2.position.set(8, 6, -6); this.scene.add(rim2);

    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.bubble = null; // transcription bubble animation state
    this.autoRotate = autoRotate;

    if (interactive) {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.enablePan = false;
    }
    this._particles();
    this._resize = () => this.resize();
    new ResizeObserver(this._resize).observe(container);
    this.resize();
    this.visible = true;
    new IntersectionObserver(([e]) => (this.visible = e.isIntersecting)).observe(container);
    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  _particles() {
    const n = 400, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[3 * i] = (Math.random() - 0.5) * 40;
      pos[3 * i + 1] = (Math.random() - 0.5) * 40;
      pos[3 * i + 2] = (Math.random() - 0.5) * 30 - 5;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.dust = new THREE.Points(g, new THREE.PointsMaterial({ color: this.dustColor, size: 0.06, transparent: true, opacity: 0.35 }));
    this.scene.add(this.dust);
  }

  setSequence(seq) {
    const s = (seq || "").toUpperCase().replace(/[^ACGTN]/g, "").slice(0, this.maxBp) || "ATGC";
    this.seq = s;
    this.group.clear();
    const n = s.length;
    const height = (n - 1) * RISE;
    const sphere = new THREE.SphereGeometry(0.16, 16, 12);
    const bbMat = [
      new THREE.MeshStandardMaterial({ color: 0xcbd5e1, metalness: 0.3, roughness: 0.35 }),
      new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.3, roughness: 0.35 }),
    ];
    const pts = [[], []];
    this.rungs = [];
    for (let i = 0; i < n; i++) {
      const theta = (2 * Math.PI * i) / BP_PER_TURN;
      const y = i * RISE - height / 2;
      const a = new THREE.Vector3(RADIUS * Math.cos(theta), y, RADIUS * Math.sin(theta));
      const b = new THREE.Vector3(RADIUS * Math.cos(theta + GROOVE_OFFSET), y, RADIUS * Math.sin(theta + GROOVE_OFFSET));
      pts[0].push(a); pts[1].push(b);
      for (const [k, p] of [[0, a], [1, b]]) {
        const m = new THREE.Mesh(sphere, bbMat[k]);
        m.position.copy(p);
        this.group.add(m);
      }
      // Base pair: two half-rungs coloured by base, meeting at the helix interior
      const base = s[i], comp = PAIR[base];
      const mid = a.clone().add(b).multiplyScalar(0.5);
      const rung = new THREE.Group();
      for (const [from, color] of [[a, COLORS[base]], [b, COLORS[comp]]]) {
        const len = from.distanceTo(mid);
        const geo = new THREE.CylinderGeometry(0.075, 0.075, len * 0.94, 10);
        const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25, roughness: 0.4 }));
        mesh.position.copy(from.clone().add(mid).multiplyScalar(0.5));
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), mid.clone().sub(from).normalize());
        mesh.userData = { from: from.clone(), mid: mid.clone() };
        rung.add(mesh);
      }
      rung.userData = { i, a: a.clone(), b: b.clone() };
      this.group.add(rung);
      this.rungs.push(rung);
    }
    // Smooth backbone tubes (sugar-phosphate)
    for (let k = 0; k < 2; k++) {
      if (pts[k].length < 2) continue;
      const curve = new THREE.CatmullRomCurve3(pts[k]);
      const tube = new THREE.TubeGeometry(curve, Math.max(8, n * 6), 0.07, 8, false);
      this.group.add(new THREE.Mesh(tube, new THREE.MeshStandardMaterial({
        color: k ? 0xc98f35 : 0x2f7d5b, emissive: k ? 0xc98f35 : 0x2f7d5b, emissiveIntensity: 0.25, roughness: 0.35,
      })));
    }
    this.group.rotation.z = Math.PI / 2.4;
    const dist = Math.max(8, height * 1.05 + 4);
    this.camera.position.set(0, 0, dist);
    this.camera.lookAt(0, 0, 0);
    if (this.controls) { this.controls.target.set(0, 0, 0); this.controls.update(); }
  }

  // A transcription bubble (~12-14 bp in vivo) travelling along the helix.
  startBubble() {
    this.bubble = { pos: 0, width: 13 };
  }

  resize() {
    const w = this.container.clientWidth || 400, h = this.container.clientHeight || 400;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
  }

  frame() {
    if (!this.visible) return;
    const dt = Math.min(this.clock.getDelta(), 0.05);
    if (this.autoRotate) this.group.rotation.y += dt * 0.35;
    this.dust.rotation.y += dt * 0.02;
    if (this.bubble && this.rungs) {
      this.bubble.pos += dt * 6;
      const n = this.rungs.length;
      if (this.bubble.pos > n + this.bubble.width) this.bubble = null;
      for (const r of this.rungs) {
        const d = this.bubble ? Math.abs(r.userData.i - this.bubble.pos) : Infinity;
        const open = this.bubble ? Math.max(0, 1 - d / (this.bubble.width / 2)) : 0;
        r.children.forEach((m) => {
          const { from, mid } = m.userData;
          const target = from.clone().lerp(mid, 0.5 * (1 - open));
          m.position.copy(target);
          m.scale.y = 1 - 0.55 * open;
        });
      }
    }
    if (this.controls) this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  get canvas() { return this.renderer.domElement; }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.controls?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
