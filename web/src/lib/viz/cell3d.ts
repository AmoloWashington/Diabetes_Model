// 3D model of a pancreatic β-cell (three.js) with semantic-zoom labels,
// three imaging modes and model-driven insulin granule exocytosis.
//
// Units: 1 scene unit ≈ 1 µm. Organelle sizes and counts are schematic
// (a real β-cell holds ~10,000 granules; a subset is drawn).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DObject, CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { PARTS } from "./cellParts";

export type ImagingMode = "illustrated" | "fluorescence" | "em";
type Part = keyof typeof PARTS;

const CELL_R = 6.0;
const NUC_C = new THREE.Vector3(-0.8, 0.4, 0.2);
const NUC_R = 2.3;
const CENTROSOME = new THREE.Vector3(1.4, 1.6, 0.4);

// Deterministic pseudo-random numbers so the cell looks the same every visit.
function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

/** Smooth irregular displacement (sum of low-frequency sinusoids). */
function bumpy(v: THREE.Vector3, amp: number, f = 1) {
  const n = Math.sin(v.x * 0.9 * f + 1.3) * Math.cos(v.y * 1.1 * f + 0.4) + 0.6 * Math.sin(v.z * 1.4 * f + v.x * 0.5 * f)
    + 0.35 * Math.sin((v.x + v.y + v.z) * 2.3 * f);
  return 1 + amp * n;
}

function blob(radius: number, amp: number, detail = 6, f = 1) {
  const g = new THREE.IcosahedronGeometry(radius, detail);
  const p = g.attributes.position as THREE.BufferAttribute;
  const v = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const k = bumpy(v.clone().normalize().multiplyScalar(3), amp, f);
    p.setXYZ(i, v.x * k, v.y * k, v.z * k);
  }
  g.computeVertexNormals();
  return g;
}

function randomInBall(r: () => number, R: number, avoid?: { c: THREE.Vector3; r: number }) {
  for (;;) {
    const v = new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).multiplyScalar(R);
    if (v.length() > R) continue;
    if (avoid && v.distanceTo(avoid.c) < avoid.r) continue;
    return v;
  }
}

interface Label { obj: CSS2DObject; part: Part; minD: number; maxD: number; anchor: THREE.Vector3 }
interface Granule { pos: THREE.Vector3; target: THREE.Vector3 | null; state: "reserve" | "transport" | "fusing"; t: number; scale: number }

export class CellExplorer {
  private renderer: THREE.WebGLRenderer;
  private labelRenderer: CSS2DRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private emPass: ShaderPass;
  private clock = new THREE.Clock();
  private labels: Label[] = [];
  private pickables: THREE.Object3D[] = [];
  private mats: Record<ImagingMode, Record<string, THREE.Material>>;
  private meshesByPart = new Map<string, THREE.Mesh[]>();
  private granules: Granule[] = [];
  private granuleCore!: THREE.InstancedMesh;
  private granuleHalo!: THREE.InstancedMesh;
  private fusionFlashes: { mesh: THREE.Mesh; t: number }[] = [];
  private secretionFold = 1;
  private exoAccumulator = 0;
  private cutPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
  private slabPlanes = [new THREE.Plane(new THREE.Vector3(0, 0, -1), 0.45), new THREE.Plane(new THREE.Vector3(0, 0, 1), 0.45)];
  private cutaway = true;
  private mode: ImagingMode = "illustrated";
  private raycaster = new THREE.Raycaster();
  private disposed = false;
  private resizeObs: ResizeObserver;
  onSelect?: (part: Part | null) => void;
  onZoom?: (distance: number) => void;
  showLabels = true;

  constructor(private container: HTMLElement) {
    const w = container.clientWidth || 800, h = container.clientHeight || 600;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.localClippingEnabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(w, h);
    Object.assign(this.labelRenderer.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
    container.appendChild(this.labelRenderer.domElement);

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.05, 200);
    this.camera.position.set(5, 3.5, 23);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 40;
    this.controls.zoomSpeed = 0.9;

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    const key = new THREE.DirectionalLight(0xfff4e6, 1.6); key.position.set(8, 10, 12); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xdfeeff, 0.6); fill.position.set(-10, -4, 6); this.scene.add(fill);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.25));

    this.mats = this.buildMaterials();
    this.build();

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.4, 0.35);
    this.composer.addPass(this.bloom);
    this.emPass = new ShaderPass({
      uniforms: { tDiffuse: { value: null }, time: { value: 0 }, enabled: { value: 0 } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: `uniform sampler2D tDiffuse; uniform float time; uniform float enabled; varying vec2 vUv;
        float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)) + time) * 43758.5453); }
        void main(){ vec4 c = texture2D(tDiffuse, vUv);
          if (enabled < 0.5) { gl_FragColor = c; return; }
          float g = dot(c.rgb, vec3(0.299, 0.587, 0.114));
          g = clamp((g - 0.5) * 1.25 + 0.5 + (h(vUv * 900.0) - 0.5) * 0.10, 0.0, 1.0);
          float vig = smoothstep(0.95, 0.35, distance(vUv, vec2(0.5)));
          gl_FragColor = vec4(vec3(g * mix(0.82, 1.0, vig)), 1.0); }`,
    });
    this.composer.addPass(this.emPass);
    this.composer.addPass(new OutputPass());

    this.renderer.domElement.addEventListener("pointerdown", this.onPointerDown);
    this.renderer.domElement.addEventListener("pointerup", this.onPointerUp);
    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.setMode("illustrated");
    this.renderer.setAnimationLoop(() => this.frame());
  }

  // ------------------------------------------------------------ materials

  private buildMaterials(): Record<ImagingMode, Record<string, THREE.Material>> {
    const P = (o: THREE.MeshPhysicalMaterialParameters) => new THREE.MeshPhysicalMaterial(o);
    const illustrated: Record<string, THREE.Material> = {
      membrane: P({ color: 0xcfe3dc, roughness: 0.25, metalness: 0, transmission: 0.55, thickness: 0.6, transparent: true, opacity: 0.38, side: THREE.DoubleSide, iridescence: 0.25, depthWrite: false }),
      envelope: P({ color: 0x8c7bb8, roughness: 0.45, transparent: true, opacity: 0.55, side: THREE.DoubleSide, clearcoat: 0.4, depthWrite: false }),
      pore: P({ color: 0xb9a98a, roughness: 0.6 }),
      nucleolus: P({ color: 0x4a3a78, roughness: 0.6 }),
      chromatin: P({ color: 0x5a4a8f, roughness: 0.75 }),
      rer: P({ color: 0xd88fa6, roughness: 0.5, transparent: true, opacity: 0.8, side: THREE.DoubleSide }),
      ribosome: P({ color: 0x7a2f4a, roughness: 0.6 }),
      golgi: P({ color: 0xd9a441, roughness: 0.4, side: THREE.DoubleSide, clearcoat: 0.35 }),
      mito: P({ color: 0xc8644a, roughness: 0.45, transparent: true, opacity: 0.85, clearcoat: 0.3 }),
      cristae: P({ color: 0x8a3524, roughness: 0.6, side: THREE.DoubleSide }),
      denseCore: P({ color: 0x1f2c4a, roughness: 0.45, clearcoat: 0.3 }),
      halo: P({ color: 0xdfe6ee, roughness: 0.5, transparent: true, opacity: 0.22, depthWrite: false }),
      lysosome: P({ color: 0x8a9a5b, roughness: 0.55 }),
      microtubule: P({ color: 0x9fb7a8, roughness: 0.6, transparent: true, opacity: 0.75 }),
      centrosome: P({ color: 0x3e7d5f, roughness: 0.4 }),
      cilium: P({ color: 0xa9cfc0, roughness: 0.3, transparent: true, opacity: 0.85 }),
      flash: new THREE.MeshBasicMaterial({ color: 0x9fd0ff, transparent: true, opacity: 0.6, depthWrite: false }),
    };
    // Fluorescence: emissive "stains" on black, styled after confocal imaging
    const F = (color: number, opacity = 1) => new THREE.MeshBasicMaterial({
      color, transparent: opacity < 1, opacity, blending: opacity < 1 ? THREE.AdditiveBlending : THREE.NormalBlending, depthWrite: opacity >= 1,
    });
    const fluorescence: Record<string, THREE.Material> = {
      membrane: F(0x40204f, 0.12), envelope: F(0x1d3bb0, 0.25), pore: F(0x6f86e8), nucleolus: F(0x14207a), chromatin: F(0x3557e8),
      rer: F(0x9c2c7a, 0.35), ribosome: F(0xc23d8e), golgi: F(0xd88a12), mito: F(0xd8352a), cristae: F(0xe2603c),
      denseCore: F(0x2fd468), halo: F(0x0f4a24, 0.25), lysosome: F(0x13a6c9), microtubule: F(0x6f6fd8, 0.3), centrosome: F(0xf0f0f0),
      cilium: F(0x9a9ae8, 0.5), flash: F(0x2fd468, 0.7),
    };
    // Electron micrograph: greyscale densities (dense = dark) on a light section
    const E = (g: number, opacity = 1) => new THREE.MeshLambertMaterial({ color: new THREE.Color(g, g, g), transparent: opacity < 1, opacity, side: THREE.DoubleSide, depthWrite: opacity >= 1 });
    const em: Record<string, THREE.Material> = {
      membrane: E(0.35, 0.9), envelope: E(0.3), pore: E(0.55), nucleolus: E(0.12), chromatin: E(0.32), rer: E(0.28), ribosome: E(0.1),
      golgi: E(0.3), mito: E(0.45), cristae: E(0.2), denseCore: E(0.06), halo: E(0.92, 0.85), lysosome: E(0.25), microtubule: E(0.45, 0.6),
      centrosome: E(0.15), cilium: E(0.35), flash: E(0.95, 0.5),
    };
    return { illustrated, fluorescence, em };
  }

  private register(part: string, mesh: THREE.Mesh, matKey = part) {
    mesh.userData.part = part;
    mesh.userData.matKey = matKey;
    const list = this.meshesByPart.get(matKey) ?? [];
    list.push(mesh);
    this.meshesByPart.set(matKey, list);
    this.pickables.push(mesh);
    this.scene.add(mesh);
    return mesh;
  }

  private label(part: Part, anchor: THREE.Vector3, minD: number, maxD: number) {
    const el = document.createElement("div");
    el.className = "cell-label";
    el.textContent = PARTS[part].name;
    el.dataset.level = String(PARTS[part].level);
    const obj = new CSS2DObject(el);
    obj.position.copy(anchor);
    this.scene.add(obj);
    this.labels.push({ obj, part, minD, maxD, anchor: anchor.clone() });
  }

  // ------------------------------------------------------------ geometry

  private build() {
    const r = rng(7);

    // Plasma membrane
    const mem = this.register("membrane", new THREE.Mesh(blob(CELL_R, 0.045, 7, 0.8), this.mats.illustrated.membrane));
    mem.renderOrder = 10;
    this.label("cell", new THREE.Vector3(0, CELL_R * 1.08, 0), 20, 999);
    this.label("membrane", new THREE.Vector3(CELL_R * 0.72, CELL_R * 0.62, 0.4), 6, 24);

    // Nucleus: double envelope, pores, nucleolus, chromatin
    const env = this.register("envelope", new THREE.Mesh(blob(NUC_R, 0.03, 6, 1.4), this.mats.illustrated.envelope), "envelope");
    env.position.copy(NUC_C);
    env.userData.part = "nucleus";
    const inner = this.register("envelope", new THREE.Mesh(blob(NUC_R * 0.965, 0.03, 5, 1.4), this.mats.illustrated.envelope), "envelope");
    inner.position.copy(NUC_C);
    const poreGeo = new THREE.TorusGeometry(0.06, 0.018, 8, 16);
    const nPores = 140;
    const pores = new THREE.InstancedMesh(poreGeo, this.mats.illustrated.pore, nPores);
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(1, 1, 1);
    for (let i = 0; i < nPores; i++) {
      // Fibonacci sphere distribution
      const y = 1 - (2 * (i + 0.5)) / nPores, rad = Math.sqrt(1 - y * y), th = i * Math.PI * (3 - Math.sqrt(5));
      const n = new THREE.Vector3(Math.cos(th) * rad, y, Math.sin(th) * rad);
      const p = n.clone().multiplyScalar(NUC_R * 1.004 * bumpy(n.clone().multiplyScalar(3), 0.03, 1.4)).add(NUC_C);
      q.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
      pores.setMatrixAt(i, m4.compose(p, q, sc));
    }
    this.register("pore", pores as unknown as THREE.Mesh);
    const nucleolus = this.register("nucleolus", new THREE.Mesh(blob(0.62, 0.1, 6, 2.5), this.mats.illustrated.nucleolus));
    nucleolus.position.copy(NUC_C).add(new THREE.Vector3(0.5, 0.35, 0.4));
    // Chromatin as fibres (random walks confined to the nucleus), denser near the envelope (heterochromatin)
    for (let f = 0; f < 120; f++) {
      const dir = new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).normalize();
      let p = dir.clone().multiplyScalar(NUC_R * (f % 3 === 0 ? 0.55 * r() : 0.75 + 0.15 * r())).add(NUC_C);
      const pts = [p.clone()];
      let heading = new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).normalize();
      for (let k = 0; k < 26; k++) {
        heading.add(new THREE.Vector3(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(0.9)).normalize();
        const next = p.clone().add(heading.clone().multiplyScalar(0.16));
        if (next.distanceTo(NUC_C) > NUC_R * 0.9) { heading.reflect(next.clone().sub(NUC_C).normalize()); continue; }
        p = next;
        pts.push(p.clone());
      }
      if (pts.length < 4) continue;
      const fib = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), pts.length * 4, 0.016 + r() * 0.012, 6, false), this.mats.illustrated.chromatin);
      this.register("chromatin", fib);
    }
    this.label("nucleus", NUC_C.clone().add(new THREE.Vector3(-1.2, NUC_R + 0.4, 0)), 6, 22);
    this.label("envelope", NUC_C.clone().add(new THREE.Vector3(-NUC_R * 0.75, -NUC_R * 0.62, 0.6)), 1.2, 7.5);
    this.label("pore", NUC_C.clone().add(new THREE.Vector3(0, 0, NUC_R + 0.05)), 1.2, 5.5);
    this.label("nucleolus", nucleolus.position.clone().add(new THREE.Vector3(0, 0.75, 0)), 1.2, 7.5);
    this.label("chromatin", NUC_C.clone().add(new THREE.Vector3(-0.9, -0.3, 1.2)), 1.2, 6.5);

    // Rough ER: curved sheets wrapped around the nucleus, studded with ribosomes
    const rerGroup: THREE.Vector3[] = [];
    for (let k = 0; k < 5; k++) {
      const rad = NUC_R + 0.45 + k * 0.32;
      const g = new THREE.SphereGeometry(rad, 48, 24, -0.6 + k * 0.15, 1.9 - k * 0.12, 0.9 + k * 0.05, 1.2);
      const sheet = this.register("rer", new THREE.Mesh(g, this.mats.illustrated.rer));
      sheet.position.copy(NUC_C);
      sheet.rotation.set(0.2 * k, -0.3 + 0.12 * k, 0.1);
      sheet.updateMatrixWorld();
      // sample ribosome positions on the sheet
      const pos = g.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < pos.count; i += 3) {
        if (r() < 0.55) rerGroup.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(sheet.matrixWorld));
      }
    }
    const ribos = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.035, 1), this.mats.illustrated.ribosome, rerGroup.length);
    rerGroup.forEach((p, i) => ribos.setMatrixAt(i, m4.compose(p, q.identity(), sc)));
    this.register("ribosome", ribos as unknown as THREE.Mesh);
    const rerAnchor = NUC_C.clone().add(new THREE.Vector3(NUC_R + 1.1, 0.4, -0.8));
    this.label("rer", rerAnchor, 3, 18);
    this.label("ribosome", rerAnchor.clone().add(new THREE.Vector3(0.2, -0.5, 0.6)), 1.2, 5);

    // Golgi: stack of curved cisternae with budding vesicles
    const golgiC = new THREE.Vector3(2.0, 0.9, -0.4);
    const golgi = new THREE.Group();
    for (let k = 0; k < 6; k++) {
      const g = new THREE.TorusGeometry(1.15 - k * 0.05, 0.1, 12, 56, Math.PI * 0.8);
      g.scale(1, 1, 0.38);
      const c = new THREE.Mesh(g, this.mats.illustrated.golgi);
      c.position.set(0, 0, k * 0.15);
      c.rotation.z = Math.PI * 0.58;
      golgi.add(c);
    }
    golgi.position.copy(golgiC);
    golgi.lookAt(NUC_C);
    golgi.updateMatrixWorld(true);
    golgi.children.forEach((c) => {
      const mesh = (c as THREE.Mesh).clone();
      mesh.applyMatrix4(golgi.matrixWorld);
      this.register("golgi", mesh);
    });
    for (let i = 0; i < 26; i++) {
      const v = new THREE.Mesh(new THREE.SphereGeometry(0.06 + r() * 0.04, 12, 8), this.mats.illustrated.golgi);
      v.position.copy(golgiC).add(new THREE.Vector3(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(2.4));
      this.register("golgi", v);
    }
    const toNuc = NUC_C.clone().sub(golgiC).normalize();
    this.label("golgi", golgiC.clone().add(new THREE.Vector3(0.3, 1.25, 0)), 3, 18);
    this.label("cisGolgi", golgiC.clone().add(toNuc.clone().multiplyScalar(0.55)).add(new THREE.Vector3(0, -0.9, 0)), 1.2, 6);
    this.label("transGolgi", golgiC.clone().add(toNuc.clone().multiplyScalar(-0.7)).add(new THREE.Vector3(0, -0.8, 0)), 1.2, 6);

    // Mitochondria: capsules with internal cristae plates
    const mitoAnchors: THREE.Vector3[] = [];
    for (let i = 0; i < 16; i++) {
      const p = randomInBall(r, CELL_R * 0.78, { c: NUC_C, r: NUC_R + 1.2 });
      const len = 0.9 + r() * 0.9;
      const grp = new THREE.Group();
      grp.add(new THREE.Mesh(new THREE.CapsuleGeometry(0.28, len, 8, 20), this.mats.illustrated.mito));
      for (let k = 0; k < 7; k++) {
        const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.02, 20, 1, false, 0, Math.PI * (1.2 + (k % 2) * 0.3)), this.mats.illustrated.cristae);
        plate.position.y = -len / 2 + (k + 0.5) * (len / 7);
        plate.rotation.y = k * 1.3;
        grp.add(plate);
      }
      grp.position.copy(p);
      grp.rotation.set(r() * Math.PI, r() * Math.PI, r() * Math.PI);
      grp.updateMatrixWorld(true);
      grp.children.forEach((c, k) => {
        const mesh = (c as THREE.Mesh).clone();
        mesh.applyMatrix4(grp.matrixWorld);
        this.register(k === 0 ? "mito" : "cristae", mesh, k === 0 ? "mito" : "cristae");
      });
      mitoAnchors.push(p);
    }
    this.label("mito", mitoAnchors[0].clone().add(new THREE.Vector3(0, 0.7, 0)), 3, 18);
    this.label("cristae", mitoAnchors[0].clone().add(new THREE.Vector3(0.4, -0.3, 0.3)), 1.2, 4.5);

    // Lysosomes
    for (let i = 0; i < 8; i++) {
      const l = this.register("lysosome", new THREE.Mesh(blob(0.22 + r() * 0.12, 0.12, 5, 3), this.mats.illustrated.lysosome));
      l.position.copy(randomInBall(r, CELL_R * 0.75, { c: NUC_C, r: NUC_R + 0.8 }));
      if (i === 0) this.label("lysosome", l.position.clone().add(new THREE.Vector3(0, 0.45, 0)), 2, 12);
    }

    // Centrosome: two orthogonal centrioles
    for (let k = 0; k < 2; k++) {
      const c = this.register("centrosome", new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 0.42, 9), this.mats.illustrated.centrosome));
      c.position.copy(CENTROSOME);
      c.rotation.set(k ? Math.PI / 2 : 0, 0, 0);
    }
    this.label("centrosome", CENTROSOME.clone().add(new THREE.Vector3(0, 0.45, 0)), 1.2, 6);

    // Microtubules radiating from the centrosome to the cortex
    const mtTargets: THREE.Vector3[] = [];
    for (let i = 0; i < 46; i++) {
      const dir = new THREE.Vector3(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1).normalize();
      const end = dir.clone().multiplyScalar(CELL_R * 0.93);
      const mid = CENTROSOME.clone().lerp(end, 0.5).add(new THREE.Vector3(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(1.2));
      const curve = new THREE.CatmullRomCurve3([CENTROSOME.clone(), mid, end]);
      if (curve.getPoints(20).some((pt) => pt.distanceTo(NUC_C) < NUC_R + 0.15)) continue;
      this.register("microtubule", new THREE.Mesh(new THREE.TubeGeometry(curve, 40, 0.018, 5, false), this.mats.illustrated.microtubule));
      mtTargets.push(end);
    }
    this.label("microtubule", CENTROSOME.clone().lerp(mtTargets[0] ?? new THREE.Vector3(5, 0, 0), 0.6), 2, 10);

    // Primary cilium protruding from the membrane near the centrosome
    const ciliumDir = CENTROSOME.clone().normalize();
    const base = ciliumDir.clone().multiplyScalar(CELL_R * 0.98);
    const tip = ciliumDir.clone().multiplyScalar(CELL_R + 2.6).add(new THREE.Vector3(0.6, 0.2, 0));
    const cil = new THREE.CatmullRomCurve3([base, base.clone().lerp(tip, 0.5).add(new THREE.Vector3(0.3, 0.2, 0)), tip]);
    this.register("cilium", new THREE.Mesh(new THREE.TubeGeometry(cil, 40, 0.09, 10, false), this.mats.illustrated.cilium));
    this.label("cilium", tip.clone().add(new THREE.Vector3(0.2, 0.3, 0)), 4, 22);

    // Insulin granules (instanced: dense core + halo)
    const N = 420;
    this.granuleCore = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.085, 2), this.mats.illustrated.denseCore, N);
    this.granuleHalo = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.135, 2), this.mats.illustrated.halo, N);
    for (let i = 0; i < N; i++) {
      let p: THREE.Vector3;
      do { p = randomInBall(r, CELL_R * 0.9, { c: NUC_C, r: NUC_R + 0.55 }); } while (p.length() < 1.2);
      this.granules.push({ pos: p, target: null, state: "reserve", t: 0, scale: 0.85 + r() * 0.3 });
    }
    this.granuleCore.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.granuleHalo.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.register("denseCore", this.granuleCore as unknown as THREE.Mesh);
    this.register("halo", this.granuleHalo as unknown as THREE.Mesh);
    this.granuleCore.userData.part = "granule";
    this.granuleHalo.userData.part = "granule";
    this.mtTargets = mtTargets;
    this.updateGranuleMatrices();
    const ga = this.granules[5].pos;
    this.label("granule", ga.clone().add(new THREE.Vector3(0, 0.45, 0)), 3, 16);
    this.label("denseCore", ga.clone().add(new THREE.Vector3(0.25, 0.05, 0.2)), 1.2, 3.2);
    this.label("halo", ga.clone().add(new THREE.Vector3(-0.3, -0.25, 0.1)), 1.2, 3.2);
  }

  private mtTargets: THREE.Vector3[] = [];

  private updateGranuleMatrices() {
    const m = new THREE.Matrix4(), q = new THREE.Quaternion();
    this.granules.forEach((g, i) => {
      const s = g.state === "fusing" ? g.scale * Math.max(0.05, 1 - g.t) : g.scale;
      const v = new THREE.Vector3(s, s, s);
      this.granuleCore.setMatrixAt(i, m.compose(g.pos, q, v));
      this.granuleHalo.setMatrixAt(i, m.compose(g.pos, q, v));
    });
    this.granuleCore.instanceMatrix.needsUpdate = true;
    this.granuleHalo.instanceMatrix.needsUpdate = true;
  }

  // ------------------------------------------------------------ modes

  setMode(mode: ImagingMode) {
    this.mode = mode;
    const mats = this.mats[mode];
    this.meshesByPart.forEach((meshes, key) => meshes.forEach((m) => { m.material = mats[key] ?? mats.membrane; }));
    this.fusionFlashes.forEach((f) => { f.mesh.material = mats.flash; });
    this.scene.background = new THREE.Color(mode === "fluorescence" ? 0x020306 : mode === "em" ? 0xd9d8d2 : 0xf3f1ea);
    this.bloom.enabled = mode === "fluorescence";
    this.scene.fog = mode === "illustrated" ? new THREE.Fog(0xf3f1ea, 24, 48) : null;
    this.emPass.uniforms.enabled.value = mode === "em" ? 1 : 0;
    this.applyClipping();
  }

  setCutaway(on: boolean) { this.cutaway = on; this.applyClipping(); }

  private applyClipping() {
    const all = Object.values(this.mats).flatMap((m) => Object.values(m));
    all.forEach((m) => { m.clippingPlanes = []; m.needsUpdate = true; });
    if (this.mode === "em") {
      // thin section through the cell, as in transmission electron microscopy
      Object.values(this.mats.em).forEach((m) => { m.clippingPlanes = this.slabPlanes; });
    } else if (this.cutaway) {
      [this.mats[this.mode].membrane, this.mats[this.mode].envelope].forEach((m) => { m.clippingPlanes = [this.cutPlane]; });
    }
  }

  setGlucose(glucoseMgDl: number) {
    // Static secretion of the Dalla Man model with published normal values: S_po = S_b + β(G − G_b)
    const Gb = 91.76, beta = 0.11, Sb = 1.5434;
    this.secretionFold = Math.max(0, (Sb + beta * (glucoseMgDl - Gb)) / Sb);
  }

  get secretion() { return this.secretionFold; }

  resetView() {
    this.camera.position.set(5, 3.5, 23);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  focus(part: Part) {
    const l = this.labels.find((x) => x.part === part);
    if (!l) return;
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.controls.target.copy(l.anchor);
    this.camera.position.copy(l.anchor.clone().add(dir.multiplyScalar(Math.max(2.2, (l.minD + l.maxD) / 4))));
    this.controls.update();
  }

  // ------------------------------------------------------------ interaction

  private down = { x: 0, y: 0 };
  private onPointerDown = (e: PointerEvent) => { this.down = { x: e.clientX, y: e.clientY }; };
  private onPointerUp = (e: PointerEvent) => {
    if (Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) > 4) return; // drag, not click
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables, false).filter((h) => {
      const planes = (h.object as THREE.Mesh).material && ((h.object as THREE.Mesh).material as THREE.Material).clippingPlanes;
      return !planes || planes.every((p) => p.distanceToPoint(h.point) >= 0);
    });
    const order = ["denseCore", "halo", "granule", "pore", "nucleolus", "cristae", "ribosome", "centrosome", "chromatin"];
    const hit = hits.find((h) => h.object.userData.part !== "membrane") ?? hits[0];
    let part = (hit?.object.userData.part as Part | undefined) ?? null;
    if (part && hit && this.camera.position.distanceTo(hit.point) < 4) {
      const fine = hits.find((h) => order.includes(h.object.userData.matKey) && h.distance - hit.distance < 0.3);
      if (fine) part = fine.object.userData.matKey as Part;
    }
    this.onSelect?.(part);
  };

  // ------------------------------------------------------------ loop

  private resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
    this.composer.setSize(w, h);
  }

  private frame() {
    if (this.disposed) return;
    const dt = Math.min(this.clock.getDelta(), 0.05);
    this.controls.update();
    this.stepGranules(dt);
    // Semantic zoom: a label is shown when the camera is within its distance band
    const camPos = this.camera.position;
    this.labels.forEach((l) => {
      const d = camPos.distanceTo(l.anchor);
      const visible = this.showLabels && d >= l.minD && d <= l.maxD;
      l.obj.visible = visible;
      (l.obj.element as HTMLElement).style.opacity = visible ? "1" : "0";
    });
    this.onZoom?.(camPos.distanceTo(this.controls.target));
    this.emPass.uniforms.time.value = this.clock.elapsedTime;
    this.composer.render();
    this.labelRenderer.render(this.scene, this.camera);
  }

  private stepGranules(dt: number) {
    // Exocytosis rate proportional to the model's secretion fold change (schematic scale)
    this.exoAccumulator += dt * 0.6 * this.secretionFold;
    while (this.exoAccumulator > 1 && this.mtTargets.length) {
      this.exoAccumulator -= 1;
      const g = this.granules.find((x) => x.state === "reserve" && x.pos.length() > 2.5);
      if (g) {
        g.state = "transport";
        // nearest microtubule end on the cortex
        g.target = this.mtTargets.reduce((a, b) => (a.distanceTo(g.pos) < b.distanceTo(g.pos) ? a : b)).clone().multiplyScalar(0.97);
      }
    }
    for (const g of this.granules) {
      if (g.state === "reserve") {
        g.pos.add(new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(dt * 0.12));
        if (g.pos.length() > CELL_R * 0.9) g.pos.multiplyScalar(0.995);
        if (g.pos.distanceTo(NUC_C) < NUC_R + 0.4) g.pos.add(g.pos.clone().sub(NUC_C).normalize().multiplyScalar(dt * 0.3));
      } else if (g.state === "transport" && g.target) {
        const d = g.target.clone().sub(g.pos);
        if (d.length() < 0.05) { g.state = "fusing"; g.t = 0; this.flash(g.target); }
        else g.pos.add(d.normalize().multiplyScalar(Math.min(d.length(), dt * 1.4)));
      } else if (g.state === "fusing") {
        g.t += dt * 1.5;
        if (g.t >= 1) {
          // replenished from the trans-Golgi network
          g.state = "reserve"; g.t = 0; g.target = null;
          g.pos.set(2.0, 0.9, -0.4).add(new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5));
        }
      }
    }
    this.updateGranuleMatrices();
    this.fusionFlashes = this.fusionFlashes.filter((f) => {
      f.t += dt;
      const s = 0.2 + f.t * 0.9;
      f.mesh.scale.setScalar(s);
      (f.mesh.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.6 - f.t * 0.7);
      if (f.t > 0.85) { this.scene.remove(f.mesh); f.mesh.geometry.dispose(); return false; }
      return true;
    });
  }

  private flash(at: THREE.Vector3) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.35, 16, 12), (this.mats[this.mode].flash as THREE.Material).clone());
    m.position.copy(at);
    this.scene.add(m);
    this.fusionFlashes.push({ mesh: m, t: 0 });
  }

  get canvas() { return this.renderer.domElement; }

  dispose() {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.resizeObs.disconnect();
    this.renderer.domElement.removeEventListener("pointerdown", this.onPointerDown);
    this.renderer.domElement.removeEventListener("pointerup", this.onPointerUp);
    this.controls.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose?.();
    });
    Object.values(this.mats).forEach((set) => Object.values(set).forEach((m) => m.dispose()));
    this.composer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
  }
}
