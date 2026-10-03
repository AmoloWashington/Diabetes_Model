// Descriptions of every labelled structure in the 3D cell explorer.
// Statements are deliberately conservative; quantitative claims carry a source.

export interface PartInfo {
  name: string;
  level: 0 | 1 | 2; // 0 whole cell, 1 organelle, 2 sub-structure (revealed on zoom)
  summary: string;
  role: string;
  source?: string;
}

export const PARTS: Record<string, PartInfo> = {
  cell: {
    name: "Pancreatic β-cell", level: 0,
    summary: "Endocrine cell of the islets of Langerhans that senses blood glucose and secretes insulin.",
    role: "Couples glucose metabolism to insulin exocytosis (stimulus–secretion coupling).",
    source: "Rorsman & Ashcroft, Physiol Rev 2018",
  },
  membrane: {
    name: "Plasma membrane", level: 1,
    summary: "Phospholipid bilayer a few nanometres thick, studded with transporters and ion channels.",
    role: "Hosts glucose transporters (mainly GLUT1 in human, GLUT2 in rodent β-cells), K_ATP channels (Kir6.2/SUR1) and voltage-gated Ca²⁺ channels, the molecular machinery of glucose sensing.",
    source: "Rorsman & Ashcroft, Physiol Rev 2018",
  },
  nucleus: {
    name: "Nucleus", level: 1,
    summary: "Holds the genome, including the insulin gene (INS) on chromosome 11p15.5.",
    role: "Site of transcription of INS into pre-mRNA, which is spliced and exported to the cytoplasm.",
  },
  envelope: {
    name: "Nuclear envelope", level: 2,
    summary: "Double membrane enclosing the nucleus; the outer membrane is continuous with the endoplasmic reticulum.",
    role: "Separates transcription (nucleus) from translation (cytoplasm).",
  },
  pore: {
    name: "Nuclear pore complex", level: 2,
    summary: "Large protein assembly spanning both membranes of the nuclear envelope.",
    role: "Gatekeeper of nucleocytoplasmic transport, including export of mature mRNA.",
  },
  nucleolus: {
    name: "Nucleolus", level: 2,
    summary: "Dense, membrane-less region of the nucleus.",
    role: "Site of ribosomal RNA transcription and ribosome subunit assembly.",
  },
  chromatin: {
    name: "Chromatin", level: 2,
    summary: "DNA wrapped around histone octamers (nucleosomes, ~147 bp per core particle) and further compacted.",
    role: "Condensed heterochromatin (often at the nuclear periphery) is largely silent; open euchromatin is transcriptionally active.",
  },
  rer: {
    name: "Rough endoplasmic reticulum", level: 1,
    summary: "Network of membrane sheets studded with ribosomes, continuous with the nuclear envelope.",
    role: "Preproinsulin is translated into the ER, its signal peptide is cleaved and proinsulin folds with three disulfide bonds.",
  },
  ribosome: {
    name: "Ribosomes", level: 2,
    summary: "RNA–protein machines that translate mRNA into protein.",
    role: "Ribosomes bound to the ER synthesise secretory proteins such as preproinsulin.",
  },
  golgi: {
    name: "Golgi apparatus", level: 1,
    summary: "Stack of flattened membrane cisternae with a cis (entry) and trans (exit) face.",
    role: "Receives proinsulin from the ER; the trans-Golgi network packages it into immature secretory granules.",
  },
  cisGolgi: {
    name: "cis-Golgi face", level: 2,
    summary: "Entry face of the stack, receiving cargo from the ER.",
    role: "First processing compartment for proteins arriving from the ER.",
  },
  transGolgi: {
    name: "trans-Golgi network", level: 2,
    summary: "Exit face where cargo is sorted into vesicles and granules.",
    role: "Buds immature insulin granules, in which proinsulin conversion begins.",
  },
  granule: {
    name: "Insulin secretory granules", level: 1,
    summary: "Membrane-bound vesicles storing insulin; a β-cell contains on the order of 10,000. Only a subset is drawn here.",
    role: "Ca²⁺ influx triggers fusion of granules with the plasma membrane (exocytosis), releasing insulin and C-peptide.",
    source: "Rorsman & Renström, Diabetologia 2003",
  },
  denseCore: {
    name: "Dense core", level: 2,
    summary: "Electron-dense crystalline core of insulin–Zn²⁺ hexamers.",
    role: "Compact, stable storage form of insulin until secretion.",
  },
  halo: {
    name: "Granule halo and membrane", level: 2,
    summary: "Paler region between the dense core and the granule membrane, typical of β-granules in electron micrographs.",
    role: "The membrane carries the proteins required for docking and fusion.",
  },
  mito: {
    name: "Mitochondria", level: 1,
    summary: "Double-membrane organelles that oxidise glucose-derived pyruvate.",
    role: "ATP production raises the cytosolic ATP/ADP ratio, which closes K_ATP channels: the metabolic step of glucose sensing.",
    source: "Rorsman & Ashcroft, Physiol Rev 2018",
  },
  cristae: {
    name: "Cristae", level: 2,
    summary: "Folds of the inner mitochondrial membrane.",
    role: "Increase membrane area for the electron transport chain and ATP synthase.",
  },
  lysosome: {
    name: "Lysosomes", level: 1,
    summary: "Acidic, enzyme-filled degradative organelles.",
    role: "In β-cells they also degrade surplus insulin granules (crinophagy).",
  },
  microtubule: {
    name: "Microtubules", level: 1,
    summary: "Hollow polymers of α/β-tubulin radiating from the centrosome.",
    role: "Tracks along which kinesin motors transport insulin granules toward the cell periphery.",
  },
  centrosome: {
    name: "Centrosome (centrioles)", level: 2,
    summary: "Pair of orthogonally arranged centrioles surrounded by pericentriolar material.",
    role: "Main microtubule-organising centre of the cell.",
  },
  cilium: {
    name: "Primary cilium", level: 1,
    summary: "Single, non-motile antenna-like projection of the plasma membrane.",
    role: "Sensory and signalling organelle; β-cells possess a primary cilium.",
  },
};
