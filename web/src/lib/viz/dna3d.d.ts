export class DNAHelix {
  constructor(container: HTMLElement, opts?: { interactive?: boolean; autoRotate?: boolean; maxBp?: number; dustColor?: number });
  setSequence(seq: string): void;
  startBubble(): void;
  dispose(): void;
  readonly canvas: HTMLCanvasElement;
}
