export const INSULIN_A: string;
export const INSULIN_B: string;
export interface Drive { glucose: number; secretion: number; insulin: number; ir: number }
export class CellTheatre {
  constructor(canvas: HTMLCanvasElement, onReadout?: (lines: string[]) => void);
  setScene(name: "beta" | "muscle" | "dogma"): string;
  setDrive(d: Partial<Drive>): void;
  click(): void;
  dispose(): void;
}
