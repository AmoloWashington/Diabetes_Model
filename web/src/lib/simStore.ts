import { useSyncExternalStore } from "react";
import type { MealResult } from "./api";

// Holds the latest meal simulation so the Cell Theatre can play it back.
let current: MealResult | null = null;
const listeners = new Set<() => void>();

export function setLastMeal(r: MealResult) {
  current = r;
  listeners.forEach((l) => l());
}

export function useLastMeal(): MealResult | null {
  return useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
    () => current,
  );
}
