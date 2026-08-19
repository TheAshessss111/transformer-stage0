import { overlaps, overlapsAny, parseTargetKey, targetKey, type HighlightTarget } from './types.ts';

/**
 * Highlight state, deliberately OUTSIDE React.
 *
 * Note the explicit `.ts` import extension: this module is DOM-free and the
 * verification harness reaches it, so it must resolve under Node's native type
 * stripping as well as under Vite. Anything the harness can reach follows the
 * same rule.
 *
 * A 16x16 TensorGrid is 256 cells. If this state lived in a Context whose value
 * changed on every pointermove, all 256 would re-render on every mouse move,
 * plus the formula, plus the code pane. That is the single most likely way to
 * make this app feel bad, and it would not become obvious until E0.5 when the
 * grids are real and it is baked in.
 *
 * So: an external store read through useSyncExternalStore with a three-valued
 * snapshot. React bails out of re-rendering any component whose own level did
 * not change, so hovering one cell re-renders that cell and whatever genuinely
 * overlaps it — not the grid.
 */

export type HighlightLevel = 'none' | 'hover' | 'pinned';

export interface HighlightStore {
  subscribe: (listener: () => void) => () => void;
  getHover: () => HighlightTarget | null;
  getPinned: () => readonly HighlightTarget[];
  setHover: (target: HighlightTarget | null) => void;
  togglePin: (target: HighlightTarget) => void;
  clearPins: () => void;
  levelFor: (target: HighlightTarget) => HighlightLevel;
  /** Same answer, addressed by key — see useHighlightLevel. */
  levelForKey: (key: string) => HighlightLevel;
}

function sameTarget(a: HighlightTarget | null, b: HighlightTarget | null): boolean {
  if (a === null || b === null) return a === b;
  return targetKey(a) === targetKey(b);
}

export function createHighlightStore(): HighlightStore {
  let hover: HighlightTarget | null = null;
  let pinned: readonly HighlightTarget[] = [];
  const listeners = new Set<() => void>();
  // Parsing a key is cheap, but this runs for every subscribed view on every
  // notification, so the results are kept.
  const parsed = new Map<string, HighlightTarget>();

  const emit = (): void => {
    for (const listener of listeners) listener();
  };

  return {
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },

    getHover: () => hover,
    getPinned: () => pinned,

    setHover(next) {
      // pointerover fires repeatedly over the same element. Without this guard
      // every mouse move would notify every subscriber for no change at all.
      if (sameTarget(hover, next)) return;
      hover = next;
      emit();
    },

    togglePin(next) {
      const key = targetKey(next);
      const index = pinned.findIndex((p) => targetKey(p) === key);
      pinned = index === -1 ? [...pinned, next] : pinned.filter((_, i) => i !== index);
      emit();
    },

    clearPins() {
      if (pinned.length === 0) return;
      pinned = [];
      emit();
    },

    /** Pinned beats hover, so a pin does not flicker as the pointer crosses it. */
    levelFor(target) {
      if (overlapsAny(target, pinned)) return 'pinned';
      if (hover !== null && overlaps(target, hover)) return 'hover';
      return 'none';
    },

    levelForKey(key) {
      let target = parsed.get(key);
      if (target === undefined) {
        target = parseTargetKey(key);
        parsed.set(key, target);
      }
      if (overlapsAny(target, pinned)) return 'pinned';
      if (hover !== null && overlaps(target, hover)) return 'hover';
      return 'none';
    },
  };
}
