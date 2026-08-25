import { overlapsAny, parseTargetKey, targetKey, type HighlightTarget } from './types.ts';

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

/**
 * What one gesture points at.
 *
 * A group rather than a single target, because pointing is usually plural: the
 * LayerNorm de-scale term is about x̂ AND g, and hovering it should light the
 * term, both tensors, and the code line together. Modelling that as one target
 * would force callers to pick which one "really" counts.
 */
export type HighlightSelection = readonly HighlightTarget[];

export interface HighlightStore {
  subscribe: (listener: () => void) => () => void;
  getHover: () => HighlightSelection | null;
  getPinned: () => readonly HighlightSelection[];
  setHover: (next: HighlightTarget | HighlightSelection | null) => void;
  togglePin: (next: HighlightTarget | HighlightSelection) => void;
  clearPins: () => void;
  levelFor: (target: HighlightTarget) => HighlightLevel;
  /** Same answer, addressed by key — see useHighlightLevel. */
  levelForKey: (key: string) => HighlightLevel;
}

function asSelection(next: HighlightTarget | HighlightSelection): HighlightSelection {
  return Array.isArray(next) ? next : [next as HighlightTarget];
}

/** Order-independent identity for a group. */
export function selectionKey(selection: HighlightSelection): string {
  return [...selection].map(targetKey).sort().join('|');
}

function sameSelection(a: HighlightSelection | null, b: HighlightSelection | null): boolean {
  if (a === null || b === null) return a === b;
  return selectionKey(a) === selectionKey(b);
}

export function createHighlightStore(): HighlightStore {
  let hover: HighlightSelection | null = null;
  let pinned: readonly HighlightSelection[] = [];
  const listeners = new Set<() => void>();
  // Parsing a key is cheap, but this runs for every subscribed view on every
  // notification, so the results are kept.
  const parsed = new Map<string, HighlightTarget>();

  const emit = (): void => {
    for (const listener of listeners) listener();
  };

  const levelOf = (target: HighlightTarget): HighlightLevel => {
    for (const group of pinned) if (overlapsAny(target, group)) return 'pinned';
    if (hover !== null && overlapsAny(target, hover)) return 'hover';
    return 'none';
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
      const selection = next === null ? null : asSelection(next);
      // pointerover fires repeatedly over the same element. Without this guard
      // every mouse move would notify every subscriber for no change at all.
      if (sameSelection(hover, selection)) return;
      hover = selection;
      emit();
    },

    togglePin(next) {
      const selection = asSelection(next);
      const key = selectionKey(selection);
      const index = pinned.findIndex((p) => selectionKey(p) === key);
      pinned = index === -1 ? [...pinned, selection] : pinned.filter((_, i) => i !== index);
      emit();
    },

    clearPins() {
      if (pinned.length === 0) return;
      pinned = [];
      emit();
    },

    /** Pinned beats hover, so a pin does not flicker as the pointer crosses it. */
    levelFor(target) {
      return levelOf(target);
    },

    levelForKey(key) {
      let target = parsed.get(key);
      if (target === undefined) {
        target = parseTargetKey(key);
        parsed.set(key, target);
      }
      return levelOf(target);
    },
  };
}
