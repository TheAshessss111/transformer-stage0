import { useCallback, useMemo, useSyncExternalStore } from 'react';
import { useHighlightStore } from './context';
import type { HighlightLevel } from './store';
import { targetKey, type HighlightTarget } from './types';

/**
 * Subscribe one view to one target.
 *
 * The snapshot is a three-valued string, so React skips re-rendering whenever
 * the level is unchanged — which is almost always, for almost every cell.
 *
 * `subscribe` and `getSnapshot` are memoized on the target's key rather than
 * its object identity; a caller building `{ kind: 'tensor', name, index }`
 * inline every render would otherwise resubscribe on every render and undo the
 * whole arrangement.
 */
export function useHighlightLevel(target: HighlightTarget): HighlightLevel {
  const store = useHighlightStore();
  const key = targetKey(target);

  const subscribe = useCallback((listener: () => void) => store.subscribe(listener), [store]);

  // Addressed by key rather than by the target object: an inline target literal
  // changes identity on every render, which would resubscribe every render and
  // undo the whole arrangement. targetKey is a bijection, so nothing is lost.
  const getSnapshot = useCallback(() => store.levelForKey(key), [store, key]);

  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export interface HighlightActions {
  setHover: (target: HighlightTarget | null) => void;
  togglePin: (target: HighlightTarget) => void;
  clearPins: () => void;
}

/**
 * Stable action handles.
 *
 * Returned from a memo so event handlers built on them never change identity —
 * otherwise every memoized child would re-render anyway and the store's whole
 * reason for existing would collapse.
 */
export function useHighlightActions(): HighlightActions {
  const store = useHighlightStore();
  return useMemo(
    () => ({
      setHover: store.setHover,
      togglePin: store.togglePin,
      clearPins: store.clearPins,
    }),
    [store],
  );
}

/** The current hover and pin set, for views that need the whole picture. */
export function useHighlightSelection(): {
  hover: HighlightTarget | null;
  pinned: readonly HighlightTarget[];
} {
  const store = useHighlightStore();
  const subscribe = useCallback((listener: () => void) => store.subscribe(listener), [store]);
  const hover = useSyncExternalStore(subscribe, store.getHover, store.getHover);
  const pinned = useSyncExternalStore(subscribe, store.getPinned, store.getPinned);
  return { hover, pinned };
}
