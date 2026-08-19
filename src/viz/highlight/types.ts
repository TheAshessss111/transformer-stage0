/**
 * The addressing protocol every linked view speaks.
 *
 * "Hover a formula term, light up the matrix cells and the code line" is not
 * three features wired to each other — pairwise wiring is O(n^2) and gets worse
 * with every new view. Instead every view describes what it is pointing at in
 * this one vocabulary, and the store broadcasts.
 *
 * ── Overlap, not equality ──────────────────────────────────────────────────
 *
 * Views never compare targets for equality. They ask whether two targets could
 * refer to the same thing: hovering the whole tensor `s` should light every
 * cell of `s`, and hovering cell [0, 2] should light the label of `s` and any
 * formula term pointing at it. So the primitive is a symmetric `overlaps`.
 *
 * Cross-kind linkage is NOT the matcher's job. A formula term declares which
 * tensor targets it corresponds to, and the store resolves through those
 * declarations. Baking content decisions into a generic utility would make it
 * useless for the next lab.
 */

export type Wildcard = '*';

/** A position, or `'*'` for "every position on this axis". */
export type TensorIndex = readonly (number | Wildcard)[];

export type HighlightTarget =
  /** A tensor by its program state-key name. Without `index`, the whole thing. */
  | { kind: 'tensor'; name: string; index?: TensorIndex }
  /** One axis of a tensor — the axis label in a grid, the axis in a shape pill. */
  | { kind: 'axis'; tensor: string; axis: number }
  /** One term of one equation. */
  | { kind: 'formula'; eq: string; term: string }
  /** A 1-indexed line of the program source. */
  | { kind: 'code'; line: number }
  /** One recorded operation — a graph node, or a row of the event table. */
  | { kind: 'event'; index: number };

export type HighlightKind = HighlightTarget['kind'];

/**
 * A total, injective serialization.
 *
 * Used as the pin-set membership key and as the memo key for subscriptions, so
 * two different targets must never produce the same string. Tensor names are
 * program state keys (JS identifiers), so `:` cannot appear inside one.
 */
export function targetKey(target: HighlightTarget): string {
  switch (target.kind) {
    case 'tensor':
      // The marker keeps "whole tensor" distinct from an explicit ['*'] index.
      return `t:${target.name}:${target.index === undefined ? '∅' : target.index.join(',')}`;
    case 'axis':
      return `a:${target.tensor}:${target.axis}`;
    case 'formula':
      return `f:${target.eq}:${target.term}`;
    case 'code':
      return `c:${target.line}`;
    case 'event':
      return `e:${target.index}`;
  }
}

/**
 * The exact inverse of {@link targetKey}.
 *
 * Exists so a subscription can be memoized on the key string alone: a hook that
 * closed over the target object would have to list it as a dependency, and an
 * inline `{ kind: 'tensor', name, index }` changes identity on every render.
 * Round-tripping through the key makes the key the identity, which is what the
 * design claims anyway.
 *
 * Splits on the LAST separator, so a name containing ':' still parses.
 */
export function parseTargetKey(key: string): HighlightTarget {
  const kind = key.slice(0, key.indexOf(':'));
  const rest = key.slice(key.indexOf(':') + 1);

  switch (kind) {
    case 't': {
      const cut = rest.lastIndexOf(':');
      const name = rest.slice(0, cut);
      const raw = rest.slice(cut + 1);
      if (raw === '∅') return { kind: 'tensor', name };
      const index = raw === '' ? [] : raw.split(',').map((p) => (p === '*' ? '*' : Number(p)));
      return { kind: 'tensor', name, index };
    }
    case 'a': {
      const cut = rest.lastIndexOf(':');
      return { kind: 'axis', tensor: rest.slice(0, cut), axis: Number(rest.slice(cut + 1)) };
    }
    case 'f': {
      const cut = rest.lastIndexOf(':');
      return { kind: 'formula', eq: rest.slice(0, cut), term: rest.slice(cut + 1) };
    }
    case 'c':
      return { kind: 'code', line: Number(rest) };
    case 'e':
      return { kind: 'event', index: Number(rest) };
    default:
      throw new Error(`parseTargetKey: unrecognised key '${key}'`);
  }
}

function indicesOverlap(a: TensorIndex | undefined, b: TensorIndex | undefined): boolean {
  // An absent index means the whole tensor, which meets any part of it.
  if (a === undefined || b === undefined) return true;
  // Different ranks cannot describe the same element.
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === '*' || b[i] === '*') continue;
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Could these two targets refer to a common thing? Symmetric.
 *
 * The one cross-kind case is tensor/axis: an axis belongs to its tensor, so
 * highlighting the tensor should light its axis labels too.
 */
export function overlaps(a: HighlightTarget, b: HighlightTarget): boolean {
  if (a.kind === 'tensor' && b.kind === 'tensor') {
    return a.name === b.name && indicesOverlap(a.index, b.index);
  }
  if (a.kind === 'tensor' && b.kind === 'axis') return a.name === b.tensor;
  if (a.kind === 'axis' && b.kind === 'tensor') return a.tensor === b.name;
  if (a.kind === 'axis' && b.kind === 'axis') return a.tensor === b.tensor && a.axis === b.axis;
  if (a.kind === 'formula' && b.kind === 'formula') return a.eq === b.eq && a.term === b.term;
  if (a.kind === 'code' && b.kind === 'code') return a.line === b.line;
  if (a.kind === 'event' && b.kind === 'event') return a.index === b.index;
  return false;
}

/** Does `target` overlap anything in `list`? */
export function overlapsAny(target: HighlightTarget, list: readonly HighlightTarget[]): boolean {
  return list.some((other) => overlaps(target, other));
}

/** Convenience constructors, so call sites read as intent rather than object literals. */
export const target = {
  tensor: (name: string, index?: TensorIndex): HighlightTarget => ({ kind: 'tensor', name, index }),
  cell: (name: string, index: TensorIndex): HighlightTarget => ({ kind: 'tensor', name, index }),
  axis: (tensor: string, axis: number): HighlightTarget => ({ kind: 'axis', tensor, axis }),
  formulaTerm: (eq: string, term: string): HighlightTarget => ({ kind: 'formula', eq, term }),
  codeLine: (line: number): HighlightTarget => ({ kind: 'code', line }),
  event: (index: number): HighlightTarget => ({ kind: 'event', index }),
};
