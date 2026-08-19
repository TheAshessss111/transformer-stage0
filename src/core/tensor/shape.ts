/**
 * Shape operations, and the one place a real memory copy can happen.
 *
 * The contract every visualization depends on (docs/architecture/overview.md A-01):
 *
 *   transpose / permute / swapAxes / select / expandDims / squeeze  → always a view
 *   reshape of a contiguous array                                   → a view
 *   reshape of a NON-contiguous array                               → copies, and says so
 *
 * "Says so" means the trace event carries didCopy plus the element count, which
 * is what Step 0.1 renders as the amber `copy · N elements` badge.
 */

import { emit } from '../trace/hook.ts';
import {
  NdArray,
  contiguousStrides,
  copyFlat,
  formatShapeTuple,
  isContiguous,
  size,
  sizeOfShape,
  type Shape,
} from './ndarray.ts';

/** A fresh contiguous array holding the same logical elements. Does not emit. */
function contiguousCopy(a: NdArray): NdArray {
  return new NdArray({ data: copyFlat(a), shape: [...a.shape] });
}

/** A view onto the same buffer with new shape/strides. Does not emit. */
function view(a: NdArray, shape: Shape, strides: Shape, offset: number): NdArray {
  return new NdArray({
    data: a.data,
    shape,
    strides,
    offset,
    base: a.base ?? a,
    readOnly: a.readOnly,
  });
}

function resolveShape(requested: Shape, total: number): number[] {
  const inferAt = requested.indexOf(-1);
  if (requested.indexOf(-1, inferAt + 1) !== -1) {
    throw new Error(`reshape: at most one -1 allowed, got ${formatShapeTuple(requested)}`);
  }

  if (inferAt === -1) {
    if (sizeOfShape(requested) !== total) {
      throw new Error(
        `reshape: cannot reshape ${total} elements into ${formatShapeTuple(requested)} (${sizeOfShape(requested)} elements)`,
      );
    }
    return [...requested];
  }

  const known = requested.reduce((acc, d, i) => (i === inferAt ? acc : acc * d), 1);
  if (known === 0 || total % known !== 0) {
    throw new Error(
      `reshape: cannot infer axis ${inferAt} of ${formatShapeTuple(requested)} from ${total} elements`,
    );
  }
  const out = [...requested];
  out[inferAt] = total / known;
  return out;
}

/**
 * Materialise `a` into contiguous memory.
 *
 * Returns `a` itself when it is already contiguous — the identity check matters,
 * because "did this cost a copy?" is a fact the UI shows.
 */
export function ascontiguousarray(a: NdArray): NdArray {
  if (isContiguous(a)) {
    emit({
      op: 'ascontiguousarray',
      phase: 'forward',
      inputs: [a],
      output: a,
      isView: true,
      didCopy: false,
    });
    return a;
  }
  const out = contiguousCopy(a);
  emit({
    op: 'ascontiguousarray',
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: false,
    didCopy: true,
    copiedElements: size(a),
  });
  return out;
}

/**
 * Reshape. Zero-copy when the input is contiguous; otherwise the elements are
 * copied into row-major order first, because reshape reinterprets *memory*
 * order and a strided view has none to reinterpret.
 *
 * This is the exact behaviour Step 0.1's "why must transpose be followed by
 * ascontiguousarray" lab demonstrates.
 */
export function reshape(a: NdArray, requested: Shape): NdArray {
  const shape = resolveShape(requested, size(a));

  if (isContiguous(a)) {
    const out = view(a, shape, contiguousStrides(shape), a.offset);
    emit({
      op: 'reshape',
      phase: 'forward',
      inputs: [a],
      output: out,
      isView: true,
      didCopy: false,
      meta: { requested: [...requested] },
    });
    return out;
  }

  const copied = contiguousCopy(a);
  const out = new NdArray({ data: copied.data, shape, strides: contiguousStrides(shape) });
  emit({
    op: 'reshape',
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: false,
    didCopy: true,
    copiedElements: size(a),
    meta: { requested: [...requested], reason: 'input was not contiguous' },
  });
  return out;
}

function normalizeAxis(axis: number, rank: number, label: string): number {
  const resolved = axis < 0 ? axis + rank : axis;
  if (!Number.isInteger(resolved) || resolved < 0 || resolved >= rank) {
    throw new Error(`${label}: axis ${axis} is out of range for a rank-${rank} array`);
  }
  return resolved;
}

/** Permute axes. Always zero-copy: only the strides are reordered. */
export function permute(a: NdArray, axes: readonly number[]): NdArray {
  const rank = a.shape.length;
  if (axes.length !== rank) {
    throw new Error(`permute: got ${axes.length} axes for a rank-${rank} array`);
  }
  const seen = new Set<number>();
  const resolved = axes.map((ax) => {
    const r = normalizeAxis(ax, rank, 'permute');
    if (seen.has(r)) throw new Error(`permute: axis ${r} listed twice in [${axes.join(', ')}]`);
    seen.add(r);
    return r;
  });

  const out = view(
    a,
    resolved.map((ax) => a.shape[ax]),
    resolved.map((ax) => a.strides[ax]),
    a.offset,
  );
  emit({
    op: 'permute',
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: true,
    didCopy: false,
    meta: { axes: resolved },
  });
  return out;
}

/** Reverse all axes. */
export function transpose(a: NdArray): NdArray {
  const axes = a.shape.map((_, i) => a.shape.length - 1 - i);
  return permute(a, axes);
}

export function swapAxes(a: NdArray, i: number, j: number): NdArray {
  const rank = a.shape.length;
  const ai = normalizeAxis(i, rank, 'swapAxes');
  const aj = normalizeAxis(j, rank, 'swapAxes');
  const axes = a.shape.map((_, k) => k);
  axes[ai] = aj;
  axes[aj] = ai;
  return permute(a, axes);
}

/**
 * Take index `i` along `axis`, dropping that axis. Zero-copy: it only advances
 * the offset. This is what E0.5's slice selector uses to show one (b, h) plane
 * of a 4-D tensor.
 */
export function select(a: NdArray, axis: number, i: number): NdArray {
  const ax = normalizeAxis(axis, a.shape.length, 'select');
  if (!Number.isInteger(i) || i < 0 || i >= a.shape[ax]) {
    throw new Error(
      `select: index ${i} out of range on axis ${ax} of ${formatShapeTuple(a.shape)}`,
    );
  }
  const shape = a.shape.filter((_, k) => k !== ax);
  const strides = a.strides.filter((_, k) => k !== ax);
  const out = view(a, shape, strides, a.offset + i * a.strides[ax]);
  emit({
    op: 'select',
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: true,
    didCopy: false,
    meta: { axis: ax, index: i },
  });
  return out;
}

/**
 * Insert an axis of extent 1.
 *
 * The inserted stride is chosen so that a contiguous input stays contiguous,
 * which keeps `isContiguous` (and therefore reshape's copy decision) honest.
 */
export function expandDims(a: NdArray, axis: number): NdArray {
  const rank = a.shape.length;
  const ax = axis < 0 ? axis + rank + 1 : axis;
  if (!Number.isInteger(ax) || ax < 0 || ax > rank) {
    throw new Error(`expandDims: axis ${axis} is out of range for a rank-${rank} array`);
  }
  const inserted = ax === rank ? 1 : a.strides[ax] * a.shape[ax];

  const shape = [...a.shape.slice(0, ax), 1, ...a.shape.slice(ax)];
  const strides = [...a.strides.slice(0, ax), inserted, ...a.strides.slice(ax)];
  const out = view(a, shape, strides, a.offset);
  emit({
    op: 'expandDims',
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: true,
    didCopy: false,
    meta: { axis: ax },
  });
  return out;
}

/** Drop axes of extent 1 — all of them, or just `axis`. */
export function squeeze(a: NdArray, axis?: number): NdArray {
  const rank = a.shape.length;
  let keep: boolean[];

  if (axis === undefined) {
    keep = a.shape.map((d) => d !== 1);
  } else {
    const ax = normalizeAxis(axis, rank, 'squeeze');
    if (a.shape[ax] !== 1) {
      throw new Error(
        `squeeze: axis ${ax} has extent ${a.shape[ax]}, not 1, in ${formatShapeTuple(a.shape)}`,
      );
    }
    keep = a.shape.map((_, k) => k !== ax);
  }

  const out = view(
    a,
    a.shape.filter((_, k) => keep[k]),
    a.strides.filter((_, k) => keep[k]),
    a.offset,
  );
  emit({
    op: 'squeeze',
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: true,
    didCopy: false,
    meta: { axis: axis ?? null },
  });
  return out;
}
