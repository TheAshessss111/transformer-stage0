/**
 * Forward operators.
 *
 * These are raw NdArray → NdArray functions with no graph attached. The tape
 * lives in autograd.ts and the backwards in vjp.ts (docs/architecture/overview.md A-03).
 * Keeping the three apart is what lets a lab that only needs a forward pass —
 * Step 0.1's shape pipeline — stay cheap, and what lets E0.2 gradcheck a single
 * VJP in isolation.
 *
 * Performance note: teaching-scale tensors are a few thousand elements, so the
 * implementations here are the obvious ones. No blocking, no SIMD. Optimising
 * would only make the code harder to read against the maths it implements.
 */

import { emit } from '../trace/hook.ts';
import {
  NdArray,
  forEachIndex,
  forEachZip,
  formatShapeTuple,
  shapesEqual,
  size,
  sizeOfShape,
  zeros,
  type Shape,
} from './ndarray.ts';
import { broadcastShapes, broadcastToQuiet } from './broadcast.ts';

// ── elementwise ─────────────────────────────────────────────────────────────

function mapUnary(op: string, a: NdArray, f: (x: number) => number): NdArray {
  const out = zeros(a.shape);
  forEachZip(a.shape, [a, out], (offsets) => {
    out.data[offsets[1]] = f(a.data[offsets[0]]);
  });
  emit({ op, phase: 'forward', inputs: [a], output: out, isView: false, didCopy: false });
  return out;
}

function mapBinary(
  op: string,
  a: NdArray,
  b: NdArray,
  f: (x: number, y: number) => number,
): NdArray {
  const shape = shapesEqual(a.shape, b.shape) ? [...a.shape] : broadcastShapes(a.shape, b.shape);
  const av = shapesEqual(a.shape, shape) ? a : broadcastToQuiet(a, shape);
  const bv = shapesEqual(b.shape, shape) ? b : broadcastToQuiet(b, shape);

  const out = zeros(shape);
  forEachZip(shape, [av, bv, out], (offsets) => {
    out.data[offsets[2]] = f(av.data[offsets[0]], bv.data[offsets[1]]);
  });

  emit({
    op,
    phase: 'forward',
    inputs: [a, b],
    output: out,
    isView: false,
    didCopy: false,
    meta: { broadcast: !shapesEqual(a.shape, b.shape), shape },
  });
  return out;
}

export function neg(a: NdArray): NdArray {
  return mapUnary('neg', a, (x) => -x);
}

export function abs(a: NdArray): NdArray {
  return mapUnary('abs', a, Math.abs);
}

export function exp(a: NdArray): NdArray {
  return mapUnary('exp', a, Math.exp);
}

export function log(a: NdArray): NdArray {
  return mapUnary('log', a, Math.log);
}

export function sqrt(a: NdArray): NdArray {
  return mapUnary('sqrt', a, Math.sqrt);
}

export function relu(a: NdArray): NdArray {
  return mapUnary('relu', a, (x) => (x > 0 ? x : 0));
}

export function add(a: NdArray, b: NdArray): NdArray {
  return mapBinary('add', a, b, (x, y) => x + y);
}

export function sub(a: NdArray, b: NdArray): NdArray {
  return mapBinary('sub', a, b, (x, y) => x - y);
}

export function mul(a: NdArray, b: NdArray): NdArray {
  return mapBinary('mul', a, b, (x, y) => x * y);
}

export function div(a: NdArray, b: NdArray): NdArray {
  return mapBinary('div', a, b, (x, y) => x / y);
}

export function maximum(a: NdArray, b: NdArray): NdArray {
  return mapBinary('maximum', a, b, (x, y) =>
    Number.isNaN(x) || Number.isNaN(y) ? NaN : Math.max(x, y),
  );
}

export function addScalar(a: NdArray, s: number): NdArray {
  return mapUnary('addScalar', a, (x) => x + s);
}

export function mulScalar(a: NdArray, s: number): NdArray {
  return mapUnary('mulScalar', a, (x) => x * s);
}

/** Boolean mask as 1.0 / 0.0 — cached by relu's forward so its VJP never needs the input. */
export function positiveMask(a: NdArray): NdArray {
  return mapUnary('positiveMask', a, (x) => (x > 0 ? 1 : 0));
}

// ── reductions ──────────────────────────────────────────────────────────────

export type Axis = number | readonly number[] | null;

/** Resolve an axis spec into a sorted list of non-negative axis indices. */
export function normalizeAxes(axis: Axis, rank: number, label: string): number[] {
  if (axis === null) return Array.from({ length: rank }, (_, i) => i);
  const raw = typeof axis === 'number' ? [axis] : [...axis];
  const seen = new Set<number>();
  for (const ax of raw) {
    const r = ax < 0 ? ax + rank : ax;
    if (!Number.isInteger(r) || r < 0 || r >= rank) {
      throw new Error(`${label}: axis ${ax} is out of range for a rank-${rank} array`);
    }
    if (seen.has(r)) throw new Error(`${label}: axis ${r} listed twice`);
    seen.add(r);
  }
  return [...seen].sort((x, y) => x - y);
}

export function reducedShape(shape: Shape, axes: readonly number[], keepdims: boolean): number[] {
  const drop = new Set(axes);
  if (keepdims) return shape.map((d, i) => (drop.has(i) ? 1 : d));
  return shape.filter((_, i) => !drop.has(i));
}

/**
 * Shared reduction skeleton: walk the input, project each multi-index onto the
 * output, and let `accumulate` fold the value in.
 */
function reduce(
  op: string,
  a: NdArray,
  axis: Axis,
  keepdims: boolean,
  init: number,
  accumulate: (previous: number, value: number) => number,
  finalize?: (value: number, count: number) => number,
): NdArray {
  const axes = normalizeAxes(axis, a.shape.length, op);
  const drop = new Set(axes);
  const outShape = reducedShape(a.shape, axes, keepdims);

  const out = zeros(outShape);
  if (init !== 0) out.data.fill(init);

  forEachIndex(a, (flat, _logical, idx) => {
    let dst = 0;
    let outAxis = 0;
    for (let i = 0; i < a.shape.length; i++) {
      if (drop.has(i)) {
        if (keepdims) outAxis += 1;
        continue;
      }
      dst += idx[i] * out.strides[outAxis];
      outAxis += 1;
    }
    out.data[dst] = accumulate(out.data[dst], a.data[flat]);
  });

  if (finalize) {
    let count = 1;
    for (const ax of axes) count *= a.shape[ax];
    for (let i = 0; i < out.data.length; i++) out.data[i] = finalize(out.data[i], count);
  }

  emit({
    op,
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: false,
    didCopy: false,
    meta: { axes, keepdims },
  });
  return out;
}

export function sum(a: NdArray, axis: Axis = null, keepdims = false): NdArray {
  return reduce('sum', a, axis, keepdims, 0, (acc, v) => acc + v);
}

export function mean(a: NdArray, axis: Axis = null, keepdims = false): NdArray {
  return reduce(
    'mean',
    a,
    axis,
    keepdims,
    0,
    (acc, v) => acc + v,
    (v, n) => v / n,
  );
}

/**
 * NaN propagates, matching NumPy: once a reduction window has seen a NaN the
 * result stays NaN regardless of what follows. Step 0.5 depends on this — a max
 * that silently skipped NaN would hide the very bug the lab is about.
 */
export function max(a: NdArray, axis: Axis = null, keepdims = false): NdArray {
  return reduce('max', a, axis, keepdims, -Infinity, (acc, v) => {
    if (Number.isNaN(acc)) return acc;
    if (Number.isNaN(v)) return NaN;
    return v > acc ? v : acc;
  });
}

export function min(a: NdArray, axis: Axis = null, keepdims = false): NdArray {
  return reduce('min', a, axis, keepdims, Infinity, (acc, v) => {
    if (Number.isNaN(acc)) return acc;
    if (Number.isNaN(v)) return NaN;
    return v < acc ? v : acc;
  });
}

// ── matrix multiply ─────────────────────────────────────────────────────────

/**
 * `(..., m, k) @ (..., k, n) → (..., m, n)`, with the leading batch dimensions
 * broadcast against each other.
 *
 * Rank < 2 is rejected on purpose rather than implementing NumPy's 1-D promotion
 * rules: Transformer code never relies on them, and the error is a better teacher
 * than a silent reinterpretation. (transformer_plan.md Step 0.1: 70% of bugs are
 * shape/transpose errors that do not raise.)
 */
export function matmul(a: NdArray, b: NdArray): NdArray {
  if (a.shape.length < 2 || b.shape.length < 2) {
    throw new Error(
      `matmul: both operands need rank >= 2, got ${formatShapeTuple(a.shape)} and ${formatShapeTuple(b.shape)}. ` +
        'Add the missing axis explicitly with expandDims so the intent is visible.',
    );
  }

  const m = a.shape[a.shape.length - 2];
  const kA = a.shape[a.shape.length - 1];
  const kB = b.shape[b.shape.length - 2];
  const n = b.shape[b.shape.length - 1];

  if (kA !== kB) {
    throw new Error(
      `matmul: inner dimensions disagree — ${formatShapeTuple(a.shape)} @ ${formatShapeTuple(b.shape)} ` +
        `(${kA} vs ${kB})`,
    );
  }

  const batch = broadcastShapes(a.shape.slice(0, -2), b.shape.slice(0, -2));
  const av = broadcastToQuiet(a, [...batch, m, kA]);
  const bv = broadcastToQuiet(b, [...batch, kB, n]);
  const out = zeros([...batch, m, n]);

  const asM = av.strides[av.strides.length - 2];
  const asK = av.strides[av.strides.length - 1];
  const bsK = bv.strides[bv.strides.length - 2];
  const bsN = bv.strides[bv.strides.length - 1];
  const osM = out.strides[out.strides.length - 2];
  const osN = out.strides[out.strides.length - 1];

  const batchCount = sizeOfShape(batch);
  const counter = new Array<number>(batch.length).fill(0);
  let aOff = av.offset;
  let bOff = bv.offset;
  let oOff = out.offset;

  for (let batchIndex = 0; batchIndex < batchCount; batchIndex++) {
    // i-p-j order: each (i, p) loads one `a` element and streams a row of `b`.
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < kA; p++) {
        const aValue = av.data[aOff + i * asM + p * asK];
        const bRow = bOff + p * bsK;
        const oRow = oOff + i * osM;
        for (let j = 0; j < n; j++) {
          out.data[oRow + j * osN] += aValue * bv.data[bRow + j * bsN];
        }
      }
    }

    for (let axis = batch.length - 1; axis >= 0; axis--) {
      counter[axis] += 1;
      aOff += av.strides[axis];
      bOff += bv.strides[axis];
      oOff += out.strides[axis];
      if (counter[axis] < batch[axis]) break;
      aOff -= batch[axis] * av.strides[axis];
      bOff -= batch[axis] * bv.strides[axis];
      oOff -= batch[axis] * out.strides[axis];
      counter[axis] = 0;
    }
  }

  emit({
    op: 'matmul',
    phase: 'forward',
    inputs: [a, b],
    output: out,
    isView: false,
    didCopy: false,
    meta: { m, k: kA, n, batch },
  });
  return out;
}

/** Total number of elements — re-exported so callers need one import fewer. */
export { size };
