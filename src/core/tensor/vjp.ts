/**
 * Vector-Jacobian products — the hand-written backward of every forward op.
 *
 * Two conventions, both deliberate:
 *
 *  1. **Each VJP takes only what the maths needs.** Where the derivative can be
 *     written in terms of the forward *output*, that is what the signature asks
 *     for (`expVjp(g, out)`, `reluVjp(g, mask)`). This is a real memory saving
 *     and it is the exact point Step 0.3 makes about softmax, so the API must
 *     not quietly cache inputs "just in case".
 *
 *  2. **Every VJP that can broadcast ends in `unbroadcast`.** Plan rule 5:
 *     forward broadcast is backward sum. The shape assertion inside unbroadcast
 *     is what catches the mistakes.
 *
 * Inner helper operations run under suppressOpHook so the backward trace has one
 * event per operator, matching the forward trace's granularity. Each event
 * carries meta.wrt: which input this gradient belongs to.
 */

import { emit, suppressOpHook } from '../trace/hook.ts';
import {
  NdArray,
  clone,
  forEachIndex,
  forEachZip,
  shapesEqual,
  zeros,
  type Shape,
} from './ndarray.ts';
import { broadcastToQuiet, unbroadcast } from './broadcast.ts';
import { reshape } from './shape.ts';
import { div, matmul, mul, mulScalar, neg, normalizeAxes, reducedShape, type Axis } from './ops.ts';

function emitBackward(
  op: string,
  wrt: number,
  inputs: readonly NdArray[],
  output: NdArray,
  meta?: Readonly<Record<string, unknown>>,
): NdArray {
  emit({
    op,
    phase: 'backward',
    inputs,
    output,
    isView: false,
    didCopy: false,
    meta: { ...meta, wrt },
  });
  return output;
}

// ── elementwise unary ───────────────────────────────────────────────────────

export function negVjp(g: NdArray): NdArray {
  return emitBackward(
    'neg',
    0,
    [g],
    suppressOpHook(() => neg(g)),
  );
}

/** d/dx exp(x) = exp(x) = the forward OUTPUT. The input is never needed. */
export function expVjp(g: NdArray, out: NdArray): NdArray {
  return emitBackward(
    'exp',
    0,
    [g, out],
    suppressOpHook(() => mul(g, out)),
  );
}

export function logVjp(g: NdArray, a: NdArray): NdArray {
  return emitBackward(
    'log',
    0,
    [g, a],
    suppressOpHook(() => div(g, a)),
  );
}

/** d/dx sqrt(x) = 1 / (2 sqrt(x)) — again expressible through the output. */
export function sqrtVjp(g: NdArray, out: NdArray): NdArray {
  return emitBackward(
    'sqrt',
    0,
    [g, out],
    suppressOpHook(() => mulScalar(div(g, out), 0.5)),
  );
}

/**
 * `mask` is the 1/0 array produced by ops.positiveMask during the forward pass.
 * Taking the mask rather than the input keeps this honest about what has to be
 * kept alive, and sidesteps the question of what relu'(0) should be.
 */
export function reluVjp(g: NdArray, mask: NdArray): NdArray {
  return emitBackward(
    'relu',
    0,
    [g, mask],
    suppressOpHook(() => mul(g, mask)),
  );
}

export function absVjp(g: NdArray, a: NdArray): NdArray {
  const out = zeros(g.shape);
  forEachZip(g.shape, [g, a, out], (offsets) => {
    const x = a.data[offsets[1]];
    out.data[offsets[2]] = g.data[offsets[0]] * (x > 0 ? 1 : x < 0 ? -1 : 0);
  });
  return emitBackward('abs', 0, [g, a], out);
}

export function addScalarVjp(g: NdArray): NdArray {
  return emitBackward('addScalar', 0, [g], clone(g));
}

export function mulScalarVjp(g: NdArray, s: number): NdArray {
  return emitBackward(
    'mulScalar',
    0,
    [g],
    suppressOpHook(() => mulScalar(g, s)),
    { scalar: s },
  );
}

// ── elementwise binary ──────────────────────────────────────────────────────

/** Addition just routes the gradient; broadcasting is undone by summation. */
export function addVjp(g: NdArray, aShape: Shape, bShape: Shape): [NdArray, NdArray] {
  const da = suppressOpHook(() => unbroadcast(g, aShape));
  const db = suppressOpHook(() => unbroadcast(g, bShape));
  return [
    emitBackward('add', 0, [g], shapesEqual(da.shape, g.shape) ? clone(da) : da),
    emitBackward('add', 1, [g], shapesEqual(db.shape, g.shape) ? clone(db) : db),
  ];
}

export function subVjp(g: NdArray, aShape: Shape, bShape: Shape): [NdArray, NdArray] {
  const da = suppressOpHook(() => unbroadcast(g, aShape));
  const db = suppressOpHook(() => unbroadcast(neg(g), bShape));
  return [
    emitBackward('sub', 0, [g], shapesEqual(da.shape, g.shape) ? clone(da) : da),
    emitBackward('sub', 1, [g], db),
  ];
}

export function mulVjp(g: NdArray, a: NdArray, b: NdArray): [NdArray, NdArray] {
  const da = suppressOpHook(() => unbroadcast(mul(g, b), a.shape));
  const db = suppressOpHook(() => unbroadcast(mul(g, a), b.shape));
  return [emitBackward('mul', 0, [g, b], da), emitBackward('mul', 1, [g, a], db)];
}

/** d/da (a/b) = 1/b ; d/db (a/b) = -a/b^2 */
export function divVjp(g: NdArray, a: NdArray, b: NdArray): [NdArray, NdArray] {
  const da = suppressOpHook(() => unbroadcast(div(g, b), a.shape));
  const db = suppressOpHook(() => unbroadcast(neg(div(mul(g, a), mul(b, b))), b.shape));
  return [emitBackward('div', 0, [g, b], da), emitBackward('div', 1, [g, a, b], db)];
}

/** Ties split the gradient evenly, matching the tie handling in maxVjp. */
export function maximumVjp(g: NdArray, a: NdArray, b: NdArray): [NdArray, NdArray] {
  const shape = g.shape;
  const av = shapesEqual(a.shape, shape) ? a : broadcastToQuiet(a, shape);
  const bv = shapesEqual(b.shape, shape) ? b : broadcastToQuiet(b, shape);

  const wideA = zeros(shape);
  const wideB = zeros(shape);
  forEachZip(shape, [g, av, bv, wideA, wideB], (offsets) => {
    const gv = g.data[offsets[0]];
    const x = av.data[offsets[1]];
    const y = bv.data[offsets[2]];
    const share = x === y ? 0.5 : x > y ? 1 : 0;
    wideA.data[offsets[3]] = gv * share;
    wideB.data[offsets[4]] = gv * (1 - share);
  });

  const da = suppressOpHook(() => unbroadcast(wideA, a.shape));
  const db = suppressOpHook(() => unbroadcast(wideB, b.shape));
  return [emitBackward('maximum', 0, [g, a, b], da), emitBackward('maximum', 1, [g, a, b], db)];
}

// ── matmul ──────────────────────────────────────────────────────────────────

/**
 * Plan rule 1, derived by shape-matching: only one arrangement type-checks.
 *
 *   dA = g @ B^T      (..., m, n) @ (..., n, k) -> (..., m, k)
 *   dB = A^T @ g      (..., k, m) @ (..., m, n) -> (..., k, n)
 *
 * then the broadcast batch dimensions are summed back down.
 */
export function matmulVjp(g: NdArray, a: NdArray, b: NdArray): [NdArray, NdArray] {
  const [da, db] = suppressOpHook(() => {
    const bT = swapLastTwo(b);
    const aT = swapLastTwo(a);
    return [unbroadcast(matmul(g, bT), a.shape), unbroadcast(matmul(aT, g), b.shape)] as const;
  });
  return [emitBackward('matmul', 0, [g, b], da), emitBackward('matmul', 1, [g, a], db)];
}

function swapLastTwo(x: NdArray): NdArray {
  const rank = x.shape.length;
  const axes = x.shape.map((_, i) => i);
  axes[rank - 2] = rank - 1;
  axes[rank - 1] = rank - 2;
  return new NdArray({
    data: x.data,
    shape: axes.map((i) => x.shape[i]),
    strides: axes.map((i) => x.strides[i]),
    offset: x.offset,
    base: x.base ?? x,
    readOnly: x.readOnly,
  });
}

// ── reductions ──────────────────────────────────────────────────────────────

/**
 * Re-insert the axes that were reduced away (as extent 1) so the gradient can
 * be broadcast back to the input shape. This is the "broadcast" half of the
 * sum/broadcast duality in plan rule 5.
 */
function restoreReducedAxes(g: NdArray, inShape: Shape, axes: readonly number[]): NdArray {
  const keptShape = reducedShape(inShape, axes, true);
  return shapesEqual(g.shape, keptShape) ? g : reshape(g, keptShape);
}

export function sumVjp(g: NdArray, inShape: Shape, axis: Axis, keepdims: boolean): NdArray {
  const axes = normalizeAxes(axis, inShape.length, 'sumVjp');
  const out = suppressOpHook(() => {
    void keepdims; // handled by restoreReducedAxes, which reshapes either form
    const restored = restoreReducedAxes(g, inShape, axes);
    return clone(broadcastToQuiet(restored, inShape));
  });
  return emitBackward('sum', 0, [g], out, { axes, keepdims });
}

export function meanVjp(g: NdArray, inShape: Shape, axis: Axis, keepdims: boolean): NdArray {
  const axes = normalizeAxes(axis, inShape.length, 'meanVjp');
  let count = 1;
  for (const ax of axes) count *= inShape[ax];

  const out = suppressOpHook(() => {
    const restored = restoreReducedAxes(g, inShape, axes);
    return mulScalar(broadcastToQuiet(restored, inShape), 1 / count);
  });
  return emitBackward('mean', 0, [g], out, { axes, keepdims, count });
}

/**
 * The gradient reaches only the argmax positions. Ties split evenly, which is
 * what PyTorch does and what keeps gradcheck honest when duplicates appear.
 */
export function maxVjp(
  g: NdArray,
  a: NdArray,
  out: NdArray,
  axis: Axis,
  keepdims: boolean,
): NdArray {
  return maxOrMinVjp('max', g, a, out, axis, keepdims);
}

export function minVjp(
  g: NdArray,
  a: NdArray,
  out: NdArray,
  axis: Axis,
  keepdims: boolean,
): NdArray {
  return maxOrMinVjp('min', g, a, out, axis, keepdims);
}

function maxOrMinVjp(
  op: 'max' | 'min',
  g: NdArray,
  a: NdArray,
  out: NdArray,
  axis: Axis,
  keepdims: boolean,
): NdArray {
  const axes = normalizeAxes(axis, a.shape.length, `${op}Vjp`);
  const drop = new Set(axes);

  const result = suppressOpHook(() => {
    const gK = broadcastToQuiet(restoreReducedAxes(g, a.shape, axes), a.shape);
    const oK = broadcastToQuiet(restoreReducedAxes(out, a.shape, axes), a.shape);

    // pass 1: mark the winners and count ties per reduction window
    const mask = zeros(a.shape);
    const tieShape = reducedShape(a.shape, axes, true);
    const ties = zeros(tieShape);

    forEachIndex(a, (flat, _logical, idx) => {
      let maskOffset = 0;
      let tieOffset = 0;
      let keptAxis = 0;
      for (let i = 0; i < a.shape.length; i++) {
        maskOffset += idx[i] * mask.strides[i];
        tieOffset += (drop.has(i) ? 0 : idx[i]) * ties.strides[keptAxis];
        keptAxis += 1;
      }
      let winner = 0;
      let oOffset = 0;
      for (let i = 0; i < a.shape.length; i++) oOffset += idx[i] * oK.strides[i];
      if (a.data[flat] === oK.data[oK.offset + oOffset]) winner = 1;
      mask.data[maskOffset] = winner;
      ties.data[tieOffset] += winner;
    });

    // pass 2: distribute
    const tiesWide = broadcastToQuiet(ties, a.shape);
    const grad = zeros(a.shape);
    forEachZip(a.shape, [mask, gK, tiesWide, grad], (offsets) => {
      const winner = mask.data[offsets[0]];
      grad.data[offsets[3]] =
        winner === 0 ? 0 : (gK.data[offsets[1]] * winner) / tiesWide.data[offsets[2]];
    });
    return grad;
  });

  return emitBackward(op, 0, [g, a, out], result, { axes, keepdims });
}

// ── shape ops ───────────────────────────────────────────────────────────────

/** Plan rule 4: the backward of a reshape is a reshape back. */
export function reshapeVjp(g: NdArray, inShape: Shape): NdArray {
  const out = suppressOpHook(() => clone(reshape(g, inShape)));
  return emitBackward('reshape', 0, [g], out, { to: [...inShape] });
}

/** The backward of a permutation is the inverse permutation. */
export function permuteVjp(g: NdArray, axes: readonly number[]): NdArray {
  const inverse = new Array<number>(axes.length);
  for (let i = 0; i < axes.length; i++) inverse[axes[i]] = i;
  const out = suppressOpHook(() => {
    const permuted = new NdArray({
      data: g.data,
      shape: inverse.map((i) => g.shape[i]),
      strides: inverse.map((i) => g.strides[i]),
      offset: g.offset,
      base: g.base ?? g,
      readOnly: g.readOnly,
    });
    return clone(permuted);
  });
  return emitBackward('permute', 0, [g], out, { axes: [...axes], inverse });
}

export function expandDimsVjp(g: NdArray, inShape: Shape): NdArray {
  return emitBackward(
    'expandDims',
    0,
    [g],
    suppressOpHook(() => clone(reshape(g, inShape))),
  );
}

export function squeezeVjp(g: NdArray, inShape: Shape): NdArray {
  return emitBackward(
    'squeeze',
    0,
    [g],
    suppressOpHook(() => clone(reshape(g, inShape))),
  );
}
