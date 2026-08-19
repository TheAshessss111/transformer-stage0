/**
 * Broadcasting, and its dual.
 *
 * transformer_plan.md 规则 5: forward broadcast ↔ backward sum. So `unbroadcast`
 * is not a helper, it *is* the backward of broadcasting, and every binary op's
 * VJP ends with a call to it.
 *
 * The trailing shape assertion in `unbroadcast` catches essentially every
 * broadcast-VJP mistake — the category the plan calls 最容易错的地方 (Step 1.3).
 *
 * The summation is written out here rather than delegated to ops.sum: ops.ts
 * depends on this module, so the reverse import would be a cycle.
 */

import { emit } from '../trace/hook.ts';
import {
  NdArray,
  formatShapeTuple,
  forEachIndex,
  shapesEqual,
  size,
  zeros,
  type Shape,
} from './ndarray.ts';

/** NumPy broadcasting: right-align, each extent must match or be 1. */
export function broadcastShapes(...shapes: readonly Shape[]): number[] {
  if (shapes.length === 0) return [];

  const rank = Math.max(...shapes.map((s) => s.length));
  const out = new Array<number>(rank).fill(1);

  for (let i = 0; i < rank; i++) {
    const axisFromRight = rank - 1 - i;
    let extent = 1;
    for (const shape of shapes) {
      const idx = shape.length - 1 - i;
      if (idx < 0) continue;
      const d = shape[idx];
      if (d === 1) continue;
      if (extent === 1) {
        extent = d;
      } else if (extent !== d) {
        throw new Error(
          `broadcast: incompatible extents ${extent} and ${d} on axis ${axisFromRight} ` +
            `(counting from the right: ${i}) of shapes ${shapes.map(formatShapeTuple).join(' and ')}`,
        );
      }
    }
    out[axisFromRight] = extent;
  }

  return out;
}

/**
 * Expand `a` to `shape` with **stride-0 axes** — the real NumPy mechanism, and a
 * teaching artifact in its own right: a stride of 0 means "read the same element
 * again", which is why Step 0.1's memory strip draws that axis as an arrow that
 * does not advance.
 *
 * The result is marked read-only: one write through a stride-0 axis would land
 * on many logical positions at once.
 */
export function broadcastToQuiet(a: NdArray, shape: Shape): NdArray {
  const rank = shape.length;
  if (rank < a.shape.length) {
    throw new Error(
      `broadcastTo: cannot broadcast ${formatShapeTuple(a.shape)} to the lower-rank ${formatShapeTuple(shape)}`,
    );
  }

  const strides = new Array<number>(rank).fill(0);
  const lead = rank - a.shape.length;
  let introducedZeroStride = lead > 0;

  for (let i = 0; i < a.shape.length; i++) {
    const outAxis = lead + i;
    const from = a.shape[i];
    const to = shape[outAxis];
    if (from === to) {
      strides[outAxis] = a.strides[i];
    } else if (from === 1) {
      strides[outAxis] = 0;
      if (to !== 1) introducedZeroStride = true;
    } else {
      throw new Error(
        `broadcastTo: cannot stretch extent ${from} to ${to} on axis ${outAxis} ` +
          `(${formatShapeTuple(a.shape)} → ${formatShapeTuple(shape)})`,
      );
    }
  }

  const out = new NdArray({
    data: a.data,
    shape: [...shape],
    strides,
    offset: a.offset,
    base: a.base ?? a,
    readOnly: a.readOnly || introducedZeroStride,
  });

  return out;
}

/**
 * Public broadcast: same view, but reported to the trace.
 *
 * `broadcastToQuiet` exists because matmul broadcasts its batch dimensions as
 * an implementation detail; surfacing that as a separate trace step would bury
 * the actual matrix multiply in noise. Broadcasts the *user* wrote go through
 * this function and do show up.
 */
export function broadcastTo(a: NdArray, shape: Shape): NdArray {
  const out = broadcastToQuiet(a, shape);
  emit({
    op: 'broadcastTo',
    phase: 'forward',
    inputs: [a],
    output: out,
    isView: true,
    didCopy: false,
    meta: { from: [...a.shape], to: [...shape], strides: [...out.strides] },
  });
  return out;
}

/** True when `a` can be broadcast to `shape` without copying. */
export function canBroadcastTo(a: Shape, shape: Shape): boolean {
  if (shape.length < a.length) return false;
  const lead = shape.length - a.length;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== shape[lead + i] && a[i] !== 1) return false;
  }
  return true;
}

/**
 * The backward of broadcasting: sum `g` back down to `targetShape`.
 *
 *   1. sum away the leading axes the target does not have
 *   2. sum (keeping the axis) wherever the target extent is 1 but g's is not
 *
 * The final shape check is deliberate and unconditional — it is the cheapest
 * bug net in the engine.
 */
export function unbroadcast(g: NdArray, targetShape: Shape): NdArray {
  if (shapesEqual(g.shape, targetShape)) return g;

  if (!canBroadcastTo(targetShape, g.shape)) {
    throw new Error(
      `unbroadcast: ${formatShapeTuple(targetShape)} is not broadcastable to the gradient's ` +
        `${formatShapeTuple(g.shape)} — the forward pass could not have produced this pairing`,
    );
  }

  const out = zeros(targetShape);
  const lead = g.shape.length - targetShape.length;
  const targetIdx = new Array<number>(targetShape.length).fill(0);

  forEachIndex(g, (flat, _logical, idx) => {
    for (let i = 0; i < targetShape.length; i++) {
      targetIdx[i] = targetShape[i] === 1 ? 0 : idx[lead + i];
    }
    let dst = 0;
    for (let i = 0; i < targetShape.length; i++) dst += targetIdx[i] * out.strides[i];
    out.data[dst] += g.data[flat];
  });

  if (!shapesEqual(out.shape, targetShape)) {
    throw new Error(
      `unbroadcast: produced ${formatShapeTuple(out.shape)} but the variable is ${formatShapeTuple(targetShape)}`,
    );
  }

  emit({
    op: 'unbroadcast',
    phase: 'backward',
    inputs: [g],
    output: out,
    isView: false,
    didCopy: true,
    copiedElements: size(g),
    meta: { from: [...g.shape], to: [...targetShape] },
  });
  return out;
}
