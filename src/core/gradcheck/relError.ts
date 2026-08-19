/**
 * Relative error and the gradcheck verdict.
 *
 * Relative rather than absolute, per transformer_plan.md: an absolute threshold
 * is meaningless when gradient magnitudes span many orders of magnitude across
 * a network.
 */

import { NdArray, forEachZip, formatShapeTuple, shapesEqual, zeros } from '../tensor/ndarray.ts';
import { numericalGrad, DEFAULT_EPS } from './numericalGrad.ts';

/** Guards against 0/0 when both values are zero. */
const DENOMINATOR_FLOOR = 1e-12;

export interface RelErrorResult {
  max: number;
  /** Multi-index of the worst element, for the error heatmap in E0.7. */
  argmax: number[];
  perElement: NdArray;
}

export function relError(a: NdArray, b: NdArray): RelErrorResult {
  if (!shapesEqual(a.shape, b.shape)) {
    throw new Error(
      `relError: shape mismatch ${formatShapeTuple(a.shape)} vs ${formatShapeTuple(b.shape)}`,
    );
  }

  const perElement = zeros(a.shape);
  let worst = -1;
  let worstLogical = 0;

  forEachZip(a.shape, [a, b, perElement], (offsets, logical) => {
    const x = a.data[offsets[0]];
    const y = b.data[offsets[1]];

    let rel: number;
    if (x === y) {
      // Covers exact zeros and matching infinities.
      rel = 0;
    } else if (Number.isNaN(x) || Number.isNaN(y) || !Number.isFinite(x) || !Number.isFinite(y)) {
      // A NaN or a lone infinity is never "close"; surface it as maximal error
      // rather than letting it vanish into an arithmetic NaN.
      rel = Infinity;
    } else {
      rel = Math.abs(x - y) / (Math.abs(x) + Math.abs(y) + DENOMINATOR_FLOOR);
    }

    perElement.data[offsets[2]] = rel;
    if (rel > worst) {
      worst = rel;
      worstLogical = logical;
    }
  });

  return { max: worst < 0 ? 0 : worst, argmax: unravel(worstLogical, a.shape), perElement };
}

function unravel(logical: number, shape: readonly number[]): number[] {
  const idx = new Array<number>(shape.length).fill(0);
  let rest = logical;
  for (let i = shape.length - 1; i >= 0; i--) {
    idx[i] = rest % shape[i];
    rest = Math.floor(rest / shape[i]);
  }
  return idx;
}

export interface GradcheckResult {
  ok: boolean;
  maxRelErr: number;
  argmax: number[];
  numerical: NdArray;
  analytic: NdArray;
  perElement: NdArray;
  report: string;
}

export const DEFAULT_TOL = 1e-7;

/**
 * Compare a hand-written gradient against the central-difference estimate.
 *
 * When checking anything with a kink (relu, max, abs), keep the inputs away
 * from the non-differentiable point: a sample straddling zero makes the two
 * sides of the difference disagree for reasons that are not a bug.
 */
export function gradcheck(
  f: (x: NdArray) => number,
  x: NdArray,
  analytic: NdArray,
  tol: number = DEFAULT_TOL,
  eps: number = DEFAULT_EPS,
): GradcheckResult {
  const numerical = numericalGrad(f, x, eps);
  const { max, argmax, perElement } = relError(numerical, analytic);
  const ok = max < tol;

  const report = ok
    ? `max rel err ${max.toExponential(2)} < ${tol.toExponential(0)}`
    : `max rel err ${max.toExponential(2)} at index [${argmax.join(', ')}] ` +
      `(numerical ${valueAt(numerical, argmax).toExponential(6)}, ` +
      `analytic ${valueAt(analytic, argmax).toExponential(6)})`;

  return { ok, maxRelErr: max, argmax, numerical, analytic, perElement, report };
}

function valueAt(a: NdArray, idx: readonly number[]): number {
  let flat = a.offset;
  for (let i = 0; i < idx.length; i++) flat += idx[i] * a.strides[i];
  return a.data[flat];
}
