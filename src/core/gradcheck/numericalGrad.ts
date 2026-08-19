/**
 * Central-difference numerical gradients — transformer_plan.md 纪律二.
 *
 * Pulled forward from E0.7 into E0.2 (impl plan section 0.1): E0.2 hand-writes
 * about fifteen VJPs, and without this it would have no way to know whether any
 * of them are right until three epics later.
 *
 * The four details the plan calls out, all load-bearing:
 *   float64          — float32 rounding drowns the difference signal
 *   central diff     — error is O(eps^2), forward difference is only O(eps)
 *   eps = 1e-5       — larger is truncation-dominated, smaller is rounding-dominated
 *   relative error   — see relError.ts
 */

import { suppressOpHook } from '../trace/hook.ts';
import { NdArray, clone, size, zeros } from '../tensor/ndarray.ts';
import { mul, sum } from '../tensor/ops.ts';

export const DEFAULT_EPS = 1e-5;

/**
 * d(f)/dx by central difference, one element at a time.
 *
 * `f` must be a scalar-valued function of the array it is handed. The working
 * copy passed to `f` is contiguous and owned by this function, so `f` may read
 * it freely but must not retain it.
 */
export function numericalGrad(
  f: (x: NdArray) => number,
  x: NdArray,
  eps: number = DEFAULT_EPS,
): NdArray {
  return suppressOpHook(() => {
    const work = clone(x);
    const grad = zeros(x.shape);
    const n = size(x);

    for (let i = 0; i < n; i++) {
      const original = work.data[i];

      work.data[i] = original + eps;
      const plus = f(work);

      work.data[i] = original - eps;
      const minus = f(work);

      work.data[i] = original;
      grad.data[i] = (plus - minus) / (2 * eps);
    }

    return grad;
  });
}

/**
 * Turn a tensor-valued forward pass into the scalar function gradcheck needs,
 * by contracting its output against a fixed random upstream gradient:
 *
 *     f(x) = sum(forward(x) * upstream)
 *
 * Then d(f)/dx is exactly the VJP of `forward` at `upstream`, which is the thing
 * being tested. The plan spells this out as the last bullet of 纪律二.
 */
export function scalarize(
  forward: (x: NdArray) => NdArray,
  upstream: NdArray,
): (x: NdArray) => number {
  return (x: NdArray) => {
    const out = forward(x);
    return sum(mul(out, upstream)).data[0];
  };
}
