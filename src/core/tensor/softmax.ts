/**
 * The softmax family, and the fused softmax + cross-entropy.
 *
 * Step 0.3 is built on three claims, and the signatures here are written so the
 * claims cannot be quietly violated later:
 *
 *  1. `softmaxVjp` takes `s`, the forward OUTPUT — never the input `x`. The
 *     backward genuinely does not need the input, which is why real frameworks
 *     cache the output instead. The signature is the lesson.
 *
 *  2. `logSoftmax` is computed with the LogSumExp identity, not as
 *     `log(softmax(x))`. The composed form takes the log of a possibly-tiny
 *     number; this one never does.
 *
 *  3. `crossEntropyVjp` is a closed form, `s - onehot(y)`, not a composition of
 *     a softmax backward with a log backward. That collapse is exactly why
 *     frameworks ship a fused operator.
 *
 * `softmaxSteps` exposes the intermediates so Step 0.3's four-stage playback
 * (F0.8.2) can drive itself from real values rather than from trace granularity.
 */

import { emit, suppressOpHook } from '../trace/hook.ts';
import { NdArray, formatShapeTuple, forEachIndex, size, zeros } from './ndarray.ts';
import { div, exp, log, max, mul, sub, sum } from './ops.ts';

export interface SoftmaxOptions {
  /**
   * Subtract the row max before exponentiating. On by default.
   *
   * Turning it off is not a footgun left lying around — it is the control for
   * Step 0.5's hazard 1, where the point is to watch exp() overflow to Inf.
   */
  subtractMax?: boolean;
}

function resolveAxis(axis: number, rank: number, label: string): number {
  const resolved = axis < 0 ? axis + rank : axis;
  if (!Number.isInteger(resolved) || resolved < 0 || resolved >= rank) {
    throw new Error(`${label}: axis ${axis} is out of range for a rank-${rank} array`);
  }
  return resolved;
}

export interface SoftmaxSteps {
  /** Row maxima, kept as an axis of extent 1. */
  rowMax: NdArray;
  /** x - max(x), the shifted logits. */
  shifted: NdArray;
  /** exp(shifted). */
  exponentials: NdArray;
  /** Sum of the exponentials along the axis, extent 1. */
  denominator: NdArray;
  /** The probabilities. */
  probs: NdArray;
}

/**
 * Softmax with every intermediate returned.
 *
 * Inner operations are suppressed from the trace: the caller wanted the stages,
 * and they are in the return value.
 */
export function softmaxSteps(x: NdArray, axis = -1, options: SoftmaxOptions = {}): SoftmaxSteps {
  const ax = resolveAxis(axis, x.shape.length, 'softmax');
  const subtractMax = options.subtractMax ?? true;

  return suppressOpHook(() => {
    const rowMax = max(x, ax, true);
    const shifted = subtractMax ? sub(x, rowMax) : x;
    const exponentials = exp(shifted);
    const denominator = sum(exponentials, ax, true);
    const probs = div(exponentials, denominator);
    return { rowMax, shifted, exponentials, denominator, probs };
  });
}

export function softmax(x: NdArray, axis = -1, options: SoftmaxOptions = {}): NdArray {
  const ax = resolveAxis(axis, x.shape.length, 'softmax');
  const { probs } = softmaxSteps(x, ax, options);
  emit({
    op: 'softmax',
    phase: 'forward',
    inputs: [x],
    output: probs,
    isView: false,
    didCopy: false,
    meta: { axis: ax, subtractMax: options.subtractMax ?? true },
  });
  return probs;
}

/**
 * log(softmax(x)) via LogSumExp: `z - log(sum(exp(z)))` where `z = x - max(x)`.
 *
 * Never forms a probability and then takes its log, so a vanishingly small
 * probability comes out as a large negative number rather than -Inf.
 */
export function logSoftmax(x: NdArray, axis = -1): NdArray {
  const ax = resolveAxis(axis, x.shape.length, 'logSoftmax');
  const out = suppressOpHook(() => {
    const shifted = sub(x, max(x, ax, true));
    return sub(shifted, log(sum(exp(shifted), ax, true)));
  });
  emit({
    op: 'logSoftmax',
    phase: 'forward',
    inputs: [x],
    output: out,
    isView: false,
    didCopy: false,
    meta: { axis: ax },
  });
  return out;
}

/**
 * The softmax VJP:
 *
 *     x̄_i = s_i (s̄_i - Σ_j s̄_j s_j)
 *
 * The subtracted term is a weighted average of the upstream gradient, so the
 * whole operation is a de-centring: whatever component of `s̄` is uniform across
 * the row cannot change the probabilities, and is removed.
 *
 * Takes `s`, the forward output. Not `x`. See the file header.
 */
export function softmaxVjp(s: NdArray, sBar: NdArray, axis = -1): NdArray {
  const ax = resolveAxis(axis, s.shape.length, 'softmaxVjp');
  const out = suppressOpHook(() => {
    const weightedMean = sum(mul(sBar, s), ax, true);
    return mul(s, sub(sBar, weightedMean));
  });
  emit({
    op: 'softmax',
    phase: 'backward',
    inputs: [sBar, s],
    output: out,
    isView: false,
    didCopy: false,
    meta: { axis: ax, wrt: 0 },
  });
  return out;
}

export function logSoftmaxVjp(g: NdArray, logProbs: NdArray, axis = -1): NdArray {
  const ax = resolveAxis(axis, logProbs.shape.length, 'logSoftmaxVjp');
  const out = suppressOpHook(() => sub(g, mul(exp(logProbs), sum(g, ax, true))));
  emit({
    op: 'logSoftmax',
    phase: 'backward',
    inputs: [g, logProbs],
    output: out,
    isView: false,
    didCopy: false,
    meta: { axis: ax, wrt: 0 },
  });
  return out;
}

export interface CrossEntropyResult {
  /** Mean negative log-likelihood over the N rows. */
  loss: number;
  /** softmax(logits) — what the backward needs. */
  probs: NdArray;
  /** log_softmax(logits) — kept because the loss is read straight off it. */
  logProbs: NdArray;
}

function checkClassificationShapes(
  logits: NdArray,
  targets: ArrayLike<number>,
  label: string,
): { rows: number; classes: number } {
  if (logits.shape.length !== 2) {
    throw new Error(
      `${label}: logits must be (N, V), got ${formatShapeTuple(logits.shape)}. ` +
        'Flatten the batch and time axes together before calling.',
    );
  }
  const rows = logits.shape[0];
  const classes = logits.shape[1];
  if (targets.length !== rows) {
    throw new Error(`${label}: ${targets.length} targets for ${rows} rows of logits`);
  }
  return { rows, classes };
}

/**
 * Cross-entropy straight from logits.
 *
 * Fused rather than composed, for the three reasons Step 0.3 lists: numerical
 * stability (no log of a probability), efficiency (the Jacobian is skipped
 * entirely), and memory (no intermediate to keep).
 */
export function crossEntropyFromLogits(
  logits: NdArray,
  targets: ArrayLike<number>,
): CrossEntropyResult {
  const { rows, classes } = checkClassificationShapes(logits, targets, 'crossEntropyFromLogits');

  const logProbs = suppressOpHook(() => logSoftmax(logits, -1));
  const probs = suppressOpHook(() => exp(logProbs));

  let total = 0;
  for (let i = 0; i < rows; i++) {
    const target = targets[i];
    if (!Number.isInteger(target) || target < 0 || target >= classes) {
      throw new Error(
        `crossEntropyFromLogits: target ${target} at row ${i} is outside [0, ${classes})`,
      );
    }
    total -=
      logProbs.data[logProbs.offset + i * logProbs.strides[0] + target * logProbs.strides[1]];
  }

  const loss = total / rows;
  emit({
    op: 'crossEntropy',
    phase: 'forward',
    inputs: [logits],
    output: probs,
    isView: false,
    didCopy: false,
    meta: { loss, rows, classes },
  });
  return { loss, probs, logProbs };
}

/**
 * The whole reason the fused operator exists:
 *
 *     ∂ℓ/∂x = (s - onehot(y)) / N
 *
 * One subtraction of 1 in the correct class's slot. No Jacobian, no chain rule
 * through the log — the softmax backward and the log backward cancel exactly.
 */
export function crossEntropyVjp(probs: NdArray, targets: ArrayLike<number>, gradLoss = 1): NdArray {
  const { rows } = checkClassificationShapes(probs, targets, 'crossEntropyVjp');

  const out = zeros(probs.shape);
  forEachIndex(probs, (flat, logical) => {
    out.data[logical] = probs.data[flat];
  });
  for (let i = 0; i < rows; i++) {
    out.data[i * out.strides[0] + targets[i] * out.strides[1]] -= 1;
  }
  const scale = gradLoss / rows;
  for (let i = 0; i < out.data.length; i++) out.data[i] *= scale;

  emit({
    op: 'crossEntropy',
    phase: 'backward',
    inputs: [probs],
    output: out,
    isView: false,
    didCopy: false,
    meta: { rows, gradLoss, wrt: 0 },
  });
  return out;
}

/**
 * The explicit (n, n) softmax Jacobian:  J_ij = s_i (δ_ij - s_j)
 *
 * Only ever built for the side-by-side comparison in Step 0.3 (F0.8.4). Real
 * code contracts against it without forming it — see `jacobian.ts` for the size
 * guard that makes the point.
 */
export function softmaxJacobian(s: NdArray): NdArray {
  if (s.shape.length !== 1) {
    throw new Error(
      `softmaxJacobian: expects a single probability vector (rank 1), got ${formatShapeTuple(s.shape)}`,
    );
  }
  const n = size(s);
  const out = zeros([n, n]);
  const values = new Float64Array(n);
  forEachIndex(s, (flat, logical) => {
    values[logical] = s.data[flat];
  });
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out.data[i * n + j] = values[i] * ((i === j ? 1 : 0) - values[j]);
    }
  }
  return out;
}
