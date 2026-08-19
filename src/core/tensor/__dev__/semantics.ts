/**
 * Softmax, autograd and Jacobian semantics — T20.
 *
 * These checks are not generic plumbing: each one is a claim Step 0.3 or Step
 * 0.4 will make on screen. If one goes red, a lab is about to teach something
 * false.
 */

import { check, expect, group } from './harness.ts';
import { flatEquals, throws } from './assertions.ts';
import { NdArray, clone, copyFlat, fromNested, size, zeros } from '../ndarray.ts';
import { randn, randomInts, uniform } from '../random.ts';
import { add as addOp, matmul as matmulOp, max as maxOp, mulScalar, sum as sumOp } from '../ops.ts';
import {
  crossEntropyFromLogits,
  crossEntropyVjp,
  logSoftmax,
  logSoftmaxVjp,
  softmax,
  softmaxJacobian,
  softmaxSteps,
  softmaxVjp,
} from '../softmax.ts';
import * as A from '../autograd.ts';
import {
  JacobianTooLargeError,
  contractJacobian,
  jacobianByVjp,
  jacobianMemoryEstimate,
} from '../jacobian.ts';
import { gradcheck } from '../../gradcheck/relError.ts';
import { scalarize } from '../../gradcheck/numericalGrad.ts';

const TOL = 1e-7;

function awayFromZero(a: NdArray, margin = 0.5): NdArray {
  const out = clone(a);
  for (let i = 0; i < out.data.length; i++) {
    const v = out.data[i];
    out.data[i] = v >= 0 ? v + margin : v - margin;
  }
  return out;
}

// ── softmax ─────────────────────────────────────────────────────────────────

group('softmax · forward claims');

check('rows sum to exactly 1 (to float64 rounding)', () => {
  const s = softmax(randn([4, 7], 301), -1);
  const totals = sumOp(s, -1);
  for (const v of copyFlat(totals)) expect(Math.abs(v - 1) < 1e-15, `row sums to ${v}`);
  return '4 rows within 1e-15 of 1';
});

check('subtracting the max changes nothing — the claim F0.8.3 visualizes', () => {
  const x = randn([3, 6], 302);
  const base = copyFlat(softmax(x, -1));

  const residualAfterShift = (c: number): number => {
    const shifted = new NdArray({
      data: Float64Array.from(copyFlat(x), (v) => v + c),
      shape: [3, 6],
    });
    const moved = copyFlat(softmax(shifted, -1));
    let worst = 0;
    for (let i = 0; i < base.length; i++) worst = Math.max(worst, Math.abs(base[i] - moved[i]));
    return worst;
  };

  const small = residualAfterShift(10);
  const large = residualAfterShift(1000);

  // The invariance is exact in real arithmetic. In float64 it is limited by how
  // precisely x + c can be represented at all: one ulp near 1000 is about
  // 2.3e-13, near 10 it is about 1.8e-15. So the residual tracks the SHIFT, not
  // the softmax — which is why it grows by roughly two orders when c does.
  // F0.8.3 should show this rather than claim bit-identical output.
  expect(large < 1e-12, `shift of 1000 left a residual of ${large}`);
  expect(small < 1e-14, `shift of 10 left a residual of ${small}`);
  expect(small <= large, `residual should grow with the shift: ${small} vs ${large}`);

  return `residual scales with the shift: c=10 -> ${small.toExponential(1)}, c=1000 -> ${large.toExponential(1)}`;
});

check('without the max shift, logits at 800 overflow — with it, they do not', () => {
  const x = fromNested([[800, 801, 802]]);

  const naive = softmaxSteps(x, -1, { subtractMax: false });
  const naiveHasInf = Array.from(copyFlat(naive.exponentials)).some((v) => !Number.isFinite(v));
  const naiveHasNaN = Array.from(copyFlat(naive.probs)).some((v) => Number.isNaN(v));
  expect(naiveHasInf, 'exp(800) should overflow to Inf');
  expect(naiveHasNaN, 'Inf/Inf should produce NaN');

  const safe = softmax(x, -1);
  const safeValues = Array.from(copyFlat(safe));
  expect(safeValues.every(Number.isFinite), `stable path produced ${safeValues.join(', ')}`);
  const total = safeValues.reduce((acc, v) => acc + v, 0);
  expect(Math.abs(total - 1) < 1e-15, `stable path sums to ${total}`);
  return 'naive: Inf then NaN · stable: finite, sums to 1';
});

check('logSoftmax matches log(softmax) where the latter is well behaved', () => {
  const x = randn([3, 5], 303);
  const viaLogSumExp = copyFlat(logSoftmax(x, -1));
  const viaCompose = Array.from(copyFlat(softmax(x, -1)), Math.log);
  for (let i = 0; i < viaLogSumExp.length; i++) {
    expect(
      Math.abs(viaLogSumExp[i] - viaCompose[i]) < 1e-12,
      `element ${i}: ${viaLogSumExp[i]} vs ${viaCompose[i]}`,
    );
  }
});

check('logSoftmax stays finite exactly where log(softmax) collapses to -Inf', () => {
  const x = fromNested([[0, -900]]);
  const composed = Math.log(copyFlat(softmax(x, -1))[1]);
  const direct = copyFlat(logSoftmax(x, -1))[1];
  expect(composed === -Infinity, `log(softmax) should underflow, got ${composed}`);
  expect(Number.isFinite(direct), `logSoftmax should stay finite, got ${direct}`);
  expect(Math.abs(direct - -900) < 1e-9, `expected about -900, got ${direct}`);
  return `log(softmax) = -Inf · logSoftmax = ${direct.toFixed(3)}`;
});

group('softmax · backward claims');

check('softmaxVjp gradchecks', () => {
  const x = randn([3, 5], 304);
  const forward = (v: NdArray) => softmax(v, -1);
  const out = forward(x);
  const upstream = awayFromZero(randn(out.shape, 404));
  const analytic = softmaxVjp(out, upstream, -1);
  const result = gradcheck(scalarize(forward, upstream), x, analytic, TOL);
  expect(result.ok, result.report);
  return result.report;
});

check('logSoftmaxVjp gradchecks', () => {
  const x = randn([3, 5], 305);
  const forward = (v: NdArray) => logSoftmax(v, -1);
  const out = forward(x);
  const upstream = awayFromZero(randn(out.shape, 405));
  const analytic = logSoftmaxVjp(upstream, out, -1);
  const result = gradcheck(scalarize(forward, upstream), x, analytic, TOL);
  expect(result.ok, result.report);
  return result.report;
});

check('the fused CE backward is literally (s - onehot(y)) / N', () => {
  const logits = randn([4, 6], 306);
  const targets = randomInts(4, 6, 307);
  const { probs } = crossEntropyFromLogits(logits, targets);
  const grad = crossEntropyVjp(probs, targets, 1);

  const expected = Array.from(copyFlat(probs));
  for (let i = 0; i < 4; i++) expected[i * 6 + targets[i]] -= 1;
  flatEquals(
    grad,
    expected.map((v) => v / 4),
    1e-15,
  );
  return 'one slot per row reduced by 1, scaled by 1/N';
});

check('fused CE gradient equals autograd through logSoftmax', () => {
  const logits = randn([4, 6], 308);
  const targets = randomInts(4, 6, 309);

  const { probs } = crossEntropyFromLogits(logits, targets);
  const fused = crossEntropyVjp(probs, targets, 1);

  // the composed route: loss = -mean(logSoftmax(x)[i, y_i])
  const xv = A.variable(logits, { label: 'logits' });
  const logProbs = A.logSoftmax(xv, -1);
  const picker = zeros([4, 6]);
  for (let i = 0; i < 4; i++) picker.data[i * 6 + targets[i]] = -1 / 4;
  const composedLoss = A.sum(A.mul(logProbs, A.constant(picker)));
  A.backward(composedLoss);

  flatEquals(A.gradOf(xv), Array.from(copyFlat(fused)), 1e-12);
  return 'fused and composed agree to 1e-12';
});

check('crossEntropy gradchecks end to end', () => {
  const logits = randn([4, 6], 310);
  const targets = randomInts(4, 6, 311);
  const analytic = crossEntropyVjp(crossEntropyFromLogits(logits, targets).probs, targets, 1);
  const f = (v: NdArray) => crossEntropyFromLogits(v, targets).loss;
  const result = gradcheck(f, logits, analytic, TOL);
  expect(result.ok, result.report);
  return result.report;
});

// ── autograd ────────────────────────────────────────────────────────────────

group('autograd · tape');

check('y = x*x + x gives 2x + 1 (fails under assignment, passes under accumulation)', () => {
  const x = A.variable(fromNested([1, 2, 3]), { label: 'x' });
  const y = A.sum(A.add(A.mul(x, x), x));
  A.backward(y);
  flatEquals(A.gradOf(x), [3, 5, 7], 1e-12);
  return 'x reached twice, both contributions summed';
});

check('a diamond graph accumulates on both branches', () => {
  const x = A.variable(randn([3, 4], 321), { label: 'x' });
  const left = A.mulScalar(x, 3);
  const right = A.exp(x);
  const merged = A.sum(A.mul(left, right));
  A.backward(merged);

  // d/dx [3x * exp(x)] = 3 exp(x) + 3x exp(x)
  const xs = copyFlat(x.value);
  const expected = Array.from(xs, (v) => 3 * Math.exp(v) + 3 * v * Math.exp(v));
  flatEquals(A.gradOf(x), expected, 1e-10);
});

check('relu(xW1+b1) -> W2,b2 -> softmax-CE gradchecks on all five variables', () => {
  const N = 4;
  const D = 3;
  const H = 5;
  const V = 6;

  // relu has a kink: a pre-activation sitting near 0 makes the central
  // difference straddle it, which reads as an error but is not one. Search for
  // a seed whose pre-activations are all comfortably off the kink.
  let seed = 0;
  let x = zeros([N, D]);
  let W1 = zeros([D, H]);
  let b1 = zeros([H]);
  for (let candidate = 1; candidate < 200; candidate++) {
    x = randn([N, D], candidate);
    W1 = randn([D, H], candidate + 1000);
    b1 = randn([H], candidate + 2000);
    const pre = addOp(matmulOp(x, W1), b1);
    const worst = Math.min(...Array.from(copyFlat(pre), Math.abs));
    if (worst > 0.05) {
      seed = candidate;
      break;
    }
  }
  expect(seed !== 0, 'no seed produced pre-activations clear of the relu kink');

  const W2 = randn([H, V], seed + 3000);
  const b2 = randn([V], seed + 4000);
  const targets = randomInts(N, V, seed + 5000);

  const build = (xv: NdArray, w1: NdArray, bb1: NdArray, w2: NdArray, bb2: NdArray) => {
    const vx = A.variable(xv, { label: 'x' });
    const vW1 = A.variable(w1, { label: 'W1' });
    const vb1 = A.variable(bb1, { label: 'b1' });
    const vW2 = A.variable(w2, { label: 'W2' });
    const vb2 = A.variable(bb2, { label: 'b2' });
    const hidden = A.relu(A.add(A.matmul(vx, vW1), vb1));
    const logits = A.add(A.matmul(hidden, vW2), vb2);
    const loss = A.crossEntropy(logits, targets);
    return { loss, vars: { vx, vW1, vb1, vW2, vb2 } };
  };

  const { loss, vars } = build(x, W1, b1, W2, b2);
  A.backward(loss);

  const scalarLoss =
    (which: 'x' | 'W1' | 'b1' | 'W2' | 'b2') =>
    (v: NdArray): number =>
      build(
        which === 'x' ? v : x,
        which === 'W1' ? v : W1,
        which === 'b1' ? v : b1,
        which === 'W2' ? v : W2,
        which === 'b2' ? v : b2,
      ).loss.value.data[0];

  const checks: Array<[string, NdArray, NdArray]> = [
    ['x', x, A.gradOf(vars.vx)],
    ['W1', W1, A.gradOf(vars.vW1)],
    ['b1', b1, A.gradOf(vars.vb1)],
    ['W2', W2, A.gradOf(vars.vW2)],
    ['b2', b2, A.gradOf(vars.vb2)],
  ];

  let worst = 0;
  for (const [name, value, analytic] of checks) {
    const result = gradcheck(
      scalarLoss(name as 'x' | 'W1' | 'b1' | 'W2' | 'b2'),
      value,
      analytic,
      TOL,
    );
    expect(result.ok, `${name}: ${result.report}`);
    worst = Math.max(worst, result.maxRelErr);
  }
  return `seed ${seed}, worst of 5 variables ${worst.toExponential(2)}`;
});

check('gradients have their variables shapes, including the broadcast bias', () => {
  const x = A.variable(randn([4, 3], 331));
  const w = A.variable(randn([3, 5], 332));
  const b = A.variable(randn([5], 333));
  const loss = A.sum(A.relu(A.add(A.matmul(x, w), b)));
  A.backward(loss);
  expect(A.gradOf(b).shape.join(',') === '5', `db is (${A.gradOf(b).shape.join(',')})`);
  expect(A.gradOf(w).shape.join(',') === '3,5', `dW is (${A.gradOf(w).shape.join(',')})`);
  expect(A.gradOf(x).shape.join(',') === '4,3', `dx is (${A.gradOf(x).shape.join(',')})`);
  return 'bias gradient summed back from (4,5) to (5,)';
});

check('backward refuses a non-scalar root', () => {
  const x = A.variable(randn([2, 3], 334));
  throws(() => A.backward(A.relu(x)), 'must be a scalar');
});

check('a VJP returning an unsummable shape names the offending op', () => {
  const x = A.variable(randn([3], 335));
  const broken = new A.Var({
    value: randn([3], 336),
    op: 'deliberatelyBroken',
    parents: [x],
    backwardFn: () => [zeros([5])],
  });
  throws(() => A.backward(A.sum(broken)), "op 'deliberatelyBroken'");
});

check('zeroGrad clears the whole reachable graph', () => {
  const x = A.variable(randn([2, 2], 337));
  const loss = A.sum(A.mul(x, x));
  A.backward(loss);
  expect(x.grad !== null, 'gradient should exist after backward');
  A.zeroGrad(loss);
  expect(x.grad === null, 'gradient should be cleared');
});

check('topoSort places every parent before its child', () => {
  const x = A.variable(randn([2], 338));
  const a = A.exp(x);
  const b = A.mul(a, x);
  const order = A.topoSort(A.sum(b));
  const position = new Map(order.map((v, i) => [v.id, i]));
  for (const v of order) {
    for (const parent of v.parents) {
      expect(position.get(parent.id)! < position.get(v.id)!, `parent ${parent.op} after ${v.op}`);
    }
  }
  return `${order.length} nodes ordered`;
});

// ── Jacobian ────────────────────────────────────────────────────────────────

group('Jacobian · the claim F0.8.4 renders');

check('softmaxJacobian contracted with sBar equals softmaxVjp', () => {
  const x = randn([8], 341);
  const s = softmax(x, -1);
  const upstream = randn([8], 342);

  const viaJacobian = contractJacobian(softmaxJacobian(s), upstream);
  const viaVjp = softmaxVjp(s, upstream, -1);

  let worst = 0;
  const a = copyFlat(viaJacobian);
  const b = copyFlat(viaVjp);
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / (Math.abs(a[i]) + Math.abs(b[i]) + 1e-12));
  }
  expect(worst < 1e-12, `max relative difference ${worst}`);
  return `(8,8) Jacobian vs 2-line VJP agree to ${worst.toExponential(1)}`;
});

check('the closed-form Jacobian matches one built from repeated VJP passes', () => {
  const x = randn([6], 343);
  const s = softmax(x, -1);
  const closed = softmaxJacobian(s);
  const built = jacobianByVjp((v) => A.softmax(v, -1), x);
  flatEquals(built, Array.from(copyFlat(closed)), 1e-12);
  return '6x6, both routes agree to 1e-12';
});

check('the softmax Jacobian is symmetric with the documented diagonal', () => {
  const s = softmax(randn([5], 344), -1);
  const j = softmaxJacobian(s);
  const values = copyFlat(s);
  for (let i = 0; i < 5; i++) {
    const diagonal = j.data[i * 5 + i];
    expect(
      Math.abs(diagonal - values[i] * (1 - values[i])) < 1e-15,
      `diagonal ${i}: ${diagonal} vs s_i(1-s_i)`,
    );
    for (let k = 0; k < 5; k++) {
      expect(Math.abs(j.data[i * 5 + k] - j.data[k * 5 + i]) < 1e-15, `asymmetry at ${i},${k}`);
    }
  }
});

check('n = 17 is refused, and the error carries the numbers the UI renders', () => {
  try {
    jacobianByVjp((v) => A.softmax(v, -1), uniform([17], -1, 1, 345));
  } catch (err) {
    expect(err instanceof JacobianTooLargeError, `wrong error type: ${String(err)}`);
    const e = err as JacobianTooLargeError;
    expect(e.n === 17, `n should be 17, got ${e.n}`);
    expect(e.elements === 289, `elements should be 289, got ${e.elements}`);
    expect(e.bytes === 289 * 8, `bytes should be ${289 * 8}, got ${e.bytes}`);
    return `refused with n=17, 289 elements`;
  }
  throw new Error('expected JacobianTooLargeError');
});

check('the memory estimate is what Step 0.3 puts on screen', () => {
  const small = jacobianMemoryEstimate(4);
  expect(small.elements === 16 && small.bytes === 64, `got ${small.elements}/${small.bytes}`);

  const seq = jacobianMemoryEstimate(4096);
  expect(seq.elements === 16777216, `got ${seq.elements}`);
  expect(seq.human === '64.0 MB', `got ${seq.human}`);

  const vocab = jacobianMemoryEstimate(50000);
  expect(vocab.human === '9.31 GB', `got ${vocab.human}`);
  return `n=4096 -> ${seq.human} · n=50000 -> ${vocab.human}`;
});

check('max over a softmax row still behaves (guards the reduction path)', () => {
  const s = softmax(randn([3, 4], 346), -1);
  const rowMax = maxOp(s, -1);
  const doubled = mulScalar(rowMax, 2);
  expect(size(doubled) === 3, `expected 3 rows, got ${size(doubled)}`);
  for (const v of copyFlat(rowMax)) expect(v > 0 && v <= 1, `row max ${v} outside (0, 1]`);
});
