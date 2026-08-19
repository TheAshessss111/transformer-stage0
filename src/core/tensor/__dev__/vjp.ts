/**
 * Gradient checks for every hand-written backward — T16.
 *
 * This is the gate for the rest of E0.2: nothing downstream is worth building
 * on a red harness. Each check contracts the forward output against a fixed
 * random upstream gradient, compares the VJP against a central-difference
 * estimate, and reports the max relative error.
 *
 * Inputs are chosen to stay away from non-differentiable points (relu, abs,
 * max) and away from domain edges (log, sqrt, div), because a sample straddling
 * a kink makes the two sides of the difference disagree for reasons that are
 * not a bug.
 */

import { check, expect, group } from './harness.ts';
import { flatEquals } from './assertions.ts';
import { NdArray, clone, fromNested, size, type Shape } from '../ndarray.ts';
import { randn, uniform } from '../random.ts';
import {
  add,
  div,
  exp,
  log,
  matmul,
  max,
  maximum,
  mean,
  min,
  mul,
  mulScalar,
  neg,
  abs as absOp,
  positiveMask,
  relu,
  sqrt,
  sub,
  sum,
  type Axis,
} from '../ops.ts';
import { expandDims, permute, reshape, squeeze } from '../shape.ts';
import {
  absVjp,
  addVjp,
  divVjp,
  expVjp,
  expandDimsVjp,
  logVjp,
  matmulVjp,
  maxVjp,
  maximumVjp,
  meanVjp,
  minVjp,
  mulScalarVjp,
  mulVjp,
  negVjp,
  permuteVjp,
  reluVjp,
  reshapeVjp,
  sqrtVjp,
  squeezeVjp,
  subVjp,
  sumVjp,
} from '../vjp.ts';
import { gradcheck } from '../../gradcheck/relError.ts';
import { scalarize } from '../../gradcheck/numericalGrad.ts';

const TOL = 1e-7;

/** Push samples away from 0 so kinked functions are sampled cleanly. */
function awayFromZero(a: NdArray, margin = 0.4): NdArray {
  const out = clone(a);
  for (let i = 0; i < out.data.length; i++) {
    const v = out.data[i];
    out.data[i] = v >= 0 ? v + margin : v - margin;
  }
  return out;
}

/**
 * Upstream gradients, drawn away from zero.
 *
 * A near-zero upstream component makes the corresponding gradient component
 * near-zero too, and *relative* error on a near-zero quantity is dominated by
 * finite-difference cancellation noise rather than by anything the VJP did.
 * Such components carry no information about correctness while contributing
 * almost all of the reported error, so they are excluded by construction —
 * otherwise the harness's verdict would depend on the seed.
 */
function upstreamFor(shape: Shape, seed: number): NdArray {
  return awayFromZero(randn(shape, seed), 0.5);
}

function gcUnary(
  label: string,
  x: NdArray,
  forward: (v: NdArray) => NdArray,
  backward: (g: NdArray, v: NdArray, out: NdArray) => NdArray,
  upstreamSeed: number,
  tol = TOL,
): string {
  const out = forward(x);
  const upstream = upstreamFor(out.shape, upstreamSeed);
  const analytic = backward(upstream, x, out);
  const result = gradcheck(scalarize(forward, upstream), x, analytic, tol);
  expect(result.ok, `${label}: ${result.report}`);
  return result.report;
}

function gcBinary(
  label: string,
  a: NdArray,
  b: NdArray,
  forward: (x: NdArray, y: NdArray) => NdArray,
  backward: (g: NdArray, x: NdArray, y: NdArray) => [NdArray, NdArray],
  upstreamSeed: number,
  tol = TOL,
): string {
  const out = forward(a, b);
  const upstream = upstreamFor(out.shape, upstreamSeed);
  const [da, db] = backward(upstream, a, b);

  expect(
    da.shape.join(',') === a.shape.join(','),
    `${label}: gradient wrt a has shape (${da.shape.join(',')}), variable is (${a.shape.join(',')})`,
  );
  expect(
    db.shape.join(',') === b.shape.join(','),
    `${label}: gradient wrt b has shape (${db.shape.join(',')}), variable is (${b.shape.join(',')})`,
  );

  const ra = gradcheck(
    scalarize((v) => forward(v, b), upstream),
    a,
    da,
    tol,
  );
  const rb = gradcheck(
    scalarize((v) => forward(a, v), upstream),
    b,
    db,
    tol,
  );
  expect(ra.ok, `${label} wrt a: ${ra.report}`);
  expect(rb.ok, `${label} wrt b: ${rb.report}`);
  return `a ${ra.maxRelErr.toExponential(1)} · b ${rb.maxRelErr.toExponential(1)}`;
}

// ── elementwise unary ───────────────────────────────────────────────────────

group('VJP · elementwise unary');

check('neg', () => gcUnary('neg', randn([3, 4], 101), neg, (g) => negVjp(g), 201));

check('exp (backward uses the OUTPUT, not the input)', () =>
  gcUnary('exp', uniform([3, 4], -1.5, 1.5, 102), exp, (g, _x, out) => expVjp(g, out), 202),
);

check('log (positive domain)', () =>
  gcUnary('log', uniform([3, 4], 0.4, 2.5, 103), log, (g, x) => logVjp(g, x), 203),
);

check('sqrt (positive domain, backward via the output)', () =>
  gcUnary('sqrt', uniform([3, 4], 0.4, 2.5, 104), sqrt, (g, _x, out) => sqrtVjp(g, out), 204),
);

check('relu (samples kept off the kink)', () =>
  gcUnary(
    'relu',
    awayFromZero(randn([4, 5], 105)),
    relu,
    (g, x) => reluVjp(g, positiveMask(x)),
    205,
  ),
);

check('abs (samples kept off the kink)', () =>
  gcUnary('abs', awayFromZero(randn([4, 5], 106)), absOp, (g, x) => absVjp(g, x), 206),
);

check('mulScalar', () =>
  gcUnary(
    'mulScalar',
    randn([3, 3], 107),
    (v) => mulScalar(v, -2.5),
    (g) => mulScalarVjp(g, -2.5),
    207,
  ),
);

// ── elementwise binary, with and without broadcasting ───────────────────────

group('VJP · elementwise binary');

check('add, same shape', () =>
  gcBinary(
    'add',
    randn([3, 4], 111),
    randn([3, 4], 112),
    add,
    (g, x, y) => addVjp(g, x.shape, y.shape),
    211,
  ),
);

check('add, broadcasting (3,1) + (2,3,4)', () =>
  gcBinary(
    'add',
    randn([3, 1], 113),
    randn([2, 3, 4], 114),
    add,
    (g, x, y) => addVjp(g, x.shape, y.shape),
    212,
  ),
);

check('sub, broadcasting (4,) - (3,4)', () =>
  gcBinary(
    'sub',
    randn([4], 115),
    randn([3, 4], 116),
    sub,
    (g, x, y) => subVjp(g, x.shape, y.shape),
    213,
  ),
);

check('mul, broadcasting (2,1,4) * (2,3,4)', () =>
  gcBinary('mul', randn([2, 1, 4], 117), randn([2, 3, 4], 118), mul, mulVjp, 214),
);

check('div, broadcasting with a denominator away from zero', () =>
  gcBinary('div', randn([2, 3], 119), uniform([1, 3], 0.6, 2.0, 120), div, divVjp, 215),
);

check('maximum, no ties', () =>
  gcBinary('maximum', randn([3, 4], 121), randn([3, 4], 122), maximum, maximumVjp, 216),
);

// ── matmul ──────────────────────────────────────────────────────────────────

group('VJP · matmul');

check('matmul 2-D', () =>
  gcBinary('matmul', randn([3, 4], 131), randn([4, 5], 132), matmul, matmulVjp, 231),
);

check('matmul with broadcast batch dims (2,1,3,4) @ (1,5,4,2)', () =>
  gcBinary('matmul', randn([2, 1, 3, 4], 133), randn([1, 5, 4, 2], 134), matmul, matmulVjp, 232),
);

check('matmul gradients have the shapes plan rule 1 predicts', () => {
  const a = randn([3, 4], 135);
  const b = randn([4, 5], 136);
  const g = randn([3, 5], 137);
  const [da, db] = matmulVjp(g, a, b);
  expect(da.shape.join(',') === '3,4', `dA is (${da.shape.join(',')}), expected (3,4)`);
  expect(db.shape.join(',') === '4,5', `dB is (${db.shape.join(',')}), expected (4,5)`);
  return 'dA = g @ Bt -> (3,4), dB = At @ g -> (4,5)';
});

// ── reductions ──────────────────────────────────────────────────────────────

group('VJP · reductions');

function gcReduction(
  label: string,
  forwardOp: (v: NdArray, axis: Axis, keepdims: boolean) => NdArray,
  backwardOp: (g: NdArray, inShape: Shape, axis: Axis, keepdims: boolean) => NdArray,
  x: NdArray,
  axis: Axis,
  keepdims: boolean,
  seed: number,
): string {
  const forward = (v: NdArray) => forwardOp(v, axis, keepdims);
  return gcUnary(label, x, forward, (g) => backwardOp(g, x.shape, axis, keepdims), seed);
}

check('sum over every axis spec, both keepdims', () => {
  const x = randn([2, 3, 4], 141);
  const reports: string[] = [];
  const specs: Array<[Axis, boolean]> = [
    [null, false],
    [0, false],
    [1, true],
    [-1, false],
    [[0, 2], false],
    [[0, 2], true],
  ];
  for (const [axis, keepdims] of specs) {
    reports.push(gcReduction('sum', sum, sumVjp, x, axis, keepdims, 241));
  }
  return `${specs.length} specs, worst ${reports.length} checked`;
});

check('mean, axis=1 keepdims=false', () =>
  gcReduction('mean', mean, meanVjp, randn([2, 3, 4], 142), 1, false, 242),
);

check('mean, axis=[0,2] keepdims=true', () =>
  gcReduction('mean', mean, meanVjp, randn([2, 3, 4], 143), [0, 2], true, 243),
);

check('max, axis=1 (no ties in random float64)', () => {
  const x = randn([3, 5], 144);
  const forward = (v: NdArray) => max(v, 1, false);
  return gcUnary('max', x, forward, (g, v, out) => maxVjp(g, v, out, 1, false), 244);
});

check('min, axis=0 keepdims=true', () => {
  const x = randn([3, 5], 145);
  const forward = (v: NdArray) => min(v, 0, true);
  return gcUnary('min', x, forward, (g, v, out) => minVjp(g, v, out, 0, true), 245);
});

check('max splits ties evenly', () => {
  //             row 0 has a 3-way tie on 7, row 1 has a clear winner
  const x = fromNested([
    [7, 7, 7, 2],
    [1, 5, 3, 2],
  ]);
  const out = max(x, 1, false);
  const g = fromNested([10, 4]);
  const grad = maxVjp(g, x, out, 1, false);
  flatEquals(grad, [10 / 3, 10 / 3, 10 / 3, 0, 0, 4, 0, 0], 1e-12);
  return 'three-way tie splits 10 into 3.333 each';
});

// ── shape ops ───────────────────────────────────────────────────────────────

group('VJP · shape ops');

check('reshape', () => {
  const x = randn([2, 6], 151);
  return gcUnary(
    'reshape',
    x,
    (v) => reshape(v, [3, 4]),
    (g) => reshapeVjp(g, x.shape),
    251,
  );
});

check('permute is inverted by its inverse permutation', () => {
  const x = randn([2, 3, 4], 152);
  const axes = [2, 0, 1];
  return gcUnary(
    'permute',
    x,
    (v) => permute(v, axes),
    (g) => permuteVjp(g, axes),
    252,
  );
});

check('expandDims / squeeze round-trip', () => {
  const x = randn([2, 3], 153);
  const a = gcUnary(
    'expandDims',
    x,
    (v) => expandDims(v, 1),
    (g) => expandDimsVjp(g, x.shape),
    253,
  );
  const y = randn([2, 1, 3], 154);
  const b = gcUnary(
    'squeeze',
    y,
    (v) => squeeze(v, 1),
    (g) => squeezeVjp(g, y.shape),
    254,
  );
  return `${a} · ${b}`;
});

// ── the invariant that catches broadcast mistakes ───────────────────────────

group('VJP · gradient-shape invariant');

check('every broadcasting VJP returns gradients shaped like their variables', () => {
  const cases: Array<[number[], number[]]> = [
    [
      [3, 1],
      [2, 3, 4],
    ],
    [[4], [3, 4]],
    [
      [1, 1],
      [5, 6],
    ],
    [
      [2, 1, 4],
      [2, 5, 4],
    ],
  ];
  for (const [sa, sb] of cases) {
    const a = randn(sa, 161);
    const b = randn(sb, 162);
    const out = mul(a, b);
    const g = randn(out.shape, 163);
    const [da, db] = mulVjp(g, a, b);
    expect(da.shape.join(',') === sa.join(','), `dA (${da.shape.join(',')}) != (${sa.join(',')})`);
    expect(db.shape.join(',') === sb.join(','), `dB (${db.shape.join(',')}) != (${sb.join(',')})`);
    expect(size(da) === size(a) && size(db) === size(b), 'element counts disagree');
  }
  return `${cases.length} broadcast pairings`;
});
