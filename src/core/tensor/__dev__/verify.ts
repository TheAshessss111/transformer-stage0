/**
 * Engine verification harness — `npm run verify:engine`.
 *
 * Runs on Node's native TypeScript type stripping: no test framework, no config,
 * no CI (DECISIONS.md D-22, impl plan section 0.4). Checks are added task by task:
 *
 *   T14 (v1) structural invariants — NdArray, shape ops, broadcasting, PRNG, formatting
 *   T16 (v2) gradcheck for every VJP
 *   T20 (v3) softmax family, autograd, Jacobian identities
 */

import { check, expect, group, report } from './harness.ts';
import {
  NdArray,
  at,
  clone,
  contiguousStrides,
  copyFlat,
  forEachOffset,
  fromNested,
  isContiguous,
  ones,
  set,
  sharesBuffer,
  shapesEqual,
  size,
  toNested,
  zeros,
} from '../ndarray.ts';
import {
  ascontiguousarray,
  expandDims,
  permute,
  reshape,
  select,
  squeeze,
  swapAxes,
  transpose,
} from '../shape.ts';
import { broadcastShapes, broadcastTo, canBroadcastTo, unbroadcast } from '../broadcast.ts';
import { add, matmul, max, mean, mul, sum } from '../ops.ts';
import { arange, mulberry32, onehot, randn, uniform } from '../random.ts';
import { classify, formatBoundShape, formatValue } from '../format.ts';
import { setOpHook, type RawOpEvent } from '../../trace/hook.ts';
import { numericalGrad, scalarize } from '../../gradcheck/numericalGrad.ts';
import { relError } from '../../gradcheck/relError.ts';

function throws(fn: () => unknown, fragment: string): void {
  try {
    fn();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    expect(
      message.includes(fragment),
      `expected the error to mention "${fragment}", got: ${message}`,
    );
    return;
  }
  throw new Error(`expected a throw mentioning "${fragment}", but nothing was thrown`);
}

/** Capture the trace events produced by `fn`. */
function record(fn: () => void): RawOpEvent[] {
  const events: RawOpEvent[] = [];
  setOpHook((e) => events.push(e));
  try {
    fn();
  } finally {
    setOpHook(null);
  }
  return events;
}

function flatEquals(a: NdArray, expected: readonly number[], tol = 0): void {
  const actual = Array.from(copyFlat(a));
  expect(actual.length === expected.length, `length ${actual.length} != ${expected.length}`);
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    expect(diff <= tol, `element ${i}: got ${actual[i]}, expected ${expected[i]} (diff ${diff})`);
  }
}

// ── NdArray ─────────────────────────────────────────────────────────────────

group('NdArray · structure');

check('constructor rejects a shape/strides rank mismatch', () => {
  throws(
    () => new NdArray({ data: new Float64Array(6), shape: [2, 3], strides: [1] }),
    'differ in rank',
  );
});

check('constructor rejects strides that reach outside the buffer', () => {
  throws(
    () => new NdArray({ data: new Float64Array(6), shape: [2, 3], strides: [100, 1] }),
    'outside a buffer',
  );
});

check('constructor rejects a negative extent', () => {
  throws(() => new NdArray({ data: new Float64Array(1), shape: [-1] }), 'bad extent');
});

check('contiguousStrides is row-major', () => {
  const s = contiguousStrides([2, 3, 4]);
  expect(s.join(',') === '12,4,1', `got ${s.join(',')}`);
  return '(2,3,4) -> [12,4,1]';
});

check('isContiguous: fresh array true, transposed false', () => {
  const a = zeros([2, 3, 4]);
  expect(isContiguous(a), 'fresh array should be contiguous');
  expect(!isContiguous(permute(a, [0, 2, 1])), 'permuted array should not be contiguous');
});

check('isContiguous ignores extent-1 axes (NumPy rule)', () => {
  const a = zeros([2, 1, 4]);
  const t = permute(a, [0, 2, 1]); // shape (2,4,1); the size-1 axis stride is unobservable
  expect(isContiguous(t), 'a permute that only moves an extent-1 axis stays contiguous');
});

check('isContiguous: empty and rank-0 arrays are contiguous', () => {
  expect(isContiguous(zeros([0, 3])), 'empty array');
  expect(isContiguous(zeros([])), 'rank-0 array');
});

check('flatIndex reports the offending axis', () => {
  throws(() => at(zeros([2, 3]), [0, 5]), 'axis 1');
});

check('forEachOffset walks a non-contiguous view in logical order', () => {
  const a = fromNested([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const t = transpose(a); // (3,2) -> logical order 1,4,2,5,3,6
  const seen: number[] = [];
  forEachOffset(t, (flat) => seen.push(t.data[flat]));
  expect(seen.join(',') === '1,4,2,5,3,6', `got ${seen.join(',')}`);
  return 'transpose reads 1,4,2,5,3,6';
});

check('toNested / fromNested round-trip through a transposed view', () => {
  const a = fromNested([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const nested = toNested(transpose(a));
  expect(JSON.stringify(nested) === '[[1,4],[2,5],[3,6]]', `got ${JSON.stringify(nested)}`);
});

check('fromNested rejects a ragged array', () => {
  throws(() => fromNested([[1, 2], [3]]), 'ragged');
});

// ── shape ops: the view/copy contract ───────────────────────────────────────

group('shape ops · view vs copy (ARCHITECTURE A-01)');

check('transpose shares the buffer (identity, not just equal contents)', () => {
  const a = randn([3, 4], 1);
  const t = transpose(a);
  expect(sharesBuffer(a, t), 'transpose must not copy');
  expect(t.base === a, 'the view should point back at its base');
  return 'a.data === transpose(a).data';
});

check('reshape of a contiguous array is a view', () => {
  const a = randn([2, 6], 2);
  const events = record(() => {
    const r = reshape(a, [3, 4]);
    expect(sharesBuffer(a, r), 'contiguous reshape must not copy');
  });
  const ev = events.find((e) => e.op === 'reshape');
  expect(ev !== undefined && !ev.didCopy, 'trace should report didCopy=false');
  return 'didCopy=false';
});

check('reshape of a NON-contiguous array copies, and says how much', () => {
  const a = randn([3, 4], 3);
  const t = transpose(a); // (4,3), not contiguous
  let copied: NdArray | null = null;
  const events = record(() => {
    copied = reshape(t, [12]);
  });
  expect(copied !== null, 'reshape returned nothing');
  expect(
    !sharesBuffer(t, copied as unknown as NdArray),
    'reshape must have allocated a new buffer',
  );

  const ev = events.find((e) => e.op === 'reshape');
  expect(ev !== undefined, 'no reshape event emitted');
  expect(ev!.didCopy, 'trace should report didCopy=true');
  expect(ev!.copiedElements === 12, `copiedElements should be 12, got ${ev!.copiedElements}`);

  // and the copy has to hold the LOGICAL order, not the buffer order
  flatEquals(copied as unknown as NdArray, Array.from(copyFlat(t)));
  return 'didCopy=true, 12 elements, logical order preserved';
});

check('ascontiguousarray returns the same object when already contiguous', () => {
  const a = randn([2, 3], 4);
  expect(ascontiguousarray(a) === a, 'no-op case should be referentially identical');
});

check('reshape rejects an element-count mismatch', () => {
  throws(() => reshape(zeros([2, 3]), [4, 2]), 'cannot reshape');
});

check('reshape infers a single -1', () => {
  const r = reshape(zeros([2, 3, 4]), [6, -1]);
  expect(shapesEqual(r.shape, [6, 4]), `got ${r.shape.join(',')}`);
  return '(2,3,4) -> (6,-1) = (6,4)';
});

check('permute round-trips', () => {
  const a = randn([2, 3, 4], 5);
  const p = permute(a, [2, 0, 1]);
  const back = permute(p, [1, 2, 0]);
  expect(shapesEqual(back.shape, a.shape), `got ${back.shape.join(',')}`);
  flatEquals(back, Array.from(copyFlat(a)));
});

check('permute rejects a repeated axis', () => {
  throws(() => permute(zeros([2, 3]), [0, 0]), 'listed twice');
});

check('select is a zero-copy view at the right offset', () => {
  const a = fromNested([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const row = select(a, 0, 1);
  expect(sharesBuffer(a, row), 'select must not copy');
  expect(row.offset === 3, `offset should be 3, got ${row.offset}`);
  flatEquals(row, [4, 5, 6]);
});

check('expandDims keeps a contiguous array contiguous', () => {
  const a = randn([2, 3], 6);
  for (const axis of [0, 1, 2]) {
    const e = expandDims(a, axis);
    expect(isContiguous(e), `expandDims(a, ${axis}) broke contiguity`);
    expect(sharesBuffer(a, e), 'expandDims must not copy');
  }
  return 'axes 0,1,2 all stay contiguous';
});

check('squeeze drops extent-1 axes', () => {
  const a = zeros([1, 3, 1, 4]);
  expect(shapesEqual(squeeze(a).shape, [3, 4]), 'squeeze-all');
  expect(shapesEqual(squeeze(a, 2).shape, [1, 3, 4]), 'squeeze one axis');
  throws(() => squeeze(a, 1), 'not 1');
});

check('swapAxes matches the equivalent permute', () => {
  const a = randn([2, 3, 4], 7);
  flatEquals(swapAxes(a, 0, 2), Array.from(copyFlat(permute(a, [2, 1, 0]))));
});

// ── broadcasting ────────────────────────────────────────────────────────────

group('broadcasting · and its dual');

check('broadcastShapes right-aligns', () => {
  expect(broadcastShapes([3, 1], [2, 1, 4]).join(',') === '2,3,4', 'case 1');
  expect(broadcastShapes([], [5]).join(',') === '5', 'scalar case');
  expect(broadcastShapes([1], [1]).join(',') === '1', 'trivial case');
});

check('broadcastShapes names the offending axis', () => {
  throws(() => broadcastShapes([3], [4]), 'incompatible extents');
});

check('broadcastTo produces stride-0 axes and shares the buffer', () => {
  const a = ones([3, 1]);
  const b = broadcastTo(a, [2, 3, 4]);
  expect(sharesBuffer(a, b), 'broadcast must not copy');
  expect(b.strides.join(',') === '0,1,0', `strides should be 0,1,0 — got ${b.strides.join(',')}`);
  return 'strides [0,1,0], buffer shared';
});

check('a broadcast view is read-only', () => {
  const b = broadcastTo(ones([3, 1]), [3, 4]);
  expect(b.readOnly, 'broadcast views must be flagged read-only');
  throws(() => set(b, [0, 0], 5), 'read-only view');
});

check('unbroadcast is the exact inverse-by-summation of broadcastTo', () => {
  const cases: Array<[number[], number[]]> = [
    [
      [3, 1],
      [2, 3, 4],
    ],
    [[1], [5]],
    [[4], [3, 4]],
    [
      [2, 1, 4],
      [2, 5, 4],
    ],
    [
      [1, 1],
      [6, 7],
    ],
  ];
  for (const [from, to] of cases) {
    const a = randn(from, 11);
    const wide = broadcastTo(a, to);
    const back = unbroadcast(wide, from);
    expect(shapesEqual(back.shape, from), `shape ${back.shape.join(',')} != ${from.join(',')}`);

    // each source element was replicated `factor` times, so summing gives a*factor
    let factor = 1;
    for (let i = 0; i < to.length; i++) {
      const fromIdx = i - (to.length - from.length);
      const fromExtent = fromIdx < 0 ? 1 : from[fromIdx];
      if (fromExtent === 1 && to[i] !== 1) factor *= to[i];
    }
    const expected = Array.from(copyFlat(a)).map((v) => v * factor);
    flatEquals(back, expected, 1e-12);
  }
  return `${cases.length} shape pairs`;
});

check('unbroadcast refuses a pairing the forward pass could not have made', () => {
  throws(() => unbroadcast(zeros([2, 3]), [4]), 'not broadcastable');
});

check('canBroadcastTo agrees with broadcastTo', () => {
  expect(canBroadcastTo([3, 1], [2, 3, 4]), 'should be broadcastable');
  expect(!canBroadcastTo([3], [4]), 'should not be broadcastable');
  expect(!canBroadcastTo([2, 3], [3]), 'lower rank target');
});

// ── operators ───────────────────────────────────────────────────────────────

group('operators · forward');

check('add broadcasts', () => {
  const a = fromNested([[1, 2, 3]]); // (1,3)
  const b = fromNested([[10], [20]]); // (2,1)
  flatEquals(add(a, b), [11, 12, 13, 21, 22, 23]);
});

check('reductions handle axis=null | number | number[], both keepdims', () => {
  const a = fromNested([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  flatEquals(sum(a), [21]);
  flatEquals(sum(a, 0), [5, 7, 9]);
  flatEquals(sum(a, 1, true), [6, 15]);
  expect(shapesEqual(sum(a, 1, true).shape, [2, 1]), 'keepdims shape');
  expect(shapesEqual(sum(a, 1, false).shape, [2]), 'no-keepdims shape');
  flatEquals(sum(a, [0, 1]), [21]);
  flatEquals(mean(a, 0), [2.5, 3.5, 4.5]);
  flatEquals(max(a, 1), [3, 6]);
});

check('reducing a non-contiguous view matches reducing its contiguous copy', () => {
  const a = randn([3, 4, 5], 12);
  const t = permute(a, [1, 2, 0]);
  flatEquals(sum(t, 1), Array.from(copyFlat(sum(clone(t), 1))), 1e-12);
  flatEquals(max(t, [0, 2]), Array.from(copyFlat(max(clone(t), [0, 2]))), 1e-12);
});

check('max propagates NaN (NumPy semantics — Step 0.5 depends on it)', () => {
  const a = fromNested([[1, NaN, 3]]);
  const m = max(a, 1);
  expect(Number.isNaN(m.data[0]), `expected NaN, got ${m.data[0]}`);
});

check('matmul: 2-D against a hand-computed result', () => {
  const a = fromNested([
    [1, 2],
    [3, 4],
  ]);
  const b = fromNested([
    [5, 6],
    [7, 8],
  ]);
  flatEquals(matmul(a, b), [19, 22, 43, 50]);
});

check('matmul broadcasts batch dims: (2,1,3,4) @ (1,5,4,6) -> (2,5,3,6)', () => {
  const a = randn([2, 1, 3, 4], 13);
  const b = randn([1, 5, 4, 6], 14);
  const out = matmul(a, b);
  expect(shapesEqual(out.shape, [2, 5, 3, 6]), `got ${out.shape.join(',')}`);

  // spot-check one batch element against an explicit 2-D multiply
  const a0 = select(select(a, 0, 1), 0, 0); // (3,4)
  const b0 = select(select(b, 0, 0), 0, 3); // (4,6)
  const ref = matmul(clone(a0), clone(b0));
  const got = select(select(out, 0, 1), 0, 3);
  flatEquals(clone(got), Array.from(copyFlat(ref)), 1e-12);
  return 'batch (1,3) spot-checked';
});

check('matmul on transposed (non-contiguous) operands is still correct', () => {
  const a = randn([4, 3], 15);
  const b = randn([4, 5], 16);
  const viaView = matmul(transpose(a), b); // (3,4) @ (4,5)
  const viaCopy = matmul(clone(transpose(a)), b);
  flatEquals(viaView, Array.from(copyFlat(viaCopy)), 1e-12);
});

check('matmul rejects rank < 2 with an actionable message', () => {
  throws(() => matmul(zeros([3]), zeros([3, 2])), 'rank >= 2');
});

check('matmul reports disagreeing inner dimensions', () => {
  throws(() => matmul(zeros([2, 3]), zeros([4, 5])), 'inner dimensions disagree');
});

// ── random & format ─────────────────────────────────────────────────────────

group('random & format');

check('the same seed produces bit-identical arrays', () => {
  const a = randn([50], 42);
  const b = randn([50], 42);
  const c = randn([50], 43);
  for (let i = 0; i < 50; i++) expect(a.data[i] === b.data[i], `element ${i} differs`);
  let anyDifferent = false;
  for (let i = 0; i < 50; i++) if (a.data[i] !== c.data[i]) anyDifferent = true;
  expect(anyDifferent, 'a different seed produced identical output');
});

check('randn is approximately standard normal', () => {
  const a = randn([100000], 7);
  let total = 0;
  for (const v of a.data) total += v;
  const mu = total / a.data.length;
  let variance = 0;
  for (const v of a.data) variance += (v - mu) * (v - mu);
  const sigma = Math.sqrt(variance / a.data.length);
  expect(Math.abs(mu) < 0.02, `mean ${mu}`);
  expect(Math.abs(sigma - 1) < 0.02, `std ${sigma}`);
  return `mu=${mu.toFixed(4)} sigma=${sigma.toFixed(4)}`;
});

check('uniform stays inside its bounds', () => {
  const a = uniform([1000], -3, 5, 9);
  for (const v of a.data) expect(v >= -3 && v < 5, `out of range: ${v}`);
});

check('mulberry32 output stays in [0, 1)', () => {
  const rng = mulberry32(123);
  for (let i = 0; i < 1000; i++) {
    const v = rng();
    expect(v >= 0 && v < 1, `out of range: ${v}`);
  }
});

check('arange and onehot', () => {
  flatEquals(arange(4), [0, 1, 2, 3]);
  flatEquals(onehot([2, 0], 3), [0, 0, 1, 1, 0, 0]);
  throws(() => onehot([5], 3), 'outside');
});

check('classify separates -0, subnormals and both infinities', () => {
  expect(classify(0) === 'zero', 'zero');
  expect(classify(-0) === 'zero', 'negative zero');
  expect(classify(5e-324) === 'subnormal', 'smallest subnormal');
  expect(classify(1e-308) === 'subnormal', '1e-308 is subnormal');
  expect(classify(1e-307) === 'finite', '1e-307 is normal');
  expect(classify(Infinity) === 'posinf', '+Inf');
  expect(classify(-Infinity) === 'neginf', '-Inf');
  expect(classify(NaN) === 'nan', 'NaN');
  expect(classify(1.5) === 'finite', 'ordinary value');
});

check('formatValue picks fixed vs exponential sensibly', () => {
  expect(formatValue(1.23456).text === '1.235', formatValue(1.23456).text);
  expect(formatValue(12.3456).text === '12.35', formatValue(12.3456).text);
  expect(formatValue(0.000012).text.includes('e-'), formatValue(0.000012).text);
  expect(formatValue(1e7).text.includes('e+'), formatValue(1e7).text);
  expect(formatValue(NaN).text === 'NaN', 'NaN');
  expect(formatValue(0).text === '0', 'zero');
});

check('formatBoundShape uses the plan symbols', () => {
  const s = formatBoundShape(['B', 'H', 'T', 'S'], [2, 4, 8, 8]);
  expect(s === '(B=2, H=4, T=8, S=8)', s);
  return s;
});

// ── gradcheck plumbing ──────────────────────────────────────────────────────

group('gradcheck · plumbing');

check('numericalGrad of sum(x*x) is 2x', () => {
  const x = randn([3, 4], 21);
  const numeric = numericalGrad((v) => sum(mul(v, v)).data[0], x);
  // The expected gradient is built by hand rather than through the ops under
  // test, so this check cannot pass by two bugs cancelling out.
  const expected = new NdArray({
    data: Float64Array.from(copyFlat(x), (v) => 2 * v),
    shape: [3, 4],
  });
  const { max: err } = relError(numeric, expected);
  expect(err < 1e-9, `max rel err ${err}`);
  return `max rel err ${err.toExponential(2)}`;
});

check('scalarize contracts the output against a fixed upstream', () => {
  const x = randn([2, 3], 22);
  const upstream = randn([2, 3], 23);
  const f = scalarize((v) => mul(v, v), upstream);
  // d/dx sum(x^2 * u) = 2*x*u
  const numeric = numericalGrad(f, x);
  const expected = new NdArray({
    data: Float64Array.from(copyFlat(x), (v, i) => 2 * v * upstream.data[i]),
    shape: [2, 3],
  });
  const { max: err } = relError(numeric, expected);
  expect(err < 1e-8, `max rel err ${err}`);
  return `max rel err ${err.toExponential(2)}`;
});

check('relError treats a lone NaN as maximal error', () => {
  const a = new NdArray({ data: Float64Array.of(1, NaN), shape: [2] });
  const b = new NdArray({ data: Float64Array.of(1, 1), shape: [2] });
  expect(relError(a, b).max === Infinity, 'NaN must not vanish into an arithmetic NaN');
});

check('numericalGrad does not disturb the caller array', () => {
  const x = randn([2, 2], 24);
  const before = Array.from(copyFlat(x));
  numericalGrad((v) => sum(v).data[0], x);
  flatEquals(x, before);
});

check('size() agrees with the shape product', () => {
  expect(size(zeros([2, 3, 4])) === 24, 'rank 3');
  expect(size(zeros([])) === 1, 'rank 0 is a single element');
  expect(size(zeros([0, 5])) === 0, 'empty');
});

report();
