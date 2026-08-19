/**
 * Explicit Jacobians — built only to show why nobody builds them.
 *
 * Step 0.3's headline visualization (F0.8.4) puts the full (n, n) softmax
 * Jacobian next to the two-line VJP and shows that they agree. The point lands
 * only if the reader can then drag n upward and watch the matrix become
 * impossible, which is what the size guard and the memory estimate are for:
 * `JacobianTooLargeError` is a feature, not a failure mode. F0.8.5 renders it.
 */

import { NdArray, forEachIndex, size, zeros } from './ndarray.ts';
import { Var, backwardWithSeed, gradOf, variable, zeroGrad } from './autograd.ts';

export { softmaxJacobian } from './softmax.ts';

/**
 * Above this, an explicit Jacobian is refused.
 *
 * Chosen small on purpose. The teaching claim is that materialising the
 * Jacobian is absurd at real sizes, so the engine should refuse well before it
 * merely becomes slow.
 */
export const JACOBIAN_MAX_N = 16;

export class JacobianTooLargeError extends Error {
  readonly n: number;
  readonly elements: number;
  readonly bytes: number;

  constructor(n: number, bytesPerElement: number) {
    const elements = n * n;
    const bytes = elements * bytesPerElement;
    super(
      `An explicit Jacobian for n=${n} would be ${n}x${n} = ${elements.toLocaleString()} ` +
        `elements (${humanBytes(bytes)}). The VJP computes the same product in O(n). ` +
        `The explicit form is capped at n=${JACOBIAN_MAX_N} here, for illustration only.`,
    );
    this.name = 'JacobianTooLargeError';
    this.n = n;
    this.elements = elements;
    this.bytes = bytes;
  }
}

export function humanBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 100 || unit === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

export interface JacobianSize {
  n: number;
  elements: number;
  bytes: number;
  human: string;
}

/**
 * What an (n, n) Jacobian would cost. Defaults to float32 because that is what
 * the comparison in Step 0.3 is against — a real model's activations.
 */
export function jacobianMemoryEstimate(n: number, bytesPerElement = 4): JacobianSize {
  const elements = n * n;
  const bytes = elements * bytesPerElement;
  return { n, elements, bytes, human: humanBytes(bytes) };
}

/**
 * Build the full Jacobian of `f` at `x` by running the reverse sweep once per
 * output element, seeded with a basis vector.
 *
 * Row i of the result is the gradient of output element i with respect to every
 * input — i.e. exactly m VJP passes. That cost ratio is the whole argument
 * against materialising it.
 */
export function jacobianByVjp(f: (x: Var) => Var, x: NdArray): NdArray {
  const n = size(x);
  if (n > JACOBIAN_MAX_N) throw new JacobianTooLargeError(n, 8);

  const input = variable(x, { label: 'x' });
  const output = f(input);
  const m = size(output.value);
  if (m > JACOBIAN_MAX_N) throw new JacobianTooLargeError(m, 8);

  const jacobian = zeros([m, n]);

  for (let i = 0; i < m; i++) {
    zeroGrad(output);
    const seed = zeros(output.value.shape);
    seed.data[i] = 1;
    backwardWithSeed(output, seed);

    const gradient = gradOf(input);
    forEachIndex(gradient, (flat, logical) => {
      jacobian.data[i * n + logical] = gradient.data[flat];
    });
  }

  return jacobian;
}

/**
 * Contract a Jacobian with an upstream gradient: `sBar @ J`.
 *
 * Present so the comparison in F0.8.4 is written the way the maths reads —
 * the VJP result should equal this, at 1e-12.
 */
export function contractJacobian(jacobian: NdArray, upstream: NdArray): NdArray {
  const m = jacobian.shape[0];
  const n = jacobian.shape[1];
  if (size(upstream) !== m) {
    throw new Error(
      `contractJacobian: upstream has ${size(upstream)} elements, the Jacobian has ${m} rows`,
    );
  }

  const flatUpstream = new Float64Array(m);
  forEachIndex(upstream, (flat, logical) => {
    flatUpstream[logical] = upstream.data[flat];
  });

  const out = zeros([n]);
  for (let i = 0; i < m; i++) {
    const weight = flatUpstream[i];
    for (let j = 0; j < n; j++) out.data[j] += weight * jacobian.data[i * n + j];
  }
  return out;
}
