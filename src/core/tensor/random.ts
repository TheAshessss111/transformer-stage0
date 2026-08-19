/**
 * Seeded randomness.
 *
 * Every lab must be reproducible: the same slider settings have to produce the
 * same matrix on every visit, or "look, this element is 0.37" stops being a
 * thing anyone can point at. So there is no Math.random() anywhere in core/.
 */

import { NdArray, sizeOfShape, type Shape } from './ndarray.ts';

/**
 * mulberry32 — a 32-bit PRNG that is short, fast, and passes gjrand.
 * Chosen because it is small enough to read and audit in one sitting.
 */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return function next(): number {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function uniform(shape: Shape, lo: number, hi: number, seed: number): NdArray {
  const rng = mulberry32(seed);
  const n = sizeOfShape(shape);
  const data = new Float64Array(n);
  for (let i = 0; i < n; i++) data[i] = lo + (hi - lo) * rng();
  return new NdArray({ data, shape: [...shape] });
}

/**
 * Standard normal samples via the Box–Muller transform.
 *
 * `rng()` can return exactly 0, whose log is -Infinity, so the low end is
 * nudged onto the smallest positive double. (An Inf here would silently poison
 * every downstream check — exactly the failure mode Step 0.5 is about.)
 */
export function randn(shape: Shape, seed: number): NdArray {
  const rng = mulberry32(seed);
  const n = sizeOfShape(shape);
  const data = new Float64Array(n);

  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(rng(), Number.MIN_VALUE);
    const u2 = rng();
    const radius = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    data[i] = radius * Math.cos(theta);
    if (i + 1 < n) data[i + 1] = radius * Math.sin(theta);
  }

  return new NdArray({ data, shape: [...shape] });
}

/** 0, 1, …, n-1 as a rank-1 array. */
export function arange(n: number): NdArray {
  const data = new Float64Array(n);
  for (let i = 0; i < n; i++) data[i] = i;
  return new NdArray({ data, shape: [n] });
}

/** Random integers in [0, high) — used for cross-entropy targets. */
export function randomInts(count: number, high: number, seed: number): Int32Array {
  const rng = mulberry32(seed);
  const out = new Int32Array(count);
  for (let i = 0; i < count; i++) out[i] = Math.floor(rng() * high);
  return out;
}

/** (N,) class indices → (N, numClasses) one-hot. */
export function onehot(indices: ArrayLike<number>, numClasses: number): NdArray {
  const n = indices.length;
  const data = new Float64Array(n * numClasses);
  for (let i = 0; i < n; i++) {
    const cls = indices[i];
    if (!Number.isInteger(cls) || cls < 0 || cls >= numClasses) {
      throw new Error(`onehot: class ${cls} at position ${i} is outside [0, ${numClasses})`);
    }
    data[i * numClasses + cls] = 1;
  }
  return new NdArray({ data, shape: [n, numClasses] });
}
