/**
 * NdArray — a strided float64 view over a shared buffer.
 *
 * Two decisions here drive the whole project (docs/architecture/overview.md A-01):
 *
 *  1. **float64 everywhere.** Central-difference gradcheck is drowned by float32
 *     rounding (transformer_plan.md 纪律二). Reduced precision is *simulated* in
 *     core/numerics/float.ts for Step 0.5, never used as storage.
 *
 *  2. **Real strides, real views.** `transpose` must share its buffer and only
 *     permute strides; `reshape` of a non-contiguous array must genuinely copy.
 *     Step 0.1 teaches memory layout, so the engine has to actually have one —
 *     the visualization reads `strides` directly rather than being drawn to match.
 */

export type Shape = readonly number[];

/** Nested plain-JS view of an array, for UI consumption. */
export type Nested = number | Nested[];

export interface NdArrayInit {
  data: Float64Array;
  shape: Shape;
  strides?: Shape;
  offset?: number;
  base?: NdArray | null;
  readOnly?: boolean;
}

/** Number of elements described by a shape. Scalars (shape []) have size 1. */
export function sizeOfShape(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** Row-major (C-order) strides for a shape, in element units. */
export function contiguousStrides(shape: Shape): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

export function shapesEqual(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function formatShapeTuple(shape: Shape): string {
  return `(${shape.join(', ')})`;
}

export class NdArray {
  readonly data: Float64Array;
  readonly shape: Shape;
  readonly strides: Shape;
  readonly offset: number;
  /** Non-null means this array is a view onto another array's buffer. */
  readonly base: NdArray | null;
  /**
   * Set by `broadcastTo`, which produces stride-0 axes. Writing through such a
   * view would scatter one write across many logical positions, so it is banned.
   */
  readonly readOnly: boolean;

  constructor(init: NdArrayInit) {
    const shape = init.shape;
    const strides = init.strides ?? contiguousStrides(shape);
    const offset = init.offset ?? 0;

    if (shape.length !== strides.length) {
      throw new Error(
        `NdArray: shape ${formatShapeTuple(shape)} and strides ${formatShapeTuple(strides)} differ in rank`,
      );
    }
    for (let i = 0; i < shape.length; i++) {
      const d = shape[i];
      if (!Number.isInteger(d) || d < 0) {
        throw new Error(`NdArray: shape ${formatShapeTuple(shape)} has a bad extent at axis ${i}`);
      }
      if (!Number.isInteger(strides[i])) {
        throw new Error(`NdArray: strides ${formatShapeTuple(strides)} must be integers`);
      }
    }
    if (!Number.isInteger(offset) || offset < 0) {
      throw new Error(`NdArray: offset must be a non-negative integer, got ${offset}`);
    }

    // Every reachable element must lie inside the buffer.
    const n = sizeOfShape(shape);
    if (n > 0) {
      let lo = offset;
      let hi = offset;
      for (let i = 0; i < shape.length; i++) {
        const span = (shape[i] - 1) * strides[i];
        if (span >= 0) hi += span;
        else lo += span;
      }
      if (lo < 0 || hi >= init.data.length) {
        throw new Error(
          `NdArray: shape ${formatShapeTuple(shape)} strides ${formatShapeTuple(strides)} ` +
            `offset ${offset} reach [${lo}, ${hi}] outside a buffer of length ${init.data.length}`,
        );
      }
    }

    this.data = init.data;
    this.shape = shape;
    this.strides = strides;
    this.offset = offset;
    this.base = init.base ?? null;
    this.readOnly = init.readOnly ?? false;
  }
}

export function ndim(a: NdArray): number {
  return a.shape.length;
}

export function size(a: NdArray): number {
  return sizeOfShape(a.shape);
}

/**
 * C-contiguity, following NumPy: axes of extent 1 are skipped because their
 * stride is unobservable, and an empty array is trivially contiguous.
 *
 * Getting this wrong would make `reshape` copy when it should not, which would
 * make Step 0.1 teach something false.
 */
export function isContiguous(a: NdArray): boolean {
  if (size(a) === 0) return true;
  let expected = 1;
  for (let i = a.shape.length - 1; i >= 0; i--) {
    if (a.shape[i] === 1) continue;
    if (a.strides[i] !== expected) return false;
    expected *= a.shape[i];
  }
  return true;
}

/** True when `a` and `b` read from the same buffer. */
export function sharesBuffer(a: NdArray, b: NdArray): boolean {
  return a.data === b.data;
}

/** Flat index into `a.data` for a logical multi-index. */
export function flatIndex(a: NdArray, idx: readonly number[]): number {
  if (idx.length !== a.shape.length) {
    throw new Error(
      `flatIndex: got ${idx.length} indices for a rank-${a.shape.length} array ${formatShapeTuple(a.shape)}`,
    );
  }
  let flat = a.offset;
  for (let i = 0; i < idx.length; i++) {
    const v = idx[i];
    if (!Number.isInteger(v) || v < 0 || v >= a.shape[i]) {
      throw new Error(
        `flatIndex: index ${v} out of range on axis ${i} of ${formatShapeTuple(a.shape)}`,
      );
    }
    flat += v * a.strides[i];
  }
  return flat;
}

export function at(a: NdArray, idx: readonly number[]): number {
  return a.data[flatIndex(a, idx)];
}

export function set(a: NdArray, idx: readonly number[], value: number): void {
  if (a.readOnly) {
    throw new Error(
      'set: this array is a read-only view (it has stride-0 axes from broadcastTo). ' +
        'Writing through it would scatter one write across many logical positions.',
    );
  }
  a.data[flatIndex(a, idx)] = value;
}

/**
 * Walk every element in logical (row-major) order, calling `fn` with the flat
 * buffer offset and the logical position. No multi-index is materialised, so
 * this is the hot path used by every operator.
 */
export function forEachOffset(
  a: NdArray,
  fn: (flatOffset: number, logicalIndex: number) => void,
): void {
  const n = size(a);
  if (n === 0) return;

  if (isContiguous(a)) {
    for (let i = 0; i < n; i++) fn(a.offset + i, i);
    return;
  }

  const rank = a.shape.length;
  const counter = new Array<number>(rank).fill(0);
  let flat = a.offset;
  for (let i = 0; i < n; i++) {
    fn(flat, i);
    // odometer: increment the last axis, carrying leftwards
    for (let axis = rank - 1; axis >= 0; axis--) {
      counter[axis] += 1;
      flat += a.strides[axis];
      if (counter[axis] < a.shape[axis]) break;
      flat -= a.shape[axis] * a.strides[axis];
      counter[axis] = 0;
    }
  }
}

/**
 * Like {@link forEachOffset} but also supplies the multi-index.
 *
 * The `idx` array is REUSED between iterations for speed — copy it if you need
 * to keep it.
 */
export function forEachIndex(
  a: NdArray,
  fn: (flatOffset: number, logicalIndex: number, idx: readonly number[]) => void,
): void {
  const n = size(a);
  if (n === 0) return;

  const rank = a.shape.length;
  const counter = new Array<number>(rank).fill(0);
  let flat = a.offset;
  for (let i = 0; i < n; i++) {
    fn(flat, i, counter);
    for (let axis = rank - 1; axis >= 0; axis--) {
      counter[axis] += 1;
      flat += a.strides[axis];
      if (counter[axis] < a.shape[axis]) break;
      flat -= a.shape[axis] * a.strides[axis];
      counter[axis] = 0;
    }
  }
}

/**
 * Walk several identically-shaped arrays in lockstep, supplying each one's flat
 * offset. Arrays may have any strides (including 0 from broadcasting), so this
 * is what every elementwise operator runs on after its inputs are broadcast.
 */
export function forEachZip(
  shape: Shape,
  arrays: readonly NdArray[],
  fn: (offsets: readonly number[], logicalIndex: number) => void,
): void {
  for (const a of arrays) {
    if (!shapesEqual(a.shape, shape)) {
      throw new Error(
        `forEachZip: input ${formatShapeTuple(a.shape)} does not match ${formatShapeTuple(shape)}`,
      );
    }
  }

  const n = sizeOfShape(shape);
  if (n === 0) return;

  const rank = shape.length;
  const count = arrays.length;
  const offsets = arrays.map((a) => a.offset);
  const counter = new Array<number>(rank).fill(0);

  for (let i = 0; i < n; i++) {
    fn(offsets, i);
    for (let axis = rank - 1; axis >= 0; axis--) {
      counter[axis] += 1;
      for (let j = 0; j < count; j++) offsets[j] += arrays[j].strides[axis];
      if (counter[axis] < shape[axis]) break;
      for (let j = 0; j < count; j++) offsets[j] -= shape[axis] * arrays[j].strides[axis];
      counter[axis] = 0;
    }
  }
}

/**
 * Elements in logical order.
 *
 * MAY ALIAS `a.data` when `a` is already contiguous — treat the result as
 * read-only. Use {@link copyFlat} when you need to own the memory.
 */
export function readFlat(a: NdArray): Float64Array {
  const n = size(a);
  if (isContiguous(a)) return a.data.subarray(a.offset, a.offset + n);
  return copyFlat(a);
}

/** Elements in logical order, always in a freshly allocated buffer. */
export function copyFlat(a: NdArray): Float64Array {
  const out = new Float64Array(size(a));
  forEachOffset(a, (flat, i) => {
    out[i] = a.data[flat];
  });
  return out;
}

// ── constructors ────────────────────────────────────────────────────────────

export function fromFlat(data: Float64Array, shape: Shape): NdArray {
  const n = sizeOfShape(shape);
  if (data.length !== n) {
    throw new Error(
      `fromFlat: buffer of length ${data.length} does not fill shape ${formatShapeTuple(shape)} (${n} elements)`,
    );
  }
  return new NdArray({ data, shape: [...shape] });
}

export function zeros(shape: Shape): NdArray {
  return new NdArray({ data: new Float64Array(sizeOfShape(shape)), shape: [...shape] });
}

export function full(shape: Shape, value: number): NdArray {
  const data = new Float64Array(sizeOfShape(shape));
  data.fill(value);
  return new NdArray({ data, shape: [...shape] });
}

export function ones(shape: Shape): NdArray {
  return full(shape, 1);
}

export function scalar(value: number): NdArray {
  return new NdArray({ data: Float64Array.of(value), shape: [] });
}

export function zerosLike(a: NdArray): NdArray {
  return zeros(a.shape);
}

/** Deep-copy `a` into a fresh contiguous array. */
export function clone(a: NdArray): NdArray {
  return new NdArray({ data: copyFlat(a), shape: [...a.shape] });
}

// ── nested-array interop (used by every viz component) ──────────────────────

function inferNestedShape(value: Nested): number[] {
  const shape: number[] = [];
  let cursor: Nested = value;
  while (Array.isArray(cursor)) {
    shape.push(cursor.length);
    if (cursor.length === 0) break;
    cursor = cursor[0];
  }
  return shape;
}

function flattenNested(value: Nested, shape: Shape, depth: number, out: number[]): void {
  if (depth === shape.length) {
    if (typeof value !== 'number') throw new Error('fromNested: ragged nesting (too deep)');
    out.push(value);
    return;
  }
  if (!Array.isArray(value)) throw new Error('fromNested: ragged nesting (too shallow)');
  if (value.length !== shape[depth]) {
    throw new Error(
      `fromNested: ragged array — expected ${shape[depth]} entries at depth ${depth}, got ${value.length}`,
    );
  }
  for (const child of value) flattenNested(child, shape, depth + 1, out);
}

export function fromNested(value: Nested): NdArray {
  const shape = inferNestedShape(value);
  const out: number[] = [];
  flattenNested(value, shape, 0, out);
  return new NdArray({ data: Float64Array.from(out), shape });
}

export function toNested(a: NdArray): Nested {
  const flat = readFlat(a);
  const build = (axis: number, start: number, stride: number): Nested => {
    if (axis === a.shape.length) return flat[start];
    const out: Nested[] = new Array<Nested>(a.shape[axis]);
    const childStride = stride / Math.max(1, a.shape[axis]);
    for (let i = 0; i < a.shape[axis]; i++) {
      out[i] = build(axis + 1, start + i * childStride, childStride);
    }
    return out;
  };
  return build(0, 0, size(a));
}
