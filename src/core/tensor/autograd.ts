/**
 * A tape-based autograd, ~200 lines.
 *
 * This is a rehearsal for 阶段 1 of transformer_plan.md, and it is deliberately
 * built the same way: a graph of Var nodes, a topological sort, and a reverse
 * sweep that turns each node's upstream gradient into gradients for its parents.
 *
 * Two invariants are enforced at runtime rather than trusted:
 *
 *  - **Gradients accumulate, never overwrite.** Step 1.2 of the plan flags this
 *    as the classic embedding bug: a value reached twice must sum both
 *    contributions. `y = x*x + x` fails under assignment and passes under
 *    accumulation, and the harness checks exactly that.
 *
 *  - **A gradient's shape equals its variable's shape.** The plan's core 心法.
 *    Checked after every accumulation, in production and not just in dev — it
 *    costs nothing next to a matmul and it names the offending op when it fires.
 */

import { NdArray, formatShapeTuple, ones, shapesEqual, size, zeros } from './ndarray.ts';
import { unbroadcast } from './broadcast.ts';
import { permute as permuteRaw, reshape as reshapeRaw } from './shape.ts';
import * as ops from './ops.ts';
import * as sm from './softmax.ts';
import * as vjp from './vjp.ts';

/** Turns this node's upstream gradient into one gradient per parent, in order. */
export type BackwardFn = (upstream: NdArray) => NdArray[];

let nextId = 0;

export interface VarInit {
  value: NdArray;
  op?: string;
  parents?: readonly Var[];
  backwardFn?: BackwardFn | null;
  requiresGrad?: boolean;
  label?: string;
}

export class Var {
  readonly id: number;
  value: NdArray;
  grad: NdArray | null;
  requiresGrad: boolean;
  readonly op: string;
  readonly parents: readonly Var[];
  readonly backwardFn: BackwardFn | null;
  /**
   * Stable display name. Present from the start because M2's GraphDAG needs
   * node names, and retrofitting it would mean touching every wrapper below.
   */
  readonly label: string | undefined;

  constructor(init: VarInit) {
    this.id = nextId++;
    this.value = init.value;
    this.grad = null;
    this.op = init.op ?? 'leaf';
    this.parents = init.parents ?? [];
    this.backwardFn = init.backwardFn ?? null;
    this.requiresGrad = init.requiresGrad ?? this.parents.some((p) => p.requiresGrad);
    this.label = init.label;
  }
}

/** A leaf: an input or a parameter. */
export function variable(
  value: NdArray,
  options: { requiresGrad?: boolean; label?: string } = {},
): Var {
  return new Var({
    value,
    op: 'leaf',
    requiresGrad: options.requiresGrad ?? true,
    label: options.label,
  });
}

/** A constant: participates in the forward pass, receives no gradient. */
export function constant(value: NdArray, label?: string): Var {
  return new Var({ value, op: 'const', requiresGrad: false, label });
}

function node(op: string, value: NdArray, parents: readonly Var[], backwardFn: BackwardFn): Var {
  return new Var({ value, op, parents, backwardFn });
}

/**
 * Post-order topological sort: every node appears after all of its parents.
 * Iterative rather than recursive so a deep graph cannot blow the stack.
 */
export function topoSort(root: Var): Var[] {
  const order: Var[] = [];
  const visited = new Set<number>();
  const stack: Array<{ node: Var; expanded: boolean }> = [{ node: root, expanded: false }];

  while (stack.length > 0) {
    const frame = stack.pop()!;
    if (frame.expanded) {
      order.push(frame.node);
      continue;
    }
    if (visited.has(frame.node.id)) continue;
    visited.add(frame.node.id);
    stack.push({ node: frame.node, expanded: true });
    for (const parent of frame.node.parents) {
      if (!visited.has(parent.id)) stack.push({ node: parent, expanded: false });
    }
  }

  return order;
}

/** Clear gradients on the whole graph reachable from `root`. */
export function zeroGrad(root: Var): void {
  for (const v of topoSort(root)) v.grad = null;
}

/**
 * Reverse sweep from a scalar root.
 *
 * The root has to be a scalar because that is the only case where "the
 * gradient of the output" is unambiguous — for anything else you would have to
 * say gradient of *what* scalar function of it.
 */
export function backward(root: Var): void {
  if (size(root.value) !== 1) {
    throw new Error(
      `backward: the root must be a scalar, got ${formatShapeTuple(root.value.shape)}. ` +
        'Reduce it first (sum or mean), or contract it against an upstream gradient.',
    );
  }
  backwardWithSeed(root, ones(root.value.shape));
}

/**
 * Reverse sweep from an arbitrary seed gradient on the root.
 *
 * `backward` is this with a seed of ones on a scalar. The general form exists
 * because building an explicit Jacobian means running the sweep once per output
 * element, each time seeded with a different basis vector — see jacobian.ts.
 */
export function backwardWithSeed(root: Var, seed: NdArray): void {
  if (!shapesEqual(seed.shape, root.value.shape)) {
    throw new Error(
      `backwardWithSeed: seed is ${formatShapeTuple(seed.shape)} but the root is ` +
        `${formatShapeTuple(root.value.shape)}`,
    );
  }

  const order = topoSort(root);
  root.grad = seed;

  for (let i = order.length - 1; i >= 0; i--) {
    const current = order[i];
    if (current.backwardFn === null || current.grad === null) continue;

    const parentGrads = current.backwardFn(current.grad);
    if (parentGrads.length !== current.parents.length) {
      throw new Error(
        `backward: op '${current.op}' produced ${parentGrads.length} gradients ` +
          `for ${current.parents.length} parents`,
      );
    }

    for (let p = 0; p < current.parents.length; p++) {
      const parent = current.parents[p];
      if (!parent.requiresGrad) continue;

      let contribution = parentGrads[p];
      if (!shapesEqual(contribution.shape, parent.value.shape)) {
        // A VJP that forgot to undo a broadcast lands here. Summing it back is
        // the right repair; anything unsummable is a genuine bug, and the op's
        // name is the single most useful thing to say about it.
        try {
          contribution = unbroadcast(contribution, parent.value.shape);
        } catch {
          throw new Error(
            `backward: op '${current.op}' produced a gradient of ` +
              `${formatShapeTuple(contribution.shape)} for parent ${p}, whose value is ` +
              `${formatShapeTuple(parent.value.shape)}. A gradient always has the shape of ` +
              'its variable (transformer_plan.md 规则总述).',
          );
        }
      }

      parent.grad = parent.grad === null ? contribution : ops.add(parent.grad, contribution);

      if (!shapesEqual(parent.grad.shape, parent.value.shape)) {
        throw new Error(
          `backward: op '${current.op}' produced a gradient of ` +
            `${formatShapeTuple(parent.grad.shape)} for a variable of ` +
            `${formatShapeTuple(parent.value.shape)}. A gradient always has the shape of ` +
            'its variable (transformer_plan.md 规则总述).',
        );
      }
    }
  }
}

/** The gradient of `v`, or zeros if nothing reached it. */
export function gradOf(v: Var): NdArray {
  return v.grad ?? zeros(v.value.shape);
}

// ── differentiable wrappers ─────────────────────────────────────────────────

export function add(a: Var, b: Var): Var {
  return node('add', ops.add(a.value, b.value), [a, b], (g) =>
    vjp.addVjp(g, a.value.shape, b.value.shape),
  );
}

export function sub(a: Var, b: Var): Var {
  return node('sub', ops.sub(a.value, b.value), [a, b], (g) =>
    vjp.subVjp(g, a.value.shape, b.value.shape),
  );
}

export function mul(a: Var, b: Var): Var {
  return node('mul', ops.mul(a.value, b.value), [a, b], (g) => vjp.mulVjp(g, a.value, b.value));
}

export function div(a: Var, b: Var): Var {
  return node('div', ops.div(a.value, b.value), [a, b], (g) => vjp.divVjp(g, a.value, b.value));
}

export function matmul(a: Var, b: Var): Var {
  return node('matmul', ops.matmul(a.value, b.value), [a, b], (g) =>
    vjp.matmulVjp(g, a.value, b.value),
  );
}

export function neg(a: Var): Var {
  return node('neg', ops.neg(a.value), [a], (g) => [vjp.negVjp(g)]);
}

export function mulScalar(a: Var, s: number): Var {
  return node('mulScalar', ops.mulScalar(a.value, s), [a], (g) => [vjp.mulScalarVjp(g, s)]);
}

export function exp(a: Var): Var {
  const out = ops.exp(a.value);
  return node('exp', out, [a], (g) => [vjp.expVjp(g, out)]);
}

export function log(a: Var): Var {
  return node('log', ops.log(a.value), [a], (g) => [vjp.logVjp(g, a.value)]);
}

export function sqrt(a: Var): Var {
  const out = ops.sqrt(a.value);
  return node('sqrt', out, [a], (g) => [vjp.sqrtVjp(g, out)]);
}

/** Caches the 1/0 mask, not the input — see vjp.reluVjp. */
export function relu(a: Var): Var {
  const mask = ops.positiveMask(a.value);
  return node('relu', ops.relu(a.value), [a], (g) => [vjp.reluVjp(g, mask)]);
}

export function sum(a: Var, axis: ops.Axis = null, keepdims = false): Var {
  return node('sum', ops.sum(a.value, axis, keepdims), [a], (g) => [
    vjp.sumVjp(g, a.value.shape, axis, keepdims),
  ]);
}

export function mean(a: Var, axis: ops.Axis = null, keepdims = false): Var {
  return node('mean', ops.mean(a.value, axis, keepdims), [a], (g) => [
    vjp.meanVjp(g, a.value.shape, axis, keepdims),
  ]);
}

export function max(a: Var, axis: ops.Axis = null, keepdims = false): Var {
  const out = ops.max(a.value, axis, keepdims);
  return node('max', out, [a], (g) => [vjp.maxVjp(g, a.value, out, axis, keepdims)]);
}

export function reshape(a: Var, shape: readonly number[]): Var {
  return node('reshape', reshapeRaw(a.value, shape), [a], (g) => [
    vjp.reshapeVjp(g, a.value.shape),
  ]);
}

export function permute(a: Var, axes: readonly number[]): Var {
  return node('permute', permuteRaw(a.value, axes), [a], (g) => [vjp.permuteVjp(g, axes)]);
}

export function softmax(a: Var, axis = -1, options: sm.SoftmaxOptions = {}): Var {
  const out = sm.softmax(a.value, axis, options);
  return node('softmax', out, [a], (g) => [sm.softmaxVjp(out, g, axis)]);
}

export function logSoftmax(a: Var, axis = -1): Var {
  const out = sm.logSoftmax(a.value, axis);
  return node('logSoftmax', out, [a], (g) => [sm.logSoftmaxVjp(g, out, axis)]);
}

/**
 * Fused softmax + cross-entropy. The backward is `(s - onehot(y)) / N`, which
 * is why this is one node in the graph rather than three.
 */
export function crossEntropy(logits: Var, targets: ArrayLike<number>): Var {
  const { loss, probs } = sm.crossEntropyFromLogits(logits.value, targets);
  const value = new NdArray({ data: Float64Array.of(loss), shape: [] });
  return node('crossEntropy', value, [logits], (g) => [
    sm.crossEntropyVjp(probs, targets, g.data[g.offset]),
  ]);
}

/** Grouped alias, so a lab can write `A.matmul(x, w)` and read as maths. */
export const A = {
  add,
  sub,
  mul,
  div,
  matmul,
  neg,
  mulScalar,
  exp,
  log,
  sqrt,
  relu,
  sum,
  mean,
  max,
  reshape,
  permute,
  softmax,
  logSoftmax,
  crossEntropy,
};
