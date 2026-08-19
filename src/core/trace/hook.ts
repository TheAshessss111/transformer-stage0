import type { NdArray } from '../tensor/ndarray.ts';

/**
 * The seam that lets E0.3's recorder observe every operation without any op
 * knowing about it.
 *
 * This module is deliberately introduced in E0.2 rather than E0.3: if the call
 * sites were added later, every operator written in E0.2 would have to be
 * rewritten. The *recorder* is still E0.3 — this is only the hook.
 *
 * When no hook is attached, `emit` is a single null check.
 */
export interface RawOpEvent {
  /** Operator name, e.g. 'matmul', 'transpose', 'softmax'. */
  op: string;
  phase: 'forward' | 'backward';
  inputs: readonly NdArray[];
  output: NdArray;
  /** Output shares its buffer with an input (zero-copy). */
  isView: boolean;
  /** This step allocated a new buffer and copied elements into it. */
  didCopy: boolean;
  /** How many elements were copied, when didCopy is true. */
  copiedElements?: number;
  /** Operator-specific detail: axis, keepdims, permutation, epsilon… */
  meta?: Readonly<Record<string, unknown>>;
}

export type OpHook = (event: RawOpEvent) => void;

let hook: OpHook | null = null;

export function setOpHook(next: OpHook | null): void {
  hook = next;
}

export function getOpHook(): OpHook | null {
  return hook;
}

export function emit(event: RawOpEvent): void {
  hook?.(event);
}

/**
 * Run `fn` with `hook` attached, restoring the previous hook afterwards.
 * Used by the recorder so nested recordings cannot leak.
 */
export function withOpHook<T>(next: OpHook, fn: () => T): T {
  const previous = hook;
  hook = next;
  try {
    return fn();
  } finally {
    hook = previous;
  }
}

/**
 * Run `fn` with no hook attached.
 *
 * Numerical gradient checking evaluates the forward pass thousands of times;
 * those evaluations are an internal procedure, not steps of the computation the
 * user is looking at, so they must never reach a recorder.
 */
export function suppressOpHook<T>(fn: () => T): T {
  const previous = hook;
  hook = null;
  try {
    return fn();
  } finally {
    hook = previous;
  }
}
