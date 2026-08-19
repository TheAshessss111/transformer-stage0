/** Assertions shared across check modules. */

import { expect } from './harness.ts';
import { NdArray, copyFlat } from '../ndarray.ts';
import { setOpHook, type RawOpEvent } from '../../trace/hook.ts';

export function throws(fn: () => unknown, fragment: string): void {
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
export function record(fn: () => void): RawOpEvent[] {
  const events: RawOpEvent[] = [];
  setOpHook((e) => events.push(e));
  try {
    fn();
  } finally {
    setOpHook(null);
  }
  return events;
}

export function flatEquals(a: NdArray, expected: readonly number[], tol = 0): void {
  const actual = Array.from(copyFlat(a));
  expect(actual.length === expected.length, `length ${actual.length} != ${expected.length}`);
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    expect(diff <= tol, `element ${i}: got ${actual[i]}, expected ${expected[i]} (diff ${diff})`);
  }
}
