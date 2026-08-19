/**
 * The recorded shape of one execution.
 *
 * `RawOpEvent` (hook.ts) is what an operator emits. `TraceEvent` is what the
 * recorder turns it into: identity, resolved producer edges, and names.
 *
 * ── The immutability contract ──────────────────────────────────────────────
 *
 * A Trace holds live NdArray references, and some of them are views sharing a
 * buffer. So a Trace is valid ONLY for the execution that produced it, and
 * nothing may mutate an array reachable from one.
 *
 * Editing an input means changing the lab's state and re-running the program,
 * which produces a new Trace. This matters for F0.5.3 (click a cell to edit):
 * the edit goes through lab state, never into a recorded array. Stated here
 * because violating it produces a bug that is impossible to find later.
 */

import type { NdArray } from '../tensor/ndarray.ts';
import type { L } from '../i18n.ts';
import type { ProgramLanguage } from './program.ts';

export interface TraceInput {
  /** Index of the event that produced this array, or null if it is a program input. */
  event: number | null;
  /** State-key name, if this array has one. */
  name: string | null;
  shape: readonly number[];
}

export interface TraceOutput {
  name: string | null;
  shape: readonly number[];
  /** See the immutability contract above. */
  array: NdArray;
}

export interface TraceEvent {
  /**
   * Position in the sequence, and the stable identity.
   *
   * NdArray has no id, and assigning one per recording would change on every
   * recompute -- which would drop whatever the user was hovering every time a
   * slider moved. The same program run with different values emits the same
   * sequence of operations, so event 7 is event 7 across recomputes.
   */
  index: number;
  /** Which program step emitted it. */
  step: number;
  phase: 'forward' | 'backward';
  op: string;
  inputs: readonly TraceInput[];
  output: TraceOutput;
  isView: boolean;
  didCopy: boolean;
  copiedElements?: number;
  /**
   * The output IS one of the inputs -- `ascontiguousarray` returning its
   * argument unchanged, which is part of the engine's view/copy contract.
   * Such events are skipped when building the graph (they would be self-edges)
   * but stay in the event list, because "this call cost nothing" is exactly
   * what Step 0.1 wants to show.
   */
  passthrough: boolean;
  meta?: Readonly<Record<string, unknown>>;
}

/** One program step, carried inside the Trace so the UI needs nothing else. */
export interface TraceStep {
  index: number;
  code: string;
  lineStart: number;
  lineEnd: number;
  note?: L;
  phase: 'forward' | 'backward';
}

export interface Trace {
  programId: string;
  language: ProgramLanguage;
  /** The joined program source, for the code pane. */
  source: string;
  steps: readonly TraceStep[];
  events: readonly TraceEvent[];
  /** Named tensors from the program state, in the order they were produced. */
  names: ReadonlyMap<string, NdArray>;
  durationMs: number;
}
