/**
 * Runs a TracedProgram and records what happened.
 *
 * Plugs into the seam planted in E0.2: every operator already calls `emit`, so
 * nothing here needs the operators' cooperation.
 */

import { NdArray } from '../tensor/ndarray.ts';
import { withOpHook, type RawOpEvent } from './hook.ts';
import { phaseOfStep, type TracedProgram } from './program.ts';
import type { Trace, TraceEvent, TraceInput, TraceStep } from './types.ts';

interface Captured {
  raw: RawOpEvent;
  step: number;
}

function isNdArray(value: unknown): value is NdArray {
  return value instanceof NdArray;
}

export interface RunResult<S> {
  state: S;
  trace: Trace;
}

/**
 * Run every step in order, then resolve identity, producer edges and names.
 *
 * Two passes on purpose: an operator's output has no name at the moment it is
 * emitted, because names come from the state keys the step returns, which only
 * exist once the step has finished.
 */
export function runProgram<S extends object>(program: TracedProgram<S>, initial: S): RunResult<S> {
  const captured: Captured[] = [];
  const names = new Map<string, NdArray>();
  const nameOf = new WeakMap<NdArray, string>();

  const register = (key: string, value: unknown): void => {
    if (!isNdArray(value)) return;
    names.set(key, value);
    nameOf.set(value, key);
  };

  for (const [key, value] of Object.entries(initial)) register(key, value);

  // A private copy: steps accumulate into it, the caller's object is untouched.
  const state: S = { ...initial };
  let currentStep = 0;

  const started = performance.now();

  withOpHook(
    (raw) => {
      captured.push({ raw, step: currentStep });
    },
    () => {
      for (let i = 0; i < program.steps.length; i++) {
        currentStep = i;
        const produced = program.steps[i].run(state);
        Object.assign(state, produced);
        for (const [key, value] of Object.entries(produced)) register(key, value);
      }
    },
  );

  const durationMs = performance.now() - started;

  // Pass 2: identity and edges.
  const producedBy = new Map<NdArray, number>();
  const events: TraceEvent[] = [];

  for (let index = 0; index < captured.length; index++) {
    const { raw, step } = captured[index];
    const passthrough = raw.inputs.some((input) => input === raw.output);

    const inputs: TraceInput[] = raw.inputs.map((input) => ({
      event: producedBy.get(input) ?? null,
      name: nameOf.get(input) ?? null,
      shape: input.shape,
    }));

    events.push({
      index,
      step,
      phase: raw.phase,
      op: raw.op,
      inputs,
      output: {
        name: nameOf.get(raw.output) ?? null,
        shape: raw.output.shape,
        array: raw.output,
      },
      isView: raw.isView,
      didCopy: raw.didCopy,
      copiedElements: raw.copiedElements,
      passthrough,
      meta: raw.meta,
    });

    // A passthrough must NOT claim authorship, or every later consumer would
    // point at the no-op instead of the operation that really produced the data.
    if (!passthrough) producedBy.set(raw.output, index);
  }

  const steps: TraceStep[] = program.steps.map((step, index) => ({
    index,
    code: step.code,
    lineStart: program.lineRanges[index].start,
    lineEnd: program.lineRanges[index].end,
    note: step.note,
    phase: phaseOfStep(program, index),
  }));

  return {
    state,
    trace: {
      programId: program.id,
      language: program.language,
      source: program.source,
      steps,
      events,
      names,
      durationMs,
    },
  };
}
