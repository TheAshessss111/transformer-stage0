/**
 * A program a lab can both run and show.
 *
 * The problem this solves: a lab needs source text for the code pane, a line
 * range to highlight, playback frames, and names for its intermediate tensors.
 * Instrumenting individual operators would deliver none of them well — it is
 * verbose at every call site, it breaks whenever the code is edited, and it
 * gives the playhead the wrong granularity, because a person reads a statement
 * rather than a primitive operation.
 *
 * So a lab declares its forward pass once, as an ordered list of steps:
 *
 *     defineProgram<SoftmaxState>({
 *       id: 'softmax-forward',
 *       language: 'python',
 *       steps: [
 *         {
 *           code: 'row_max = x.max(axis=-1, keepdims=True)',
 *           note: { zh: '每一行取最大值', en: 'Row maxima' },
 *           run: (s) => ({ rowMax: max(s.x, -1, true) }),
 *         },
 *         …
 *       ],
 *     })
 *
 * and gets all four back. The last one is the important one: **the state keys
 * are the tensor names**. `s.rowMax` becomes addressable as
 * `{ kind: 'tensor', name: 'rowMax' }`, which is what gives a formula term
 * something to point at.
 */

import type { L } from '../i18n.ts';

export type ProgramLanguage = 'python' | 'typescript';

export interface ProgramStep<S> {
  /** Source text for the code pane. May span several lines. */
  code: string;
  /** "What is happening here", in both languages. */
  note?: L;
  /**
   * Runs the step. The returned keys are merged into the running state, and
   * each key becomes the name of the tensor it holds.
   */
  run: (state: Readonly<S>) => Partial<S>;
  /**
   * Which pass this step belongs to. Defaults to 'forward'; set 'backward' on
   * the steps that compute gradients so the two can be replayed separately.
   */
  phase?: 'forward' | 'backward';
}

/** Inclusive, 1-indexed line range within the joined source. */
export interface LineRange {
  start: number;
  end: number;
}

export interface TracedProgram<S> {
  id: string;
  language: ProgramLanguage;
  steps: readonly ProgramStep<S>[];
  /** All step sources joined with newlines — what the code pane renders. */
  source: string;
  /** Line range of each step, parallel to `steps`. */
  lineRanges: readonly LineRange[];
}

export interface ProgramSpec<S> {
  id: string;
  language: ProgramLanguage;
  steps: readonly ProgramStep<S>[];
}

/**
 * Precomputes the joined source and each step's line range, so the code pane
 * can highlight a range without re-deriving it on every frame.
 */
export function defineProgram<S>(spec: ProgramSpec<S>): TracedProgram<S> {
  if (spec.steps.length === 0) {
    throw new Error(`defineProgram: program '${spec.id}' has no steps`);
  }

  const lineRanges: LineRange[] = [];
  let cursor = 1;

  for (const step of spec.steps) {
    const lineCount = step.code.split('\n').length;
    lineRanges.push({ start: cursor, end: cursor + lineCount - 1 });
    cursor += lineCount;
  }

  return {
    id: spec.id,
    language: spec.language,
    steps: spec.steps,
    source: spec.steps.map((s) => s.code).join('\n'),
    lineRanges,
  };
}

/** The step containing a 1-indexed source line, or -1. */
export function stepAtLine<S>(program: TracedProgram<S>, line: number): number {
  return program.lineRanges.findIndex((r) => line >= r.start && line <= r.end);
}

/** The phase a step belongs to, applying the 'forward' default. */
export function phaseOfStep<S>(program: TracedProgram<S>, step: number): 'forward' | 'backward' {
  return program.steps[step]?.phase ?? 'forward';
}
