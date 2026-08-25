import { useEffect, useRef, useState } from 'react';
import { runProgram } from '../core/trace/recorder';
import type { TracedProgram } from '../core/trace/program';
import type { Trace } from '../core/trace/types';

/**
 * Run a TracedProgram from React, at most once per animation frame.
 *
 * The risk is not that a run is slow — teaching-scale tensors are a few
 * thousand elements — but that a slider drag fires several runs per frame.
 * Scheduling through requestAnimationFrame coalesces them: nothing queues up
 * behind a drag, so releasing the slider never leaves a backlog to chew through.
 *
 * ── The recompute contract ─────────────────────────────────────────────────
 *
 * `makeInitial` should be wrapped in `useCallback` by the caller; its dependency
 * list is what decides when the program re-runs. That is the ordinary React
 * idiom rather than a bespoke deps array or key, and it keeps this hook's own
 * dependency list a literal, which exhaustive-deps requires.
 *
 * Forgetting the useCallback is not a correctness bug — it re-runs on every
 * render, and coalescing still caps that at once per frame — but the timing
 * readout on /dev/trace will show the extra runs.
 */

export interface RunTiming {
  /** Duration of the most recent run, in ms. */
  last: number;
  /** Over the last {@link HISTORY} runs, not over `runs`. */
  p50: number;
  p95: number;
  /** Cumulative since mount — the number the rAF coalescing claim is read off. */
  runs: number;
}

export interface TracedRun<S> {
  state: S | null;
  trace: Trace | null;
  timing: RunTiming;
}

/** NFR-2: 60fps means a 16ms budget; half of it is left for React and paint. */
export const RUN_BUDGET_MS = 8;

const HISTORY = 64;
const warned = new Set<string>();

/**
 * Nearest-rank percentile.
 *
 * `floor` lands one rank too high: with 20 samples it puts p95 at index 19, the
 * maximum. The readout on /dev/trace is the evidence for NFR-2, so it has to
 * measure the thing it names.
 */
function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return 0;
  const rank = Math.ceil(fraction * sorted.length) - 1;
  return sorted[Math.min(sorted.length - 1, Math.max(0, rank))];
}

export function useTracedProgram<S extends object>(
  program: TracedProgram<S>,
  makeInitial: () => S,
): TracedRun<S> {
  const [result, setResult] = useState<{ state: S; trace: Trace } | null>(null);
  const [timing, setTiming] = useState<RunTiming>({ last: 0, p50: 0, p95: 0, runs: 0 });

  const durations = useRef<number[]>([]);
  const handle = useRef<number | null>(null);

  useEffect(() => {
    // The coalescing is the cleanup's doing: React runs it before re-running
    // this effect, so a pending callback is cancelled and re-scheduled with the
    // newer makeInitial. The latest inputs always win, and a drag that changes
    // inputs several times within one frame still produces exactly one run.
    handle.current = requestAnimationFrame(() => {
      handle.current = null;

      const run = runProgram(program, makeInitial());
      setResult(run);

      const history = durations.current;
      history.push(run.trace.durationMs);
      if (history.length > HISTORY) history.shift();

      const sorted = [...history].sort((a, b) => a - b);
      const p50 = percentile(sorted, 0.5);
      const p95 = percentile(sorted, 0.95);
      setTiming((previous) => ({
        last: run.trace.durationMs,
        p50,
        p95,
        runs: previous.runs + 1,
      }));

      if (import.meta.env.DEV && p95 > RUN_BUDGET_MS && !warned.has(program.id)) {
        warned.add(program.id);
        console.warn(
          `[useTracedProgram] '${program.id}' p95 is ${p95.toFixed(1)}ms, over the ` +
            `${RUN_BUDGET_MS}ms budget (NFR-2). Reduce the tensor sizes or split the program.`,
        );
      }
    });

    return () => {
      if (handle.current !== null) {
        cancelAnimationFrame(handle.current);
        handle.current = null;
      }
    };
  }, [program, makeInitial]);

  return { state: result?.state ?? null, trace: result?.trace ?? null, timing };
}
