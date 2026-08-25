/**
 * Grouping a recorded trace into something a playhead can step through.
 *
 * Two granularities, on purpose:
 *
 *   'frame'  one source statement — what a person reads, and what the code
 *            pane highlights. The default.
 *   'event'  one primitive operation — what Step 0.2's graph wants to
 *            single-step through.
 *
 * Forward and backward are separable because most of Steps 0.3 and 0.4 is about
 * replaying only the backward pass.
 */

import type { L } from '../i18n.ts';
import type { Trace, TraceEvent } from './types.ts';

export interface TraceFrame {
  /** Position within the (possibly filtered) frame list. */
  index: number;
  /** Index of the program step this frame came from. */
  step: number;
  phase: 'forward' | 'backward';
  code: string;
  lineStart: number;
  lineEnd: number;
  note?: L;
  /** The primitive operations this statement produced. May be empty. */
  events: readonly TraceEvent[];
}

export type Granularity = 'frame' | 'event';

/**
 * One frame per program step, in order.
 *
 * A step that emitted no operations still gets a frame — the code line should
 * still be highlightable while nothing computes.
 */
export function framesFor(trace: Trace, phase?: 'forward' | 'backward'): TraceFrame[] {
  const byStep = new Map<number, TraceEvent[]>();
  for (const event of trace.events) {
    const bucket = byStep.get(event.step);
    if (bucket) bucket.push(event);
    else byStep.set(event.step, [event]);
  }

  const frames: TraceFrame[] = [];
  for (const step of trace.steps) {
    if (phase !== undefined && step.phase !== phase) continue;
    frames.push({
      index: frames.length,
      step: step.index,
      phase: step.phase,
      code: step.code,
      lineStart: step.lineStart,
      lineEnd: step.lineEnd,
      note: step.note,
      events: byStep.get(step.index) ?? [],
    });
  }
  return frames;
}

/** The frame containing a given event, by its trace-wide index. */
export function frameAtEvent(
  frames: readonly TraceFrame[],
  eventIndex: number,
): TraceFrame | undefined {
  return frames.find((frame) => frame.events.some((e) => e.index === eventIndex));
}

/** How many playhead positions this frame list has, at the given granularity. */
export function positionCount(frames: readonly TraceFrame[], granularity: Granularity): number {
  if (granularity === 'frame') return frames.length;
  let total = 0;
  for (const frame of frames) total += frame.events.length;
  return total;
}

/**
 * Everything that has happened through `position`, inclusive.
 *
 * `position < 0` means "nothing yet", so a playhead can sit before the first
 * step. Monotone by construction: position n+1 is always a superset of n.
 */
export function eventsUpTo(
  frames: readonly TraceFrame[],
  position: number,
  granularity: Granularity = 'frame',
): TraceEvent[] {
  if (position < 0) return [];

  if (granularity === 'frame') {
    const out: TraceEvent[] = [];
    for (let i = 0; i <= position && i < frames.length; i++) out.push(...frames[i].events);
    return out;
  }

  const flat = flattenEvents(frames);
  return flat.slice(0, Math.min(position + 1, flat.length));
}

/** All events across the frame list, in order. */
export function flattenEvents(frames: readonly TraceFrame[]): TraceEvent[] {
  const out: TraceEvent[] = [];
  for (const frame of frames) out.push(...frame.events);
  return out;
}

/** The frame a playhead position lands in, at either granularity. */
export function frameAtPosition(
  frames: readonly TraceFrame[],
  position: number,
  granularity: Granularity = 'frame',
): TraceFrame | undefined {
  if (position < 0) return undefined;
  if (granularity === 'frame') return frames[position];

  let remaining = position;
  for (const frame of frames) {
    if (remaining < frame.events.length) return frame;
    remaining -= frame.events.length;
  }
  return undefined;
}
