/**
 * Trace checks — T21 onward.
 *
 * The program model, the recorder's identity and edge resolution, and the frame
 * grouping that the playhead and code pane read.
 */

import { check, expect, group } from './harness.ts';
import { throws } from './assertions.ts';
import { defineProgram, phaseOfStep, stepAtLine } from '../../trace/program.ts';
import { runProgram } from '../../trace/recorder.ts';
import {
  eventsUpTo,
  flattenEvents,
  frameAtEvent,
  frameAtPosition,
  framesFor,
  positionCount,
} from '../../trace/replay.ts';
import { NdArray, zeros } from '../ndarray.ts';
import { exp, matmul, mul, relu, sum } from '../ops.ts';
import { ascontiguousarray, reshape, transpose } from '../shape.ts';
import { softmax, softmaxVjp } from '../softmax.ts';
import { randn } from '../random.ts';

group('trace · program model');

const SAMPLE = defineProgram<{ a: unknown; b: unknown }>({
  id: 'sample',
  language: 'python',
  steps: [
    { code: 'a = 1', run: () => ({ a: 1 }) },
    { code: 'b = (\n    a\n    + 1\n)', run: () => ({ b: 2 }) },
    { code: 'c = b', phase: 'backward', run: () => ({}) },
  ],
});

check('line ranges are 1-indexed and inclusive, across multi-line steps', () => {
  const got = SAMPLE.lineRanges.map((r) => `${r.start}-${r.end}`).join(' ');
  expect(got === '1-1 2-5 6-6', `got ${got}`);
  return got;
});

check('source round-trips: slicing by each range returns that step back', () => {
  const lines = SAMPLE.source.split('\n');
  SAMPLE.steps.forEach((step, i) => {
    const range = SAMPLE.lineRanges[i];
    const sliced = lines.slice(range.start - 1, range.end).join('\n');
    expect(sliced === step.code, `step ${i}: got ${JSON.stringify(sliced)}`);
  });
  return `${SAMPLE.steps.length} steps`;
});

check('stepAtLine maps every source line back to its step', () => {
  const expected = [0, 1, 1, 1, 1, 2];
  expected.forEach((step, i) => {
    const got = stepAtLine(SAMPLE, i + 1);
    expect(got === step, `line ${i + 1}: got step ${got}, expected ${step}`);
  });
  expect(stepAtLine(SAMPLE, 99) === -1, 'a line past the end should be -1');
});

check('phase defaults to forward and is honoured when set', () => {
  expect(phaseOfStep(SAMPLE, 0) === 'forward', 'step 0');
  expect(phaseOfStep(SAMPLE, 2) === 'backward', 'step 2');
});

check('a program with no steps is refused', () => {
  throws(() => defineProgram({ id: 'empty', language: 'python', steps: [] }), 'has no steps');
});

check('a single-line program is the degenerate case, not a special case', () => {
  const one = defineProgram<{ x: unknown }>({
    id: 'one',
    language: 'typescript',
    steps: [{ code: 'const x = zeros([2]);', run: () => ({ x: zeros([2]) }) }],
  });
  expect(one.lineRanges.length === 1, 'one range');
  expect(one.lineRanges[0].start === 1 && one.lineRanges[0].end === 1, 'range is 1-1');
  expect(one.source === 'const x = zeros([2]);', 'source is the step');
});

// ── recorder ────────────────────────────────────────────────────────────────

group('trace · recorder');

interface ChainState {
  x: NdArray;
  w: NdArray;
  h: NdArray;
  y: NdArray;
  loss: NdArray;
}

function chainProgram() {
  return defineProgram<ChainState>({
    id: 'chain',
    language: 'typescript',
    steps: [
      { code: 'h = x @ w', run: (s) => ({ h: matmul(s.x, s.w) }) },
      { code: 'y = relu(h)', run: (s) => ({ y: relu(s.h) }) },
      { code: 'loss = y.sum()', phase: 'backward', run: (s) => ({ loss: sum(s.y) }) },
    ],
  });
}

function chainInputs(seed: number): ChainState {
  return {
    x: randn([3, 4], seed),
    w: randn([4, 5], seed + 1),
    h: zeros([]),
    y: zeros([]),
    loss: zeros([]),
  };
}

check('producer edges resolve along a chain', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(1));
  const matmulEvent = trace.events.find((e) => e.op === 'matmul');
  const reluEvent = trace.events.find((e) => e.op === 'relu');
  expect(matmulEvent !== undefined && reluEvent !== undefined, 'missing events');
  expect(
    matmulEvent!.inputs.every((i) => i.event === null),
    'matmul inputs are program inputs, so their producer is null',
  );
  expect(
    reluEvent!.inputs[0].event === matmulEvent!.index,
    `relu should consume event ${matmulEvent!.index}, got ${reluEvent!.inputs[0].event}`,
  );
  return `matmul#${matmulEvent!.index} -> relu#${reluEvent!.index}`;
});

check('program inputs and produced tensors both get their state-key names', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(2));
  const matmulEvent = trace.events.find((e) => e.op === 'matmul')!;
  expect(matmulEvent.inputs[0].name === 'x', `got ${matmulEvent.inputs[0].name}`);
  expect(matmulEvent.inputs[1].name === 'w', `got ${matmulEvent.inputs[1].name}`);
  expect(matmulEvent.output.name === 'h', `got ${matmulEvent.output.name}`);
  expect([...trace.names.keys()].join(',') === 'x,w,h,y,loss', [...trace.names.keys()].join(','));
  return 'x, w -> h';
});

check('a diamond resolves both branches to the same producer', () => {
  interface S {
    a: NdArray;
    left: NdArray;
    right: NdArray;
    out: NdArray;
  }
  const program = defineProgram<S>({
    id: 'diamond',
    language: 'typescript',
    steps: [
      { code: 'left = exp(a)', run: (s) => ({ left: exp(s.a) }) },
      { code: 'right = relu(a)', run: (s) => ({ right: relu(s.a) }) },
      { code: 'out = left * right', run: (s) => ({ out: mul(s.left, s.right) }) },
    ],
  });
  const { trace } = runProgram(program, {
    a: randn([2, 3], 3),
    left: zeros([]),
    right: zeros([]),
    out: zeros([]),
  });
  const mulEvent = trace.events.find((e) => e.op === 'mul')!;
  const expEvent = trace.events.find((e) => e.op === 'exp')!;
  const reluEvent = trace.events.find((e) => e.op === 'relu')!;
  expect(mulEvent.inputs[0].event === expEvent.index, 'left branch');
  expect(mulEvent.inputs[1].event === reluEvent.index, 'right branch');
});

check('a no-op ascontiguousarray is marked passthrough and does not claim authorship', () => {
  interface S {
    x: NdArray;
    c: NdArray;
    y: NdArray;
  }
  const program = defineProgram<S>({
    id: 'passthrough',
    language: 'typescript',
    steps: [
      // x is already contiguous, so this returns the very same object
      { code: 'c = ascontiguousarray(x)', run: (s) => ({ c: ascontiguousarray(s.x) }) },
      { code: 'y = relu(c)', run: (s) => ({ y: relu(s.c) }) },
    ],
  });
  const { trace } = runProgram(program, { x: randn([2, 3], 4), c: zeros([]), y: zeros([]) });

  const pass = trace.events.find((e) => e.op === 'ascontiguousarray')!;
  expect(pass.passthrough, 'a no-op ascontiguousarray must be flagged passthrough');
  expect(!pass.didCopy, 'and must not report a copy');

  const reluEvent = trace.events.find((e) => e.op === 'relu')!;
  expect(
    reluEvent.inputs[0].event === null,
    `relu should still see a program input, not the no-op (got ${reluEvent.inputs[0].event})`,
  );
  return 'passthrough does not become a self-edge';
});

check('a real copy is NOT passthrough and reports its element count', () => {
  interface S {
    x: NdArray;
    t: NdArray;
    flat: NdArray;
  }
  const program = defineProgram<S>({
    id: 'copy',
    language: 'typescript',
    steps: [
      { code: 't = x.T', run: (s) => ({ t: transpose(s.x) }) },
      { code: 'flat = t.reshape(-1)', run: (s) => ({ flat: reshape(s.t, [-1]) }) },
    ],
  });
  const { trace } = runProgram(program, { x: randn([3, 4], 5), t: zeros([]), flat: zeros([]) });
  const reshapeEvent = trace.events.find((e) => e.op === 'reshape')!;
  expect(!reshapeEvent.passthrough, 'a copying reshape is not a passthrough');
  expect(reshapeEvent.didCopy, 'it should report a copy');
  expect(reshapeEvent.copiedElements === 12, `got ${reshapeEvent.copiedElements}`);
  return 'copy - 12 elements';
});

check('identity is stable across recomputes with different values', () => {
  const a = runProgram(chainProgram(), chainInputs(10)).trace;
  const b = runProgram(chainProgram(), chainInputs(99)).trace;
  const signature = (t: typeof a) =>
    t.events.map((e) => `${e.index}:${e.step}:${e.op}:${e.phase}`).join('|');
  expect(signature(a) === signature(b), 'event sequence drifted between runs');
  expect(a.events.length > 0, 'no events recorded at all');
  return `${a.events.length} events, identical signature`;
});

check('steps carry their code, line range, note and phase', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(11));
  expect(trace.steps.length === 3, `got ${trace.steps.length} steps`);
  expect(trace.steps[0].code === 'h = x @ w', trace.steps[0].code);
  expect(trace.steps[2].phase === 'backward', 'step 2 is backward');
  expect(trace.steps[1].lineStart === 2 && trace.steps[1].lineEnd === 2, 'line range');
  expect(trace.source.split('\n').length === 3, 'source has one line per step here');
});

check('suppression holds: softmaxVjp contributes one backward event, not four', () => {
  interface S {
    x: NdArray;
    s: NdArray;
    g: NdArray;
    dx: NdArray;
  }
  const program = defineProgram<S>({
    id: 'softmax-backward',
    language: 'typescript',
    steps: [
      { code: 's = softmax(x)', run: (st) => ({ s: softmax(st.x, -1) }) },
      {
        code: 'dx = softmax_vjp(s, g)',
        phase: 'backward',
        run: (st) => ({ dx: softmaxVjp(st.s, st.g, -1) }),
      },
    ],
  });
  const { trace } = runProgram(program, {
    x: randn([2, 4], 6),
    s: zeros([]),
    g: randn([2, 4], 7),
    dx: zeros([]),
  });
  const backward = trace.events.filter((e) => e.phase === 'backward');
  expect(backward.length === 1, `expected 1 backward event, got ${backward.length}`);
  expect(backward[0].op === 'softmax', `got op ${backward[0].op}`);
  const forward = trace.events.filter((e) => e.phase === 'forward');
  expect(forward.length === 1, `expected 1 forward event, got ${forward.length}`);
  return 'one event per operator, not per primitive';
});

check('nested runProgram restores the outer hook', () => {
  const outer = defineProgram<{ x: NdArray; y: NdArray }>({
    id: 'outer',
    language: 'typescript',
    steps: [
      {
        code: 'y = relu(x)  # runs an inner program first',
        run: (s) => {
          runProgram(
            defineProgram<{ a: NdArray; b: NdArray }>({
              id: 'inner',
              language: 'typescript',
              steps: [{ code: 'b = exp(a)', run: (i) => ({ b: exp(i.a) }) }],
            }),
            { a: randn([2], 8), b: zeros([]) },
          );
          return { y: relu(s.x) };
        },
      },
    ],
  });
  const { trace } = runProgram(outer, { x: randn([2], 9), y: zeros([]) });
  const ops = trace.events.map((e) => e.op);
  expect(ops.includes('relu'), 'the outer relu must still be recorded');
  expect(!ops.includes('exp'), `the inner program leaked into the outer trace: ${ops.join(',')}`);
  return 'inner events stayed inside';
});

check('durationMs is measured', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(12));
  expect(trace.durationMs >= 0 && Number.isFinite(trace.durationMs), `got ${trace.durationMs}`);
  return `${trace.durationMs.toFixed(3)} ms`;
});

// ── frames and replay ───────────────────────────────────────────────────────

group('trace · frames and replay');

check('one frame per step, including steps that computed nothing', () => {
  interface S {
    x: NdArray;
    y: NdArray;
  }
  const program = defineProgram<S>({
    id: 'quiet-step',
    language: 'typescript',
    steps: [
      { code: '# nothing happens on this line', run: () => ({}) },
      { code: 'y = relu(x)', run: (s) => ({ y: relu(s.x) }) },
    ],
  });
  const { trace } = runProgram(program, { x: randn([2], 20), y: zeros([]) });
  const frames = framesFor(trace);
  expect(frames.length === 2, `got ${frames.length} frames`);
  expect(frames[0].events.length === 0, 'the quiet step still gets a frame');
  expect(frames[1].events.length === 1, 'the working step has its event');
  return 'a step that computes nothing is still highlightable';
});

check('forward and backward split into independently replayable lists', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(21));
  const all = framesFor(trace);
  const fwd = framesFor(trace, 'forward');
  const bwd = framesFor(trace, 'backward');
  expect(all.length === 3, `all: ${all.length}`);
  expect(fwd.length === 2, `forward: ${fwd.length}`);
  expect(bwd.length === 1, `backward: ${bwd.length}`);
  expect(bwd[0].index === 0, 'a filtered list renumbers from 0');
  expect(bwd[0].step === 2, 'but keeps the original step index');
  return 'forward 2 / backward 1, renumbered but traceable';
});

check('eventsUpTo is monotone at both granularities', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(22));
  const frames = framesFor(trace);
  for (const granularity of ['frame', 'event'] as const) {
    const total = positionCount(frames, granularity);
    let previous: number[] = [];
    for (let p = -1; p < total; p++) {
      const ids = eventsUpTo(frames, p, granularity).map((e) => e.index);
      expect(
        previous.every((id) => ids.includes(id)),
        `${granularity} position ${p} dropped an event that position ${p - 1} had`,
      );
      expect(ids.length >= previous.length, `${granularity} position ${p} shrank`);
      previous = ids;
    }
    expect(
      previous.length === trace.events.length,
      `${granularity}: the final position should hold every event`,
    );
  }
  return 'frame and event granularity both monotone';
});

check('position -1 means nothing has happened yet', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(23));
  const frames = framesFor(trace);
  expect(eventsUpTo(frames, -1).length === 0, 'should be empty');
  expect(frameAtPosition(frames, -1) === undefined, 'no frame before the start');
});

check('frameAtEvent and frameAtPosition agree', () => {
  const { trace } = runProgram(chainProgram(), chainInputs(24));
  const frames = framesFor(trace);
  const flat = flattenEvents(frames);
  flat.forEach((event, position) => {
    const viaEvent = frameAtEvent(frames, event.index);
    const viaPosition = frameAtPosition(frames, position, 'event');
    expect(viaEvent !== undefined, `no frame for event ${event.index}`);
    expect(viaEvent === viaPosition, `event ${event.index}: the two lookups disagree`);
  });
  return `${flat.length} events cross-checked`;
});
