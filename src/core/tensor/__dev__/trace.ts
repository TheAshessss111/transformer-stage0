/**
 * Trace checks — T21 onward.
 *
 * The program model, the recorder's identity and edge resolution, and the frame
 * grouping that the playhead and code pane read.
 */

import { check, expect, group } from './harness.ts';
import { throws } from './assertions.ts';
import { defineProgram, phaseOfStep, stepAtLine } from '../../trace/program.ts';
import { zeros } from '../ndarray.ts';

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
