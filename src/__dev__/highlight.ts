/**
 * Highlight protocol checks — T28 onward.
 *
 * The matcher is generic infrastructure every linked view depends on, and it is
 * pure logic with no DOM, so it belongs in the harness rather than being
 * eyeballed on a page.
 */

import { check, expect, group } from './harness.ts';
import {
  overlaps,
  overlapsAny,
  target,
  targetKey,
  type HighlightTarget,
} from '../viz/highlight/types.ts';

const SAMPLES: HighlightTarget[] = [
  target.tensor('s'),
  target.tensor('s', [0, 1]),
  target.tensor('s', [0, '*']),
  target.tensor('s', ['*', 1]),
  target.tensor('s', [1, 1]),
  target.tensor('probs'),
  target.tensor('probs', [0, 1]),
  target.axis('s', 0),
  target.axis('s', 1),
  target.axis('probs', 0),
  target.formulaTerm('vjp', 'weighted-mean'),
  target.formulaTerm('vjp', 'direct'),
  target.formulaTerm('ln', 'weighted-mean'),
  target.codeLine(3),
  target.codeLine(4),
  target.event(0),
  target.event(7),
];

group('highlight · overlap matcher');

check('overlaps is symmetric across every sample pair', () => {
  let pairs = 0;
  for (const a of SAMPLES) {
    for (const b of SAMPLES) {
      expect(overlaps(a, b) === overlaps(b, a), `asymmetric: ${targetKey(a)} vs ${targetKey(b)}`);
      pairs += 1;
    }
  }
  return `${pairs} ordered pairs`;
});

check('overlaps is reflexive', () => {
  for (const a of SAMPLES) expect(overlaps(a, a), `${targetKey(a)} does not overlap itself`);
});

check('a whole tensor meets any cell of itself, in both orders', () => {
  const whole = target.tensor('s');
  const cell = target.tensor('s', [2, 3]);
  expect(overlaps(whole, cell), 'whole -> cell');
  expect(overlaps(cell, whole), 'cell -> whole');
});

check('wildcards match any position on their axis', () => {
  expect(overlaps(target.tensor('s', [0, '*']), target.tensor('s', [0, 5])), 'column wildcard');
  expect(overlaps(target.tensor('s', ['*', 1]), target.tensor('s', [9, 1])), 'row wildcard');
  expect(!overlaps(target.tensor('s', [0, '*']), target.tensor('s', [1, 5])), 'wrong row');
  expect(overlaps(target.tensor('s', ['*', '*']), target.tensor('s', [3, 4])), 'both wildcards');
});

check('a rank mismatch never overlaps', () => {
  expect(!overlaps(target.tensor('s', [0]), target.tensor('s', [0, 0])), 'rank 1 vs 2');
  expect(!overlaps(target.tensor('s', [0, 0, 0]), target.tensor('s', [0, 0])), 'rank 3 vs 2');
});

check('different tensor names never overlap', () => {
  expect(!overlaps(target.tensor('s'), target.tensor('probs')), 'whole tensors');
  expect(!overlaps(target.tensor('s', [0, 0]), target.tensor('probs', [0, 0])), 'same cell index');
});

check('an axis belongs to its tensor — the one cross-kind case', () => {
  expect(overlaps(target.tensor('s'), target.axis('s', 1)), 'tensor -> axis');
  expect(overlaps(target.axis('s', 1), target.tensor('s', [4, 4])), 'axis -> cell of that tensor');
  expect(!overlaps(target.axis('s', 1), target.tensor('probs')), 'axis of a different tensor');
  expect(!overlaps(target.axis('s', 0), target.axis('s', 1)), 'different axes');
});

check('formula terms are scoped to their equation', () => {
  expect(
    overlaps(target.formulaTerm('vjp', 'direct'), target.formulaTerm('vjp', 'direct')),
    'same',
  );
  expect(
    !overlaps(
      target.formulaTerm('vjp', 'weighted-mean'),
      target.formulaTerm('ln', 'weighted-mean'),
    ),
    'the same term name in two equations must not collide',
  );
});

check('code lines and events match only themselves', () => {
  expect(overlaps(target.codeLine(3), target.codeLine(3)), 'same line');
  expect(!overlaps(target.codeLine(3), target.codeLine(4)), 'different lines');
  expect(overlaps(target.event(7), target.event(7)), 'same event');
  expect(!overlaps(target.event(7), target.event(0)), 'different events');
});

check('unrelated kinds never overlap', () => {
  const byKind = new Map<string, HighlightTarget>();
  for (const s of SAMPLES) if (!byKind.has(s.kind)) byKind.set(s.kind, s);
  const kinds = [...byKind.entries()];
  for (const [ka, a] of kinds) {
    for (const [kb, b] of kinds) {
      if (ka === kb) continue;
      const isTensorAxisPair =
        (ka === 'tensor' && kb === 'axis') || (ka === 'axis' && kb === 'tensor');
      if (isTensorAxisPair) continue;
      expect(!overlaps(a, b), `${ka} should not overlap ${kb}`);
    }
  }
  return `${kinds.length} kinds cross-checked`;
});

check('overlapsAny agrees with a manual scan', () => {
  const pins = [target.tensor('probs'), target.codeLine(4)];
  expect(overlapsAny(target.tensor('probs', [1, 2]), pins), 'cell of a pinned tensor');
  expect(overlapsAny(target.codeLine(4), pins), 'pinned line');
  expect(!overlapsAny(target.codeLine(3), pins), 'unpinned line');
  expect(!overlapsAny(target.event(2), pins), 'unpinned event');
});

group('highlight · target keys');

check('targetKey is injective across near-identical targets', () => {
  const tricky: HighlightTarget[] = [
    ...SAMPLES,
    target.tensor('s', ['*']),
    target.tensor('s', [0]),
    target.tensor('s:0', [1]),
    target.axis('s', 10),
    target.event(70),
    target.codeLine(34),
  ];
  const seen = new Map<string, HighlightTarget>();
  for (const t of tricky) {
    const key = targetKey(t);
    const clash = seen.get(key);
    expect(
      clash === undefined || JSON.stringify(clash) === JSON.stringify(t),
      `key '${key}' is shared by ${JSON.stringify(clash)} and ${JSON.stringify(t)}`,
    );
    seen.set(key, t);
  }
  return `${seen.size} distinct keys from ${tricky.length} targets`;
});

check('a whole tensor and an all-wildcard index are distinct keys', () => {
  const whole = targetKey(target.tensor('s'));
  const wild = targetKey(target.tensor('s', ['*']));
  expect(whole !== wild, `both serialized to '${whole}'`);
  return `${whole} vs ${wild}`;
});

check('targetKey is stable for equal targets', () => {
  expect(
    targetKey(target.tensor('s', [1, 2])) === targetKey(target.tensor('s', [1, 2])),
    'two structurally equal targets must key the same',
  );
});
