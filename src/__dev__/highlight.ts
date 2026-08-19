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
  parseTargetKey,
  target,
  targetKey,
  type HighlightTarget,
} from '../viz/highlight/types.ts';
import { createHighlightStore } from '../viz/highlight/store.ts';

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

check('targetKey and parseTargetKey are inverses', () => {
  const tricky: HighlightTarget[] = [
    ...SAMPLES,
    target.tensor('s', ['*']),
    target.tensor('weird:name', [0, 1]),
    target.formulaTerm('eq:with:colons', 'term'),
    target.tensor('empty', []),
  ];
  for (const t of tricky) {
    const round = parseTargetKey(targetKey(t));
    expect(
      targetKey(round) === targetKey(t),
      `round trip changed ${targetKey(t)} into ${targetKey(round)}`,
    );
    expect(overlaps(round, t), `round trip of ${targetKey(t)} no longer overlaps the original`);
  }
  return `${tricky.length} targets round-tripped`;
});

check('parseTargetKey rejects a malformed key', () => {
  let threw = false;
  try {
    parseTargetKey('zzz:nonsense');
  } catch {
    threw = true;
  }
  expect(threw, 'an unrecognised kind should throw rather than return a wrong target');
});

group('highlight · store');

check('levelFor and levelForKey agree for every sample', () => {
  const store = createHighlightStore();
  store.setHover(target.tensor('s', [0, 1]));
  store.togglePin(target.codeLine(4));
  for (const t of SAMPLES) {
    expect(
      store.levelFor(t) === store.levelForKey(targetKey(t)),
      `disagreement on ${targetKey(t)}`,
    );
  }
  return `${SAMPLES.length} targets`;
});

check('pinned beats hover on the same target', () => {
  const store = createHighlightStore();
  const t = target.codeLine(3);
  store.setHover(t);
  expect(store.levelFor(t) === 'hover', 'hover first');
  store.togglePin(t);
  expect(store.levelFor(t) === 'pinned', 'pin must win, or it would flicker under the pointer');
});

check('setHover on an equivalent target does not notify', () => {
  const store = createHighlightStore();
  let notifications = 0;
  store.subscribe(() => {
    notifications += 1;
  });

  store.setHover(target.tensor('s', [0, 1]));
  expect(notifications === 1, `first hover should notify once, got ${notifications}`);

  // pointerover fires repeatedly over the same element
  for (let i = 0; i < 50; i++) store.setHover(target.tensor('s', [0, 1]));
  expect(notifications === 1, `repeat hovers notified ${notifications} times`);

  store.setHover(null);
  expect(notifications === 2, 'clearing should notify');
  return '50 repeat hovers produced 0 extra notifications';
});

check('pins are a set: toggling twice removes, and several can coexist', () => {
  const store = createHighlightStore();
  store.togglePin(target.tensor('s'));
  store.togglePin(target.tensor('probs'));
  expect(store.getPinned().length === 2, 'multi-pin is the point — comparison needs it');
  store.togglePin(target.tensor('s'));
  expect(store.getPinned().length === 1, 'toggling the same target removes it');
  expect(targetKey(store.getPinned()[0]) === targetKey(target.tensor('probs')), 'the right one');
  store.clearPins();
  expect(store.getPinned().length === 0, 'clearPins');
});

check('an unsubscribed listener stops receiving', () => {
  const store = createHighlightStore();
  let count = 0;
  const off = store.subscribe(() => {
    count += 1;
  });
  store.setHover(target.codeLine(1));
  off();
  store.setHover(target.codeLine(2));
  expect(count === 1, `expected 1 notification after unsubscribing, got ${count}`);
});

check('a pinned whole tensor lights its cells and axes', () => {
  const store = createHighlightStore();
  store.togglePin(target.tensor('probs'));
  expect(store.levelFor(target.tensor('probs', [2, 3])) === 'pinned', 'a cell');
  expect(store.levelFor(target.axis('probs', 1)) === 'pinned', 'an axis');
  expect(store.levelFor(target.tensor('s', [2, 3])) === 'none', 'a different tensor');
});
