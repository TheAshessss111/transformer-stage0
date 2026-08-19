# Trace & Highlight — Authoring Reference

> How a lab declares a program, and how views point at the same thing.
> Every lab from E0.8 onward is written against this.

Live proofs: [`/dev/trace`](../../src/pages/DevTrace.tsx) ·
[`/dev/highlight`](../../src/pages/DevHighlight.tsx) ·
[`/dev/formula`](../../src/pages/DevFormula.tsx)

---

## 1. Declare the program, don't instrument the ops

A lab writes its forward pass **once**, as an ordered list of statements:

```ts
export const SOFTMAX_PROGRAM = defineProgram<SoftmaxState>({
  id: 'softmax-forward-backward',
  language: 'python',
  steps: [
    {
      code: 'row_max = x.max(axis=-1, keepdims=True)',
      note: { zh: '每一行取最大值', en: 'Row maxima' },
      run: (s) => ({ rowMax: max(s.x, -1, true) }),
    },
    {
      code: 'dx = probs * (upstream - weighted_mean)',
      phase: 'backward',
      run: (s) => ({ dx: mul(s.probs, sub(s.upstream, s.weightedMean)) }),
    },
  ],
});
```

One declaration yields all of:

| What you get | From |
|---|---|
| Code-pane source | the `code` strings, joined |
| Line highlighting | `lineRanges`, precomputed per step |
| Playback frames | one frame per step |
| "What is happening now" | `note` |
| **Tensor names** | **the state keys** |

That last row is why this exists. `s.rowMax` is addressable as
`{ kind: 'tensor', name: 'rowMax' }`, which is what a formula term can point at.
Without named tensors there is nothing to link.

**Write one statement per step.** Not one operation — a person reads statements,
and the playhead's default granularity follows the source.

---

## 2. Run it

```ts
const { state, trace } = runProgram(program, initialState);
```

From React, use the hook instead — it coalesces runs to one per animation frame:

```ts
const makeInitial = useCallback(() => softmaxInputs(rows, cols, seed), [rows, cols, seed]);
const { state, trace, timing } = useTracedProgram(SOFTMAX_PROGRAM, makeInitial);
```

**Wrap `makeInitial` in `useCallback`.** Its dependency list is what decides when
the program re-runs. Forgetting it is not a correctness bug — coalescing still
caps runs at one per frame — but `/dev/trace`'s timing readout will show the
extra work.

Budget: **8 ms per run** at ≤ 4096 elements, half a 60fps frame. Exceeding it
logs a dev warning naming the program. Measured at the ceiling: 3.1 ms.

---

## 3. What a Trace contains

```ts
interface Trace {
  programId: string;
  source: string;              // joined, for the code pane
  steps: readonly TraceStep[]; // code, line range, note, phase
  events: readonly TraceEvent[];
  names: ReadonlyMap<string, NdArray>;
  durationMs: number;
}
```

Each `TraceEvent` is one primitive operation:

- **`index`** — the stable identity. Not the array object: a per-recording id
  would change on every recompute and drop whatever the user was hovering when a
  slider moved. The same program run with different values emits the same
  sequence, so event 7 is event 7 across recomputes.
- **`inputs[].event`** — which event produced this input, or `null` for a program
  input. Resolved by object identity, which is what makes the graph drawable.
- **`isView` / `didCopy` / `copiedElements`** — the memory facts Step 0.1 renders
  as the cyan `view` and amber `copy · N` badges.
- **`passthrough`** — the output *is* one of the inputs. Only `ascontiguousarray`
  on already-contiguous input does this. Such events are skipped when building a
  graph (they would be self-edges) but stay in the event list, because "this call
  cost nothing" is the point.

### The immutability contract

> **A Trace is valid only for the execution that produced it. Nothing may mutate
> an array reachable from one.**

It holds live `NdArray` references, some of them views sharing a buffer. Editing
an input means changing lab state and re-running, which produces a new Trace.
F0.5.3 (click a cell to edit) must go through lab state, never write into a
recorded array.

---

## 4. Frames and playback

```ts
const frames = framesFor(trace);                 // all
const backwardOnly = framesFor(trace, 'backward');

eventsUpTo(frames, position, 'frame' | 'event');
frameAtPosition(frames, position, granularity);
```

Two granularities: `'frame'` is one statement (what a person reads, and what the
code pane highlights); `'event'` is one primitive operation (what Step 0.2's
graph single-steps). `position < 0` means nothing has happened yet, so a playhead
can sit before the first step.

A step that emitted no operations still produces a frame — its code line stays
highlightable while nothing computes.

---

## 5. Pointing at things

Every view describes what it is pointing at in one vocabulary:

```ts
target.tensor('probs')            // the whole tensor
target.cell('probs', [0, 2])      // one element
target.tensor('probs', [0, '*'])  // a row
target.axis('probs', 1)
target.formulaTerm('softmax-vjp', 'weighted-mean')
target.codeLine(7)
target.event(3)
```

The primitive is **symmetric overlap**, not equality: the whole tensor meets any
of its cells, in either direction. `tensor`/`axis` is the only cross-kind
overlap — formula-to-tensor linkage is done by *declaration*, not by the matcher.

A gesture points at a **group**, because pointing is usually plural:

```ts
setHover([target.tensor('xhat'), target.tensor('g'), target.codeLine(9)]);
```

`togglePin` takes the same, and pins the group as one unit.

---

## 6. Consuming highlight state

```ts
const level = useHighlightLevel(spot);      // 'none' | 'hover' | 'pinned'
const { setHover, togglePin, clearPins } = useHighlightActions();
```

**Two rules for leaf views, both load-bearing:**

1. **Memoize the leaf.** A grid label subscribing to the whole tensor re-renders
   the grid whenever any cell is hovered. Without `memo` on the cell, that
   re-renders every child — which is exactly what the external store exists to
   prevent. Measured on `/dev/highlight`: unmemoized, 48/48 cells re-rendered on
   one hover; memoized, **1 of 96**.

2. **Build the target with `useMemo`.** An inline `{ kind: 'tensor', name, index }`
   changes identity every render. Subscriptions are keyed by `targetKey`, so this
   is not fatal, but it churns.

---

## 7. Writing an equation

```ts
const VJP: EquationSpec<SoftmaxState> = {
  id: 'softmax-vjp',
  latex: String.raw`\bar x_i = \term{outer}{s_i(\bar s_i - \term{mean}{\sum_j \bar s_j s_j})}`,
  terms: {
    outer: {
      label: { zh: '整个 VJP', en: 'the whole VJP' },
      purpose: { zh: '…', en: 'Turns the upstream gradient into the input gradient.' },
      shape: '(B, T)',
      read: (s) => s.dx,
      highlight: [target.tensor('dx'), target.codeLine(7)],
    },
    // …
  },
};
```

`\term{id}{content}` is a macro expanding to KaTeX `\htmlData`, giving each
marked sub-expression a `data-term` attribute (D-25). It survives `\frac`,
`\sum` limits and nesting, and two equations may reuse a term name.

The tooltip shows four panes in a fixed order: **what / shape / value now /
purpose**. `read(ctx)` is what makes the third one live — dragging a slider
changes the number inside an open tooltip.

**Every `\term` in the LaTeX needs an entry in `terms`, and vice versa.**
`assertValidEquationSpec` throws in dev on either mismatch. Without it a renamed
term renders a tooltip that silently does nothing and the page still looks
finished.

### A gotcha worth knowing

React re-applies `dangerouslySetInnerHTML` on every render even when the html
string is byte-identical — the container node survives, the children do not. So
`Formula` re-applies its highlight attributes in a **layout effect after every
commit**, with no dependency array. Anything else writing onto KaTeX output must
do the same, or it will work until the first unrelated state change.

---

## 8. Stepwise derivations

```ts
<DerivationSteps steps={[
  { equation: JACOBIAN, justification: { zh: '…', en: '…' } },
  { equation: CONTRACTED, justification: {…}, correspondence: [{ from: 'jac', to: 'expanded' }] },
  { equation: VJP, justification: {…}, correspondence: [{ from: 'expanded', to: 'outer' }] },
]} ctx={state} />
```

`correspondence` links a term to the term it came from in the previous step, so
hovering the compact form lights its ancestor in the expanded one. That is what
makes a four-line collapse legible rather than magic — and it is what Step 0.4's
four-step LayerNorm derivation is built on.
