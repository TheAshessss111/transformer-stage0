# Implementation Plan — E0.1 (Scaffold) & E0.2 (Tensor Engine MVP)

> Scope: the first two epics of M0. When both are done you have a deployable dark-dashboard shell
> with routing and design tokens, plus a float64 strided tensor engine whose every VJP is verified
> against central-difference numerical gradients.
>
> Reading order: §0 (deviations — read first, they need your OK) → §1 (E0.1 tasks) → §2 (E0.2 design)
> → §3 (E0.2 tasks) → §4 (sign-off).

---

## 0. Deviations from the roadmap — please confirm

Five adjustments I want to make. Each has a reason; reject any and I'll re-plan around it.

### 0.1 Pull `core/gradcheck/` forward from E0.7 into E0.2

`ROADMAP.md` puts gradcheck in E0.7 as a Python-side facility (F0.7.7). But E0.2 hand-writes ~15 VJPs,
and D-22 removed Vitest while D-21 put the parity panel in E0.7. That leaves **E0.2 with no way to know
whether any of its backward passes are correct** until three epics later.

Central-difference gradcheck needs no NumPy and no test framework — it is self-contained, and it is
literally 纪律二 from `transformer_plan.md`. Adding `core/gradcheck/numericalGrad.ts` + `relError.ts`
(~80 lines, pure functions, zero deps) to E0.2 makes the epic self-verifying.

**These same two files are later reused by E0.7's Python harness as the TS-side reference.** No waste.

### 0.2 Plant the trace hook seam in E0.2

F0.3.2 says "all ops auto-record when the recorder is active". If the hook is introduced in E0.3,
every op written in E0.2 gets rewritten. So E0.2 defines `core/trace/hook.ts` (a nullable module-level
callback, ~20 lines, zero cost when unset) and every op calls `emit(...)` at its tail.

E0.3 then only has to implement the recorder that plugs into the seam. **The hook is E0.2; the recorder stays E0.3.**

### 0.3 Five small additions to the F0.2.4 op list

| Addition | Why now | Downstream consumer |
|---|---|---|
| `sqrt` | `/sqrt(Dh)` in Step 0.3, `sqrt(σ²+ε)` in Step 0.4 | F0.8.8, F3.3.1 |
| `select(a, axis, i)` | returns a view by advancing `offset` — 6 lines | `SliceSelector` (F0.5.4) |
| `expandDims` / `squeeze` | needed by every `keepdims=false` VJP path | F1.2.4 keepdims trap |
| `toNested(a)` | NdArray → nested JS array | every viz component in E0.5 |

### 0.4 Verification harness runs on Node's native TypeScript, not Vitest

D-22 declined Vitest and CI. Node 24.17 (already installed) strips TS types natively, so:

```bash
node src/core/tensor/__dev__/verify.ts
```

runs with **zero test dependencies, zero config, zero CI**. It is a plain script that prints a
pass/fail table and exits non-zero on failure. This is not a test framework, but it *is* automated
verification — flagging it explicitly since D-22 leaned the other way.

Two constraints it imposes on `core/`:
- **Erasable syntax only**: no `enum`, no `namespace`, no constructor parameter properties, no legacy decorators. (Classes, interfaces, generics, `as`, `satisfies` are all fine.)
- **Explicit `.ts` extensions on imports inside `core/`** (Node requires real paths). Enabled by `allowImportingTsExtensions: true` in tsconfig; Vite resolves them fine. App code outside `core/` keeps extensionless imports.

### 0.5 E0.1 ships a minimal `LocaleContext`, not the E0.6 content system

F0.1.4 requires a working locale toggle in the top bar, but the `L<T>` content system is F0.6.2.
E0.1 ships only: `Locale = 'zh' | 'en'`, a context, a `localStorage` key, and the toggle button.
E0.6 builds `L<T>` and `useL()` on top without changing the context.

---

## 1. E0.1 — Scaffold & Engineering Standards

### T1 · Initialize Vite + React 19 + TS  →  F0.1.1

```bash
cd /Users/barrywsmacbookair/Documents/MathBase/transformer-stage0
npm create vite@latest . -- --template react-ts
npm install
npm install react-router
npm install -D gh-pages
```

`npm create vite` into a non-empty dir will prompt; choose "Ignore files and continue" (only
`README.md`, `.gitignore`, `docs/`, `.git/` are present, none of which it overwrites — verify
`README.md` survives, restore from git if not).

**Record resolved versions.** Do not hand-pin; run `npm ls --depth=0` after install and paste the
output into the commit body so the lockfile decisions are auditable.

`vite.config.ts`:

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig(({ command }) => ({
  // GitHub Pages project site lives under /transformer-stage0/, dev server at /
  base: command === 'build' ? '/transformer-stage0/' : '/',
  plugins: [react(), tailwindcss()],
}));
```

`tsconfig.app.json` — add to `compilerOptions`:

```jsonc
{
  "allowImportingTsExtensions": true,  // core/ imports use explicit .ts (see §0.4)
  "verbatimModuleSyntax": true,        // keeps core/ erasable-syntax clean
  "noUncheckedIndexedAccess": false    // deliberately off: strided index math would drown in `!`
}
```

> `noUncheckedIndexedAccess` is off on purpose. The tensor engine does thousands of
> `data[offset + i * stride]` reads that are provably in range; turning it on would fill `core/`
> with non-null assertions and hide the real errors. `strict: true` (Vite's default) stays on.

**DoD**
- [ ] `npm run dev` serves a blank page with no console errors
- [ ] `npm run build` succeeds
- [ ] `docs/` and `README.md` intact
- [ ] Commit: `chore: scaffold vite + react 19 + ts`

---

### T2 · Design tokens & dark dashboard theme  →  F0.1.2

```bash
npm install -D tailwindcss @tailwindcss/vite
```

Create `src/styles/theme.css`, imported once from `src/main.tsx`:

```css
@import 'tailwindcss';

@theme {
  /* ── surfaces (3 levels, per ARCHITECTURE §8) ── */
  --color-bg-base:   oklch(0.17 0.012 265);
  --color-bg-panel:  oklch(0.22 0.014 265);
  --color-bg-well:   oklch(0.13 0.010 265);
  --color-border:    oklch(0.32 0.015 265);
  --color-text:      oklch(0.93 0.005 265);
  --color-text-dim:  oklch(0.68 0.010 265);

  /* ── diverging scale: gradient sign. zero MUST equal bg-well, not grey ── */
  --color-grad-neg:  oklch(0.62 0.19 25);    /* red-orange */
  --color-grad-zero: oklch(0.13 0.010 265);  /* == bg-well */
  --color-grad-pos:  oklch(0.62 0.16 255);   /* blue */

  /* ── sequential scale: non-negative magnitudes (probabilities, |x|) ── */
  --color-mag-0: oklch(0.16 0.020 275);
  --color-mag-1: oklch(0.32 0.090 270);
  --color-mag-2: oklch(0.48 0.130 230);
  --color-mag-3: oklch(0.66 0.140 190);
  --color-mag-4: oklch(0.85 0.130 160);

  /* ── memory semantics: BADGE AND BORDER ONLY, never a data fill ── */
  --color-view:  oklch(0.75 0.13 195);  /* zero-copy view */
  --color-copy:  oklch(0.78 0.15 75);   /* a real memory copy happened */

  /* ── non-finite values ── */
  --color-nan:   oklch(0.70 0.25 330);  /* magenta */
  --color-inf:   oklch(0.68 0.21 35);   /* orange-red */

  /* ── status ── */
  --color-ok:    oklch(0.74 0.16 150);
  --color-warn:  oklch(0.80 0.15 85);
  --color-err:   oklch(0.65 0.21 25);

  --font-mono: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  --font-sans: system-ui, -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
}
```

**Two rules that must hold for the rest of the project:**
1. `--color-view` / `--color-copy` are used for **badges, borders and pipeline connectors only** —
   never as a cell fill — so they can never be confused with the diverging data scale.
2. The diverging scale's zero point is `bg-well`, so a matrix of zeros looks empty, not grey.

No web fonts (self-hosting is not worth a dependency here; system mono stacks are fine on macOS).

**Token preview page** at route `/dev/tokens`: renders every token as a labelled swatch, plus a
21-step ramp of the diverging and sequential scales with sample numbers overlaid at each stop so
NFR-7 (text legibility on every fill) can be eyeballed in one screen.

**DoD**
- [ ] `/dev/tokens` renders all tokens; every ramp stop shows a legible numeral
- [ ] Tailwind utilities like `bg-bg-panel text-text-dim font-mono` work
- [ ] Commit: `feat(theme): dark dashboard design tokens + /dev/tokens preview`

---

### T3 · ESLint + Prettier + commit hooks  →  F0.1.3

```bash
npm install -D eslint @eslint/js typescript-eslint eslint-plugin-react-hooks \
               eslint-plugin-react-refresh globals \
               prettier prettier-plugin-tailwindcss \
               husky lint-staged
npx husky init
```

`.husky/pre-commit` → replace generated content with `npx lint-staged`.

`package.json`:

```jsonc
{
  "scripts": {
    "lint": "eslint . --max-warnings 0",
    "format": "prettier --write .",
    "verify:engine": "node src/core/tensor/__dev__/verify.ts",
    "deploy": "npm run build && cp dist/index.html dist/404.html && touch dist/.nojekyll && gh-pages -d dist -t"
  },
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix --max-warnings 0", "prettier --write"],
    "*.{css,md,json}": ["prettier --write"]
  }
}
```

`.prettierrc`: `{ "singleQuote": true, "printWidth": 100, "plugins": ["prettier-plugin-tailwindcss"] }`

**ESLint rule specific to this project** — add to the flat config:

```js
// core/ must never import React or anything UI (ARCHITECTURE §2 "铁律")
{
  files: ['src/core/**/*.ts', 'src/py/**/*.ts'],
  rules: {
    'no-restricted-imports': ['error', {
      patterns: ['react', 'react-*', '@react-three/*', '../viz/*', '../../viz/*'],
    }],
  },
}
```

This makes the layering rule mechanically enforced instead of a docs promise.

**DoD**
- [ ] `npm run lint` → 0 errors, 0 warnings
- [ ] A deliberate `import React from 'react'` inside `src/core/` fails lint
- [ ] Staging a badly formatted file and committing auto-fixes it
- [ ] Commit: `chore: eslint + prettier + husky, enforce core/ layer boundary`

---

### T4 · Router, AppShell, Sidebar, locale toggle  →  F0.1.4

Files:

```
src/shell/AppShell.tsx        layout frame: sidebar | main
src/shell/Sidebar.tsx         5 steps from registry, active route highlighted
src/shell/TopBar.tsx          title + LocaleToggle
src/shell/LocaleToggle.tsx    中 / EN
src/shell/LocaleContext.tsx   minimal (see §0.5)
src/steps/registry.ts         Step metadata
src/steps/StepPage.tsx        stub: renders title + "not built yet"
src/pages/Landing.tsx         stub
src/pages/DevTokens.tsx       from T2
```

`src/steps/registry.ts`:

```ts
export interface StepMeta {
  id: '0-1' | '0-2' | '0-3' | '0-4' | '0-5';
  number: string;             // '0.1'
  title: { zh: string; en: string };
  tagline: { zh: string; en: string };
  signatureViz: { zh: string; en: string };  // the headline visualization
  status: 'planned' | 'in-progress' | 'done';
}

export const STEPS: readonly StepMeta[] = [ /* 5 entries, all status:'planned' */ ];
```

Routing — `BrowserRouter` with `basename={import.meta.env.BASE_URL}`:

| Route | Component |
|---|---|
| `/` | `Landing` |
| `/step/:id` | `StepPage` (unknown id → redirect `/`) |
| `/dev/tokens` | `DevTokens` |

Sidebar shows `0.1 … 0.5` with title and a status dot driven by `StepMeta.status`.

**DoD**
- [ ] All 5 step routes reachable; active item highlighted; deep-link + refresh works in dev
- [ ] Locale toggle flips a visible string and survives reload
- [ ] Commit: `feat(shell): router, app shell, sidebar, locale toggle`

---

### T5 · Small-screen gate  →  F0.1.5

`src/shell/SmallScreenGate.tsx` — wraps `AppShell`. Below `1280px` render a centered panel
(bilingual, uses the same tokens) and **do not mount the app tree** — this matters because E0.5's
labs and E1.1's R3F canvas must never mount on a phone.

Use `window.matchMedia('(min-width: 1280px)')` in a `useSyncExternalStore` hook — not a resize
listener — so it doesn't thrash on window drag.

**DoD**
- [ ] Below 1280px only the gate renders (verify children are unmounted, not hidden by CSS)
- [ ] Commit: `feat(shell): small-screen gate at 1280px`

---

### T6 · GitHub Pages deploy  →  F0.1.6

The `deploy` script from T3 already covers it. Two things to know:

- **`cp dist/index.html dist/404.html`** — GitHub Pages has no SPA fallback, so a refresh on
  `/step/0-3` would 404. Serving the app as the 404 page makes deep links work. Caveat: the HTTP
  status is 404 even though the page renders. Acceptable here; if it ever bothers you, switch to
  `HashRouter` and delete the `cp`.
- **`.nojekyll`** stops Pages from filtering files, and `gh-pages -t` publishes dotfiles.

First deploy needs a GitHub remote and repo Settings → Pages → source `gh-pages` branch.
**Ask before pushing** — first publish is outward-facing.

**DoD**
- [ ] `npm run build && npx serve dist` renders correctly under the `/transformer-stage0/` base
- [ ] Deep link + refresh works against the built output
- [ ] Commit: `chore: github pages deploy script`

---

## 2. E0.2 — Design Notes

Read this before §3. These are the decisions that make the engine teachable rather than merely correct.

### 2.1 The core data structure

```ts
// src/core/tensor/ndarray.ts
export class NdArray {
  readonly data: Float64Array;        // underlying buffer, possibly shared with other views
  readonly shape: readonly number[];
  readonly strides: readonly number[]; // ELEMENT units, not bytes. may be 0 (broadcast) 
  readonly offset: number;             // element index into `data` of logical [0,0,...]
  readonly base: NdArray | null;       // non-null ⇒ this is a view
}
```

**Invariants** (assert in the constructor, they are the cheapest bug net in the project):
- `shape.length === strides.length`
- every `shape[i] >= 0`
- for any in-range logical index, `offset + Σ idx[i]·strides[i]` lies within `data`
- `base === null` ⇔ this array owns `data` exclusively

**Why float64, not float32**: `transformer_plan.md` 纪律二 — gradcheck's central difference is drowned
by float32 rounding. Everything is float64 all the way through. (Reduced precision is *simulated*
later in `core/numerics/float.ts` for M4, never used as storage.)

**Why `strides` may be 0**: `broadcastTo` produces stride-0 axes, meaning "read the same element
repeatedly". This is both the real NumPy mechanism and a teaching artifact — F1.1.2's `MemoryStrip`
will draw a stride-0 axis as an arrow that doesn't advance. **A stride-0 view is read-only; writing
through it is a bug.** Enforce with a `readonly: boolean` flag set by `broadcastTo`, checked in `set`.

**`isContiguous()`** follows NumPy's rule: compare against C-contiguous strides *skipping axes of
size 1* (their stride is unobservable), and treat empty arrays as contiguous. Getting this wrong
makes `reshape` copy when it shouldn't, which would make Step 0.1 teach a falsehood.

### 2.2 View vs copy is a first-class, visible fact

This is A-01 from `ARCHITECTURE.md`, and it drives the API shape:

| Operation | Result | UI marking |
|---|---|---|
| `transpose` / `permute` / `select` / `expandDims` / `squeeze` / `broadcastTo` | always a view, `data` shared | `--color-view` badge |
| `reshape` on a contiguous array | view | `--color-view` badge |
| `reshape` on a non-contiguous array | calls `ascontiguousarray` first | `--color-copy` badge + element count |
| `ascontiguousarray` on a contiguous array | returns the same object | no badge |

So every shape op returns `{ out: NdArray; didCopy: boolean; copiedElements: number }` internally
and reports it through the trace hook. The public function returns just `NdArray`; the metadata
travels via `emit()`.

### 2.3 Two layers: raw ops and autograd ops

```
ops.ts   NdArray → NdArray      pure forward, no graph
vjp.ts   (gradOut, saved…) → gradIn(s)      hand-written backwards
────────────────────────────────────────────
autograd.ts   Var → Var         tape wrapper composing the two
```

Labs that only need a forward pass (Step 0.1's shape pipeline) use `ops.ts` directly and stay cheap.
Labs that need gradients (Steps 0.2–0.4) use `autograd.ts`. Keeping them separate is also what makes
E0.2's own gradcheck possible: you can verify a VJP in isolation, without the tape.

**Save the output, not the input, wherever the math allows.** `exp`, `softmax`, `relu`(via mask) all
cache their output. This is a real memory optimization *and* the exact point Step 0.3 makes
("反向只需要前向的输出 s"), so F0.8.6 will visualize it. The API must not quietly cache inputs.

### 2.4 Broadcasting and its dual

Per 规则 5 in the plan: forward `broadcast` ↔ backward `sum`. So `unbroadcast` is not a helper,
it is *the* backward of broadcasting, and every binary op's VJP ends with it:

```ts
export function unbroadcast(g: NdArray, targetShape: readonly number[]): NdArray;
// 1. sum away leading axes that the target doesn't have
// 2. sum with keepdims over axes where target[i] === 1 && g.shape[i] > 1
// 3. assert result.shape deep-equals targetShape
```

That final assert is the single highest-value line in the engine: it catches essentially every
broadcast-VJP mistake, which is the category `transformer_plan.md` Step 1.3 calls "最容易错的地方".

### 2.5 The gradient-shape invariant

From 规则总述: *梯度形状永远等于对应变量的形状*. Encode it as a hard runtime check in
`autograd.backward()` — after accumulating into `p.grad`, assert `p.grad.shape` equals `p.value.shape`
and throw with both shapes and the producing op name. Do not make this a dev-only assert; it costs
nothing next to the matmuls and it will save hours in M2 and M3.

### 2.6 Gradient accumulation, never assignment

`transformer_plan.md` Step 1.2 flags this as the classic embedding bug. In `backward()`:

```ts
p.grad = p.grad === null ? gi : add(p.grad, gi);   // never `p.grad = gi`
```

The verify harness must contain a case that fails under assignment and passes under accumulation —
e.g. `y = x * x + x`, where `x` is reached twice.

### 2.7 The trace seam (§0.2)

```ts
// src/core/trace/hook.ts
export interface RawOpEvent {
  op: string;
  inputs: NdArray[];
  output: NdArray;
  phase: 'forward' | 'backward';
  isView: boolean;
  didCopy: boolean;
  copiedElements?: number;
  meta?: Record<string, unknown>;   // axis, keepdims, permutation…
}
let hook: ((e: RawOpEvent) => void) | null = null;
export function setOpHook(h: ((e: RawOpEvent) => void) | null): void { hook = h; }
export function emit(e: RawOpEvent): void { hook?.(e); }
```

Every function in `ops.ts`, `shape.ts`, `broadcast.ts` and `vjp.ts` calls `emit(...)` before returning.
When no recorder is attached this is one null check. E0.3 implements the recorder; E0.2 only plants
the call sites.

---

## 3. E0.2 — Task Breakdown

Sequential unless noted. Every task ends with a commit.

### T7 · `ndarray.ts` — structure, indexing, iteration  →  F0.2.1

```ts
// construction
zeros(shape), ones(shape), full(shape, v), fromFlat(data, shape), fromNested(nested)
// introspection
size(a), ndim(a), isContiguous(a), contiguousStrides(shape)
// element access
at(a, idx: number[]): number
set(a, idx: number[], v: number): void        // throws if a.readonly
flatIndex(a, idx: number[]): number
// iteration — the workhorse for every op
forEachLogical(a, fn: (flatOffset: number, logicalIndex: number, idx: number[]) => void): void
toFlat(a): Float64Array                        // logical order; zero-copy iff contiguous
toNested(a): NestedArray                       // for viz (§0.3)
```

`forEachLogical` uses an odometer over `shape` with an incremental offset (no per-element multiply),
and takes a fast path when `isContiguous(a)`.

**DoD**
- [ ] Constructor invariants (§2.1) throw with readable messages on violation
- [ ] `isContiguous` correct for: C-contiguous, transposed, size-1 axes, empty, 0-d
- [ ] Commit: `feat(core): NdArray with strided storage and logical iteration`

---

### T8 · `shape.ts` — views and the one real copy  →  F0.2.2

```ts
reshape(a, shape: number[]): NdArray          // supports one -1
transpose(a): NdArray                         // reverses all axes
permute(a, axes: number[]): NdArray
swapAxes(a, i, j): NdArray
select(a, axis, i): NdArray                   // drops the axis; offset += i * strides[axis]
expandDims(a, axis): NdArray
squeeze(a, axis?): NdArray
ascontiguousarray(a): NdArray                 // identity if already contiguous
```

Contract per §2.2. `reshape` on a non-contiguous array must **not** silently produce garbage and must
**not** throw — it copies, and reports `didCopy` through `emit`.

**DoD**
- [ ] `transpose(a).data === a.data` (identity, not just equal contents)
- [ ] `reshape(transpose(a), [...])`: `.data !== a.data`, emitted event has `didCopy: true` with the right element count
- [ ] `reshape(contiguous)`: `.data === a.data`, `didCopy: false`
- [ ] Round trip `permute(permute(a, p), inverse(p))` equals `a` elementwise
- [ ] Commit: `feat(core): shape ops with true view/copy semantics`

---

### T9 · `broadcast.ts`  →  F0.2.3

```ts
broadcastShapes(...shapes: readonly number[][]): number[]
broadcastTo(a, shape): NdArray          // stride-0 view, marked readonly
unbroadcast(g, targetShape): NdArray    // §2.4, with the trailing shape assert
```

Errors must name the offending axis and both shapes, not just "shapes incompatible".

**DoD**
- [ ] `broadcastTo(ones([3,1]), [2,3,4]).data === original.data`, strides contain zeros
- [ ] Writing through a broadcast view throws
- [ ] For 20 random shape pairs: `unbroadcast(broadcastTo(a, s), a.shape).shape === a.shape` and values equal `a * (broadcastFactor)`
- [ ] Commit: `feat(core): broadcasting with stride-0 views and unbroadcast`

---

### T10 · `random.ts` + `format.ts`  →  F0.2.9, F0.2.10

```ts
// random.ts
mulberry32(seed: number): () => number
randn(shape, seed): NdArray             // Box–Muller on top of mulberry32
uniform(shape, lo, hi, seed): NdArray
arange(n): NdArray
onehot(indices: Int32Array, numClasses: number): NdArray

// format.ts
type ValueKind = 'finite' | 'zero' | 'nan' | 'posinf' | 'neginf' | 'subnormal';
classify(v: number): ValueKind
formatValue(v: number, opts?: { sigDigits?: number }): { text: string; kind: ValueKind }
formatShape(shape: readonly number[]): string   // '(B,T,D)' style when names given, else '(2,4,8)'
```

`formatValue` switches to exponential outside `[1e-3, 1e5)`. The returned `kind` is what
`TensorGrid` will map to `--color-nan` / `--color-inf` in E0.5 — **no component should ever
re-implement `isNaN` checks**.

Independent of T7–T9; can be done in parallel.

**DoD**
- [ ] Same seed ⇒ bit-identical arrays across runs
- [ ] `randn(1e5)` has mean ≈ 0 (|μ| < 0.02) and std ≈ 1 (|σ−1| < 0.02)
- [ ] `classify` distinguishes `-0`, subnormals, and both infinities
- [ ] Commit: `feat(core): seeded PRNG and numeric formatting`

---

### T11 · `trace/hook.ts` — the seam  →  (E0.3 prep, §0.2)

~20 lines exactly as written in §2.7. Must land **before** T12 so op call sites are planted as they
are written rather than retrofitted.

**DoD**
- [ ] `emit` with no hook set is a no-op; setting a hook receives events
- [ ] Commit: `feat(core): op trace hook seam`

---

### T12 · `ops.ts` — forward operators  →  F0.2.4 (+ §0.3)

```ts
// elementwise unary
neg, exp, log, sqrt, relu, abs
// elementwise binary (broadcasting)
add, sub, mul, div, maximum
// scalar
addScalar, mulScalar
// reductions — axis: number | number[] | null, keepdims: boolean
sum, mean, max, min
// linear algebra
matmul(a, b)      // (...,m,k) @ (...,k,n), batch dims broadcast, ndim >= 2 required
```

`matmul`: broadcast the batch dims via `broadcastShapes`, then loop batches with a naive `m×k×n`
triple loop over `toFlat` views. Teaching-scale tensors (≤ ~4096 elements) make this plenty fast; do
not optimize now. Reject `ndim < 2` with a message pointing at the plan's shape table rather than
implementing NumPy's 1-D promotion rules (which Transformer code never relies on).

Every op ends with `emit({ op, inputs, output, phase: 'forward', ... })`.

**DoD**
- [ ] `sum/mean/max` correct for `axis = null | number | number[]`, both `keepdims` values
- [ ] Reductions over a transposed (non-contiguous) input give the same result as over its contiguous copy
- [ ] `matmul` batch broadcasting: `(2,1,3,4) @ (1,5,4,6) → (2,5,3,6)`
- [ ] Commit: `feat(core): forward operators with broadcasting and keepdims`

---

### T13 · `gradcheck/` — numerical gradients  →  §0.1

```ts
// numericalGrad.ts — central difference, float64, per 纪律二
numericalGrad(f: (x: NdArray) => number, x: NdArray, eps?: number): NdArray   // eps = 1e-5
// relError.ts
relError(a: NdArray, b: NdArray): { max: number; argmax: number[]; perElement: NdArray };
gradcheck(f, x, analytic, tol?): { ok: boolean; maxRelErr: number; report: string };
```

Follow the plan exactly: central difference (O(ε²)), float64, ε = 1e-5, **relative** error with the
`|a| + |b| + 1e-12` denominator, tolerance 1e-6…1e-7. `perElement` exists so F0.7.7 can render the
error heatmap without recomputation.

The scalarizing wrapper the plan describes — `f = x => sum(forward(x) * randomUpstream)` — belongs
here as a helper `scalarize(forward, upstream)`, since every VJP check needs it.

**DoD**
- [ ] `numericalGrad` on `x => sum(x*x)` matches `2x` to < 1e-9 relative
- [ ] Non-differentiable-point guard documented: ReLU checks must use inputs bounded away from 0
- [ ] Commit: `feat(core): central-difference gradcheck utilities`

---

### T14 · Verify harness v1 — structural invariants  →  §0.4

`src/core/tensor/__dev__/verify.ts`, run by `npm run verify:engine`.

A plain script: a `check(name, fn)` collector, a printed table, `process.exitCode = 1` on any failure.
No framework, no globals, no config.

v1 covers T7–T10: constructor invariants, `isContiguous` cases, view-vs-copy identity checks,
broadcast round-trips, PRNG reproducibility, `classify` edge cases.

**DoD**
- [ ] `npm run verify:engine` prints a pass table and exits 0
- [ ] Deliberately breaking `isContiguous` makes it exit non-zero with a readable line
- [ ] Commit: `chore(core): engine verification harness (structural)`

---

### T15 · `vjp.ts` — hand-written backwards  →  F0.2.5 (partial)

One VJP per forward op. Signatures take *only what the math needs* (§2.3):

```ts
addVjp(g, aShape, bShape): [NdArray, NdArray]              // unbroadcast both
mulVjp(g, a, b): [NdArray, NdArray]
divVjp(g, a, b): [NdArray, NdArray]
matmulVjp(g, a, b): [NdArray, NdArray]                     // ḡBᵀ , Aᵀḡ  then unbroadcast batch dims
expVjp(g, out): NdArray                                     // uses OUTPUT (§2.3)
logVjp(g, a): NdArray
sqrtVjp(g, out): NdArray
reluVjp(g, mask): NdArray                                   // mask cached at forward, not `a`
sumVjp(g, inShape, axis, keepdims): NdArray                 // re-insert axes then broadcast
meanVjp(g, inShape, axis, keepdims): NdArray
maxVjp(g, a, out, axis, keepdims): NdArray                  // ties split evenly
reshapeVjp / permuteVjp / expandDimsVjp / squeezeVjp        // inverse of the forward
```

Each ends with `emit({ ..., phase: 'backward' })`.

**DoD**
- [ ] Each VJP passes `gradcheck` at rel-err < 1e-7 on random float64 inputs
- [ ] Broadcasting cases explicitly covered (`(3,1) + (2,3,4)` etc.), asserting output shapes
- [ ] `max` tie-splitting verified on a crafted input with duplicates
- [ ] Commit: `feat(core): VJPs for all forward operators`

---

### T16 · Verify harness v2 — gradcheck every VJP

Extend `verify.ts` with one `check()` per operator from T15, each using `scalarize` + `gradcheck`.
This is the first moment the engine is trustworthy. **Do not proceed to T17 with any red line.**

**DoD**
- [ ] ~18 gradcheck lines, all green, max rel-err printed per op
- [ ] Total runtime < 5 s
- [ ] Commit: `chore(core): gradcheck every VJP in the verify harness`

---

### T17 · softmax family and fused cross-entropy  →  F0.2.5, F0.2.6

```ts
softmax(x, axis?): NdArray                    // z = x - max(x, axis, keepdims); exp; normalize
logSoftmax(x, axis?): NdArray                 // z - log(sum(exp(z)))  — LogSumExp, never log(softmax)
softmaxVjp(s, sBar, axis?): NdArray           // s * (sBar - sum(sBar*s, axis, keepdims))

crossEntropyFromLogits(logits: NdArray, targets: Int32Array):
  { loss: number; probs: NdArray }            // mean over N
crossEntropyVjp(probs, targets, gradLoss?): NdArray   // (probs - onehot(y)) / N
```

**API notes that are also teaching points, and must not be "simplified" later:**
- `softmaxVjp` takes `s` (the forward *output*), never `x`. The signature is the lesson.
- `logSoftmax` is computed by LogSumExp, not as `log(softmax(x))`.
- `crossEntropyVjp` is a closed form, not a composition — that is why frameworks fuse it.

**DoD**
- [ ] `softmax(x)` vs `softmax(x + 1000)` agree to < 1e-15 per element (max-shift invariance — the exact claim F0.8.3 will visualize)
- [ ] `softmaxVjp` gradchecks at < 1e-7
- [ ] `crossEntropyVjp` equals autograd-through-`logSoftmax` to < 1e-10, and gradchecks
- [ ] `softmax` of logits at 800 without the max shift overflows to NaN; with it, does not (this case gets reused by F4.1.2)
- [ ] Commit: `feat(core): softmax, log_softmax, fused softmax-CE and their VJPs`

---

### T18 · `autograd.ts` — tape  →  F0.2.7

```ts
export class Var {
  value: NdArray;
  grad: NdArray | null;
  requiresGrad: boolean;
  readonly parents: readonly Var[];
  readonly op: string;
  readonly backwardFn: ((g: NdArray) => NdArray[]) | null;   // grads wrt parents, in order
  readonly label?: string;                                    // for DAG node names in M2
}

export function variable(value: NdArray, opts?): Var;
export function backward(root: Var): void;                    // root must be scalar
// differentiable wrappers mirroring ops.ts:
export const A = { add, sub, mul, div, matmul, exp, log, sqrt, relu, sum, mean, max,
                   reshape, permute, softmax, crossEntropy };
```

`backward`: DFS post-order topological sort with a visited set → reverse iterate → `root.grad = ones`
→ per node call `backwardFn`, `unbroadcast` each result to the parent's shape, **accumulate** (§2.6),
then assert the gradient-shape invariant (§2.5).

`label` is added now because M2's `GraphDAG` needs stable node names, and retrofitting it means
touching every wrapper.

**DoD**
- [ ] `y = x*x + x` gives `dy/dx = 2x + 1` (the accumulation test — fails under assignment)
- [ ] `relu(x@W + b) → softmax → CE` end-to-end gradcheck on `x`, `W`, `b` all < 1e-7
- [ ] Diamond graph (a value consumed by two branches that re-merge) gradchecks
- [ ] Shape-invariant violation throws naming the op and both shapes
- [ ] Commit: `feat(core): tape-based autograd with accumulation and shape invariants`

---

### T19 · `jacobian.ts`  →  F0.2.8

```ts
softmaxJacobian(s: NdArray): NdArray          // closed form J_ij = s_i(δ_ij − s_j), 1-D input
jacobianByVjp(f: (x: Var) => Var, x: NdArray): NdArray   // m VJP passes → (m, n)
jacobianMemoryEstimate(n: number, bytesPerElement?: number):
  { elements: number; bytes: number; human: string }     // for F0.8.5's scale slider

export class JacobianTooLargeError extends Error {
  readonly n: number; readonly elements: number; readonly bytes: number;
}
```

Hard guard at `n > 16`: throw `JacobianTooLargeError` carrying the numbers **so the UI can render the
"a 4096×4096 Jacobian is 64 MB" message instead of crashing**. The error is a feature, not a failure
mode — F0.8.5 depends on it.

**DoD**
- [ ] `softmaxJacobian(s) @ sBar` equals `softmaxVjp(s, sBar)` to < 1e-12 — the identity F0.8.4 visualizes
- [ ] `jacobianByVjp` agrees with `softmaxJacobian` for n ≤ 16
- [ ] `n = 17` throws `JacobianTooLargeError` with populated fields
- [ ] Commit: `feat(core): explicit Jacobian builders with size guard`

---

### T20 · Verify harness v3 + epic sign-off

Extend `verify.ts` with T17–T19 cases (softmax invariance, fused-CE equivalence, autograd
accumulation and diamond graphs, Jacobian↔VJP identity). Then:

- Run `npm run verify:engine` — all green
- Run `npm run lint` — 0 warnings
- Write `docs/ENGINE.md`: the public API surface of `core/tensor`, the view/copy contract, and the
  gradient-shape invariant. This is what E0.5/E0.7 will code against, and it is the file that will be
  copied into your `handmade-transformer` repo later.
- Commit: `docs: engine API reference; chore: verify harness v3`

---

## 4. Epic Sign-off

**E0.1 done when**
- [ ] `npm run dev` / `build` / `lint` / `deploy` all work
- [ ] `/`, `/step/0-1`…`/step/0-5`, `/dev/tokens` all render; sidebar highlights the active step
- [ ] Locale toggle works and persists
- [ ] Below 1280px the app tree is unmounted behind the gate
- [ ] A `react` import inside `src/core/` fails lint

**E0.2 done when**
- [ ] `npm run verify:engine` exits 0 with every VJP gradchecking below 1e-7
- [ ] `transpose` provably shares its buffer; `reshape` of a non-contiguous array provably copies and reports the element count
- [ ] `softmax(x) == softmax(x + 1000)` to 1e-15
- [ ] `softmaxJacobian(s) @ sBar == softmaxVjp(s, sBar)` to 1e-12
- [ ] `core/` imports nothing from React or `viz/`
- [ ] `docs/ENGINE.md` exists and matches the code

**Explicitly still missing after both epics** (so nobody thinks M0 is close to done):
E0.3 trace recorder · E0.4 highlight bus and formula engine · E0.5 all viz primitives ·
E0.6 content system · E0.7 Pyodide and the parity panel · E0.8 the actual Step 0.3 page.

---

## 5. Task Dependency Graph

```
T1 scaffold
 ├─ T2 tokens ──────────────┐
 ├─ T3 lint/format/hooks    │
 └─ T4 shell/router ── T5 gate ── T6 deploy      [E0.1]
                            │
       ┌────────────────────┘
       ▼
T7 ndarray ── T8 shape ── T9 broadcast ─┐
T10 random+format (parallel) ───────────┤
T11 trace hook seam ────────────────────┤
                                        ▼
                                  T12 ops forward
                                        │
                                  T13 gradcheck
                                        │
                                  T14 verify v1
                                        │
                                  T15 vjp ── T16 verify v2   ← engine becomes trustworthy here
                                        │
                        ┌───────────────┴───────────────┐
                  T17 softmax/CE                   T18 autograd
                        └───────────────┬───────────────┘
                                  T19 jacobian
                                        │
                                  T20 verify v3 + ENGINE.md    [E0.2]
```

T16 is the gate: nothing downstream is worth building on a red harness.
