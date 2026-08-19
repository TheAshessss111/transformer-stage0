# `core/tensor` — Engine Reference

> The float64 strided tensor engine behind every number on screen.
> E0.5's visualizations and E0.7's parity panel code against this surface.
>
> Written to be portable: `core/` imports nothing from React or `viz/` (enforced
> by an oxlint rule), so this directory can be lifted into your
> `handmade-transformer` repo later as a reference implementation.

Verify it at any time:

```bash
npm run verify
```

102 checks. Runs on Node's native TypeScript type stripping — no test framework,
no config, no CI.

---

## 1. The three contracts

Everything else is detail. These three are what callers may rely on.

### C1 · View vs copy is real, and observable

| Operation | Result | Buffer |
|---|---|---|
| `transpose` `permute` `swapAxes` `select` `expandDims` `squeeze` | always a view | shared |
| `broadcastTo` | always a view, with stride-0 axes | shared, **read-only** |
| `reshape` on a contiguous array | view | shared |
| `reshape` on a non-contiguous array | copies | new |
| `ascontiguousarray` on a contiguous array | returns the *same object* | shared |

`sharesBuffer(a, b)` answers the question directly. The trace event for each
operation carries `isView`, `didCopy` and `copiedElements` — that is what Step
0.1 renders as the cyan `view` / amber `copy · N elements` badge.

`isContiguous` follows NumPy exactly: axes of extent 1 are skipped because their
stride is unobservable, and empty arrays are contiguous. This matters — a
stricter rule would make `reshape` copy when it should not, and Step 0.1 would
teach something false.

### C2 · A gradient has the shape of its variable

Checked at runtime in `backward()`, on every accumulation, in production and not
just in dev. When it fires, the message names the op and both shapes.

`unbroadcast(g, targetShape)` ends with the same assertion. Between them they
catch essentially every broadcast-VJP mistake — the category
`transformer_plan.md` Step 1.3 calls 最容易错的地方.

### C3 · Gradients accumulate, never overwrite

`p.grad = p.grad === null ? contribution : add(p.grad, contribution)`.

A value reached twice sums both contributions. `y = x*x + x` is the canonical
case and the harness checks it.

---

## 2. Module map

| Module | Holds |
|---|---|
| `ndarray.ts` | the `NdArray` structure, indexing, the odometer walkers, constructors, nested-array interop |
| `shape.ts` | reshape / transpose / permute / select / expandDims / squeeze / ascontiguousarray |
| `broadcast.ts` | `broadcastShapes` `broadcastTo` `unbroadcast` |
| `ops.ts` | forward operators — elementwise, reductions, matmul |
| `vjp.ts` | one hand-written backward per forward operator |
| `softmax.ts` | softmax / logSoftmax / fused cross-entropy and their backwards |
| `autograd.ts` | `Var`, topological sort, reverse sweep, differentiable wrappers |
| `jacobian.ts` | explicit Jacobians and the size guard that argues against them |
| `random.ts` | seeded PRNG and initializers |
| `format.ts` | `ValueKind` classification and display formatting |
| `../trace/hook.ts` | the seam E0.3's recorder plugs into |
| `../gradcheck/` | central-difference numerical gradients and relative error |

> Two files here are not in `docs/architecture/overview.md`'s original listing: `softmax.ts`
> (split out of `ops.ts` because Step 0.3 is built entirely on it) and
> `jacobian.ts` (as planned in F0.2.8).

---

## 3. `NdArray`

```ts
class NdArray {
  readonly data: Float64Array;   // buffer, possibly shared with other views
  readonly shape: readonly number[];
  readonly strides: readonly number[];  // ELEMENT units; may be 0 (broadcast)
  readonly offset: number;       // index in `data` of logical [0, 0, …]
  readonly base: NdArray | null; // non-null ⇒ this is a view
  readonly readOnly: boolean;    // set by broadcastTo; writing is banned
}
```

**Why float64 everywhere**: central-difference gradcheck is drowned by float32
rounding (`transformer_plan.md` 纪律二). Reduced precision is *simulated* in
`core/numerics/float.ts` for Step 0.5, never used as storage.

**Why stride-0 is allowed**: it is the real NumPy broadcasting mechanism, and a
teaching artifact — Step 0.1's memory strip draws such an axis as an arrow that
does not advance. One write through it would land on many logical positions, so
those views are flagged `readOnly` and `set()` refuses them.

### Iteration

| Function | Use |
|---|---|
| `forEachOffset(a, fn)` | hot path: flat offset + logical index, no multi-index materialised |
| `forEachIndex(a, fn)` | also supplies the multi-index — **reused between iterations**, copy it if you keep it |
| `forEachZip(shape, arrays, fn)` | several identically-shaped arrays in lockstep; what every elementwise op runs on |
| `readFlat(a)` | logical order; **may alias `a.data`** — treat as read-only |
| `copyFlat(a)` | logical order, always a fresh buffer |

### Constructors

`zeros` `ones` `full` `scalar` `zerosLike` `clone` `fromFlat` `fromNested` `toNested`

---

## 4. Operators

```
unary      neg abs exp log sqrt relu positiveMask
binary     add sub mul div maximum            (broadcasting)
scalar     addScalar mulScalar
reduce     sum mean max min                   axis: number | number[] | null, keepdims
linalg     matmul                             (…,m,k) @ (…,k,n), batch dims broadcast
```

Two behaviours worth knowing:

- **`max` / `min` propagate NaN**, matching NumPy. A reduction that silently
  skipped NaN would hide the very bug Step 0.5 is about.
- **`matmul` rejects rank < 2** rather than implementing NumPy's 1-D promotion
  rules. Transformer code never relies on them, and an error teaches more than a
  silent reinterpretation.

---

## 5. VJPs

Each backward asks for **only what the maths needs**:

```ts
expVjp(g, out)         // the OUTPUT — exp' = exp
sqrtVjp(g, out)        // the OUTPUT
reluVjp(g, mask)       // the 1/0 mask from ops.positiveMask, not the input
softmaxVjp(s, sBar)    // the OUTPUT s. Never x.
logVjp(g, a)           // genuinely needs the input
mulVjp(g, a, b)        // genuinely needs both
```

This is a real memory saving and it is the exact point Step 0.3 makes. Do not
"simplify" these signatures into taking the input as well.

Every VJP that can broadcast ends in `unbroadcast` (plan rule 5: forward
broadcast ↔ backward sum). `max`/`min` split ties evenly.

Inner helper operations run under `suppressOpHook`, so the backward trace has one
event per operator — the same granularity as the forward trace. Each backward
event carries `meta.wrt`: which input the gradient belongs to.

---

## 6. Softmax and cross-entropy

```ts
softmax(x, axis?, { subtractMax? })   // subtractMax defaults true
softmaxSteps(x, axis?, opts)          // { rowMax, shifted, exponentials, denominator, probs }
logSoftmax(x, axis?)                  // LogSumExp, never log(softmax(x))
softmaxVjp(s, sBar, axis?)            // s * (sBar - Σ sBar·s)
logSoftmaxVjp(g, logProbs, axis?)

crossEntropyFromLogits(logits, targets) -> { loss, probs, logProbs }
crossEntropyVjp(probs, targets, gradLoss?)   // (s - onehot(y)) / N
softmaxJacobian(s)                    // explicit (n,n), rank-1 input only
```

- `subtractMax: false` is not a footgun left lying around — it is the control for
  Step 0.5's hazard ①. With logits at 800 the naive path overflows to `Inf` and
  then `Inf/Inf = NaN`; the harness asserts both halves.
- `softmaxSteps` exists so Step 0.3's four-stage playback (F0.8.2) drives itself
  from real intermediates instead of depending on trace granularity.
- The max-shift invariance is **exact in real arithmetic but not in float64**:
  the residual tracks how precisely `x + c` can be represented, so it grows with
  the shift (c=10 → 2e-16, c=1000 → 2e-14). F0.8.3 should show that rather than
  claim bit-identical output.

---

## 7. Autograd

```ts
variable(value, { requiresGrad?, label? })   // leaf
constant(value, label?)                       // no gradient
backward(root)                                // root must be scalar
backwardWithSeed(root, seed)                  // arbitrary seed; used by jacobianByVjp
zeroGrad(root)  gradOf(v)  topoSort(root)

A.{ add sub mul div matmul neg mulScalar exp log sqrt relu
    sum mean max reshape permute softmax logSoftmax crossEntropy }
```

`Var.label` exists from the start because M2's `GraphDAG` needs stable node
names, and retrofitting it would mean touching every wrapper.

The topological sort is iterative, not recursive, so a deep graph cannot blow the
stack.

---

## 8. Jacobians

```ts
softmaxJacobian(s)                      // closed form, J_ij = s_i(δ_ij − s_j)
jacobianByVjp(f, x)                     // m reverse sweeps, each seeded with a basis vector
contractJacobian(J, upstream)           // sBar @ J
jacobianMemoryEstimate(n, bytes?)       // { elements, bytes, human }
JacobianTooLargeError                   // carries { n, elements, bytes }
JACOBIAN_MAX_N = 16
```

The size guard is a **feature**. F0.8.5 catches `JacobianTooLargeError` and
renders its numbers — "a 4096×4096 Jacobian is 64.0 MB, and the VJP does the same
job in O(n)" — instead of the page crashing.

Verified identity, the one F0.8.4 puts on screen:

```
contractJacobian(softmaxJacobian(s), sBar)  ==  softmaxVjp(s, sBar)     to 5e-16
```

---

## 9. Gradcheck

```ts
numericalGrad(f, x, eps = 1e-5)        // central difference, float64
scalarize(forward, upstream)           // f(x) = sum(forward(x) * upstream)
relError(a, b)                         // { max, argmax, perElement }
gradcheck(f, x, analytic, tol = 1e-7)  // { ok, maxRelErr, report, perElement, … }
```

`numericalGrad` runs under `suppressOpHook`: those thousands of forward passes
are an internal procedure, not steps of the computation the user is watching.

`relError` treats a lone NaN or infinity as **maximal** error rather than letting
it vanish into an arithmetic NaN, and returns `perElement` so E0.7 can draw the
error heatmap without recomputing.

When checking anything with a kink (`relu`, `abs`, `max`), keep the samples away
from the non-differentiable point, and draw the upstream gradient away from zero
— a near-zero upstream makes the gradient near-zero, and relative error there is
dominated by finite-difference cancellation rather than by anything the VJP did.

---

## 10. Trace hook

```ts
setOpHook(fn | null)        withOpHook(fn, body)        suppressOpHook(body)

interface RawOpEvent {
  op: string;
  phase: 'forward' | 'backward';
  inputs: readonly NdArray[];
  output: NdArray;
  isView: boolean;
  didCopy: boolean;
  copiedElements?: number;
  meta?: Record<string, unknown>;   // axis, keepdims, permutation, wrt, …
}
```

When no hook is attached, `emit` is a single null check. E0.3 implements the
recorder that plugs in here; E0.2 only planted the call sites.
