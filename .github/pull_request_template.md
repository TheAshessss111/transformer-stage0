## What and why

<!-- What changed, and why it needed to change. The diff already says "what" —
     spend the words on "why". -->

## Roadmap

<!-- Which roadmap items does this close? e.g. E0.3 / F0.3.1–F0.3.4 -->

Closes:

## How it was verified

<!-- CI covers format/lint/types/engine/build. Say what CI could NOT check:
     what you looked at in the browser, what numbers you compared, what you
     deliberately broke to confirm the check fails. -->

## Deviations from plan

<!-- Anything that turned out differently from the roadmap or implementation
     plan. If none, say "none". If some, they must also be recorded in the
     relevant doc — see docs/contributing/definition-of-done.md. -->

---

- [ ] Every commit compiles and passes `verify` on its own (rebase-merge lands them all on `main` — D-24)
- [ ] New `core/` behaviour has a check in `src/core/tensor/__dev__/`
- [ ] New public API is reflected in `docs/architecture/engine.md`
- [ ] No threshold was loosened to make a check pass
