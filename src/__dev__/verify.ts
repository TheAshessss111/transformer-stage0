/**
 * Verification harness — `npm run verify`.
 *
 * Runs on Node's native TypeScript type stripping: no test framework, no
 * config, no watch mode (D-22's minimal-tooling spirit). It is not a unit-test
 * suite; it is a list of claims the project makes, each one checkable.
 *
 * D-23 later made it a required CI gate, which is what closed risk R-04 — the
 * engine being wrong would make every visualization above it a lie, and nothing
 * on screen would show it.
 *
 * Grouped by area, added task by task:
 *
 *   structural.ts  T14 — NdArray, shape ops, broadcasting, operators, PRNG, formatting
 *   vjp.ts         T16 — gradcheck for every hand-written backward
 *   semantics.ts   T20 — softmax family, autograd, Jacobian identities
 *   trace.ts       T21–T23 — program model, recorder, frames and replay
 */

import './structural.ts';
import './vjp.ts';
import './semantics.ts';
import './trace.ts';
import { report } from './harness.ts';

report();
