/**
 * Engine verification harness — `npm run verify:engine`.
 *
 * Runs on Node's native TypeScript type stripping: no test framework, no config,
 * no CI (DECISIONS.md D-22, impl plan section 0.4). Checks are grouped by area
 * and added task by task through E0.2:
 *
 *   structural.ts  T14 — NdArray, shape ops, broadcasting, operators, PRNG, formatting
 *   vjp.ts         T16 — gradcheck for every hand-written backward
 *   semantics.ts   T20 — softmax family, autograd, Jacobian identities
 */

import './structural.ts';
import './vjp.ts';
import './semantics.ts';
import { report } from './harness.ts';

report();
