/**
 * Engine verification harness — run with `npm run verify:engine`.
 *
 * Checks are added task by task through E0.2:
 *   T14 (v1) structural invariants: NdArray, shape ops, broadcasting, PRNG, formatting
 *   T16 (v2) gradcheck for every VJP
 *   T20 (v3) softmax family, autograd, Jacobian identities
 */

import { report } from './harness.ts';

report();
