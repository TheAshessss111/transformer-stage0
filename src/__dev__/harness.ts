/**
 * Minimal check collector for the engine verification harness.
 *
 * Runs under Node's native TypeScript type stripping:
 *     node src/core/tensor/__dev__/verify.ts
 *
 * Deliberately not a test framework (see docs/product/decisions.md D-22 and
 * docs/planning/implementation/m0-e0.1-e0.2.md §0.4): no globals, no config, no watch mode.
 * It collects results, prints a table, and sets a non-zero exit code on failure.
 */

export interface CheckResult {
  group: string;
  name: string;
  ok: boolean;
  detail: string;
}

const results: CheckResult[] = [];
let currentGroup = 'ungrouped';

export function group(name: string): void {
  currentGroup = name;
}

/**
 * Run one check. `fn` returns a detail string on success (e.g. the measured
 * relative error) or throws / returns a falsy-ok object on failure.
 */
export function check(name: string, fn: () => string | void): void {
  try {
    const detail = fn() ?? '';
    results.push({ group: currentGroup, name, ok: true, detail });
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    results.push({ group: currentGroup, name, ok: false, detail });
  }
}

/** Assertion helper that produces readable failure messages. */
export function expect(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const DIM = '\x1b[2m';
const BOLD = '\x1b[1m';
const RESET = '\x1b[0m';

export function report(): void {
  let lastGroup = '';
  let passed = 0;
  let failed = 0;

  for (const r of results) {
    if (r.group !== lastGroup) {
      console.log(`\n${BOLD}${r.group}${RESET}`);
      lastGroup = r.group;
    }
    const mark = r.ok ? `${GREEN}pass${RESET}` : `${RED}FAIL${RESET}`;
    const detail = r.detail ? ` ${DIM}${r.detail}${RESET}` : '';
    console.log(`  ${mark}  ${r.name}${detail}`);
    if (r.ok) passed += 1;
    else failed += 1;
  }

  const summary =
    failed === 0
      ? `${GREEN}${passed} passed${RESET}`
      : `${GREEN}${passed} passed${RESET}, ${RED}${failed} FAILED${RESET}`;
  console.log(`\n${summary}\n`);

  if (failed > 0) process.exitCode = 1;
}
