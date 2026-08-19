/**
 * Numeric formatting.
 *
 * `classify` is the single place that decides "is this value special". No
 * visualization component re-implements an isNaN check: they read ValueKind and
 * map it onto --color-nan / --color-inf from the theme.
 */

import { formatShapeTuple, type Shape } from './ndarray.ts';

export type ValueKind = 'finite' | 'zero' | 'nan' | 'posinf' | 'neginf' | 'subnormal';

/** Smallest positive *normal* double; below this, precision degrades. */
export const MIN_NORMAL_F64 = 2.2250738585072014e-308;

export function classify(v: number): ValueKind {
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return 'posinf';
  if (v === -Infinity) return 'neginf';
  if (v === 0) return 'zero'; // also catches -0
  if (Math.abs(v) < MIN_NORMAL_F64) return 'subnormal';
  return 'finite';
}

export interface FormatOptions {
  /** Significant digits for finite values. Default 4. */
  sigDigits?: number;
  /** Below this magnitude, switch to exponential. Default 1e-3. */
  expBelow?: number;
  /** At or above this magnitude, switch to exponential. Default 1e5. */
  expAbove?: number;
}

export interface FormattedValue {
  text: string;
  kind: ValueKind;
}

/**
 * Format one value for display in a matrix cell.
 *
 * Fixed decimal places (rather than toPrecision) so that a column of numbers
 * lines up under `font-variant-numeric: tabular-nums`.
 */
export function formatValue(v: number, options: FormatOptions = {}): FormattedValue {
  const kind = classify(v);
  const sigDigits = options.sigDigits ?? 4;
  const expBelow = options.expBelow ?? 1e-3;
  const expAbove = options.expAbove ?? 1e5;

  switch (kind) {
    case 'nan':
      return { text: 'NaN', kind };
    case 'posinf':
      return { text: '+Inf', kind };
    case 'neginf':
      return { text: '-Inf', kind };
    case 'zero':
      return { text: '0', kind };
    case 'subnormal':
      return { text: v.toExponential(2), kind };
    default:
      break;
  }

  const magnitude = Math.abs(v);
  if (magnitude >= expAbove || magnitude < expBelow) {
    return { text: v.toExponential(Math.max(1, sigDigits - 2)), kind };
  }

  const decimals = Math.min(6, Math.max(0, sigDigits - 1 - Math.floor(Math.log10(magnitude))));
  return { text: v.toFixed(decimals), kind };
}

/** "(2, 4, 8)" */
export function formatShape(shape: Shape): string {
  return formatShapeTuple(shape);
}

/** "(B, T, D)" — the symbol names from transformer_plan.md 纪律一. */
export function formatNamedShape(names: readonly string[]): string {
  return `(${names.join(', ')})`;
}

/** "(B=2, T=4, D=8)" — used in shape pipelines where both matter. */
export function formatBoundShape(names: readonly string[], shape: Shape): string {
  if (names.length !== shape.length) {
    throw new Error(
      `formatBoundShape: ${names.length} names for a rank-${shape.length} shape ${formatShapeTuple(shape)}`,
    );
  }
  return `(${names.map((n, i) => `${n}=${shape[i]}`).join(', ')})`;
}
