import type { NdArray } from '../../core/tensor/ndarray';
import type { L } from '../../core/i18n';
import type { HighlightTarget } from '../highlight/types';

/**
 * What one term of an equation is, means, and points at.
 *
 * Metadata lives beside the LaTeX rather than inside it. Cramming shapes and
 * value accessors into the markup would make both unreadable, and the LaTeX has
 * to stay something a person can proofread against a textbook.
 */
export interface TermSpec<Ctx = unknown> {
  /** "What this term is." */
  label: L;
  /** "What it is doing" — the reason it appears at all. */
  purpose: L;
  /** Shape in the plan's symbols, e.g. "(B, T, 1)". */
  shape?: string;
  /** "The value right now", read live from the lab's state. */
  read?: (ctx: Ctx) => NdArray | number | undefined;
  /** Which tensors, code lines or events this term corresponds to. */
  highlight?: readonly HighlightTarget[];
}

export interface EquationSpec<Ctx = unknown> {
  /** Scopes term ids, so two equations may reuse a term name. */
  id: string;
  /** LaTeX using `\term{id}{…}` for anything interactive. */
  latex: string;
  /** Display mode (centred, full size). Default true. */
  display?: boolean;
  terms: Readonly<Record<string, TermSpec<Ctx>>>;
}
