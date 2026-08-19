/**
 * The bilingual string primitive.
 *
 * Lives in core/ because it is a data shape, not a UI concern, and because
 * core/trace carries display text (a program step's "what is happening here"
 * note) that must be written in both languages at once.
 *
 * ARCHITECTURE.md A-04: writing one language means writing both, enforced by the
 * type. A key-value catalogue in two files would drift; this cannot.
 */

export type Locale = 'zh' | 'en';

export type L<T = string> = { zh: T; en: T };

export function resolveL<T>(value: L<T>, locale: Locale): T {
  return value[locale];
}
