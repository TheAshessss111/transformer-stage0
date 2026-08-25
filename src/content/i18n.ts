import { createContext, use } from 'react';
import type { L, Locale } from '../core/i18n';

/**
 * Locale plumbing for the UI layers.
 *
 * The `L<T>` type itself lives in core/i18n.ts, because it is a data shape and
 * because core/trace carries bilingual display text. This module adds the React
 * side: the context, the hook, and persistence.
 *
 * The full content system (typed Block lists, BlockRenderer) is F0.6.2 and will
 * be built on top of this without changing it.
 */

export type { L, Locale } from '../core/i18n';
export { resolveL } from '../core/i18n';

export interface LocaleState {
  locale: Locale;
  setLocale: (next: Locale) => void;
  /** Resolve a bilingual value against the active locale. */
  t: <T>(value: L<T>) => T;
}

export const LOCALE_STORAGE_KEY = 'stage-zero:locale';

export const LocaleContext = createContext<LocaleState | null>(null);

export function useLocale(): LocaleState {
  const state = use(LocaleContext);
  if (!state) throw new Error('useLocale must be used inside <LocaleProvider>');
  return state;
}

export function readStoredLocale(): Locale {
  const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
  if (stored === 'zh' || stored === 'en') return stored;
  return navigator.language.startsWith('zh') ? 'zh' : 'en';
}
