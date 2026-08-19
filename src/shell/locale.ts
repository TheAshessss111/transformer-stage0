import { createContext, use } from 'react';

/**
 * Minimal locale plumbing for E0.1.
 *
 * The full content system (typed `L<T>` blocks, BlockRenderer) is F0.6.2 and will
 * be built on top of this context without changing it. See impl plan section 0.5.
 */
export type Locale = 'zh' | 'en';

/** A string that exists in both languages.缺一个字段就编译报错。 */
export interface Bilingual {
  zh: string;
  en: string;
}

export interface LocaleState {
  locale: Locale;
  setLocale: (next: Locale) => void;
  /** Resolve a bilingual string against the active locale. */
  t: (value: Bilingual) => string;
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
