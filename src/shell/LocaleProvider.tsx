import { useCallback, useMemo, useState } from 'react';
import {
  LOCALE_STORAGE_KEY,
  LocaleContext,
  readStoredLocale,
  type Bilingual,
  type Locale,
  type LocaleState,
} from './locale';

export default function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readStoredLocale);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    localStorage.setItem(LOCALE_STORAGE_KEY, next);
    document.documentElement.lang = next === 'zh' ? 'zh-CN' : 'en';
  }, []);

  const value = useMemo<LocaleState>(
    () => ({ locale, setLocale, t: (v: Bilingual) => v[locale] }),
    [locale, setLocale],
  );

  return <LocaleContext value={value}>{children}</LocaleContext>;
}
