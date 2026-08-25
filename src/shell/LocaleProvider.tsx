import { useCallback, useMemo, useState } from 'react';
import {
  LOCALE_STORAGE_KEY,
  LocaleContext,
  readStoredLocale,
  type L,
  type Locale,
  type LocaleState,
} from '../content/i18n';

export default function LocaleProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readStoredLocale);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    localStorage.setItem(LOCALE_STORAGE_KEY, next);
    document.documentElement.lang = next === 'zh' ? 'zh-CN' : 'en';
  }, []);

  const value = useMemo<LocaleState>(
    () => ({ locale, setLocale, t: <T,>(v: L<T>): T => v[locale] }),
    [locale, setLocale],
  );

  return <LocaleContext value={value}>{children}</LocaleContext>;
}
