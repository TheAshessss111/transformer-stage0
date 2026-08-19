import LocaleToggle from './LocaleToggle';
import { useLocale } from './locale';

export default function TopBar() {
  const { t } = useLocale();

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-line bg-surface px-6">
      <div className="text-xs text-ink-dim">
        {t({
          zh: '把阶段 0 的数学变成看得见的东西',
          en: 'Making the Stage 0 mathematics visible',
        })}
      </div>
      <LocaleToggle />
    </header>
  );
}
