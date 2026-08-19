import { useLocale } from './locale';

const OPTIONS = [
  { value: 'zh', label: '中' },
  { value: 'en', label: 'EN' },
] as const;

export default function LocaleToggle() {
  const { locale, setLocale } = useLocale();

  return (
    <div
      className="flex overflow-hidden rounded border border-line"
      role="group"
      aria-label="Language"
    >
      {OPTIONS.map((opt) => {
        const active = locale === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => setLocale(opt.value)}
            aria-pressed={active}
            className={`px-2.5 py-1 font-mono text-xs transition-colors ${
              active ? 'bg-line-strong text-ink' : 'text-ink-dim hover:bg-panel hover:text-ink'
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
