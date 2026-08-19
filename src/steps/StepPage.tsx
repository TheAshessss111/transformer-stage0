import { Navigate, useParams } from 'react-router';
import { findStep } from './registry';
import { useLocale } from '../shell/locale';

/**
 * Placeholder step page. The five-section template (F0.6.5) replaces this in E0.6.
 */
export default function StepPage() {
  const { id } = useParams();
  const step = id ? findStep(id) : undefined;
  const { t } = useLocale();

  if (!step) return <Navigate to="/" replace />;

  return (
    <div className="mx-auto max-w-4xl p-8">
      <div className="font-mono tabular text-sm text-ink-faint">Step {step.number}</div>
      <h1 className="mt-1 text-2xl font-semibold text-ink">{t(step.title)}</h1>
      <p className="mt-2 text-ink-dim">{t(step.tagline)}</p>

      <div className="mt-8 rounded-lg border border-line bg-panel p-6">
        <div
          className="inline-block rounded border px-2 py-1 font-mono text-xs"
          style={{ borderColor: 'var(--color-view)', color: 'var(--color-view)' }}
        >
          {t({ zh: '招牌可视化', en: 'signature visualization' })}
        </div>
        <p className="mt-3 text-ink">{t(step.signatureViz)}</p>
        <p className="mt-4 text-sm text-ink-faint">
          {t({
            zh: '此页面尚未实现。内容系统见 E0.6，交互实验室见 E0.5。',
            en: 'Not built yet. The content system is E0.6; the interactive labs are E0.5.',
          })}
        </p>
      </div>
    </div>
  );
}
