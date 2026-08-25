import { Link } from 'react-router';
import { STEPS } from '../steps/registry';
import { useLocale } from '../content/i18n';

export default function Landing() {
  const { t } = useLocale();

  return (
    <div className="mx-auto max-w-4xl p-8">
      <h1 className="text-3xl font-semibold text-ink">Stage Zero</h1>
      <p className="mt-3 max-w-2xl leading-relaxed text-ink-dim">
        {t({
          zh: '阶段 0 的目标是：能在白纸上，不查资料，手推出 softmax 和 LayerNorm 的反向。这个页面把那条路上的每一步变成可以动手拨弄的东西。',
          en: 'The goal of Stage 0 is to derive the softmax and LayerNorm backward passes on a blank sheet of paper, from memory. This site turns every step of that path into something you can poke at.',
        })}
      </p>

      <ul className="mt-10 grid gap-3">
        {STEPS.map((step) => (
          <li key={step.id}>
            <Link
              to={`/step/${step.id}`}
              className="block rounded-lg border border-line bg-panel p-5 transition-colors hover:border-line-strong"
            >
              <div className="flex items-baseline gap-3">
                <span className="font-mono tabular text-sm text-ink-faint">{step.number}</span>
                <span className="font-medium text-ink">{t(step.title)}</span>
              </div>
              <p className="mt-1 text-sm text-ink-dim">{t(step.tagline)}</p>
              <p className="mt-2 font-mono text-xs text-ink-faint">{t(step.signatureViz)}</p>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
