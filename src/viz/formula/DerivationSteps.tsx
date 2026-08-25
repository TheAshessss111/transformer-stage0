import { useCallback, useMemo, useState } from 'react';
import Formula from './Formula';
import type { EquationSpec } from './types';
import { useLocale, type L } from '../../content/i18n';
import { useHighlightActions } from '../highlight/useHighlight';
import { target } from '../highlight/types';

/**
 * A derivation revealed one rewrite at a time.
 *
 * Two properties the LayerNorm four-step derivation (F3.1.3) needs:
 *
 *  - every step carries its own justification, because "why is this rewrite
 *    allowed" is the part a reader actually gets stuck on
 *  - `correspondence` links terms across adjacent steps, so hovering a term in
 *    the compact form lights the terms it came from in the expanded one. That
 *    is what makes a four-line collapse legible instead of magic.
 */

export interface DerivationStep<Ctx> {
  equation: EquationSpec<Ctx>;
  /** "Why this rewrite is allowed." */
  justification: L;
  /** Terms of the PREVIOUS step that became terms of this one. */
  correspondence?: readonly { from: string; to: string }[];
}

export interface DerivationStepsProps<Ctx> {
  steps: readonly DerivationStep<Ctx>[];
  ctx?: Ctx;
  /** How many steps are visible before the reader asks for more. */
  initiallyVisible?: number;
}

export default function DerivationSteps<Ctx>({
  steps,
  ctx,
  initiallyVisible = 1,
}: DerivationStepsProps<Ctx>) {
  const { t } = useLocale();
  const [visible, setVisible] = useState(Math.min(initiallyVisible, steps.length));
  const { setHover } = useHighlightActions();

  const allShown = visible >= steps.length;

  /** For step i, the terms of step i-1 that it came from. */
  const linkFor = useCallback(
    (index: number, term: string) => {
      const links = steps[index]?.correspondence ?? [];
      const previous = steps[index - 1];
      if (!previous) return [];
      return links
        .filter((link) => link.to === term)
        .map((link) => target.formulaTerm(previous.equation.id, link.from));
    },
    [steps],
  );

  const handlers = useMemo(
    () =>
      steps.map((step, index) => (term: string | null) => {
        if (term === null) return;
        const ancestors = linkFor(index, term);
        if (ancestors.length === 0) return;
        // Extend the selection the Formula just set with where the term came
        // from, so the two forms light up together.
        setHover([target.formulaTerm(step.equation.id, term), ...ancestors]);
      }),
    [steps, linkFor, setHover],
  );

  return (
    <div className="space-y-4">
      {steps.slice(0, visible).map((step, index) => (
        <div key={step.equation.id} className="rounded-lg border border-line bg-panel p-4">
          <div className="mb-2 font-mono text-xs text-ink-faint">
            {index + 1} / {steps.length}
          </div>
          <Formula
            spec={step.equation}
            ctx={ctx}
            onTermHover={handlers[index]}
            className="overflow-x-auto"
          />
          <p className="mt-3 border-t border-line pt-3 text-sm leading-relaxed text-ink-dim">
            {t(step.justification)}
          </p>
        </div>
      ))}

      {!allShown ? (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setVisible((v) => Math.min(v + 1, steps.length))}
            className="rounded border border-line px-3 py-1.5 text-xs text-ink-dim hover:border-line-strong hover:text-ink"
          >
            {t({ zh: '下一步', en: 'next step' })}
          </button>
          <button
            type="button"
            onClick={() => setVisible(steps.length)}
            className="rounded px-3 py-1.5 text-xs text-ink-faint hover:text-ink-dim"
          >
            {t({ zh: '全部展开', en: 'expand all' })}
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setVisible(Math.min(initiallyVisible, steps.length))}
          className="rounded px-3 py-1.5 text-xs text-ink-faint hover:text-ink-dim"
        >
          {t({ zh: '收起', en: 'collapse' })}
        </button>
      )}
    </div>
  );
}
