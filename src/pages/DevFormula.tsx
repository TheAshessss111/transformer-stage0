import { useEffect, useRef, useState } from 'react';
import 'katex/dist/katex.min.css';
import { TERM_SELECTOR, renderLatex, termsInLatex } from '../viz/formula/render';

/**
 * T27 spike proof (/dev/formula).
 *
 * Confirms in a real DOM that `\term{…}{…}` survives nesting and yields
 * addressable elements. The production Formula component is T30; this page
 * stays as the evidence behind D-25.
 */

const SAMPLES: { title: string; latex: string }[] = [
  {
    title: 'LayerNorm backward — three terms, one inside a fraction',
    latex: String.raw`\bar x_i = \frac{1}{\hat\sigma}\left[ \term{direct}{g_i} - \term{de-mean}{\frac{1}{D}\sum_j g_j} - \term{de-scale}{\hat x_i \frac{1}{D}\sum_j g_j \hat x_j} \right]`,
  },
  {
    title: 'Softmax VJP — a term nested inside another term, under a sum',
    latex: String.raw`\bar x_i = \term{outer}{s_i \left( \bar s_i - \term{weighted-mean}{\sum_j \bar s_j s_j} \right)}`,
  },
  {
    title: 'Same term name in a second equation — no id collision',
    latex: String.raw`\term{outer}{a + b}`,
  },
];

function Sample({ title, latex }: { title: string; latex: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [found, setFound] = useState<string[]>([]);
  const [declared, setDeclared] = useState<string[]>([]);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    setDeclared(termsInLatex(latex));
    setFound(
      [...root.querySelectorAll(TERM_SELECTOR)].map((el) => el.getAttribute('data-term') ?? ''),
    );
  }, [latex]);

  // One delegated listener, not one per term — the pattern T30 will formalise.
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const over = (e: PointerEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.(TERM_SELECTOR);
      setActive(el ? (el.getAttribute('data-term') ?? null) : null);
    };
    const out = () => setActive(null);
    root.addEventListener('pointerover', over);
    root.addEventListener('pointerleave', out);
    return () => {
      root.removeEventListener('pointerover', over);
      root.removeEventListener('pointerleave', out);
    };
  }, []);

  const matches = declared.length === found.length && declared.every((t) => found.includes(t));

  return (
    <section className="rounded-lg border border-line bg-panel p-5">
      <h2 className="text-sm font-medium text-ink">{title}</h2>
      <div
        ref={ref}
        className="mt-3 overflow-x-auto rounded border border-line bg-well p-4 [&_[data-term]]:cursor-pointer [&_[data-term]]:rounded [&_[data-term]]:px-0.5 [&_[data-term]:hover]:bg-line-strong"
        dangerouslySetInnerHTML={{ __html: renderLatex(latex) }}
      />
      <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-xs text-ink-dim">
        <dt className="text-ink-faint">declared</dt>
        <dd>{declared.join(', ') || '—'}</dd>
        <dt className="text-ink-faint">in DOM</dt>
        <dd>{found.join(', ') || '—'}</dd>
        <dt className="text-ink-faint">hovering</dt>
        <dd style={{ color: active ? 'var(--color-view)' : undefined }}>{active ?? '—'}</dd>
      </dl>
      <p
        className="mt-3 font-mono text-xs"
        style={{ color: matches ? 'var(--color-ok)' : 'var(--color-err)' }}
      >
        {matches ? '✓ every declared term is addressable in the DOM' : '✗ mismatch'}
      </p>
    </section>
  );
}

export default function DevFormula() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Per-term markup — T27 spike</h1>
        <p className="mt-1 text-sm text-ink-dim">
          Evidence behind D-25. Authors write <code className="font-mono">{'\\term{id}{…}'}</code>,
          which expands to KaTeX <code className="font-mono">\htmlData</code> and yields{' '}
          <code className="font-mono">data-term</code> on the rendered element. Hover a term.
        </p>
      </header>
      {SAMPLES.map((s) => (
        <Sample key={s.title} {...s} />
      ))}
    </div>
  );
}
