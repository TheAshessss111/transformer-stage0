import { renderToString, type KatexOptions, type TrustContext } from 'katex';

/**
 * KaTeX rendering, with the per-term markup mechanism locked by the T27 spike.
 *
 * Authors write `\term{id}{content}`, which expands to KaTeX's `\htmlData`, so
 * every marked sub-expression carries `data-term="id"` in the output. Verified
 * to survive `\frac`, `\sum` limits and two levels of nesting.
 *
 * An ATTRIBUTE rather than an id on purpose: `\htmlId` also works, but ids are
 * document-global, so two equations using the same term name would collide.
 * See docs/product/decisions.md D-25.
 */

export const TERM_ATTRIBUTE = 'data-term';
export const TERM_SELECTOR = `[${TERM_ATTRIBUTE}]`;

/**
 * `\htmlData` requires `trust`, which KaTeX's strict mode then complains about
 * on every single render. Silence that one code and leave every other strict
 * warning switched on — they are how a malformed formula gets noticed.
 */
const strict = (errorCode: string): 'ignore' | 'warn' =>
  errorCode === 'htmlExtension' ? 'ignore' : 'warn';

/**
 * `trust` gates every HTML-emitting command at once. D-25 needs exactly one of
 * them, so grant exactly one.
 *
 * `trust: true` would also enable `\href`, `\url`, `\includegraphics`,
 * `\htmlId`, `\htmlClass` and `\htmlStyle` for every equation on the site — and
 * `EquationSpec.latex` is an unconstrained string that F0.6.2's content system
 * will start filling from content files, rendered through
 * dangerouslySetInnerHTML.
 */
const trust = (context: TrustContext): boolean => context.command === '\\htmlData';

export const KATEX_OPTIONS: KatexOptions = {
  trust,
  throwOnError: false,
  strict,
  macros: {
    '\\term': '\\htmlData{term=#1}{#2}',
  },
};

/** Rendering is not free and formulas re-render on every highlight change. */
const CACHE_LIMIT = 256;
const cache = new Map<string, string>();

export function renderLatex(latex: string, displayMode = true): string {
  const key = `${displayMode ? 'd' : 'i'}:${latex}`;
  const hit = cache.get(key);
  if (hit !== undefined) return hit;

  const html = renderToString(latex, { ...KATEX_OPTIONS, displayMode });

  if (cache.size >= CACHE_LIMIT) {
    const oldest = cache.keys().next();
    if (!oldest.done) cache.delete(oldest.value);
  }
  cache.set(key, html);
  return html;
}
