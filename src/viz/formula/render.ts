import { renderToString, type KatexOptions } from 'katex';

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

export const KATEX_OPTIONS: KatexOptions = {
  trust: true,
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

/** Term ids referenced by a LaTeX source, in order of appearance. */
export function termsInLatex(latex: string): string[] {
  const found: string[] = [];
  const pattern = /\\term\s*\{([^}]*)\}/g;
  let match = pattern.exec(latex);
  while (match !== null) {
    found.push(match[1]);
    match = pattern.exec(latex);
  }
  return found;
}

/** Test seam: renderLatex memoizes, so a test needs a way back to a cold start. */
export function clearRenderCache(): void {
  cache.clear();
}
