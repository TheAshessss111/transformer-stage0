import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import 'katex/dist/katex.min.css';
import { TERM_SELECTOR, renderLatex } from './render';
import { assertValidEquationSpec } from './validate';
import type { EquationSpec } from './types';
import { useHighlightActions, useHighlightSelection } from '../highlight/useHighlight';
import { overlapsAny, target, type HighlightTarget } from '../highlight/types';
import type { HighlightLevel } from '../highlight/store';
import TermTooltip from './TermTooltip';

/**
 * A rendered equation whose terms are addressable.
 *
 * Two things worth knowing about the implementation:
 *
 *  1. **One delegated listener**, not one per term. An equation can carry a
 *     dozen terms and re-render on every hover; attaching and detaching that
 *     many listeners each time would be silly.
 *
 *  2. **Highlight state is applied by attribute, not by re-render.** The KaTeX
 *     HTML is memoized on the LaTeX and written once; hover only toggles
 *     `data-highlight` on the term elements. Re-rendering the markup on every
 *     pointer move would throw away KaTeX's work several times a second.
 */

export interface FormulaProps<Ctx> {
  spec: EquationSpec<Ctx>;
  className?: string;
  /** Lab state, so a term's `read` can show the value as it is right now. */
  ctx?: Ctx;
  /** Called when a term is hovered, in case a lab wants to react too. */
  onTermHover?: (term: string | null) => void;
}

const RANK: Record<HighlightLevel, number> = { none: 0, hover: 1, pinned: 2 };

export default function Formula<Ctx>({ spec, className, ctx, onTermHover }: FormulaProps<Ctx>) {
  const root = useRef<HTMLDivElement>(null);
  const { setHover, togglePin } = useHighlightActions();
  const { hover, pinned } = useHighlightSelection();
  const [anchored, setAnchored] = useState<{ term: string; rect: DOMRect } | null>(null);

  if (import.meta.env.DEV) assertValidEquationSpec(spec as unknown as EquationSpec<never>);

  const html = useMemo(
    () => renderLatex(spec.latex, spec.display ?? true),
    [spec.latex, spec.display],
  );

  // Memoized so the prop object identity is stable, not just its contents.
  const inner = useMemo(() => ({ __html: html }), [html]);

  /** Everything a term points at: itself, plus whatever it declares. */
  const targetsFor = useCallback(
    (term: string): HighlightTarget[] => [
      target.formulaTerm(spec.id, term),
      ...(spec.terms[term]?.highlight ?? []),
    ],
    [spec],
  );

  // Delegated input. Bound once; `data-term` resolution walks up from the event
  // target, so it works no matter how deeply KaTeX nested the span.
  useEffect(() => {
    const node = root.current;
    if (!node) return;

    const termAt = (event: Event): string | null => {
      const el = (event.target as HTMLElement | null)?.closest?.(TERM_SELECTOR);
      return el?.getAttribute('data-term') ?? null;
    };

    const onOver = (event: Event) => {
      const el = (event.target as HTMLElement | null)?.closest?.(TERM_SELECTOR);
      const term = el?.getAttribute('data-term') ?? null;
      setHover(term === null ? null : targetsFor(term));
      setAnchored(term === null || !el ? null : { term, rect: el.getBoundingClientRect() });
      onTermHover?.(term);
    };
    const onLeave = () => {
      setHover(null);
      setAnchored(null);
      onTermHover?.(null);
    };
    const onClick = (event: Event) => {
      const term = termAt(event);
      if (term !== null) togglePin(targetsFor(term));
    };

    node.addEventListener('pointerover', onOver);
    node.addEventListener('pointerleave', onLeave);
    node.addEventListener('click', onClick);
    return () => {
      node.removeEventListener('pointerover', onOver);
      node.removeEventListener('pointerleave', onLeave);
      node.removeEventListener('click', onClick);
    };
  }, [setHover, togglePin, targetsFor, onTermHover]);

  // Reflect highlight state onto the DOM.
  //
  // Deliberately runs after EVERY commit, with no dependency array, and as a
  // layout effect rather than a passive one. React re-applies
  // dangerouslySetInnerHTML on every render even when the html string is
  // byte-identical -- verified: the container node survives, the child nodes do
  // not -- so anything written onto those children is destroyed on the next
  // render. A dependency-filtered effect would leave the terms un-highlighted
  // after any unrelated state change, which is exactly the kind of bug that
  // looks like "the hover just stopped working sometimes".
  //
  // useLayoutEffect so the attributes are back before paint, with no flicker.
  useLayoutEffect(() => {
    const node = root.current;
    if (!node) return;

    for (const el of node.querySelectorAll(TERM_SELECTOR)) {
      const term = el.getAttribute('data-term');
      if (term === null) continue;

      // A term is lit when the selection meets the term itself OR anything it
      // declares — which is what makes the link work in both directions.
      let level: HighlightLevel = 'none';
      for (const candidate of targetsFor(term)) {
        let candidateLevel: HighlightLevel = 'none';
        for (const group of pinned) {
          if (overlapsAny(candidate, group)) {
            candidateLevel = 'pinned';
            break;
          }
        }
        if (candidateLevel === 'none' && hover !== null && overlapsAny(candidate, hover)) {
          candidateLevel = 'hover';
        }
        if (RANK[candidateLevel] > RANK[level]) level = candidateLevel;
      }

      if (level === 'none') el.removeAttribute('data-highlight');
      else el.setAttribute('data-highlight', level);

      const label = spec.terms[term]?.label;
      if (label) el.setAttribute('aria-label', label.en);
    }
  });

  const anchoredTerm = anchored ? spec.terms[anchored.term] : undefined;

  return (
    <>
      <div
        ref={root}
        className={className}
        data-equation={spec.id}
        dangerouslySetInnerHTML={inner}
      />
      {anchored && anchoredTerm ? (
        <TermTooltip term={anchoredTerm} ctx={ctx} anchor={anchored.rect} />
      ) : null}
    </>
  );
}
