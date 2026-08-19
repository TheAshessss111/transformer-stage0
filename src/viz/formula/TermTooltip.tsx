import { useLayoutEffect, useRef, useState } from 'react';
import { NdArray, copyFlat, size } from '../../core/tensor/ndarray';
import { formatShape, formatValue } from '../../core/tensor/format';
import { useLocale } from '../../content/i18n';
import type { TermSpec } from './types';

/**
 * Four panes, always in this order: what / shape / value now / purpose.
 *
 * Fixed order so the eye learns where to look, and every pane present even when
 * empty — a missing `read` shows "not bound" rather than collapsing the layout
 * and moving the other three.
 *
 * The value pane is what makes this more than a glossary: it reads live from the
 * lab's state, so dragging a slider changes the number inside an open tooltip.
 */

const PREVIEW_VALUES = 6;
const GAP = 10;

function ValuePreview({ value }: { value: NdArray | number | undefined }) {
  if (value === undefined) {
    return <span className="text-ink-faint">not bound</span>;
  }

  if (typeof value === 'number') {
    const { text, kind } = formatValue(value);
    return (
      <span className="tabular" style={{ color: colorFor(kind) }}>
        {text}
      </span>
    );
  }

  const flat = copyFlat(value);
  const shown = Math.min(PREVIEW_VALUES, flat.length);
  return (
    <span className="tabular">
      {Array.from({ length: shown }, (_, i) => {
        const { text, kind } = formatValue(flat[i]);
        return (
          <span key={i} style={{ color: colorFor(kind) }}>
            {text}
            {i < shown - 1 ? ' ' : ''}
          </span>
        );
      })}
      {flat.length > shown ? <span className="text-ink-faint"> … {size(value)} total</span> : null}
    </span>
  );
}

function colorFor(kind: string): string | undefined {
  if (kind === 'nan') return 'var(--color-nan)';
  if (kind === 'posinf' || kind === 'neginf') return 'var(--color-inf)';
  return undefined;
}

export interface TermTooltipProps<Ctx> {
  term: TermSpec<Ctx>;
  ctx: Ctx | undefined;
  /** Viewport rect of the term element this is anchored to. */
  anchor: DOMRect;
}

export default function TermTooltip<Ctx>({ term, ctx, anchor }: TermTooltipProps<Ctx>) {
  const { t } = useLocale();
  const box = useRef<HTMLDivElement>(null);
  const [placement, setPlacement] = useState<{ left: number; top: number }>({
    left: anchor.left,
    top: anchor.bottom + GAP,
  });

  // Clamp into the viewport, flipping above the term when there is no room
  // below. Measured after layout so the real size is known.
  useLayoutEffect(() => {
    const node = box.current;
    if (!node) return;
    const rect = node.getBoundingClientRect();

    let left = anchor.left + anchor.width / 2 - rect.width / 2;
    left = Math.max(GAP, Math.min(left, window.innerWidth - rect.width - GAP));

    let top = anchor.bottom + GAP;
    if (top + rect.height > window.innerHeight - GAP) top = anchor.top - rect.height - GAP;
    top = Math.max(GAP, top);

    setPlacement({ left, top });
  }, [anchor]);

  const value = ctx === undefined ? undefined : term.read?.(ctx);

  return (
    <div
      ref={box}
      role="tooltip"
      className="pointer-events-none fixed z-50 max-w-sm rounded-lg border border-line-strong bg-panel p-3 shadow-xl"
      style={{ left: placement.left, top: placement.top }}
    >
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
        <dt className="text-ink-faint">{t({ zh: '是什么', en: 'what' })}</dt>
        <dd className="text-ink">{t(term.label)}</dd>

        <dt className="text-ink-faint">{t({ zh: '形状', en: 'shape' })}</dt>
        <dd className="font-mono text-ink-dim">
          {term.shape ??
            (value instanceof NdArray ? (
              formatShape(value.shape)
            ) : (
              <span className="text-ink-faint">scalar</span>
            ))}
        </dd>

        <dt className="text-ink-faint">{t({ zh: '现在的值', en: 'value now' })}</dt>
        <dd className="font-mono break-all text-ink-dim">
          <ValuePreview value={value} />
        </dd>

        <dt className="text-ink-faint">{t({ zh: '在干什么', en: 'purpose' })}</dt>
        <dd className="leading-relaxed text-ink-dim">{t(term.purpose)}</dd>
      </dl>
    </div>
  );
}
