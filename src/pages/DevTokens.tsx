/**
 * Design token preview (/dev/tokens).
 *
 * Exists to make NFR-7 checkable in one screen: every ramp stop carries a sample
 * numeral, so "is the text still legible on this fill?" is a look, not a guess.
 *
 * The interpolation and contrast helpers below are local on purpose — E0.5's
 * `viz/scales.ts` will own the real implementation and this page will switch to it.
 */

const RAMP_STOPS = 21;

/** Sample numerals rendered on top of each ramp stop. */
function sampleFor(t: number): string {
  return (t === 0 ? 0 : t).toFixed(2).replace('-0.00', '0.00');
}

/**
 * Diverging fill for a signed value in [-1, 1].
 * Zero maps to --color-grad-zero (== --color-well), so a zero matrix reads empty.
 */
function divergingFill(t: number): string {
  const anchor = t < 0 ? 'var(--color-grad-neg)' : 'var(--color-grad-pos)';
  const pct = Math.abs(t) * 100;
  return `color-mix(in oklab, ${anchor} ${pct}%, var(--color-grad-zero))`;
}

/** Sequential fill for a magnitude in [0, 1] across the five --color-mag-* anchors. */
function sequentialFill(t: number): string {
  const scaled = t * 4;
  const lo = Math.min(3, Math.floor(scaled));
  const frac = (scaled - lo) * 100;
  return `color-mix(in oklab, var(--color-mag-${lo + 1}) ${frac}%, var(--color-mag-${lo}))`;
}

/** Approximate OKLCH lightness of each scale so text contrast can flip. */
const DIVERGING_L = { neg: 0.62, zero: 0.13, pos: 0.62 };
const MAG_L = [0.16, 0.32, 0.48, 0.66, 0.85];

function inkForLightness(l: number): string {
  return l > 0.68 ? 'var(--color-surface)' : 'var(--color-ink)';
}

function divergingInk(t: number): string {
  const end = t < 0 ? DIVERGING_L.neg : DIVERGING_L.pos;
  return inkForLightness(DIVERGING_L.zero + (end - DIVERGING_L.zero) * Math.abs(t));
}

function sequentialInk(t: number): string {
  const scaled = t * 4;
  const lo = Math.min(3, Math.floor(scaled));
  const frac = scaled - lo;
  return inkForLightness(MAG_L[lo] + (MAG_L[lo + 1] - MAG_L[lo]) * frac);
}

interface SwatchProps {
  token: string;
  note?: string;
}

function Swatch({ token, note }: SwatchProps) {
  return (
    <div className="flex items-center gap-3">
      <div
        className="h-10 w-10 shrink-0 rounded border border-line"
        style={{ background: `var(${token})` }}
      />
      <div className="min-w-0">
        <div className="truncate font-mono text-xs text-ink">{token}</div>
        {note ? <div className="truncate text-xs text-ink-faint">{note}</div> : null}
      </div>
    </div>
  );
}

interface SectionProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}

function Section({ title, subtitle, children }: SectionProps) {
  return (
    <section className="rounded-lg border border-line bg-panel p-5">
      <h2 className="text-sm font-semibold tracking-wide text-ink uppercase">{title}</h2>
      {subtitle ? (
        <p className="mt-1 mb-4 text-xs text-ink-dim">{subtitle}</p>
      ) : (
        <div className="mb-4" />
      )}
      {children}
    </section>
  );
}

export default function DevTokens() {
  const stops = Array.from({ length: RAMP_STOPS }, (_, i) => i / (RAMP_STOPS - 1));

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Design tokens</h1>
        <p className="mt-1 text-sm text-ink-dim">
          Tailwind&apos;s default palette is reset to <code className="font-mono">initial</code>, so
          every colour on this page is the only colour the app can use.
        </p>
      </header>

      <Section title="Surfaces" subtitle="Three levels plus borders and three ink weights.">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Swatch token="--color-surface" note="page background" />
          <Swatch token="--color-panel" note="raised card / sidebar" />
          <Swatch token="--color-well" note="recessed: grids, code" />
          <Swatch token="--color-line" note="borders, dividers" />
          <Swatch token="--color-line-strong" note="emphasised border" />
          <Swatch token="--color-ink" note="primary text" />
          <Swatch token="--color-ink-dim" note="secondary, axis labels" />
          <Swatch token="--color-ink-faint" note="tertiary, disabled" />
        </div>
      </Section>

      <Section
        title="Diverging scale — signed values"
        subtitle="For gradients. Zero equals --color-well, so a zero matrix reads as empty rather than grey."
      >
        <div className="flex overflow-hidden rounded">
          {stops.map((s) => {
            const t = s * 2 - 1;
            return (
              <div
                key={s}
                className="flex h-14 flex-1 items-center justify-center font-mono tabular text-[10px]"
                style={{ background: divergingFill(t), color: divergingInk(t) }}
              >
                {sampleFor(t)}
              </div>
            );
          })}
        </div>
        <div className="mt-2 flex justify-between font-mono text-xs text-ink-faint">
          <span>--color-grad-neg</span>
          <span>--color-grad-zero</span>
          <span>--color-grad-pos</span>
        </div>
      </Section>

      <Section
        title="Sequential scale — non-negative magnitudes"
        subtitle="For probabilities and absolute values. Five anchors, interpolated in oklab."
      >
        <div className="flex overflow-hidden rounded">
          {stops.map((s) => (
            <div
              key={s}
              className="flex h-14 flex-1 items-center justify-center font-mono tabular text-[10px]"
              style={{ background: sequentialFill(s), color: sequentialInk(s) }}
            >
              {s.toFixed(2)}
            </div>
          ))}
        </div>
        <div className="mt-3 grid grid-cols-5 gap-2">
          {MAG_L.map((_, i) => `--color-mag-${i}`).map((token) => (
            <Swatch key={token} token={token} />
          ))}
        </div>
      </Section>

      <Section
        title="Memory semantics"
        subtitle="RULE: badges, borders and pipeline connectors only — never a data fill, so these can never be mistaken for the diverging scale."
      >
        <div className="mb-4 flex gap-3">
          <span
            className="rounded border px-2 py-1 font-mono text-xs"
            style={{ borderColor: 'var(--color-view)', color: 'var(--color-view)' }}
          >
            view · zero-copy
          </span>
          <span
            className="rounded border px-2 py-1 font-mono text-xs"
            style={{ borderColor: 'var(--color-copy)', color: 'var(--color-copy)' }}
          >
            copy · 512 elements
          </span>
        </div>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Swatch token="--color-view" note="zero-copy view" />
          <Swatch token="--color-copy" note="memory copy happened" />
        </div>
      </Section>

      <Section
        title="Non-finite values"
        subtitle="format.ts ValueKind maps onto these. No component re-implements isNaN checks."
      >
        <div className="mb-4 flex gap-2 font-mono text-xs">
          {[
            { label: 'NaN', color: 'var(--color-nan)' },
            { label: '+Inf', color: 'var(--color-inf)' },
            { label: '-Inf', color: 'var(--color-inf)' },
          ].map((v) => (
            <div
              key={v.label}
              className="flex h-10 w-20 items-center justify-center rounded"
              style={{ background: v.color, color: 'var(--color-surface)' }}
            >
              {v.label}
            </div>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Swatch token="--color-nan" />
          <Swatch token="--color-inf" />
        </div>
      </Section>

      <Section title="Status">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Swatch token="--color-ok" />
          <Swatch token="--color-warn" />
          <Swatch token="--color-err" />
        </div>
      </Section>

      <Section title="Type" subtitle="System stacks only — no web fonts.">
        <p className="font-sans text-base text-ink">
          Sans — 形状契约 · Shape contract · 0123456789
        </p>
        <p className="mt-2 font-mono tabular text-base text-ink">
          Mono — scores.shape == (B, H, T, S) · -1.2345e-07 · 0123456789
        </p>
        <p className="mt-2 font-mono text-xs text-ink-dim">
          Tabular numerals are on via the <code>tabular</code> utility so matrix cells do not jitter
          as digits change.
        </p>
      </Section>
    </div>
  );
}
