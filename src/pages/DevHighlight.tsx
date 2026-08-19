import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { copyFlat } from '../core/tensor/ndarray';
import { formatValue } from '../core/tensor/format';
import { framesFor } from '../core/trace/replay';
import { SOFTMAX_PROGRAM, softmaxInputs, type SoftmaxState } from '../labs/softmaxProgram';
import { useTracedProgram } from '../viz/useTracedProgram';
import Formula from '../viz/formula/Formula';
import DerivationSteps, { type DerivationStep } from '../viz/formula/DerivationSteps';
import type { EquationSpec } from '../viz/formula/types';
import { useHighlightActions, useHighlightLevel } from '../viz/highlight/useHighlight';
import { target } from '../viz/highlight/types';
import { useLocale } from '../content/i18n';

/**
 * T33 — bidirectional linking, proved.
 *
 * The grid here is a throwaway: E0.5's TensorGrid is the real one and will
 * inherit this wiring. What matters is that the link works in BOTH directions,
 * and that hovering one cell does not re-render the whole grid.
 */

const VJP: EquationSpec<SoftmaxState> = {
  id: 'softmax-vjp',
  latex: String.raw`\bar x_i = \term{outer}{s_i \left( \term{upstream}{\bar s_i} - \term{weighted-mean}{\sum_j \bar s_j s_j} \right)}`,
  terms: {
    outer: {
      label: { zh: '整个 VJP', en: 'the whole VJP' },
      purpose: {
        zh: '把上游梯度转成对输入的梯度。只用到输出 s，没有用到输入 x。',
        en: 'Turns the upstream gradient into the input gradient. Uses only the output s, never x.',
      },
      shape: '(B, T)',
      read: (s) => s.dx,
      highlight: [target.tensor('dx'), target.codeLine(7)],
    },
    upstream: {
      label: { zh: '上游梯度', en: 'the upstream gradient' },
      purpose: {
        zh: '从损失一路传下来的、对 softmax 输出的梯度。',
        en: 'The gradient of the loss with respect to the softmax output.',
      },
      shape: '(B, T)',
      read: (s) => s.upstream,
      highlight: [target.tensor('upstream')],
    },
    'weighted-mean': {
      label: { zh: '按概率加权的平均', en: 'the probability-weighted mean' },
      purpose: {
        zh: '减掉它，就是 softmax 反向的全部内容 —— 一次去中心化。上游梯度里任何"整行统一平移"的成分都改变不了概率，所以会被这一项吃掉。',
        en: 'Subtracting it is the entire backward pass: a de-centring. Any component of the upstream gradient that is uniform across the row cannot change the probabilities, and is removed here.',
      },
      shape: '(B, 1)',
      read: (s) => s.weightedMean,
      highlight: [target.tensor('weightedMean'), target.codeLine(6)],
    },
  },
};

const FORWARD: EquationSpec<SoftmaxState> = {
  id: 'softmax-forward',
  latex: String.raw`s_i = \frac{\term{numerator}{e^{z_i}}}{\term{denominator}{\sum_j e^{z_j}}}, \quad \term{shift}{z = x - \max_j x_j}`,
  terms: {
    numerator: {
      label: { zh: '分子', en: 'the numerator' },
      purpose: { zh: '每个位置的未归一化权重。', en: 'The unnormalised weight at each position.' },
      shape: '(B, T)',
      read: (s) => s.e,
      highlight: [target.tensor('e'), target.codeLine(3)],
    },
    denominator: {
      label: { zh: '分母', en: 'the normalizer' },
      purpose: { zh: '把整行拉成和为 1。', en: 'Pulls the row to sum to 1.' },
      shape: '(B, 1)',
      read: (s) => s.denom,
      highlight: [target.tensor('denom'), target.codeLine(4)],
    },
    shift: {
      label: { zh: '减去行最大值', en: 'the max shift' },
      purpose: {
        zh: '数学上不改变结果，但它是 exp 不溢出的唯一原因。',
        en: 'Changes nothing mathematically, and is the only reason exp does not overflow.',
      },
      shape: '(B, T)',
      read: (s) => s.z,
      highlight: [target.tensor('z'), target.tensor('rowMax'), target.codeLine(2)],
    },
  },
};

const DERIVATION: DerivationStep<SoftmaxState>[] = [
  {
    equation: {
      id: 'jac',
      latex: String.raw`\frac{\partial s_i}{\partial x_j} = \term{jac}{s_i (\delta_{ij} - s_j)}`,
      terms: {
        jac: {
          label: { zh: '雅可比的一项', en: 'one entry of the Jacobian' },
          purpose: {
            zh: '完整雅可比是 (n, n)。n=4096 时它有 64MB，所以从不显式构造。',
            en: 'The full Jacobian is (n, n). At n=4096 that is 64MB, which is why it is never built.',
          },
        },
      },
    },
    justification: {
      zh: '直接对 softmax 求偏导得到的雅可比。',
      en: 'The Jacobian, straight from differentiating softmax.',
    },
  },
  {
    equation: {
      id: 'contract',
      latex: String.raw`\bar x_i = \sum_j \bar s_j \frac{\partial s_j}{\partial x_i} = \term{expanded}{s_i \bar s_i - s_i \sum_j \bar s_j s_j}`,
      terms: {
        expanded: {
          label: { zh: '展开后的两项', en: 'the two expanded terms' },
          purpose: {
            zh: 'δ 项留下 s_i s̄_i，另一项收成一个标量和。',
            en: 'The delta leaves s_i s̄_i; the rest collapses into a single scalar sum.',
          },
        },
      },
    },
    justification: {
      zh: '把上游梯度和雅可比缩并。δ_ij 让第一项只剩对角，第二项对 j 求和收成一个标量。',
      en: 'Contract the upstream gradient with the Jacobian. The delta reduces the first term to its diagonal; the second sums over j into a scalar.',
    },
    correspondence: [{ from: 'jac', to: 'expanded' }],
  },
  {
    equation: VJP,
    justification: {
      zh: '提出公因子 s_i。两行代码，O(n) —— 雅可比从未出现过。',
      en: 'Factor out s_i. Two lines, O(n), and the Jacobian never appeared.',
    },
    correspondence: [{ from: 'expanded', to: 'outer' }],
  },
];

/**
 * A cell that subscribes to exactly one target, and counts its own renders.
 *
 * memo is not an optimisation here, it is what makes the design work. The Grid
 * label subscribes to the WHOLE tensor, so hovering any cell re-renders the
 * Grid -- and without memo that would re-render all 24 children, which is
 * precisely the behaviour the external store exists to avoid. E0.5's real
 * TensorGrid must do the same.
 */
const Cell = memo(function Cell({
  name,
  row,
  col,
  value,
  counter,
}: {
  name: string;
  row: number;
  col: number;
  value: number;
  counter: React.RefObject<Map<string, number>>;
}) {
  const spot = useMemo(() => target.tensor(name, [row, col]), [name, row, col]);
  const level = useHighlightLevel(spot);
  const { setHover, togglePin } = useHighlightActions();

  const id = `${name}:${row},${col}`;
  counter.current.set(id, (counter.current.get(id) ?? 0) + 1);

  const { text, kind } = formatValue(value, { sigDigits: 3 });
  const background =
    level === 'pinned'
      ? 'color-mix(in oklab, var(--color-view) 34%, var(--color-well))'
      : level === 'hover'
        ? 'color-mix(in oklab, var(--color-view) 20%, var(--color-well))'
        : 'var(--color-well)';

  return (
    <button
      type="button"
      onPointerEnter={() => setHover(spot)}
      onPointerLeave={() => setHover(null)}
      onClick={() => togglePin(spot)}
      className="rounded border border-line px-1 py-0.5 text-right font-mono tabular text-[10px]"
      style={{
        background,
        color: kind === 'nan' ? 'var(--color-nan)' : 'var(--color-ink)',
        outline: level === 'pinned' ? '1px solid var(--color-view)' : undefined,
      }}
    >
      {text}
    </button>
  );
});

function Grid({
  name,
  values,
  rows,
  cols,
  counter,
}: {
  name: string;
  values: Float64Array;
  rows: number;
  cols: number;
  counter: React.RefObject<Map<string, number>>;
}) {
  const whole = useMemo(() => target.tensor(name), [name]);
  const level = useHighlightLevel(whole);
  const { setHover } = useHighlightActions();

  return (
    <div>
      <button
        type="button"
        onPointerEnter={() => setHover(whole)}
        onPointerLeave={() => setHover(null)}
        className="mb-1 font-mono text-xs"
        style={{ color: level === 'none' ? 'var(--color-ink-dim)' : 'var(--color-view)' }}
      >
        {name} ({rows}, {cols})
      </button>
      <div
        className="grid gap-0.5"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {Array.from({ length: rows * cols }, (_, i) => (
          <Cell
            key={i}
            name={name}
            row={Math.floor(i / cols)}
            col={i % cols}
            value={values[i]}
            counter={counter}
          />
        ))}
      </div>
    </div>
  );
}

export default function DevHighlight() {
  const { t } = useLocale();
  const [seed, setSeed] = useState(1);
  const rows = 4;
  const cols = 6;

  const makeInitial = useCallback(() => softmaxInputs(rows, cols, seed), [seed]);
  const { state, trace } = useTracedProgram(SOFTMAX_PROGRAM, makeInitial);
  const { clearPins } = useHighlightActions();

  const renders = useRef(new Map<string, number>());
  const [snapshot, setSnapshot] = useState<{ total: number; touched: number } | null>(null);

  // Exposed so render isolation can be measured from the console without
  // clicking anything -- a button click re-renders the page and would pollute
  // exactly the number being measured.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    (window as unknown as { devRenderCounts?: Map<string, number> }).devRenderCounts =
      renders.current;
  }, []);

  const sampleRenders = () => {
    const counts = [...renders.current.values()];
    setSnapshot({ total: counts.length, touched: counts.filter((c) => c > 1).length });
  };
  const resetRenders = () => {
    renders.current.clear();
    setSnapshot(null);
  };

  const frames = trace ? framesFor(trace) : [];

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Highlight bus — E0.4</h1>
        <p className="mt-1 text-sm text-ink-dim">
          {t({
            zh: '公式项、矩阵单元、代码行三者互相点亮。悬停任意一处，另两处同步响应；点击钉住。',
            en: 'Formula terms, matrix cells and code lines all light each other. Hover any one, the other two respond; click to pin.',
          })}
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-line bg-panel p-4">
        <label className="flex items-center gap-3">
          <span className="font-mono text-xs text-ink-dim">seed</span>
          <input
            type="range"
            min={1}
            max={40}
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-40"
          />
          <span className="w-8 font-mono tabular text-xs text-ink">{seed}</span>
        </label>
        <button
          type="button"
          onClick={clearPins}
          className="rounded border border-line px-3 py-1 font-mono text-xs text-ink-dim hover:text-ink"
        >
          clear pins
        </button>
        <button
          type="button"
          onClick={resetRenders}
          className="rounded border border-line px-3 py-1 font-mono text-xs text-ink-dim hover:text-ink"
        >
          reset render counts
        </button>
        <button
          type="button"
          onClick={sampleRenders}
          className="rounded border border-line px-3 py-1 font-mono text-xs text-ink-dim hover:text-ink"
        >
          sample
        </button>
        {snapshot ? (
          <span className="font-mono text-xs" style={{ color: 'var(--color-ok)' }}>
            {snapshot.touched} / {snapshot.total} cells re-rendered
          </span>
        ) : null}
      </div>

      <section className="space-y-5 rounded-lg border border-line bg-panel p-5">
        <Formula spec={FORWARD} ctx={state ?? undefined} className="overflow-x-auto" />
        <Formula spec={VJP} ctx={state ?? undefined} className="overflow-x-auto" />
      </section>

      <div className="grid grid-cols-[1fr_1fr] gap-5">
        <section className="space-y-4 rounded-lg border border-line bg-panel p-5">
          <h2 className="text-xs font-semibold tracking-wide text-ink-dim uppercase">tensors</h2>
          {state
            ? (['z', 'e', 'upstream', 'dx'] as const).map((name) => (
                <Grid
                  key={name}
                  name={name}
                  values={copyFlat(state[name])}
                  rows={rows}
                  cols={cols}
                  counter={renders}
                />
              ))
            : null}
        </section>

        <section className="rounded-lg border border-line bg-panel p-5">
          <h2 className="mb-3 text-xs font-semibold tracking-wide text-ink-dim uppercase">
            source
          </h2>
          <pre className="overflow-x-auto rounded border border-line bg-well p-3 font-mono text-xs leading-relaxed">
            {frames.map((frame) => (
              <CodeLine key={frame.step} line={frame.lineStart} code={frame.code} />
            ))}
          </pre>
        </section>
      </div>

      <section>
        <h2 className="mb-3 text-xs font-semibold tracking-wide text-ink-dim uppercase">
          derivation — hover a term to light where it came from
        </h2>
        <DerivationSteps steps={DERIVATION} ctx={state ?? undefined} />
      </section>
    </div>
  );
}

function CodeLine({ line, code }: { line: number; code: string }) {
  const spot = useMemo(() => target.codeLine(line), [line]);
  const level = useHighlightLevel(spot);
  const { setHover, togglePin } = useHighlightActions();

  return (
    <div
      onPointerEnter={() => setHover(spot)}
      onPointerLeave={() => setHover(null)}
      onClick={() => togglePin(spot)}
      className="-mx-1 cursor-pointer rounded px-1"
      style={{
        background:
          level === 'pinned'
            ? 'color-mix(in oklab, var(--color-view) 30%, transparent)'
            : level === 'hover'
              ? 'color-mix(in oklab, var(--color-view) 18%, transparent)'
              : undefined,
        color: level === 'none' ? 'var(--color-ink-dim)' : 'var(--color-ink)',
      }}
    >
      <span className="mr-3 inline-block w-4 text-right text-ink-faint select-none">{line}</span>
      {code}
    </div>
  );
}
