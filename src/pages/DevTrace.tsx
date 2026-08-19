import { useCallback, useMemo, useState } from 'react';
import { SOFTMAX_PROGRAM, softmaxInputs } from '../labs/softmaxProgram';
import { RUN_BUDGET_MS, useTracedProgram } from '../viz/useTracedProgram';
import { framesFor } from '../core/trace/replay';
import { formatShape } from '../core/tensor/format';
import { size } from '../core/tensor/ndarray';
import { useLocale } from '../shell/locale';

/**
 * T26 — the E0.3 proof, and where NFR-2 stops being a hope and becomes a number.
 *
 * Everything here reads from one recorded Trace: the source pane, the frame
 * list, the event table and the names all render the same object.
 */

type PhaseFilter = 'all' | 'forward' | 'backward';

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-line bg-panel">
      <h2 className="border-b border-line px-4 py-2 text-xs font-semibold tracking-wide text-ink-dim uppercase">
        {title}
      </h2>
      <div className="p-4">{children}</div>
    </section>
  );
}

export default function DevTrace() {
  const { t, locale } = useLocale();
  const [rows, setRows] = useState(4);
  const [cols, setCols] = useState(8);
  const [seed, setSeed] = useState(1);
  const [phase, setPhase] = useState<PhaseFilter>('all');
  const [active, setActive] = useState(0);

  const makeInitial = useCallback(() => softmaxInputs(rows, cols, seed), [rows, cols, seed]);
  const { trace, timing } = useTracedProgram(SOFTMAX_PROGRAM, makeInitial);

  const frames = useMemo(
    () => (trace ? framesFor(trace, phase === 'all' ? undefined : phase) : []),
    [trace, phase],
  );
  const current = frames[Math.min(active, Math.max(0, frames.length - 1))];
  const sourceLines = trace ? trace.source.split('\n') : [];
  const overBudget = timing.p95 > RUN_BUDGET_MS;

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-8">
      <header>
        <h1 className="text-2xl font-semibold text-ink">Trace recorder — E0.3</h1>
        <p className="mt-1 text-sm text-ink-dim">
          {t({
            zh: '一个 TracedProgram 跑一次，下面所有面板读的是同一份 Trace。',
            en: 'One TracedProgram, run once. Every panel below reads the same Trace object.',
          })}
        </p>
      </header>

      <Panel title="inputs">
        <div className="flex flex-wrap items-center gap-6">
          {(
            [
              ['B (rows)', rows, setRows, 1, 64],
              ['T (cols)', cols, setCols, 2, 64],
              ['seed', seed, setSeed, 1, 50],
            ] as const
          ).map(([label, value, setter, min, max]) => (
            <label key={label} className="flex items-center gap-3">
              <span className="w-20 font-mono text-xs text-ink-dim">{label}</span>
              <input
                type="range"
                min={min}
                max={max}
                value={value}
                onChange={(e) => setter(Number(e.target.value))}
                className="w-48"
              />
              <span className="w-10 font-mono tabular text-xs text-ink">{value}</span>
            </label>
          ))}
          <span className="font-mono text-xs text-ink-faint">{rows * cols} elements</span>
        </div>
      </Panel>

      <div className="grid grid-cols-[1fr_1fr] gap-5">
        <Panel title="source · current step highlighted">
          <pre className="overflow-x-auto rounded border border-line bg-well p-3 font-mono text-xs leading-relaxed">
            {sourceLines.map((line, i) => {
              const lineNo = i + 1;
              const lit = current && lineNo >= current.lineStart && lineNo <= current.lineEnd;
              return (
                <div
                  key={lineNo}
                  className={lit ? '-mx-1 rounded bg-line-strong px-1 text-ink' : 'text-ink-dim'}
                >
                  <span className="mr-3 inline-block w-4 text-right text-ink-faint select-none">
                    {lineNo}
                  </span>
                  {line}
                </div>
              );
            })}
          </pre>
          {current?.note ? (
            <p className="mt-3 text-sm leading-relaxed text-ink-dim">{current.note[locale]}</p>
          ) : null}
        </Panel>

        <Panel title="frames">
          <div className="mb-3 flex gap-1">
            {(['all', 'forward', 'backward'] as const).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => {
                  setPhase(p);
                  setActive(0);
                }}
                className={`rounded px-2 py-1 font-mono text-xs ${
                  phase === p ? 'bg-line-strong text-ink' : 'text-ink-dim hover:text-ink'
                }`}
              >
                {p}
              </button>
            ))}
          </div>
          <ol className="space-y-1">
            {frames.map((frame) => (
              <li key={frame.step}>
                <button
                  type="button"
                  onClick={() => setActive(frame.index)}
                  className={`w-full rounded px-2 py-1.5 text-left font-mono text-xs ${
                    current?.step === frame.step
                      ? 'bg-well text-ink'
                      : 'text-ink-dim hover:bg-well/60'
                  }`}
                >
                  <span className="mr-2 text-ink-faint">
                    {frame.index}·L{frame.lineStart}
                  </span>
                  <span
                    className="mr-2"
                    style={{
                      color: frame.phase === 'backward' ? 'var(--color-copy)' : 'var(--color-view)',
                    }}
                  >
                    {frame.phase === 'backward' ? 'bwd' : 'fwd'}
                  </span>
                  {frame.code}
                  <span className="ml-2 text-ink-faint">
                    {frame.events.length} {frame.events.length === 1 ? 'op' : 'ops'}
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </Panel>
      </div>

      <Panel title="events · every primitive operation, with its producer edges">
        <div className="overflow-x-auto">
          <table className="w-full font-mono text-xs">
            <thead className="text-ink-faint">
              <tr className="border-b border-line text-left">
                <th className="py-1 pr-4">#</th>
                <th className="py-1 pr-4">step</th>
                <th className="py-1 pr-4">op</th>
                <th className="py-1 pr-4">inputs</th>
                <th className="py-1 pr-4">output</th>
                <th className="py-1 pr-4">memory</th>
              </tr>
            </thead>
            <tbody>
              {(trace?.events ?? []).map((event) => {
                const inFrame = current?.events.some((e) => e.index === event.index);
                return (
                  <tr
                    key={event.index}
                    className={`border-b border-line/50 ${inFrame ? 'bg-well text-ink' : 'text-ink-dim'}`}
                  >
                    <td className="py-1 pr-4">{event.index}</td>
                    <td className="py-1 pr-4">{event.step}</td>
                    <td className="py-1 pr-4">{event.op}</td>
                    <td className="py-1 pr-4">
                      {event.inputs
                        .map(
                          (i) =>
                            `${i.name ?? (i.event === null ? '?' : `#${i.event}`)}${formatShape(i.shape)}`,
                        )
                        .join('  ')}
                    </td>
                    <td className="py-1 pr-4">
                      {event.output.name ? `${event.output.name} ` : ''}
                      {formatShape(event.output.shape)}
                    </td>
                    <td className="py-1 pr-4">
                      {event.passthrough ? (
                        <span className="text-ink-faint">no-op</span>
                      ) : event.didCopy ? (
                        <span
                          className="rounded border px-1"
                          style={{ borderColor: 'var(--color-copy)', color: 'var(--color-copy)' }}
                        >
                          copy · {event.copiedElements}
                        </span>
                      ) : event.isView ? (
                        <span
                          className="rounded border px-1"
                          style={{ borderColor: 'var(--color-view)', color: 'var(--color-view)' }}
                        >
                          view
                        </span>
                      ) : (
                        <span className="text-ink-faint">new</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid grid-cols-[2fr_1fr] gap-5">
        <Panel title="names · state keys are tensor names">
          <div className="grid grid-cols-3 gap-x-6 gap-y-1 font-mono text-xs">
            {[...(trace?.names.entries() ?? [])].map(([name, array]) => (
              <div key={name} className="flex justify-between gap-2">
                <span className="text-ink">{name}</span>
                <span className="text-ink-faint">{formatShape(array.shape)}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="timing · NFR-2">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-xs">
            <dt className="text-ink-faint">last</dt>
            <dd className="tabular text-ink">{timing.last.toFixed(3)} ms</dd>
            <dt className="text-ink-faint">p50</dt>
            <dd className="tabular text-ink">{timing.p50.toFixed(3)} ms</dd>
            <dt className="text-ink-faint">p95</dt>
            <dd
              className="tabular"
              style={{ color: overBudget ? 'var(--color-err)' : 'var(--color-ok)' }}
            >
              {timing.p95.toFixed(3)} ms
            </dd>
            <dt className="text-ink-faint">budget</dt>
            <dd className="tabular text-ink-dim">{RUN_BUDGET_MS} ms</dd>
            <dt className="text-ink-faint">runs</dt>
            <dd className="tabular text-ink-dim">{timing.runs}</dd>
            <dt className="text-ink-faint">elements</dt>
            <dd className="tabular text-ink-dim">{trace ? size(trace.names.get('x')!) : 0}</dd>
          </dl>
          <p className="mt-3 text-xs leading-relaxed text-ink-faint">
            {t({
              zh: '拖动滑块：runs 每帧最多加一，因为 rAF 把同一帧内的多次输入变化合并成一次。',
              en: 'Drag a slider: runs increments at most once per frame, because rAF coalesces every input change within a frame into a single run.',
            })}
          </p>
        </Panel>
      </div>
    </div>
  );
}
