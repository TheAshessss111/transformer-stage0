import { NavLink } from 'react-router';
import { STEPS, type StepStatus } from '../steps/registry';
import { useLocale } from './locale';

const STATUS_DOT: Record<StepStatus, string> = {
  planned: 'bg-line-strong',
  'in-progress': 'bg-warn',
  done: 'bg-ok',
};

export default function Sidebar() {
  const { t } = useLocale();

  return (
    <nav className="flex w-72 shrink-0 flex-col border-r border-line bg-panel">
      <NavLink to="/" className="block border-b border-line px-5 py-4">
        <div className="font-mono text-sm font-semibold tracking-wide text-ink">Stage Zero</div>
        <div className="mt-0.5 text-xs text-ink-faint">
          {t({ zh: '阶段 0 · 数学与工具预备', en: 'Stage 0 · Math & Tooling Prep' })}
        </div>
      </NavLink>

      <ul className="flex-1 overflow-y-auto py-2">
        {STEPS.map((step) => (
          <li key={step.id}>
            <NavLink
              to={`/step/${step.id}`}
              className={({ isActive }) =>
                `block border-l-2 px-5 py-3 transition-colors ${
                  isActive
                    ? 'border-l-view bg-well text-ink'
                    : 'border-l-transparent text-ink-dim hover:bg-well/50 hover:text-ink'
                }`
              }
            >
              <div className="flex items-center gap-2">
                <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${STATUS_DOT[step.status]}`} />
                <span className="font-mono tabular text-xs">{step.number}</span>
                <span className="truncate text-sm">{t(step.title)}</span>
              </div>
              <div className="mt-1 pl-5 text-xs leading-snug text-ink-faint">
                {t(step.signatureViz)}
              </div>
            </NavLink>
          </li>
        ))}
      </ul>

      <div className="border-t border-line px-5 py-3 font-mono text-xs text-ink-faint">
        <NavLink to="/dev/tokens" className="hover:text-ink-dim">
          /dev/tokens
        </NavLink>
      </div>
    </nav>
  );
}
