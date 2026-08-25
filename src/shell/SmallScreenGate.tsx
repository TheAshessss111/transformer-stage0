import { useMediaQuery } from './useMediaQuery';
import { useLocale } from '../content/i18n';
import LocaleToggle from './LocaleToggle';

/** Desktop-only, per docs/product/decisions.md D-14. */
export const MIN_WIDTH_PX = 1280;

/**
 * Below MIN_WIDTH_PX the children are *unmounted*, not hidden with CSS. That
 * matters: the labs in E0.5 and the R3F canvas in E1.1 must never mount on a
 * small screen, and a `display:none` wrapper would still run all of them.
 */
export default function SmallScreenGate({ children }: { children: React.ReactNode }) {
  const wideEnough = useMediaQuery(`(min-width: ${MIN_WIDTH_PX}px)`);
  const { t } = useLocale();

  if (wideEnough) return children;

  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="max-w-md rounded-lg border border-line bg-panel p-8 text-center">
        <div className="font-mono text-sm font-semibold text-ink">Stage Zero</div>
        <p className="mt-4 leading-relaxed text-ink-dim">
          {t({
            zh: '这个页面需要至少 1280px 宽的窗口。多栏矩阵视图和代码沙盒在小屏上没法读，所以没有做移动端布局。请用桌面浏览器打开。',
            en: 'This site needs a window at least 1280px wide. Multi-column matrix views and the code sandbox are unreadable on a small screen, so there is no mobile layout. Please open it on a desktop browser.',
          })}
        </p>
        <div className="mt-6 flex justify-center">
          <LocaleToggle />
        </div>
      </div>
    </div>
  );
}
