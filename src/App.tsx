import { BrowserRouter, Navigate, Route, Routes } from 'react-router';
import AppShell from './shell/AppShell';
import LocaleProvider from './shell/LocaleProvider';
import HighlightProvider from './viz/highlight/HighlightProvider';
import SmallScreenGate from './shell/SmallScreenGate';
import Landing from './pages/Landing';
import DevTokens from './pages/DevTokens';
import DevFormula from './pages/DevFormula';
import DevTrace from './pages/DevTrace';
import DevHighlight from './pages/DevHighlight';
import StepPage from './steps/StepPage';

export default function App() {
  return (
    <LocaleProvider>
      <HighlightProvider>
        <SmallScreenGate>
          <BrowserRouter basename={import.meta.env.BASE_URL}>
            <Routes>
              <Route element={<AppShell />}>
                <Route index element={<Landing />} />
                <Route path="step/:id" element={<StepPage />} />
                <Route path="dev/tokens" element={<DevTokens />} />
                <Route path="dev/formula" element={<DevFormula />} />
                <Route path="dev/trace" element={<DevTrace />} />
                <Route path="dev/highlight" element={<DevHighlight />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </SmallScreenGate>
      </HighlightProvider>
    </LocaleProvider>
  );
}
