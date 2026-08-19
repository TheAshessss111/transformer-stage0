import { BrowserRouter, Navigate, Route, Routes } from 'react-router';
import AppShell from './shell/AppShell';
import LocaleProvider from './shell/LocaleProvider';
import SmallScreenGate from './shell/SmallScreenGate';
import Landing from './pages/Landing';
import DevTokens from './pages/DevTokens';
import StepPage from './steps/StepPage';

export default function App() {
  return (
    <LocaleProvider>
      <SmallScreenGate>
        <BrowserRouter basename={import.meta.env.BASE_URL}>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<Landing />} />
              <Route path="step/:id" element={<StepPage />} />
              <Route path="dev/tokens" element={<DevTokens />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </SmallScreenGate>
    </LocaleProvider>
  );
}
