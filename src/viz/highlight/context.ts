import { createContext, use } from 'react';
import type { HighlightStore } from './store';

export const HighlightContext = createContext<HighlightStore | null>(null);

export function useHighlightStore(): HighlightStore {
  const store = use(HighlightContext);
  if (!store) throw new Error('highlight hooks must be used inside <HighlightProvider>');
  return store;
}
