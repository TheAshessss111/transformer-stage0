import { useState } from 'react';
import { HighlightContext } from './context';
import { createHighlightStore } from './store';

/**
 * The store is created once and never replaced, so the context value is
 * referentially stable for the lifetime of the provider. That is the point:
 * nothing re-renders because highlight state changed, only because a specific
 * subscription's level changed.
 */
export default function HighlightProvider({ children }: { children: React.ReactNode }) {
  const [store] = useState(createHighlightStore);
  return <HighlightContext value={store}>{children}</HighlightContext>;
}
