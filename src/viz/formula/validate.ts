import type { EquationSpec } from './types.ts';

/**
 * Term ids referenced by a LaTeX source, in order of appearance.
 *
 * Lives here rather than in render.ts so this module stays free of any katex
 * import, which is what lets the Node verification harness reach it.
 */
export function termsInLatex(latex: string): string[] {
  const found: string[] = [];
  const pattern = /\\term\s*\{([^}]*)\}/g;
  let match = pattern.exec(latex);
  while (match !== null) {
    found.push(match[1]);
    match = pattern.exec(latex);
  }
  return found;
}

/**
 * Catch the failure mode that looks like success.
 *
 * A term marked up in the LaTeX but missing from `terms` renders a tooltip that
 * silently does nothing: the page looks finished and one term is quietly dead.
 * A `terms` entry never referenced in the LaTeX is the same bug from the other
 * end — usually a rename that only got applied on one side.
 *
 * Pure and DOM-free, so the verification harness checks it too.
 */
export interface ValidationProblem {
  kind: 'missing-metadata' | 'unused-metadata' | 'duplicate-markup';
  term: string;
  message: string;
}

export function validateEquationSpec(spec: EquationSpec<never>): ValidationProblem[] {
  const inLatex = termsInLatex(spec.latex);
  const declared = Object.keys(spec.terms);
  const problems: ValidationProblem[] = [];

  const seen = new Set<string>();
  for (const term of inLatex) {
    if (seen.has(term)) {
      problems.push({
        kind: 'duplicate-markup',
        term,
        message: `equation '${spec.id}': \\term{${term}} appears more than once; term ids must be unique within an equation`,
      });
    }
    seen.add(term);
  }

  for (const term of seen) {
    if (!declared.includes(term)) {
      problems.push({
        kind: 'missing-metadata',
        term,
        message: `equation '${spec.id}': \\term{${term}} is marked up but has no entry in terms, so its tooltip would be silently dead`,
      });
    }
  }

  for (const term of declared) {
    if (!seen.has(term)) {
      problems.push({
        kind: 'unused-metadata',
        term,
        message: `equation '${spec.id}': terms.${term} is declared but never marked up in the LaTeX`,
      });
    }
  }

  return problems;
}

/** Dev-only guard. Throws so the problem cannot be scrolled past. */
export function assertValidEquationSpec(spec: EquationSpec<never>): void {
  const problems = validateEquationSpec(spec);
  if (problems.length === 0) return;
  throw new Error(
    `Invalid EquationSpec '${spec.id}':\n` + problems.map((p) => `  - ${p.message}`).join('\n'),
  );
}
