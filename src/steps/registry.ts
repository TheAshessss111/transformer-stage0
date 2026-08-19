import type { Bilingual } from '../shell/locale';

export type StepId = '0-1' | '0-2' | '0-3' | '0-4' | '0-5';

export type StepStatus = 'planned' | 'in-progress' | 'done';

export interface StepMeta {
  id: StepId;
  /** Display number, matching transformer_plan.md ("0.1"). */
  number: string;
  title: Bilingual;
  tagline: Bilingual;
  /** The headline visualization this step is built around (ROADMAP.md D-12). */
  signatureViz: Bilingual;
  status: StepStatus;
}

export const STEPS: readonly StepMeta[] = [
  {
    id: '0-1',
    number: '0.1',
    title: { zh: '张量形状代数', en: 'Tensor Shape Algebra' },
    tagline: {
      zh: '看到操作就知道 shape 怎么变',
      en: 'See an operation, know the shape it produces',
    },
    signatureViz: {
      zh: '内存布局 / stride 3D 视图',
      en: 'Memory layout / stride 3D view',
    },
    status: 'planned',
  },
  {
    id: '0-2',
    number: '0.2',
    title: { zh: '矩阵微积分五条规则', en: 'Five Rules of Matrix Calculus' },
    tagline: {
      zh: '反向传播只做一件事：把上游梯度转成各输入的梯度',
      en: 'Backprop does one thing: turn the upstream gradient into input gradients',
    },
    signatureViz: {
      zh: '计算图 DAG + 反向传播动画',
      en: 'Computation graph DAG with backward animation',
    },
    status: 'planned',
  },
  {
    id: '0-3',
    number: '0.3',
    title: { zh: 'softmax 的反向', en: 'The Softmax Backward Pass' },
    tagline: {
      zh: '反向只需要前向的输出 s，不需要输入 x',
      en: 'The backward pass needs only the output s, never the input x',
    },
    signatureViz: {
      zh: '显式雅可比矩阵 vs VJP 对照',
      en: 'Explicit Jacobian vs VJP, side by side',
    },
    status: 'planned',
  },
  {
    id: '0-4',
    number: '0.4',
    title: { zh: 'LayerNorm 的反向', en: 'The LayerNorm Backward Pass' },
    tagline: {
      zh: '梯度被投影到零均值、与 x̂ 正交的子空间',
      en: 'The gradient is projected onto the zero-mean subspace orthogonal to x̂',
    },
    signatureViz: {
      zh: '梯度投影几何图',
      en: 'Gradient projection geometry',
    },
    status: 'planned',
  },
  {
    id: '0-5',
    number: '0.5',
    title: { zh: '数值稳定性', en: 'Numerical Stability' },
    tagline: {
      zh: '建立「哪里会 NaN」的直觉',
      en: 'Build the instinct for where NaN comes from',
    },
    signatureViz: {
      zh: '危险操作沙盒（七条清单）',
      en: 'Hazard sandbox (the seven-row checklist)',
    },
    status: 'planned',
  },
];

export function findStep(id: string): StepMeta | undefined {
  return STEPS.find((s) => s.id === id);
}
