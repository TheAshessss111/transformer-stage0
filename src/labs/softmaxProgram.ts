import { NdArray, zeros } from '../core/tensor/ndarray';
import { div, exp, max, mul, sub, sum } from '../core/tensor/ops';
import { randn } from '../core/tensor/random';
import { defineProgram } from '../core/trace/program';

/**
 * Softmax forward and backward, written as a TracedProgram.
 *
 * Stands in for what Step 0.3's labs will run. Deliberately spelled out one
 * statement at a time rather than calling ops.softmax: the point of the code
 * pane is that each line is a thing you can point at.
 */

export interface SoftmaxState {
  /** (B, T) logits. */
  x: NdArray;
  rowMax: NdArray;
  z: NdArray;
  e: NdArray;
  denom: NdArray;
  probs: NdArray;
  /** (B, T) upstream gradient. */
  upstream: NdArray;
  weightedMean: NdArray;
  dx: NdArray;
}

export const SOFTMAX_PROGRAM = defineProgram<SoftmaxState>({
  id: 'softmax-forward-backward',
  language: 'python',
  steps: [
    {
      code: 'row_max = x.max(axis=-1, keepdims=True)',
      note: {
        zh: '每一行取最大值，形状留成 (B, 1)。keepdims 是为了下一步能广播回去。',
        en: 'Row maxima, kept as an axis of extent 1 so the next step can broadcast back.',
      },
      run: (s) => ({ rowMax: max(s.x, -1, true) }),
    },
    {
      code: 'z = x - row_max',
      note: {
        zh: '平移。数学上不改变结果，但它是 exp 不溢出的唯一原因。',
        en: 'The shift. It changes nothing mathematically, and is the only reason exp does not overflow.',
      },
      run: (s) => ({ z: sub(s.x, s.rowMax) }),
    },
    {
      code: 'e = np.exp(z)',
      note: {
        zh: '现在每个 z ≤ 0，所以 e ∈ (0, 1]。',
        en: 'Every z is now at most 0, so e lands in (0, 1].',
      },
      run: (s) => ({ e: exp(s.z) }),
    },
    {
      code: 'denom = e.sum(axis=-1, keepdims=True)',
      note: { zh: '每行的归一化分母。', en: 'The per-row normalizer.' },
      run: (s) => ({ denom: sum(s.e, -1, true) }),
    },
    {
      code: 'probs = e / denom',
      note: { zh: '每行加起来是 1。', en: 'Each row now sums to 1.' },
      run: (s) => ({ probs: div(s.e, s.denom) }),
    },
    {
      code: 'weighted_mean = (upstream * probs).sum(axis=-1, keepdims=True)',
      phase: 'backward',
      note: {
        zh: '上游梯度按概率加权的平均。softmax 反向的全部内容就是减掉它。',
        en: 'The probability-weighted mean of the upstream gradient. Subtracting it is the whole backward pass.',
      },
      run: (s) => ({ weightedMean: sum(mul(s.upstream, s.probs), -1, true) }),
    },
    {
      code: 'dx = probs * (upstream - weighted_mean)',
      phase: 'backward',
      note: {
        zh: '注意这里只用到 probs，没有用到 x —— 所以实现时缓存的是输出。',
        en: 'Note this uses probs and never x, which is why an implementation caches the output.',
      },
      run: (s) => ({ dx: mul(s.probs, sub(s.upstream, s.weightedMean)) }),
    },
  ],
});

export function softmaxInputs(rows: number, cols: number, seed: number): SoftmaxState {
  const empty = zeros([]);
  return {
    x: randn([rows, cols], seed),
    upstream: randn([rows, cols], seed + 977),
    rowMax: empty,
    z: empty,
    e: empty,
    denom: empty,
    probs: empty,
    weightedMean: empty,
    dx: empty,
  };
}
