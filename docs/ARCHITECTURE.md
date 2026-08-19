# 架构设计：Stage Zero

---

## 1. 技术栈

| 层 | 选型 | 版本 | 为什么是它 |
|---|---|---|---|
| 构建 | **Vite** | 7.x | 秒级 HMR；Web Worker 与 WASM 一等公民支持（Pyodide 需要） |
| 框架 | **React** | 19 | 组件模型贴合「每 Step 若干独立实验室」；生态最全 |
| 语言 | **TypeScript** | 5.x | 张量形状、双语文案、公式项元数据都靠类型兜底 |
| 样式 | **Tailwind CSS** | v4 | CSS-first `@theme` 令牌，暗色仪表盘配色集中管理 |
| 公式 | **KaTeX** | 0.16+ | 同步渲染（无布局抖动）；`\htmlId` + `trust` 支持逐项挂载 |
| 动画 | **motion** | 12.x | 单步播放的过渡编排；`useMotionValue` 驱动滑块联动 |
| 3D | **@react-three/fiber** + **drei** | — | Step 0.1 stride 立方体、Step 0.4 梯度投影几何 |
| 图布局 | **dagre** | — | Step 0.2 计算图 DAG 的分层布局 |
| 代码编辑 | **CodeMirror 6** | — | 比 Monaco 轻一个量级；Python 语法与主题定制简单 |
| Python | **Pyodide** | 0.28+ | 浏览器内真 NumPy；跑在 Web Worker 里 |
| 状态 | **Zustand** | 5.x | 每个 Lab 一个小 store；避免 Context 重渲染风暴 |
| 规范 | ESLint + Prettier + husky + lint-staged | — | D-22 |

> **不引入**：Redux、Next.js、图表库（D3 之类）。可视化全部手写 SVG / Canvas / R3F —— 因为要画的是张量而不是折线图，通用图表库帮不上忙反而添乱。

---

## 2. 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│  shell/        AppShell · Sidebar · LocaleToggle · 小屏拦截  │
├──────────────────────────────────────────────────────────────┤
│  steps/        StepLayout（统一五段模板）· registry          │
├──────────────────────────────────────────────────────────────┤
│  content/      StepContent 数据（双语内联）· BlockRenderer   │
├──────────────────────────────────────────────────────────────┤
│  labs/         各 Step 专属交互实验室（组合 viz 原语）        │
├──────────────────────────────────────────────────────────────┤
│  viz/          TensorGrid · Cube3D · DAG · Formula · Playhead │
│                └── highlight/  联动总线（贯穿全层）           │
├──────────────────────────────────────────────────────────────┤
│  sandbox/      CodeMirror · ChallengeRunner · 持久化          │
├──────────────────────────────────────────────────────────────┤
│  core/         tensor · trace · gradcheck · numerics          │← 纯计算，零 UI 依赖
├──────────────────────────────────────────────────────────────┤
│  py/           Pyodide Worker · RPC · 序列化 · harness.py     │← 另一条计算通路
└──────────────────────────────────────────────────────────────┘
```

**铁律**：`core/` 与 `py/` 不 import 任何 React。可视化层只消费它们的输出。
这样 `core/tensor` 将来可以整包搬去你的 `handmade-transformer` 仓库当参考实现。

---

## 3. 四个核心设计决策

### A-01 · 引擎必须有真实的 view / copy 语义

普通教学可视化会把张量当成一个嵌套数组，`transpose` 就地换个下标顺序完事。
**本项目不能这么做**，因为 Step 0.1 的整个教学点就是 stride 与内存布局。

```ts
interface NdArray {
  data: Float64Array;   // 底层缓冲，可被多个 NdArray 共享
  shape: number[];
  strides: number[];    // 以元素为单位
  offset: number;
  base: NdArray | null; // 非 null 表示这是一个 view
}
```

- `transpose(0,2,1,3)` → 返回新 NdArray，`data` 是**同一个引用**，只有 `strides` 变了 → UI 标记「零拷贝 view」
- `reshape(...)` → 若当前非连续，必须先 `ascontiguousarray()` → UI 标记「⚠ 发生了一次内存复制，N 个元素」
- `isContiguous()` 是一等公民 API，Step 0.1 会直接把它的返回值显示在界面上

**收益**：可视化的正确性不是靠美工画对，而是引擎本来就对。`MemoryStrip` 直接读 `strides` 画跳跃箭头。

### A-02 · Trace 驱动可视化（本项目的中枢神经）

**不写动画脚本**。所有单步播放、代码行高亮、DAG 边流动，都是同一份执行记录的不同渲染。

```ts
type TraceEvent = {
  id: string;
  phase: 'forward' | 'backward';
  op: string;                    // 'matmul' | 'softmax' | 'transpose' ...
  inputs: TensorRef[];
  output: TensorRef;
  shapeIn: number[][];
  shapeOut: number[];
  isView: boolean;               // 输出是否与输入共享内存
  didCopy: boolean;              // 本步是否发生了内存复制
  codeLine?: number;             // 绑定到 CodePane 的行号
  note?: L<string>;              // 「这一步在干什么」的双语说明
};
```

数据流：

```
滑块/矩阵编辑改变输入
      ↓
recorder.start() → 引擎重新执行 → recorder.stop()
      ↓
TraceEvent[]  ──┬──→ Playhead（帧序列，可单步/回放）
                ├──→ ShapePipeline（每步一个形状胶囊）
                ├──→ CodePane（当前帧 → 高亮 codeLine）
                ├──→ GraphDAG（事件即节点，TensorRef 即边）
                └──→ TensorGrid（当前帧的输入/输出张量）
```

**收益**：加一个新 Lab 时，绝大部分工作是「用引擎写一遍前向」，可视化几乎白送。

### A-03 · 联动总线：统一寻址协议

「悬停公式项 → 矩阵行发亮 → 代码行发亮 → DAG 边发亮」如果手工连线，是 N² 的复杂度。改成广播总线：

```ts
type HighlightTarget =
  | { kind: 'tensor';  name: string; index?: (number | '*')[] }  // 支持通配：['*', 2] = 第 2 列整列
  | { kind: 'formula'; eq: string; term: string }
  | { kind: 'code';    line: number }
  | { kind: 'dag';     node: string }
  | { kind: 'axis';    tensor: string; axis: number };
```

- 两级状态：`hover`（临时）与 `pinned`（点击钉住，可多选做对比）
- 内容作者在**数据里声明**绑定关系，例如某个公式项 `term: 'sum-term'` 声明它对应 `{ kind:'tensor', name:'s', index:['*'] }`
- 任何组件只需 `useHighlight()` 订阅，自己决定怎么表现「被点亮」

### A-04 · 双语内联，靠类型防漂移

不用 i18next 的 key-value 文件 —— 两份文件必然漂移。改成：

```ts
type L<T = string> = { zh: T; en: T };

// 内容里长这样：
{ kind: 'prose', text: {
    zh: 'softmax 的反向只需要前向的输出 s，不需要输入 x。',
    en: 'The softmax backward pass needs only the forward output s, not the input x.',
}}
```

写一段就必须同时写两语，缺一个字段 TS 直接报错。`useL()` 按当前 locale 取值。

**代价**：内容文件更长。**收益**：不可能漏翻译、不可能语义漂移。

---

## 4. 内容系统：为什么不用 MDX

MDX 的诱惑是「文案里直接写 `<Lab />`」。但本项目有两个硬约束让它失效：

1. **公式项需要结构化元数据**（id / 形状 / 取值器 / 双语说明）—— 在 MDX 里写这个会变成一堆难读的 props
2. **双语** —— MDX 意味着 `step-0-3.zh.mdx` 与 `step-0-3.en.mdx` 两份文件，必然漂移

所以内容是**类型化的 TS 数据**，由 `BlockRenderer` 统一渲染：

```ts
type Block =
  | { kind: 'prose';      text: L }
  | { kind: 'formula';    eq: EquationSpec }              // 见下
  | { kind: 'derivation'; steps: DerivationStep[] }        // 逐步展开
  | { kind: 'callout';    tone: 'insight'|'warn'|'interview'; title: L; text: L }
  | { kind: 'pitfall';    text: L; symptom: L; fix: L }
  | { kind: 'lab';        component: string; props?: unknown }   // 实验室插槽
  | { kind: 'challenge';  id: string };

type EquationSpec = {
  id: string;
  latex: string;                  // 内含 \htmlId{term-x}{...}
  terms: Record<string, {
    label: L;                     // 「这一项是什么」
    purpose: L;                   // 「它在干什么」
    shape?: string;               // '(B,T,1)'
    read?: (ctx: LabState) => NdArray | number;   // 「现在的值是多少」
    highlight?: HighlightTarget[]; // 悬停时点亮谁
  }>;
};
```

---

## 5. Pyodide 通路

```
主线程                          Web Worker
────────                        ──────────
bridge.ts  ──postMessage──→     worker.ts
   │                              │ loadPyodide()（懒，首次调用时）
   │                              │ micropip 装 numpy
   │                              │ 注入 harness.py
   │  ←──结果 + traceback──        │ exec 用户代码
   ▼                              ▼
ChallengeRunner                 numpy 计算
```

- **懒加载时机**：用户滚动到「实现挑战」段落或点开对拍面板。加载中显示明确进度条（NFR-3）。
- **序列化**：`Float64Array` + `shape` 直传（`postMessage` 可转移 ArrayBuffer），Python 侧 `np.frombuffer(...).reshape(shape)`。
- **gradcheck 跑在 Python 侧**：因为你最终要交付的就是 NumPy 代码，误差数字必须是真 NumPy 算出来的。页面只提供 harness 与结果渲染。
- **对拍面板**（D-21）：同一 `Float64Array` 输入，TS 引擎与 NumPy 各算一遍，回传比较。这是本项目唯一的正确性防线，因此**每次进入一个 Step 时自动静默跑一次自检**，失败则在顶栏显示红色告警。

---

## 6. 目录结构

```
transformer-stage0/
├── README.md
├── package.json  vite.config.ts  tsconfig.json  eslint.config.js  .prettierrc
├── .husky/pre-commit
├── index.html
├── docs/
│   ├── DECISIONS.md          # 22 项拍板（宪法）
│   ├── REQUIREMENTS.md       # 需求规格
│   ├── ARCHITECTURE.md       # 本文件
│   ├── ROADMAP.md            # 里程碑 → epic → 功能
│   └── ADDING_A_STEP.md      # M5 产出：如何新增一个 Step
└── src/
    ├── main.tsx  App.tsx
    │
    ├── core/                         # ── 纯计算层，零 React 依赖 ──
    │   ├── tensor/
    │   │   ├── ndarray.ts            #   NdArray：data/shape/strides/offset/base
    │   │   ├── shape.ts              #   reshape/transpose/permute/ascontiguousarray
    │   │   ├── broadcast.ts          #   广播规则 + unbroadcast（反向求和）
    │   │   ├── ops.ts                #   前向算子
    │   │   ├── vjp.ts                #   每个算子的反向
    │   │   ├── autograd.ts           #   tape + 拓扑排序 + backward()
    │   │   ├── jacobian.ts           #   显式雅可比（n ≤ 16，教学专用）
    │   │   ├── random.ts             #   mulberry32 固定种子 PRNG
    │   │   └── format.ts             #   数值格式化（NaN/Inf/subnormal 标记）
    │   ├── trace/
    │   │   ├── types.ts  recorder.ts  replay.ts
    │   ├── gradcheck/
    │   │   ├── numericalGrad.ts      #   中心差分 float64
    │   │   └── relError.ts
    │   └── numerics/
    │       ├── float.ts              #   fp32/fp16/bf16 范围与舍入模拟
    │       └── hazards.ts            #   七条危险操作的开关与检测
    │
    ├── py/                           # ── Pyodide 通路 ──
    │   ├── worker.ts  bridge.ts  serialize.ts
    │   └── snippets/ harness.py  parity.py
    │
    ├── viz/                          # ── 与 Step 无关的可视化原语 ──
    │   ├── highlight/ HighlightContext.tsx  types.ts
    │   ├── TensorGrid.tsx  SliceSelector.tsx
    │   ├── TensorCube3D.tsx  MemoryStrip.tsx
    │   ├── ShapePipeline.tsx  GraphDAG.tsx
    │   ├── Formula.tsx  DerivationSteps.tsx
    │   ├── Vector3DScene.tsx  DistributionStrip.tsx
    │   ├── Playhead.tsx  SliderRow.tsx  CodePane.tsx  DiffBadge.tsx
    │   └── scales.ts                 #   发散色阶 / 顺序色阶
    │
    ├── sandbox/
    │   ├── SandboxEditor.tsx  ChallengeRunner.tsx
    │   ├── challenges.ts             #   骨架代码 + gradcheck 配置
    │   └── persistence.ts            #   localStorage
    │
    ├── content/
    │   ├── types.ts  i18n.ts  BlockRenderer.tsx
    │   └── step-0-1.ts … step-0-5.ts
    │
    ├── labs/
    │   ├── shape/      StrideCubeLab  MhaShapeFlowLab  ReshapeTrapLab  KeepdimsLab
    │   ├── calculus/   DagBackwardLab  ShapeMatchLab  BroadcastDualityLab
    │   ├── softmax/    SoftmaxForwardLab  MaxShiftLab  JacobianVsVjpLab  FusedCeLab
    │   ├── layernorm/  LnForwardLab  FourStepDerivationLab  ProjectionLab  EpsilonLab
    │   └── numerics/   HazardSandboxLab  NanPropagationLab  PrecisionRangeLab
    │
    ├── steps/  StepLayout.tsx  registry.ts
    ├── shell/  AppShell.tsx  Sidebar.tsx  TopBar.tsx  LocaleToggle.tsx  SmallScreenGate.tsx
    └── styles/ theme.css  katex-overrides.css
```

---

## 7. 符号约定

全站沿用 `transformer_plan.md` 纪律一的符号，代码、UI、文案三处一致：

`B` batch · `T` 目标序列长 · `S` 源序列长 · `D` d_model · `H` 头数 · `Dh` = D/H · `Df` d_ff · `V` 词表 · `L` 层数

`TensorGrid` 的轴标签直接渲染这些字母，滑块也用这些名字。**不允许出现 `n`、`m`、`dim0` 这类临时命名。**

---

## 8. 视觉系统（暗色技术仪表盘）

Tailwind v4 `@theme` 中定义的令牌：

| 令牌组 | 用途 |
|---|---|
| `--color-bg-*` | 三级深色背景（页面 / 面板 / 凹槽） |
| `--color-grad-neg` → `--color-grad-pos` | **发散色阶**：梯度正负。零值处必须是背景色而非灰色 |
| `--color-mag-*` | **顺序色阶**：非负量（概率、幅值） |
| `--color-view` / `--color-copy` | view = 青色（零拷贝）/ copy = 琥珀色（发生了内存复制），全站统一 |
| `--color-nan` / `--color-inf` | NaN 品红、Inf 橙红，任何数值渲染处一致 |
| `--font-mono` | 所有数值、shape、代码 |

**硬约束**：数值文字在任何色块上都必须可读 —— 单元格背景饱和度高时自动切换文字色。
