# Stage Zero

> 把《手搓 Transformer》实施方案 **阶段 0：数学与工具预备** 变成一个可以动手玩的桌面端网页。
>
> 看到操作就知道 shape 怎么变；看到公式就知道每一项是什么形状、现在的值是多少、它在干什么。

**状态**：规划完成，尚未开始实现。

## 覆盖的五个 Step

| Step | 主题 | 招牌可视化 |
|---|---|---|
| 0.1 | 张量形状代数 | 内存布局 / stride 3D 视图 |
| 0.2 | 矩阵微积分五条规则 | 计算图 DAG + 反向传播动画 |
| 0.3 | softmax 的反向 | 显式雅可比矩阵 vs VJP 对照 |
| 0.4 | LayerNorm 的反向 | 梯度投影到零均值 ∩ 正交子空间的几何图 |
| 0.5 | 数值稳定性 | 危险操作沙盒（七条清单，现场制造 NaN） |

## 规划文档

- [`docs/DECISIONS.md`](docs/DECISIONS.md) —— 22 项需求拍板，本项目的宪法
- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) —— 需求规格、功能/非功能需求、明确不做的事
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) —— 技术栈、分层、四个核心设计决策、目录结构
- [`docs/ROADMAP.md`](docs/ROADMAP.md) —— M0–M5 里程碑 → Epic → 功能，含风险清单

## 技术栈一览

React 19 · TypeScript · Vite · Tailwind v4 · KaTeX · react-three-fiber · CodeMirror 6 · Pyodide（浏览器内真 NumPy）

页面上每一个数字都由自研的 TypeScript float64 张量引擎当场算出，
并由 Pyodide 里的真 NumPy **在线对拍**验证 —— 这既是测试，也是「纪律三：对拍」的教学演示。

## 源计划

`~/Downloads/transformer_plan.md` · Part 1 · 阶段 0
