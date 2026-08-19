# Stage Zero

[![CI](https://github.com/TheAshessss111/transformer-stage0/actions/workflows/ci.yml/badge.svg)](https://github.com/TheAshessss111/transformer-stage0/actions/workflows/ci.yml)

> 把《手搓 Transformer》实施方案 **阶段 0：数学与工具预备** 变成一个可以动手玩的桌面端网页。
>
> 看到操作就知道 shape 怎么变；看到公式就知道每一项是什么形状、现在的值是多少、它在干什么。

**状态**：M0 进行中。E0.1（脚手架与工程规范）、E0.2（张量引擎）已完成 —— 引擎有 102 项检查，每个手写反向都过 gradcheck。

## 覆盖的五个 Step

| Step | 主题 | 招牌可视化 |
|---|---|---|
| 0.1 | 张量形状代数 | 内存布局 / stride 3D 视图 |
| 0.2 | 矩阵微积分五条规则 | 计算图 DAG + 反向传播动画 |
| 0.3 | softmax 的反向 | 显式雅可比矩阵 vs VJP 对照 |
| 0.4 | LayerNorm 的反向 | 梯度投影到零均值 ∩ 正交子空间的几何图 |
| 0.5 | 数值稳定性 | 危险操作沙盒（七条清单，现场制造 NaN） |

## 快速开始

```bash
npm install
npm run dev              # 开发服务器
npm run verify           # 152 项检查（无测试框架，跑在 Node 原生 TS 上）
```

## 文档

完整索引在 **[docs/README.md](docs/README.md)**，按职能分四组：

| 分组 | 内容 |
|---|---|
| [product/](docs/product/) | [需求规格](docs/product/requirements.md) · [24 条决策记录](docs/product/decisions.md) |
| [architecture/](docs/architecture/) | [架构总览](docs/architecture/overview.md) · [引擎 API 参考](docs/architecture/engine.md) |
| [planning/](docs/planning/) | [路线图](docs/planning/roadmap.md) · [实施计划](docs/planning/implementation/) |
| [contributing/](docs/contributing/) | [分支策略](docs/contributing/branching.md) · [提交规范](docs/contributing/commits.md) · [DoD](docs/contributing/definition-of-done.md) |

## 贡献流程

`main` 受保护：禁止直推，一律走 PR，CI 五道 gate 全绿才能合并。
合并方式是 **rebase 而非 squash** —— 逐 Step 的 commit history 本身就是产出物（[D-24](docs/product/decisions.md)）。

细节见 [分支策略](docs/contributing/branching.md)。

## 技术栈一览

React 19 · TypeScript · Vite · Tailwind v4 · KaTeX · react-three-fiber · CodeMirror 6 · Pyodide（浏览器内真 NumPy）

页面上每一个数字都由自研的 TypeScript float64 张量引擎当场算出，
并由 Pyodide 里的真 NumPy **在线对拍**验证 —— 这既是测试，也是「纪律三：对拍」的教学演示。

## 源计划

`~/Downloads/transformer_plan.md` · Part 1 · 阶段 0
