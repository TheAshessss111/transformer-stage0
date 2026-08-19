# 文档索引

按职能分四组。想找什么，先看这里。

## 从哪开始

| 你是 | 读这个顺序 |
|---|---|
| 第一次看这个项目 | 根 [README](../README.md) → [需求规格](product/requirements.md) → [路线图](planning/roadmap.md) |
| 准备动手写代码 | [架构总览](architecture/overview.md) → [引擎参考](architecture/engine.md) → [分支策略](contributing/branching.md) |
| 想知道"为什么是这样" | [决策记录](product/decisions.md) —— 24 条拍板，每条都带理由和代价 |

---

## product/ · 要做什么，为什么

| 文档 | 内容 |
|---|---|
| [requirements.md](product/requirements.md) | 问题陈述、使用场景、7 组功能需求、8 条非功能指标、9 项明确不做 |
| [decisions.md](product/decisions.md) | **24 条决策记录**。项目的宪法：任何实现分歧先查这里；要改，改这里并注明日期与理由 |

## architecture/ · 怎么搭的

| 文档 | 内容 |
|---|---|
| [overview.md](architecture/overview.md) | 技术栈、8 层分层、四个核心设计决策（真实 stride 语义 / trace 驱动可视化 / 联动总线 / 双语内联）、完整目录结构 |
| [engine.md](architecture/engine.md) | `core/tensor` 的 API 参考与**三条契约**。E0.5/E0.7 照着它写代码；将来整包搬进 `handmade-transformer` 的也是它 |

## planning/ · 什么时候做

| 文档 | 内容 |
|---|---|
| [roadmap.md](planning/roadmap.md) | M0–M5 里程碑 → 24 个 Epic → 约 100 项带完成定义的功能，含 7 条风险 |
| [implementation/](planning/implementation/) | 逐 Epic 的可执行实施计划，含任务分解、依赖图、签收记录与实施中的偏差 |

现有实施计划：

- [m0-e0.1-e0.2.md](planning/implementation/m0-e0.1-e0.2.md) — E0.1 脚手架 + E0.2 张量引擎（**已完成**）
- [m0-e0.3-e0.4.md](planning/implementation/m0-e0.3-e0.4.md) — E0.3 执行追踪器 + E0.4 联动总线与公式引擎（**待动工**，含 7 条待确认偏差）

## contributing/ · 怎么协作

| 文档 | 内容 |
|---|---|
| [branching.md](contributing/branching.md) | 分支模型、命名、main 保护规则、CI gate、**为什么用 rebase 而不是 squash** |
| [commits.md](contributing/commits.md) | Conventional Commits 规范与 scope 列表，由 commit-msg 钩子强制 |
| [definition-of-done.md](contributing/definition-of-done.md) | 三层 DoD：机械可检 / 人工可检 / 项目红线 |

---

## 源计划

这一切来自 `transformer_plan.md` 的 **Part 1 · 阶段 0**（Step 0.1–0.5）。那份文档是地图，这个仓库是把地图上阶段 0 那一段变成可以动手玩的东西。
