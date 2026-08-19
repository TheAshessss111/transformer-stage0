# 完成的定义（DoD）

一个功能"做完了"要同时满足三层。`docs/planning/roadmap.md` 里每个功能条目的 DoD 是**在这三层之上的**额外要求，不是替代。

## 第一层 · 机械可检（CI 会拦你）

- [ ] `npm run format:check` 通过
- [ ] `npm run lint` 零警告（`--deny-warnings`）
- [ ] `npx tsc -b` 通过
- [ ] `npm run verify:engine` 全绿
- [ ] `npm run build` 成功

## 第二层 · 人工可检（CI 拦不住，但必须做到）

- [ ] **改动了 `core/`？** 在 `src/core/tensor/__dev__/` 里加了对应检查。新的算子没有 gradcheck 就是没写完
- [ ] **改动了可见行为？** 在浏览器里亲眼确认过，不是"应该没问题"
- [ ] **加了新的公开 API？** `docs/architecture/engine.md` 同步更新
- [ ] **偏离了计划？** 记进对应的实施计划文档（像 [m0-e0.1-e0.2.md](../planning/implementation/m0-e0.1-e0.2.md) §4 那样），不要让文档和代码悄悄分叉
- [ ] **推翻了某条决策？** 在 [decisions.md](../product/decisions.md) 里标注修订，而不是直接改掉原文

## 第三层 · 项目特有（这个项目的红线）

- [ ] **`core/` 没有引入任何 UI 依赖**（oxlint 会拦，但心里要有这根弦）
- [ ] **每个 commit 单独可编译、单独能过 `verify:engine`** —— 因为 rebase-merge 会让它们原样进入 main（[D-24](../product/decisions.md)）
- [ ] **没有为了让检查变绿而放宽阈值。** gradcheck 容差、色阶对比度、误差阈值：数字变差就是有东西坏了，先查原因
- [ ] **可视化展示的每个数字都是真算出来的**（[D-05](../product/decisions.md)），不是硬编码、不是预录
- [ ] **文案是双语的**（[D-10](../product/decisions.md)）—— `L<T>` 类型会强制，但别用空字符串糊弄

## 一个反复出现的陷阱

**检查变红时，先怀疑检查写错了，再怀疑代码。**

E0.2 里有两次是检查本身写错了（softmax 减 max 的不变性在 float64 下并非位相等；`humanBytes` 的位数规则），修的是检查不是代码 —— 而且第一次还改进了原本要做的教学内容。

但**反过来的诱惑更危险**：不要因为"大概是精度问题吧"就把容差从 1e-7 调到 1e-5。先把误差的来源解释清楚，解释不通就是有 bug。
