# 决策记录（ADR）

> 来源：2026-08-19 与 Claude 的需求访谈，共 4 轮 16 个问题、22 项拍板。
> 本文件是**规划的宪法**。后续任何实现分歧，先回来查这里；要改，改这里并注明日期与理由。

## D-01 · 技术栈：React 19 + TypeScript + Vite

**理由**：五个 Step 各自是独立的「交互实验室」，组件模型最贴合；KaTeX / react-three-fiber / CodeMirror / Motion 的 React 生态最成熟；后续若把内容扩展到阶段 1–8，组件化结构扛得住。

## D-02 · 交付：本地 dev + 静态部署

`npm run dev` 日常开发，`npm run build` 产出纯静态站点，部署到 GitHub Pages。无后端、无数据库、无账号系统。

## D-03 · 范围：只做阶段 0 的五个 Step，但每个都做透

Step 0.1 形状代数 / 0.2 矩阵微积分 / 0.3 softmax 反向 / 0.4 LayerNorm 反向 / 0.5 数值稳定性。
架构上预留扩展点，但**第一版不写阶段 1+ 的任何内容**。

## D-04 · 受众：自己学习 + 面试作品集

推论：文案可以假定读者有线性代数与 Python 基础，不铺垫基础概念；但导航、落地页、视觉完成度要按公开作品的标准做。

## D-05 · 数值来源：浏览器内 TypeScript 引擎实时真算

页面上每一个数字都是当场算出来的，不是预录的。改任何输入都真实重算。
**推论**：这个 TS 张量引擎本身就是你阶段 1「微型 autograd」的预演，投入不浪费。

## D-06 · 交互方式：全部采用

滑块调参即时重算 / 单步播放（下一步·回放）/ 悬停·点击联动高亮 / 直接编辑矩阵元素。
原话补充：**"任何可以帮助理解的交互方式都采用"** —— 这些做成跨 Step 复用的原语，而不是每个页面单独造。

## D-07 · 公式：KaTeX 逐项可交互 **且** 逐步推导可展开

- 每一项可悬停／点击，弹出「这一项是什么形状、现在的值是多少、它在干什么」。
- 推导过程可一步步展开，每步配一句「为什么可以这样变」。

## D-08 · 代码沙盒：Pyodide 按需懒加载，写真 Python + NumPy

页面主体用 TS 引擎（秒开），只有点开代码沙盒时才加载 Pyodide。
**理由**：你最终要交付的是 NumPy 实现，沙盒里写的代码必须能直接拷回你的仓库。
**代价**：首次点开有加载等待；需要维护 TS / Python 两套引擎 —— 这个代价由 D-21 的对拍面板转化为教学资产。

## D-09 · 练习形态：实现挑战 + 自动 gradcheck

给一个空的 `softmax_backward`，你在沙盒里写，页面自动跑中心差分数值梯度对拍并报出相对误差。
**不做**：形状推断小测、面试闪卡、验收 checklist 勾选。

## D-10 · 语言：中英双语可切换

全站 i18n，一个开关切换。文案在数据结构里**内联双语**（见 A-04），从类型上杜绝中英漂移。

## D-11 · 视觉：暗色技术仪表盘

深色底 + 高饱和数据色 + 等宽字体 + 强对比热力图。单主题，不做亮色。

## D-12 · 四个招牌可视化：全要

1. 内存布局 / stride 3D 视图（Step 0.1）
2. 计算图 DAG + 反向传播动画（Step 0.2）
3. 雅可比矩阵 vs VJP 对照（Step 0.3）
4. LayerNorm 梯度投影几何图（Step 0.4）

## D-13 · 页面结构：统一五段模板

目标 → 形状契约 → 数学要点（交互实验室）→ 实现要点与坑 → 实现挑战。
与 `transformer_plan.md` 中每个 Step 的五段结构一一对应。

## D-14 · 设备：仅桌面宽屏（≥ 1280px）

小屏只给一个「请用桌面浏览器打开」的提示页。不为移动端重做布局。

## D-15 · 开发顺序：垂直切片优先

先用 **Step 0.3 softmax** 把全链路打通（引擎 → 可视化 → 公式联动 → Pyodide 沙盒 → gradcheck），拿到一个真正可用的页面后再横向复制到其余四个 Step。

## D-16 · Step 0.5 形态：危险操作沙盒（七条清单）

把计划里那张「危险操作」表的每一行做成可现场触发的开关，当场看到 Inf/NaN 出现在哪一步。
**不做**：完整的浮点位级可视化组件。
**补充约定**：计划的验收标准里有「说得出 fp16 与 bf16 的动态范围差异」，用一个轻量的动态范围对比条覆盖，不做可点击的位格子。

## D-17 · 文案：由 Claude 从 `transformer_plan.md` 提取并扩写

结构化搬运原文要点，并补写交互所需的引导语、提示、错误反馈文案。你审阅修改。

## D-18 · 持久化：只存沙盒代码

`localStorage` 按 challenge id 保存你写的实现，刷新不丢。
**不做**：阅读进度追踪、URL 状态编码分享。

## D-19 · 节奏：不赶时间，质量优先

允许 M0 有较长的基础设施阶段。宁可多花几轮把引擎、组件库、视觉系统做扎实。

## D-20 · 位置：`MathBase/transformer-stage0/`，独立 git 仓库

与你未来的 `handmade-transformer` 代码仓库解耦，可单独部署为作品集。

## D-21 · 引擎验证：Pyodide 在线对拍，做成可见功能

不做离线 golden fixtures，不做 CI 单测。取而代之：页面上有一个**对拍面板**，同一输入下当场跑真 NumPy 与 TS 引擎并显示最大相对误差。
**这既是测试，也是你计划里「纪律三：对拍」的教学演示。**

## D-22 · 工程规范：lint + Prettier + 提交钩子

> **2026-08-19 修订**：本条中「不做 GitHub Actions CI」的部分已被 **D-23 推翻**。其余不变。

**不做**：Vitest 单元测试、TS 额外严格选项（`strict` 走模板默认值，不额外收紧）。
**实现时的偏差**：Vite 8 模板已改用 **oxlint** 而非 ESLint。实测 oxlint 支持 `no-restricted-imports` 与 per-file `overrides`，`core/` 分层铁律照常机械化执行，因此照单接受。

~~**已知风险**：无自动化回归防线（R-04）。~~ → 已由 D-23 的 CI gate 消除。

---

## D-23 · GitHub 工作流：public 仓库 + 分支保护 + PR + CI gates

**推翻 D-22 中「不上 CI」的部分。** 理由：required status checks 是分支保护唯一有意义的 gate；没有 CI，「保护」只剩下「必须开 PR」这一条形式。

- **仓库 public**。免费账号下分支保护、rulesets、Actions 全部可用（private 需 GitHub Pro）；且 D-04 本来就把它定为面试作品集。
- **main 受保护**：禁止直推、禁止 force push、禁止删除、要求线性历史、要求对话解决完毕、`enforce_admins` 打开（你自己也绕不过）。
- **不要求 approval**（`required_approving_review_count: 0`）。单人项目里 GitHub 不允许你 approve 自己的 PR，要求 1 个 approval 等于永远无法合并。PR 仍然强制，只是靠 CI 而非人来把关。
- **CI gate 集**：`format:check` → `lint` → `typecheck` → `verify:engine` → `build`。全部通过才能合并。
- **`verify:engine` 是这套 gate 的核心**：引擎算错了，上面所有可视化都在骗人（见 `docs/product/requirements.md` 验收第 5 条）。把它放进 CI，`docs/planning/roadmap.md` 风险清单 R-04 才算真正被堵上。

## D-24 · 合并策略：rebase-merge，不是 squash

**这一条和通行做法相反，理由是项目特有的。**

业界默认 squash-merge：一个 PR 塌缩成一个 commit，主分支干净。但 `transformer_plan.md` 的 Git 纪律写得很明确：

> 每完成一个 Step 提交一次，commit message 写清楚"实现了什么 + 验收怎么过的"。阶段 8 时这份 commit history 本身就是你的能力证明。

**squash 会把这份证明删掉。** E0.2 那 8 个逐任务 commit 一旦被压成一个 `feat: tensor engine`，"我是怎么一步步搭起来并逐步验证的"这个叙事就没了 —— 而那正是要给面试官看的东西。

所以：
- ✅ **Rebase merge**（默认）—— 保留每一个 commit，同时保证线性历史
- ✅ Squash merge（保留开关，仅用于琐碎的修补型 PR）
- ❌ Merge commit（关闭，与 `required_linear_history` 冲突）

代价：每个 commit 都必须单独说得过去、且单独可编译，因为它们会原样进入 main。这正是我们想要的纪律。

## D-25 · 公式逐项标记：KaTeX `\htmlData` + `\term` 宏

**T27 spike 的结论**（证据页：`/dev/formula`）。D-07 要求公式每一项可悬停/点击，这依赖于能给任意子表达式挂一个稳定的钩子。实测 KaTeX 0.18.4：

作者写 `\term{id}{内容}`，通过宏展开成 `\htmlData{term=id}{内容}`，渲染出的元素带 `data-term="id"`。

| 场景 | 结果 |
|---|---|
| 平铺 | ✅ |
| `\frac` 的分子与分母内部 | ✅ 两项都拿到 |
| `\sum` 上下标范围内 | ✅ |
| 嵌套两层（term 套 term） | ✅ 三层全部拿到 |
| 两个公式用同名 term | ✅ 不冲突 |

**为什么用属性而不是 `\htmlId`**：`\htmlId` 同样可用，但 id 是全文档唯一的，两个公式复用同一个 term 名就会撞。属性没有这个问题，也就不需要给每个公式加前缀。

**两个必须记住的配置**：

- `trust: true` —— 不开 `\htmlData` 直接被丢弃。
- `strict: (code) => code === 'htmlExtension' ? 'ignore' : 'warn'` —— `trust` 会让 KaTeX 在**每一次渲染**都打印一条 `htmlExtension` 警告。只静音这一条，其余 strict 警告全部保留（公式写错了要能看见）。

**风险 R-02 就此关闭**：备选方案（`\htmlClass` / `\htmlId` / 拆成多个 span）都不需要了。
