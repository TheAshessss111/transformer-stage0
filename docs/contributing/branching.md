# 分支策略与合并流程

> 决策依据：[D-23](../product/decisions.md)（分支保护 + CI gates）、[D-24](../product/decisions.md)（rebase-merge）。

## 模型：主干开发 + 短生命周期特性分支

`main` 永远是可部署、CI 全绿的状态。没有 `develop`，没有 release 分支 —— 这个项目是持续交付的静态站点，git-flow 的那套只会增加开销。

```
main ──●──●──────────────●──●──────────●──▶
        \                /    \        /
         ●──●──●────────╯      ●──●───╯      ← feat/e03-trace-recorder
         逐任务 commit，rebase 合并后原样进入 main
```

**分支寿命目标：不超过一个 Epic。** 一个分支对应 `docs/planning/roadmap.md` 里的一个 Epic（或一组紧密相关的任务）。分支越长，rebase 越痛。

## 分支命名

```
<type>/<scope>-<slug>
```

`<type>` 与提交类型一致（见 [commits.md](commits.md)）：

| 前缀 | 用途 | 例 |
|---|---|---|
| `feat/` | 新功能 | `feat/e03-trace-recorder` |
| `fix/` | 修 bug | `fix/unbroadcast-rank0` |
| `docs/` | 只动文档 | `docs/regroup-and-github-workflow` |
| `refactor/` | 不改行为的重构 | `refactor/split-ops-module` |
| `chore/` | 工具链、依赖、配置 | `chore/bump-vite` |
| `ci/` | 只动 CI | `ci/cache-node-modules` |

Epic 分支用 Epic 编号去掉点号：E0.3 → `e03`，E1.1 → `e11`。

## 工作流程

```bash
git switch main && git pull --ff-only
git switch -c feat/e03-trace-recorder
# ... 逐任务提交，每个 commit 都要能单独编译并通过 verify ...
git push -u origin feat/e03-trace-recorder
gh pr create --fill
```

合并前本地自检（和 CI 跑的是同一套）：

```bash
npm run format:check && npm run lint && npx tsc -b && npm run verify && npm run build
```

## main 的保护规则

在 GitHub 上对 `main` 生效，`enforce_admins` 打开 —— 仓库拥有者也绕不过去：

| 规则 | 值 | 为什么 |
|---|---|---|
| 必须走 PR | 是 | 每次变更都有一个可回顾的载体 |
| 需要的 approval 数 | **0** | 单人项目里 GitHub 不允许你 approve 自己的 PR；要求 1 个等于永远无法合并。把关交给 CI |
| 必须通过的检查 | `verify` | 见下 |
| 分支必须是最新的 | 是 | 合并前必须先 rebase 到最新 main，避免"各自通过、合起来爆炸" |
| 线性历史 | 是 | 禁止 merge commit |
| 对话必须解决 | 是 | review 意见不能被静默忽略 |
| force push | 禁止 | |
| 删除分支 | 禁止 | |

## CI gate：`verify`

`.github/workflows/ci.yml` 里的单个 job，五个步骤，全绿才能合并：

| 步骤 | 命令 | 拦住什么 |
|---|---|---|
| Format | `npm run format:check` | 格式漂移 |
| Lint | `npm run lint` | 代码问题，**以及 `core/` 引入 UI 依赖**（分层铁律） |
| Typecheck | `npx tsc -b` | 类型错误 |
| **Engine** | `npm run verify` | **引擎回归 —— 102 项检查，含每个 VJP 的 gradcheck** |
| Build | `npm run build` | 生产构建失败 |

第四项是这套 gate 存在的理由。引擎算错了，上面所有可视化都在骗人，而这件事在页面上看不出来 —— 它只会安静地教你错的数学。这也是 `docs/planning/roadmap.md` 风险 R-04 的正式解法。

## 合并方式：rebase，不是 squash

**默认用 Rebase and merge。** 完整理由见 [D-24](../product/decisions.md)，一句话概括：

> `transformer_plan.md` 说这份 commit history 本身就是能力证明。squash 会把它删掉。

所以每个 commit 都必须：

- 单独说得过去（message 写清"做了什么 + 怎么验收的"）
- 单独可编译、单独能过 `verify`

做不到就在推送前用 `git rebase -i` 整理干净。**代价是真实的，但这正是我们要的纪律。**

Squash merge 保留开关，仅用于琐碎的修补型 PR（改个错字、调个依赖版本）。Merge commit 已关闭。

## 保持分支最新

因为要求线性历史和"分支必须最新"，用 rebase 而不是 merge：

```bash
git switch feat/e03-trace-recorder
git fetch origin
git rebase origin/main
git push --force-with-lease      # 注意是 --force-with-lease，不是 --force
```

`--force-with-lease` 会在远端有你没见过的提交时拒绝推送。永远用它。
