# 提交信息规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)。由 `.husky/commit-msg` 钩子机械化检查（零依赖，一段正则）。

## 格式

```
<type>(<scope>): <subject>

<body —— 做了什么，为什么，怎么验收的>
```

首行 ≤ 72 字符，用祈使语气（"add"，不是"added"/"adds"）。

## type

| type | 用途 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修 bug |
| `docs` | 只动文档 |
| `refactor` | 不改外部行为的重构 |
| `perf` | 性能 |
| `chore` | 工具链、依赖、配置 |
| `ci` | CI 配置 |
| `build` | 构建系统 |
| `style` | 纯格式（极少用，Prettier 已自动处理） |
| `test` | 测试/校验脚本 |
| `revert` | 回滚 |

破坏性变更在 type 后加 `!`：`feat(core)!: ...`

## scope

对应源码分层（见 [架构总览](../architecture/overview.md)）：

`core` · `viz` · `shell` · `content` · `labs` · `steps` · `sandbox` · `py` · `theme` · `docs` · `ci` · `deps`

scope 可省略（例如纯文档提交 `docs: ...`）。

## body 要写什么

因为 [D-24](../product/decisions.md) 规定 rebase-merge，**每个 commit 都会原样进入 main 并被面试官看到**。body 至少回答两个问题：

1. **为什么这样做**（不是"做了什么"——diff 已经说了）
2. **怎么验证的**

好的例子（本仓库真实提交）：

```
feat(core): VJPs for all forward operators

Each VJP takes only what the maths needs: expVjp/sqrtVjp take the forward
OUTPUT, reluVjp takes the 1/0 mask. That is a real memory saving and it is the
exact point Step 0.3 makes about softmax, so the signatures encode it rather
than caching inputs just in case.

Every broadcasting VJP ends in unbroadcast (plan rule 5), whose unconditional
shape assertion is what catches the mistakes.
```

差的例子：

```
feat: add vjp        ← 没说为什么，没说怎么验的
fix: bug             ← 什么 bug
update files         ← 连 type 都没有，钩子会拒绝
```

## 引用计划文档

提到路线图条目时用编号，方便反查：`F0.2.5`、`E0.3`、`D-24`、`R-04`、`NFR-6`。
