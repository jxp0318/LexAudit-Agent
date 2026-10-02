# LexAudit Agent — 项目蓝图（Phase 0 产出）

> 本文是删库重开后的第一份产出：**只做分析与设计，不含任何业务代码**。
> 旧 MVP（已完成并验证）已归档于 `archive/pre-rewrite-2026-10-02/`，其 `docs/ARCHITECTURE.md`
> 中的技术决策记录（D1~D7）与踩坑清单是重建过程的"答案库"，决策依据直接复用，不凭空另起炉灶。
>
> 图例约定：🧱 = 代码预先决定（Workflow 侧）　🤖 = LLM 运行时决定（Agent 侧）

---

## 1. 项目目标

**产品目标**：LexAudit —— 合同智能审查 Agent。输入一份合同 + 审查要求，输出带原文定位、
风险等级、修订建议的结构化审查报告。

**求职目标**：第二个作品集项目。与项目一（电商问数，固定 workflow）构成对照——
项目一回答"流程确定时如何做到可控"，本项目回答"流程不确定时如何让 Agent 自主规划并执行"。

**学习目标**：每个技术决策都能回答三问——**解决什么问题 / 为什么现在需要它 / 不用它会怎样**。
最终能向面试官完整讲清 Agent Loop、State、Tool Calling、Planning、Re-planning 的设计与取舍。

## 2. 用户使用场景

- **场景 A（主）**：用户上传合同（md/docx）+ 自然语言审查要求（"帮我看这份采购合同有什么风险"）
  → Agent 自主生成审查计划 → 逐项/并行执行检查 → 执行中发现新证据时调整计划 → 输出审查报告。
- **场景 B**：指定关注点审查（"只看违约责任和争议解决条款"）→ 计划应相应收窄。
- **边界**：辅助审查工具，输出非法律意见（免责声明写入报告）；不处理真实在办案件。

## 3. 核心功能

| 功能 | 说明 | 引入阶段 |
|---|---|---|
| 文档解析 | 合同 → 条款树（编号/标题/层级/原文） | Phase 1 |
| 风险识别 | 缺失条款 / 有毒条款 / 交叉引用错误 / 金额日期不一致 | Phase 2~5 |
| 审查要点检索（RAG） | 给 LLM 提供可评测的审查依据 | Phase 4 |
| 自主规划 + 重规划 | 任务 DAG 运行时生成、按发现增量调整 | Phase 5~6 |
| 反思与失败恢复 | 结果质量校验、工具错误自愈 | Phase 7 |
| 报告生成 | 结构化 md/json + 审查覆盖率 | Phase 5+ |
| HTTP API | FastAPI 薄适配层 | Phase 9 |
| 三版本评测 | 固定 workflow / Plan-Execute / 裸 LLM 对比 | Phase 10 |

## 4. 整体架构

```
api/                FastAPI 薄适配层（只做 HTTP 进出翻译，不含业务逻辑）
 └─ src/lexaudit    核心包（框架无关的纯逻辑，可脱离 HTTP 被 pytest/CLI/eval 直接驱动）
     ├─ graph.py     静态图组装（少量节点构成的状态机，逐阶段生长）
     ├─ nodes/       planner / executor / observer / reflector / reporter
     ├─ tools/       工具实现 + 注册表（白名单）
     ├─ rag/         审查要点库 + 检索
     ├─ runtime.py   LLM 注入点（MockLLM ↔ 真实模型一键切换）
     └─ state.py     State schema + reducer
```

三条架构原则：
1. **依赖方向单向**：api → core，core 永不 import api。为什么：core 必须能被测试、CLI、评测直接调用。
2. **图拓扑静态**：动态性来自"State 里的 plan + 条件边读 State 路由"（详见 §5、§8）。🧱
3. **LLM 一律经 runtime 注入**：测试期用 MockLLM。为什么：不花钱、可离线、可精确断言
   （如断言"不同合同类型产出不同 DAG"，守住"非伪动态"）。

## 5. Agent Loop

先纠正一个直觉：**LangGraph 里没有 while 循环**。Agent Loop = "节点函数 + 条件边"构成的
状态机循环——执行到某节点后，条件边函数决定下一步去哪，直到走到 END。

```
analyze → plan → schedule ⇄ execute → observe(=replan_check)
                        ↑                │ ├─ 还有就绪任务 ────→ execute
                        │                │ ├─ 有新证据且未超限 → plan（重规划）
                        └────────────────┘ └─ 全部完成 ────────→ reflect → synthesize → END
```

- **Workflow 侧 🧱**：节点集合、边集合、循环退出条件、最大步数、reducer 合并方式——编译期写死。
- **Agent 侧 🤖**：plan 的内容（生成什么任务）、每个任务的分析结论、重规划时追加什么任务。
- **面试标准表述**："图是静态的，执行序列由运行时 State 中的 plan 驱动——这是状态驱动的
  调度循环，不是运行时动态生成的图。"这句话本身就是对图执行模型理解的证明。

## 6. State 设计

State 是全图唯一的共享黑板，一切动态性都存在这里。**写入纪律比字段本身更重要**（并行正确性的根）：

| 字段 | 类型 | 谁写 | 写入纪律 |
|---|---|---|---|
| document_text / clause_tree | str / ClauseTree | ingest 🧱 | 只写一次 |
| contract_profile | Profile | analyze 🤖 | 顺序节点独占写 |
| plan（任务 DAG） | Plan | plan/replan 🤖 | **只能由顺序节点写**，并行 executor 绝不碰 |
| findings | list（reducer: add） | execute 🤖 | 并行 executor 只写 reducer 字段 |
| completed_task_ids | list（reducer: add） | execute 🧱 | 同上 |
| evidence_queue | list（reducer: add） | execute 🤖 | 新证据（缺附件/阴阳条款）经此传给 planner |
| events | list（reducer: add） | 所有节点 | 审计轨迹：覆盖率统计与调试都靠它 |
| replan_count / step_count | int | observer 🧱 | 死循环防护计数器 |

**为什么 findings 必须带 reducer**：多个并行 executor 同时返回写 state，普通字段会触发
写冲突（InvalidUpdateError）；reducer（`operator.add`）把并发更新定义为"追加合并"。
**为什么 plan 不能用 reducer**：计划需要"整表替换"语义而非追加，且 DAG 依赖关系必须保持
一致视图——所以它只能由单个顺序节点独占写。

## 7. Tool 设计

工具契约：name + description + args_schema（Pydantic）→ 供 LLM tool calling；实现为
**纯函数**（不调 LLM、无副作用）；**错误以返回值（观察结果）回传而非抛异常**——让 LLM
有机会看到错误、改参数重试（自我纠正的前提）。

| 工具 | 职责 | 引入 |
|---|---|---|
| parse_document | 文本 → 条款树（结构化信息用解析器确定性获取，绝不让 LLM 从纯文本里抽） | Phase 1 |
| locate_clause / read_clause | 条号 → 定位与切片（context 管理的基础） | Phase 2 |
| clause_cross_ref | 交叉引用一致性核对（确定性正则，不是 LLM） | Phase 2 |
| risk_kb_search | 审查要点库检索（RAG） | Phase 4 |
| amount_consistency_check | 金额/日期一致性粗筛（召回型工具，报告中须明示其误报边界） | Phase 5 |

**白名单 🧱**：planner 生成的任务只能引用注册表里的工具名——这是"防乱规划"的第一道闸。

## 8. Planning 设计（含 Re-planning）

- **产出物**：planner 🤖 用 structured output 生成 `Plan{tasks: [{id, 描述, 工具, 依赖, 涉及条款, 验收标准}]}`，DAG 存 State（带环检测）。
- **调度**：schedule 🧱 挑出"依赖全部完成"的任务 → `Send` fan-out 并行执行（图级并行，不是节点内 asyncio）。

Re-planning 四问（规则 13 逐项回答）：

1. **什么情况重新规划**：① 任务失败且换路径可能成功；② evidence_queue 出现新证据
   （规划时看不到的 unknown unknowns：附件缺失、阴阳条款）；③ reflection 发现覆盖缺口；
   ④ 预算未耗尽且要点库还有未覆盖的核心主题。
2. **谁触发**：replan_check 是条件边 🧱——**代码**读 State 的信号字段（失败标记/新证据/反思
   结论）决定"要不要重新规划"；**LLM** 🤖 只决定"追加什么任务"。
   ⚠️ 触发器是确定性的，内容是概率性的——这是 Workflow 与 Agent 的协作，
   **不要吹成"LLM 自主决定重规划"**。
3. **State 中保存什么**：plan（含每个任务 status）、findings、evidence_queue、replan_count、step_count。
4. **如何避免无限循环 / 最大执行步数**：四重护栏——replan_count ≤ 3；任务总数 ≤ 15；
   **diff 式增量**（只追加新任务，不重排已完成的，任务幂等）；连续两次 replan 无新任务 → 强制收尾。
   另设 step_count 硬上限（schedule→execute→check 循环 ≤ 30 步），到顶强制进 synthesize
   并在报告中标注"审查未完成"。

## 9. RAG 接入方式

**知识库选择**（继承旧版决策 D7）：自建小型结构化"审查要点库"
（每条：合同类型 + 条款主题 → 审查要点 + 风险等级 + 关联法条），**不做通用法条语料检索**。
为什么：可控、可标注 ground truth、可测召回；通用语料噪音大且无法评测。

检索两级，接口抽象为 `Retriever`：
1. **结构化过滤 🧱**：按合同类型/主题精确匹配（Phase 4 先只做这层——确定性、可测试）；
2. **语义检索**：embedding + 余弦 top-k（Phase 4 进阶选项，换实现不动接口）。

**不用 Qdrant 的理由**：几百条要点的规模，内存中检索足够。不用会怎样？——现在不会怎样，
这正是不用的原因。面试要能反向讲清"什么时候才需要向量数据库"
（百万级向量、多租户隔离、复杂元数据过滤组合需求）。

## 10. Reflection 设计

- **位置**：所有任务完成后、报告生成前，插入 reflect 节点 🤖。
- **检查三件事**：① plan 完成度（有无 failed/skipped 任务）；② 证据率（findings 是否都有
  条款定位与依据）；③ 覆盖缺口（要点库核心主题是否都被某任务覆盖）。
- **输出** verdict（pass / needs_replan + 缺口信号）→ 条件边 🧱 决定进 synthesize 还是回 plan
  （占用 replan 预算）。
- **失败恢复（另一条线，贯穿执行期）**：工具错误 → 以观察结果返回 executor 🤖 → LLM 改参数
  重试一次 → 仍失败标记 task failed，交 replan 判断"换路径还是放弃"。
- **区分两个概念**：Reflection = 评估结果质量（该不该再做点别的）；Re-planning = 实际修改计划。
  反思的输出是重规划的输入之一，但两者不是一回事。

## 11. Memory 设计

三层，各解决不同问题，**不为"看起来高级"而加层**：

| 层 | 载体 | 解决什么 | 不用会怎样 |
|---|---|---|---|
| 短期（工作记忆） | State + 任务切片 | 单次审查内的信息流转 | executor 上下文塞全文 → 膨胀、贵、注意力稀释 |
| 中期（轨迹持久化） | LangGraph Checkpointer（SqliteSaver） | 断点续跑（thread_id 恢复）、HITL 暂停恢复、全轨迹回放（法律可审计刚需） | 长审查中断即全部重跑；无法人工介入 |
| 长期（跨会话） | 审查历史 sqlite 表 | 历史风险模式 | **MVP 明确不做**，Phase 8 再评估是否纳入 |

context 管理核心手段：executor 每个任务只带"相关条款切片 + 检索到的要点"，不带全文；
全量信息存 State，由报告层汇总。

## 12. 项目目录结构（目标态）

```
LexAudit Agent/
├── docs/BLUEPRINT.md          # 本文件（Phase 0 产出）
├── src/lexaudit/              # 核心包（Phase 1 起逐阶段生长）
│   ├── state.py               # Phase 1：State + reducer
│   ├── graph.py               # Phase 1 起：图组装（逐阶段加节点）
│   ├── nodes/                 # Phase 5 起：planner/executor/observer/reflector/reporter
│   ├── tools/                 # Phase 1 起：parse_document → registry → 其余工具
│   ├── rag/                   # Phase 4：要点库 + 检索
│   ├── runtime.py             # Phase 1：LLM 注入点（MockLLM 起步）
│   └── report.py              # Phase 5+：报告渲染
├── api/                       # Phase 9：FastAPI 薄层
├── tests/                     # 每阶段配套 pytest
├── eval/                      # Phase 10：三版本对比
├── samples/                   # 样例合同（md 起步）
├── scripts/demo.py            # 每阶段可运行入口
└── archive/                   # 旧版归档（只读，进 .gitignore）
```

原则：**目录跟着阶段长，不预先建空壳**。

## 13~15. 开发阶段划分 + 每阶段学习目标 + 面试知识

沿用 Phase 0~10 划分，仅两处调整，理由如下：
① 不设"文档解析"独立阶段——解析作为 Phase 1/2 的工具顺势引入（没有文档就没有审查对象，
独立成阶段会推迟你看到第一个 Agent Loop）；
② 并行执行（Send）并入 Phase 5 Planning——它是 plan-execute 范式的自然组成部分，单独成阶段反而割裂。

| Phase | 核心问题 | 交付物 | 学习目标 | 面试考点 |
|---|---|---|---|---|
| 0 蓝图 | 为什么这么设计 | 本文 | 三范式区别；"动态性"的真实含义 | Plan-Execute vs ReAct vs Workflow |
| 1 最小 Agent Loop | 循环怎么转起来 | 单循环图：LLM→parse_document→观察→LLM→END | State / 节点 / 条件边 / 循环退出 | LangGraph 执行模型；"loop 不是 while" |
| 2 Tool Calling | LLM 怎么选工具 | locate/read/cross_ref 工具 | tool schema、观察结果、代码 vs LLM 边界 | function calling 原理；"错误当观察" |
| 3 Tool Registry | 工具怎么管理 | 注册表 + 白名单 | 解耦与扩展；统一错误处理 | 为什么白名单是防乱规划第一道闸 |
| 4 RAG Tool | 审查依据从哪来 | risk_kb_search | 分块-检索-注入三步；检索质量评估 | 为什么自建要点库；何时才需要向量库 |
| 5 Planning | 计划从哪来 | planner+executor 拆分；DAG 入 State；Send 并行 | DAG、reducer 写入纪律 | 动态性 = 状态驱动路由（不是动态图） |
| 6 Re-planning | 计划怎么变 | observer + replan_check 三岔路由 | 反馈闭环；死循环防护 | replan 触发器；diff 式追加 |
| 7 Reflection | 结果可信吗 | reflect 节点 + 证据校验 + 失败恢复 | 反思 vs 重规划 | 错误恢复机制设计 |
| 8 Memory | 状态怎么存续 | SqliteSaver + context 管理 | checkpoint / thread_id | 为什么用 Checkpointer 而不是自己存 |
| 9 工程化 | 别人怎么用 | FastAPI 薄层 + 异常 + 日志 + 后台任务 | 适配层与 core 解耦 | 为什么薄 API 层；依赖方向 |
| 10 评测 | 怎么证明好 | 三版本对比 eval + ground truth 测试集 | 指标设计 | 怎么证明你的 Agent 比裸 LLM 好 |

每阶段结束：独立 git commit（Phase 1 起步先 `git init`）+ 知识检查 + **停止等你确认**。

## 附：与项目一的对照（面试必讲）

| | 项目一（电商问数） | 本项目（LexAudit） |
|---|---|---|
| 范式 | 固定 workflow（图开发期固化） | Plan-and-Execute（计划运行时生成） |
| 适合场景 | 流程确定、追求可控 | 流程不确定、需要自主 |
| 共用点 | 同一框架 LangGraph——证明你理解的是框架原语，不是只会一个用法 |
