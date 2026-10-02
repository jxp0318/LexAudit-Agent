"""parse_document —— 唯一的解析工具：合同原文 → 结构化条款列表。

设计原则（蓝图 §7）：
    1. 工具是「纯函数」：不调 LLM、无副作用、不修改外部状态，同样的输入永远得到同样的输出。
       为什么：纯函数可单独测试、可在并行中安全执行、不依赖执行顺序。
    2. 结构化信息（条号、标题）用代码解析器「确定性」获取，绝不让 LLM 从纯文本里「概率性」地抽。
       为什么：能用代码拿到的结构绝不用 LLM（旧版决策 D6）——正则是可预测的，LLM 是不可预测的。
    3. 工具参数刻意为「空」：要解析的合同属于执行环境数据，从 State 注入；
       工具参数只承载 LLM 决策出来的信息。合同全文若经过 tool_call 参数传递，
       既浪费 token 又不符合真实工程的做法。
"""

import re

from pydantic import BaseModel, Field

# 条款编号行的匹配模式：如「第一条 合同期限」「第十条、合同生效」。
# 中文数字只覆盖到「二十」以内的常见写法——MVP 合同样例足够，更大的数字（如「第一百零一条」）
# 留给 Phase 2 按需扩展（克制原则：现在加就是过度设计）。
_CLAUSE_RE = re.compile(r"^第[一二三四五六七八九十]{1,3}条[、\s]*(.*)$")


class Clause(BaseModel):
    """解析出的单个条款。

    字段说明：
        id: 条款在本次解析中的稳定标识（如 "c1"），供后续阶段的工具定位/引用。
        number: 条款原文编号（如 "第一条"）。
        title: 条款标题（编号行里跟在条号后面的部分，可能为空字符串）。
        text: 条款正文（不含编号行本身；后续行按原样累积）。
    """

    id: str
    number: str
    title: str
    text: str


class ParseDocumentArgs(BaseModel):
    """parse_document 的参数 schema。

    刻意定义为空 schema（无参数）：LLM 调用这个工具时不需要传达任何决策信息，
    合同文本由 tools 节点执行时从 State.contract_text 读取。
    为什么保留这个空类而不是不定义：args_schema 是工具协议的一部分，
    LLM 侧的 function calling 需要一个明确的 JSON Schema（空对象）来生成合法调用。
    """

    placeholder: str = Field(default="", description="保留字段，无实际用途")


def parse_document(text: str) -> list[Clause]:
    """把合同原文解析成条款列表（纯函数，不调 LLM、无副作用）。

    参数:
        text: 合同原文（markdown 纯文本）。

    返回:
        按文档顺序排列的 Clause 列表；若原文没有任何条款编号行，返回空列表。

    具体做了什么（分步）:
        1. 逐行扫描原文；
        2. 匹配到「第X条 ...」行时，关闭上一条、开新条款（id 按 c1/c2/... 递增分配）；
        3. 非编号行追加到「当前条款」的正文；没有当前条款时该行被忽略；
        4. 返回累积完成的条款列表。

    已知限制（刻意的，Phase 2 再修）:
        编号行之前的文本（如「甲方：... 乙方：...」的当事人信息）当前被整体丢弃——
        Phase 1 只需要条款结构。这个限制留着，是 Phase 2 改进解析器时的现成素材。

    为什么这么设计:
        「状态机扫描」是最朴素也最可调试的实现：一行代码一个动作，出错能精确定位行号。
        不上 AST/markdown 解析库，因为合同文本的语义结构（条号）比 md 标题更可靠，
        引入库反而引入与业务无关的复杂度。
    """
    clauses: list[Clause] = []
    current: Clause | None = None
    for line in text.splitlines():
        matched = _CLAUSE_RE.match(line.strip())
        if matched:
            # 关闭上一条，开新条
            if current is not None:
                clauses.append(current)
            current = Clause(
                id=f"c{len(clauses) + 1}",
                number=line.strip().split()[0],  # 形如「第一条」
                title=matched.group(1).strip(),
                text="",
            )
        elif current is not None:
            # 非编号行：追加进当前条款正文（保留行间换行，维持原文结构）
            current.text += line + "\n"
    if current is not None:
        clauses.append(current)
    return clauses


def format_clauses_as_observation(clauses: list[Clause]) -> str:
    """把条款列表序列化为喂给 LLM 的「观察」文本（工具结果的字符串表示）。

    参数:
        clauses: parse_document 的返回值。

    返回:
        适合放入 ToolMessage 的多行文本：先给条款总数，再逐条给出「编号 标题 | 正文预览」。

    具体做了什么（分步）:
        1. 第一行写总条数（给 LLM 一个全局概览）；
        2. 每条一行：编号、标题、正文前 60 个字符（单行化，避免观察文本过长）。

    为什么这么设计:
        观察（ToolMessage 内容）是 LLM 唯一能看到的工具结果，必须「自解释」；
        但全文塞进去会让上下文膨胀（蓝图 §11 的 context 管理问题），
        所以 Phase 1 只给「结构概览 + 预览」——后续阶段引入 read_clause 时按需取全文。
    """
    lines = [f"解析完成，共 {len(clauses)} 个条款："]
    for c in clauses:
        preview = c.text.strip().replace("\n", " ")[:60]
        lines.append(f"  {c.number} {c.title} | {preview}")
    return "\n".join(lines)
