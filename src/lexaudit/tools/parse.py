"""parse_document —— 解析工具：合同原文 → 结构化文档（当事人信息 + 条款列表）。

设计原则：
    - 结构化信息（条号、标题）用代码「确定性」获取，不让 LLM 从纯文本「概率性」地抽；
    - 环境数据（合同全文）由框架从 State 注入，不经过 LLM 决策。

产物的两种归宿：观察文本给 LLM（进 messages）、ParsedDocument 给后续工具（进 State.parsed）。
"""

import re

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from lexaudit.models import Clause, ParsedDocument
from lexaudit.state import AgentState

# 条款编号行的匹配模式：如「第一条 合同期限」「第十条、合同生效」。
# 中文数字覆盖常见写法，更大数字按需扩展（克制原则：现在加就是过度设计）。
_CLAUSE_RE = re.compile(r"^第[一二三四五六七八九十]{1,3}条[、\s]*(.*)$")


def parse_document(text: str) -> ParsedDocument:
    """把合同原文解析成结构化文档（纯函数，不调 LLM、无副作用）。

    参数:
        text: 合同原文（markdown 纯文本）。

    返回:
        ParsedDocument：preamble（编号行之前的全部文本）+ clauses（条款列表）。

    两段式状态机：先累积前言，遇到第一个编号行后开始收条款，非编号行追加进当前条款正文。
    """
    preamble_lines: list[str] = []
    clauses: list[Clause] = []
    current: Clause | None = None
    for line in text.splitlines():
        matched = _CLAUSE_RE.match(line.strip())
        if matched:
            if current is not None:
                clauses.append(current)
            current = Clause(
                id=f"c{len(clauses) + 1}",
                number=line.strip().split()[0],  # 形如「第一条」
                title=matched.group(1).strip(),
                text="",
            )
        elif current is not None:
            current.text += line + "\n"
        else:
            # 尚未遇到第一个编号行：这段是当事人信息等前言
            preamble_lines.append(line)
    if current is not None:
        clauses.append(current)
    return ParsedDocument(preamble="\n".join(preamble_lines).strip(), clauses=clauses)


def format_document_as_observation(doc: ParsedDocument) -> str:
    """把解析结果序列化为喂给 LLM 的「观察」文本。

    参数:
        doc: parse_document 的返回值。

    返回:
        多行文本：当事人信息摘要 + 条款总数 + 逐条「编号 标题 | 正文前 60 字预览」。

    给结构概览而非全文：全文塞进观察会让上下文膨胀，需要全文时 LLM 应调 read_clause
    按条号取——这就是「按需读取」的 context 管理。
    """
    lines: list[str] = []
    if doc.preamble:
        preview = doc.preamble.replace("\n", " ")[:80]
        lines.append(f"当事人信息：{preview}")
    lines.append(f"解析完成，共 {len(doc.clauses)} 个条款：")
    for c in doc.clauses:
        text_preview = c.text.strip().replace("\n", " ")[:60]
        lines.append(f"  {c.number} {c.title} | {text_preview}")
    return "\n".join(lines)


@tool(
    "parse_document",
    description=(
        "解析合同原文，返回当事人信息与全部条款的结构清单（编号/标题/正文预览）。"
        "审查任何合同前必须先调用本工具。"
    ),
)
def parse_document_tool(runtime: ToolRuntime[None, AgentState]) -> Command:
    """工具壳；实现体是上面的纯函数 parse_document。

    description 由装饰器显式指定，**不取本 docstring**——@tool 默认把整段 docstring 当
    描述，会把写给人看的中文长注释塞进模型上下文（「给人看的注释」≠「给 LLM 看的描述」）。

    参数:
        runtime: 官方统一运行时入口（ToolRuntime[ContextT, StateT]），由框架注入，
                 **不会出现在发给 LLM 的工具 schema 里**。一个参数即可拿到 state /
                 tool_call_id / store / stream_writer / context，为 Phase 8/9 铺路。
                 第二个泛型参数写成 AgentState，这要求它在运行时可见——正是共享模型
                 必须独立于工具模块的直接原因（见 models.py）。

    返回:
        Command：一次更新两处——messages 追加配对 ToolMessage（给 LLM 的观察），
        parsed 写入结构化文档（给后续工具的数据）。

    为什么返回 Command 而不是直接 return str：默认返回值只能变成 ToolMessage，
    写不了 State 的其他字段；而本工具有两个消费者。LangGraph 里「从工具写 State」的
    官方机制就是 Command.update（goto 是另一半能力，本项目未用）。
    为什么注入而不是让 LLM 传合同全文：数据不经过 LLM——注入参数会被框架从 schema 剔除。
    """
    doc = parse_document(runtime.state["contract_text"])
    return Command(
        update={
            "parsed": doc,
            "messages": [
                ToolMessage(
                    content=format_document_as_observation(doc),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
