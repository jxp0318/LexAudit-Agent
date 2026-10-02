"""read_clause —— 读取指定条款的全文（按需读取，对付长文档的 context 管理）。

工具观（Phase 2 核心概念）：
    参数 = LLM 的决策信息。「读哪一条」由 LLM 看完观察（条款清单）后决定；
    合同数据本身（条款树）由 tools 节点从 State.parsed 注入——数据不经过 LLM。

错误当观察的进阶（蓝图 §7 原则的强化）：
    错误信息不只说明失败，还返回「可用条号列表」——错误本身就是引导 LLM
    下一步自我纠正的原料。真实 LLM 会传「第6条」「第六條」等变体，
    空泛的「未找到」只会让它瞎试；给出合法值域能把重试收敛到一次。
"""

from pydantic import BaseModel, Field

from lexaudit.tools.parse import ParsedDocument


class ReadClauseArgs(BaseModel):
    """read_clause 的参数 schema。

    字段说明:
        number: 要读取的条款编号。description 写得越精确，LLM 传参越准——
                description 是「给 LLM 看的 API 文档」，是 prompt 工程在工具层的延伸。
    """

    number: str = Field(description="要读取的条款编号，必须是「第X条」形式，如：第六条")


TOOL_READ_CLAUSE = {
    "type": "function",
    "function": {
        "name": "read_clause",
        "description": (
            "读取指定条款的完整正文。先用 parse_document 拿到条款清单，"
            "再按条号读取需要细审的条款全文。"
        ),
        "parameters": ReadClauseArgs.model_json_schema(),
    },
}


def read_clause(parsed: ParsedDocument | None, number: str | None) -> str:
    """读取指定条款全文（纯函数）。

    参数:
        parsed: 由 State.parsed 注入的解析结果（环境数据，不经过 LLM）。
        number: LLM 决策的条号（决策数据，必须经过 LLM）。

    返回:
        观察文本：条款全文，或引导性的错误说明。

    具体做了什么（分步）:
        1. parsed 为空 → 错误观察「请先调用 parse_document」（引导修复依赖顺序）；
        2. number 缺失/非字符串 → 错误观察「缺少参数 number」；
        3. 条号不在条款里 → 错误观察 + 可用条号列表；
        4. 命中 → 返回「编号 标题 + 全文」。

    为什么这么设计:
        每个错误分支都返回「下一步怎么办」的信息，而不是干巴巴的失败——
        这是「错误当观察」的完整形态：观察的价值在于让 LLM 的下一步有据可依。
    """
    if parsed is None:
        return "错误：尚未解析合同。请先调用 parse_document。"
    if not isinstance(number, str) or not number:
        return "错误：缺少参数 number（条款编号，如'第六条'）。"
    for c in parsed.clauses:
        if c.number == number:
            return f"{c.number} {c.title}\n{c.text.strip()}"
    available = "、".join(c.number for c in parsed.clauses)
    return f"错误：未找到条款「{number}」。可用条款：{available}"
