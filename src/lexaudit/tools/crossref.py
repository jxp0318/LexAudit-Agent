"""clause_cross_ref —— 交叉引用一致性核对（形式核对，不是语义判断）。

分工边界（Phase 2 最重要的教学点）：
    本工具只回答「条款正文里引用的第X条存不存在」这种**形式**问题
    （正则提取 + 查表，确定性、可穷举）；
    「引用的内容对不对题」是**语义**问题，留给 LLM。
    所以核对报告里附带被引条款的标题——把语义判断的原料递给 LLM，而不是替它下结论。
"""

import re

from pydantic import BaseModel, Field

from lexaudit.tools.parse import ParsedDocument

# 从条款正文里提取「第X条」引用的正则。比解析器的模式宽（不带行首锚定），
# 因为引用出现在句子中间，如「具体范围见第十条」。
_REF_RE = re.compile(r"第[一二三四五六七八九十]{1,3}条")


class ClauseCrossRefArgs(BaseModel):
    """clause_cross_ref 的参数 schema。

    字段说明:
        number: 要核对的条款编号（LLM 的决策信息：查哪一条的引用）。
    """

    number: str = Field(description="要核对的条款编号，如：第六条")


TOOL_CLAUSE_CROSS_REF = {
    "type": "function",
    "function": {
        "name": "clause_cross_ref",
        "description": (
            "核对指定条款正文里引用的其他条款是否存在（形式核对）。"
            "返回引用清单及被引条款的标题；语义是否匹配需另行判断。"
        ),
        "parameters": ClauseCrossRefArgs.model_json_schema(),
    },
}


def clause_cross_ref(parsed: ParsedDocument | None, number: str | None) -> str:
    """核对指定条款正文里引用的其他条款是否存在（纯函数）。

    参数:
        parsed: 由 State.parsed 注入的解析结果（环境数据）。
        number: LLM 决策的待核对条号。

    返回:
        观察文本：引用核对报告（无引用 / 每个引用的存在性 + 被引条款标题），或引导性错误说明。

    具体做了什么（分步）:
        1. parsed/number 防御（与 read_clause 同一套约定）；
        2. 定位条款；
        3. 正则提取正文中的全部「第X条」引用，剔除自身编号；
        4. 无引用 → 明确报告「该条无跨条引用」；
        5. 有引用 → 逐个核对存在性，报告附被引条款标题（给 LLM 的语义判断原料）。

    为什么剔除自身：
        解析时编号行已被剥离，正文理论上不含自身条号，但仍防御性排除——
        一行成本换取「引用自身」这种病态输入的确定性。
    """
    if parsed is None:
        return "错误：尚未解析合同。请先调用 parse_document。"
    if not isinstance(number, str) or not number:
        return "错误：缺少参数 number（条款编号，如'第六条'）。"
    target = next((c for c in parsed.clauses if c.number == number), None)
    if target is None:
        available = "、".join(c.number for c in parsed.clauses)
        return f"错误：未找到条款「{number}」。可用条款：{available}"

    referenced = [ref for ref in _REF_RE.findall(target.text) if ref != number]
    if not referenced:
        return f"{number}（{target.title}）正文中没有引用其他条款。"

    # 条号 → 标题 的查询表，供报告附带语义原料
    title_map = {c.number: c.title for c in parsed.clauses}
    lines = [f"{number}（{target.title}）的引用核对结果："]
    for ref in referenced:
        if ref in title_map:
            lines.append(f"  {ref} → 存在（标题：{title_map[ref]}）")
        else:
            lines.append(f"  {ref} → 不存在，引用缺失⚠")
    return "\n".join(lines)
