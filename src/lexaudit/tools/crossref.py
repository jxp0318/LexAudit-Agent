"""clause_cross_ref —— 交叉引用一致性核对（形式核对，不是语义判断）。

本工具只回答「正文里引用的第X条存不存在」这种**形式**问题（正则 + 查表，确定性、可穷举）；
「引用的内容对不对题」是**语义**问题，留给 LLM。所以报告里附带被引条款的标题——
把语义判断的原料递过去，而不是替 LLM 下结论。
"""

import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from lexaudit.models import ParsedDocument

# 从正文里提取「第X条」引用。比解析器的模式宽（不带行首锚定），
# 因为引用出现在句子中间，如「具体范围见第十条」。
_REF_RE = re.compile(r"第[一二三四五六七八九十]{1,3}条")


def clause_cross_ref(parsed: ParsedDocument | None, number: str | None) -> str:
    """核对指定条款正文里引用的其他条款是否存在（纯函数）。

    参数:
        parsed: 由 State.parsed 注入的解析结果（环境数据）。
        number: LLM 决策的待核对条号。

    返回:
        引用核对报告（每个引用的存在性 + 被引条款标题），或引导性错误观察。
    """
    if parsed is None:
        return "错误：尚未解析合同。请先调用 parse_document。"
    if not isinstance(number, str) or not number:
        return "错误：缺少参数 number（条款编号，如'第六条'）。"
    target = next((c for c in parsed.clauses if c.number == number), None)
    if target is None:
        available = "、".join(c.number for c in parsed.clauses)
        return f"错误：未找到条款「{number}」。可用条款：{available}"

    # 剔除自身条号：编号行已在解析时剥离，理论不会出现，但一行成本的防御换取确定性
    referenced = [ref for ref in _REF_RE.findall(target.text) if ref != number]
    if not referenced:
        return f"{number}（{target.title}）正文中没有引用其他条款。"

    # 条号 → 标题，供报告附带语义判断的原料
    title_map = {c.number: c.title for c in parsed.clauses}
    lines = [f"{number}（{target.title}）的引用核对结果："]
    for ref in referenced:
        if ref in title_map:
            lines.append(f"  {ref} → 存在（标题：{title_map[ref]}）")
        else:
            lines.append(f"  {ref} → 不存在，引用缺失⚠")
    return "\n".join(lines)


@tool(
    "clause_cross_ref",
    description=(
        "核对指定条款正文里引用的其他条款是否存在（形式核对）。"
        "返回引用清单及被引条款的标题；语义是否匹配需另行判断。"
    ),
)
def clause_cross_ref_tool(
    number: Annotated[
        str,
        Field(description="要核对的条款编号，必须是「第X条」形式，如：第六条"),
    ],
    parsed: Annotated[ParsedDocument | None, InjectedState("parsed")],
) -> str:
    """工具壳；实现体是上面的纯函数 clause_cross_ref。

    description 由装饰器显式指定，不取本 docstring（原因见 parse.py 同一处说明）。

    参数:
        number: LLM 的决策参数——核对哪一条的引用。
        parsed: 框架从 State.parsed 注入的环境数据，不经过 LLM。

    为什么报告要附带被引条款标题：工具只做形式核对，而「引用内容与自身主题是否匹配」
    是语义判断——附上标题等于把判断原料递给 LLM，工具不越界替它下结论。
    """
    return clause_cross_ref(parsed, number)
