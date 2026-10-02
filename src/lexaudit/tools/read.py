"""read_clause —— 读取指定条款全文（按需读取，对付长文档的 context 管理）。

参数 = LLM 的决策信息：「读哪一条」由 LLM 看完条款清单后决定；
合同数据本身由框架从 State.parsed 注入——数据不经过 LLM。

错误信息里附「可用条号列表」：错误当观察，给出合法值域能把 LLM 的重试收敛到一次。
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from lexaudit.models import ParsedDocument


def read_clause(parsed: ParsedDocument | None, number: str | None) -> str:
    """读取指定条款全文（纯函数）。

    参数:
        parsed: 由 State.parsed 注入的解析结果（环境数据，不经过 LLM）。
        number: LLM 决策的条号（决策数据，必须经过 LLM）。

    返回:
        条款全文；或带「下一步怎么办」引导的错误观察（未解析 / 缺参数 / 条号不存在）。
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


@tool(
    "read_clause",
    description=(
        "读取指定条款的完整正文。先用 parse_document 拿到条款清单，"
        "再按条号读取需要细审的条款全文。"
    ),
)
def read_clause_tool(
    number: Annotated[
        str,
        Field(description="要读取的条款编号，必须是「第X条」形式，如：第六条"),
    ],
    parsed: Annotated[ParsedDocument | None, InjectedState("parsed")],
) -> str:
    """工具壳；实现体是上面的纯函数 read_clause。

    description 同样由装饰器显式指定，不取本 docstring（原因见 parse.py 同一处说明）。

    参数:
        number: LLM 的决策参数——读哪一条。它的 description 是「给 LLM 看的 API 文档」，
                写得越精确，LLM 传参越准。
        parsed: 框架从 State.parsed 注入的环境数据，不出现在发给 LLM 的 schema 里。

    为什么工具壳与实现体分开：实现体是可脱离框架单测的纯函数；工具壳只声明
    「哪些参数来自 LLM、哪些来自框架」，换框架/换协议不动业务逻辑。
    """
    return read_clause(parsed, number)
