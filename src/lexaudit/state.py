"""AgentState —— 全图唯一的共享状态定义。

节点函数接收 State、返回「要更新的字段子集」，框架负责把返回值合并回 State 再传给下一节点。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from lexaudit.models import ParsedDocument


class AgentState(TypedDict):
    """三个字段，覆盖「LLM 视角」与「程序视角」两类数据。

    messages: 对话轨迹。带 add_messages reducer —— 节点返回 {"messages": [新消息]} 时
              框架做的是「追加合并」而非「整表覆盖」，多节点写入可自动汇合
              （这是 Phase 5 并行 executor 不冲突的根基）。
    contract_text: 合同原文。普通字段（覆盖语义），只由调用方在初始 State 写入一次。
    parsed: parse_document 的结构化结果。写入纪律：只有 parse_document 写它（经 Command
            回写），其余工具只读；初始为 None，缺失时的处理见各工具的「错误当观察」分支。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    contract_text: str
    parsed: ParsedDocument | None
