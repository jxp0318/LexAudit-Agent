"""AgentState —— 全图唯一的共享状态定义。

State 是 LangGraph 的核心概念之一：图里每个节点函数都接收 State、返回"要更新的字段子集"，
框架负责把返回值合并回 State 再传给下一个节点。一切"动态性"最终都存在这里。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from lexaudit.tools.parse import ParsedDocument


class AgentState(TypedDict):
    """Phase 2 的 State：三个字段，覆盖「LLM 视角」与「程序视角」两类数据。

    字段说明：
        messages: 对话轨迹（human / ai / tool 消息列表）。
                  带 add_messages reducer：节点返回 {"messages": [新消息]} 时，
                  框架做的是「追加合并」而不是「整表覆盖」——多个节点的写入可以自动汇合。
                  这也是 Phase 5 并行 executor 不冲突的根基。
        contract_text: 合同原文（普通字段）。
                      无 reducer：谁返回了这个 key 就是「整表覆盖」语义，
                      只由调用方在初始 State 写入一次，图内节点不写它。
        parsed: parse_document 的结构化结果（普通字段，ParsedDocument | None）。
                写入纪律：**只有 tools 节点的 parse_document 分支写它**（覆盖语义），
                其余工具只读——「环境数据给工具用」的程序视角数据，
                与 messages 里的观察文本（LLM 视角）是同一份解析的两种归宿。
                初始为 None，read_clause / clause_cross_ref 依赖它，
                缺失时的处理见各工具的「错误当观察」分支。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    contract_text: str
    parsed: ParsedDocument | None
