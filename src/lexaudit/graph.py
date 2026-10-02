"""graph —— 图组装：把节点和边拼成可运行的 Agent Loop。

    START → agent ──(最后一条消息有 tool_calls?)──yes──→ tools ──→ agent  (回到环上)
                   └────────────no────────────────────→ END

🧱 代码决定：节点集合、边集合、条件边的判断逻辑、循环的退出条件。
🤖 LLM 决定：是否产生 tool_calls（走哪条边）、调什么工具、传什么参数、最终回答。
图是静态的，执行序列由运行时的消息轨迹驱动。
"""

from collections.abc import Callable

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from lexaudit.state import AgentState
from lexaudit.tools import TOOLS


def agent_node(state: AgentState, llm: Runnable, system_prompt: str | None = None) -> dict:
    """agent 节点：把完整对话轨迹交给 LLM，取回它的下一步决策。

    参数:
        state: 当前 State（本节点实际只读 messages）。
        llm: 注入的 LLM。类型用 langchain 官方的 Runnable——它只要求有 invoke，
             真实模型（build_llm 的产物）与测试里自备的替身都能直接注入。
        system_prompt: 可选。给定且轨迹开头还没有 system 消息时前置一条——模型靠它
                       知道有哪些工具、该怎么干活。

    返回:
        {"messages": [AIMessage]} —— 只返回要「更新」的字段子集，经 add_messages
        reducer 追加合并（节点不直接改 State，合并由框架负责）。

    Loop 里唯一的 🤖 决策点：它不做业务逻辑，只做「LLM ↔ State」的翻译。
    """
    messages = state["messages"]
    if system_prompt and (not messages or messages[0].type != "system"):
        messages = [SystemMessage(content=system_prompt), *messages]
    return {"messages": [llm.invoke(messages)]}


def route_after_agent(state: AgentState) -> str:
    """条件边函数：agent 之后走 tools 还是 END。

    参数:
        state: 当前 State（只读 messages 的最后一条）。

    返回:
        "tools"（继续环）或 END —— 返回值是 add_conditional_edges 注册表里的 key。

    全图唯一控制循环退出的地方：🧱 判断逻辑是代码（有没有 tool_calls），
    🤖 但「产不产生 tool_calls」是 LLM 的输出——退出条件由代码定义，何时满足由 LLM 决定。
    """
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(llm: Runnable, system_prompt: str | None = None) -> Callable:
    """组装并编译图。

    参数:
        llm: 注入的 LLM（真实模型，或测试自备的替身）。
        system_prompt: 可选，透传给 agent 节点。

    返回:
        编译后的可执行图（支持 invoke / stream）。

    以 AgentState 为 schema，注册 agent（闭包捕获注入的 llm）与 tools（官方 ToolNode），
    画三条边，compile() 做拓扑校验并产出可执行对象。

    为什么 tools 用 ToolNode：工具分发、ToolMessage 配对、未知工具的错误观察、参数
    schema 校验这四件事框架全包，不再手写（BLUEPRINT 首要原则）。
    为什么 llm 经闭包注入：同一张图定义能用不同模型编译出多个实例，互不干扰。
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: agent_node(state, llm, system_prompt))
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")  # 环的回边：观察喂回 LLM
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    return graph.compile()
