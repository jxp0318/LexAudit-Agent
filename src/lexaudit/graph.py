"""graph —— 图组装：把节点和边拼成可运行的 Agent Loop。

Phase 1 的图（一个环 + 一个条件边）：

    START → agent ──(最后一条消息有 tool_calls?)──yes──→ tools ──→ agent  (回到环上)
                   └────────────no────────────────────→ END

Workflow 侧（🧱 代码决定）：节点集合、边集合、条件边的判断逻辑、循环的退出条件。
Agent 侧（🤖 LLM 决定）：是否产生 tool_calls（决定走哪条边）、tool_calls 的内容、最终回答。
"图是静态的，执行序列由运行时的消息轨迹驱动"——这就是 Phase 0 蓝图 §5 的最小实证。
"""

from collections.abc import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from lexaudit.runtime import LLMAdapter
from lexaudit.state import AgentState
from lexaudit.tools.parse import format_clauses_as_observation, parse_document


def agent_node(state: AgentState, llm: LLMAdapter) -> dict:
    """agent 节点：把完整对话轨迹交给 LLM，取回它的下一步决策。

    参数:
        state: 当前 State（本节点实际只读 messages）。
        llm: 注入的 LLM 适配器（MockLLM 或未来的真实模型）。

    返回:
        {"messages": [AIMessage]} —— 只返回要「更新」的字段子集，
        经 add_messages reducer 追加合并进 State（节点不直接改 State，框架负责合并）。

    具体做了什么（分步）:
        1. 取出 state["messages"]（完整轨迹，含最初请求、历史 AI 消息、工具观察）；
        2. 调用 llm.invoke 得到一条 AIMessage；
        3. 把它包进返回 dict，交回框架。

    为什么这么设计:
        agent 节点是 Loop 里唯一的 🤖 决策点——它不做任何业务逻辑，只做「LLM ↔ State」的翻译。
        这样图结构、业务工具、模型决策三者互不污染，任何一层都能被单独替换/测试。
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tools_node(state: AgentState) -> dict:
    """tools 节点：执行上一条 AIMessage 里的所有 tool_calls，把结果作为「观察」追加回轨迹。

    参数:
        state: 当前 State（读 messages 的最后一条 AI 消息 + contract_text）。

    返回:
        {"messages": [ToolMessage, ...]} —— 每个 tool_call 恰好对应一条 ToolMessage。

    具体做了什么（分步）:
        1. 取轨迹最后一条 AIMessage 的 tool_calls；
        2. 按工具名分发执行（Phase 1 只注册了 parse_document）；
        3. 执行结果序列化为观察文本，包成 ToolMessage（tool_call_id 与调用配对）；
        4. 未知工具名 → 返回「错误说明文本」作为观察，而不是抛异常。

    为什么这么设计:
        a) 错误当观察（蓝图 §7）：工具失败的信息必须让 LLM 看见，它才有机会在下一步改参数重试；
           抛异常会让整个图崩溃，LLM 失去纠正机会——这是 Phase 7「失败恢复」的地基。
        b) 手写而不是用 langgraph.prebuilt.ToolNode：ToolNode 内部做了批量执行、错误策略等封装，
           但 Phase 1 的学习目标恰恰是看清「tool_call → 执行 → ToolMessage」这条链路本身。
           等 Phase 2 工具多了，再把「换成 ToolNode」作为替代方案讨论。
        c) 合同文本从 State 读取而不是 tool_call 参数：环境数据不经过 LLM 决策（省 token，
           也是真实工程做法——LLM 只传「要解析哪份」这类决策信息，不搬运数据本身）。
    """
    last_ai: AIMessage = state["messages"][-1]
    results: list[ToolMessage] = []
    for tc in last_ai.tool_calls:
        name = tc["name"]
        call_id = tc["id"]
        if name == "parse_document":
            clauses = parse_document(state["contract_text"])
            observation = format_clauses_as_observation(clauses)
        else:
            # 未知工具：错误文本本身就是观察，交给 LLM 在下一步自我纠正
            observation = f"错误：工具 {name} 不存在。可用工具：parse_document"
        results.append(ToolMessage(content=observation, tool_call_id=call_id))
    return {"messages": results}


def route_after_agent(state: AgentState) -> str:
    """条件边函数：agent 之后走 tools 还是 END。

    参数:
        state: 当前 State（只读 messages 的最后一条）。

    返回:
        "tools"（继续环）或 END（收尾）——返回值是 add_conditional_edges 注册表里的 key。

    具体做了什么（分步）:
        1. 取最后一条消息（必然是 agent 刚产出的 AIMessage）；
        2. 有 tool_calls → 返回 "tools"；没有 → 返回 END。

    为什么这么设计:
        这是「Loop 终止条件」的唯一所在，全图只有这一处代码控制循环退出。
        🧱 判断逻辑是代码（检查 tool_calls 是否存在）；
        🤖 但「产生不产生 tool_calls」本身是 LLM 的输出——
        即：退出条件由代码定义，何时满足由 LLM 决定。这就是 Agent Loop 的本质分工。
    """
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(llm: LLMAdapter) -> Callable:
    """组装并编译 Phase 1 的图。

    参数:
        llm: 注入的 LLM 适配器。

    返回:
        编译后的可执行图（LangGraph CompiledGraph，支持 invoke / stream）。

    具体做了什么（分步）:
        1. 以 AgentState 为 State schema 建 StateGraph；
        2. 注册两个节点（agent 闭包捕获注入的 llm——依赖注入在这里发生）；
        3. 画三条边：START→agent（入口）、tools→agent（环的回边）、
           agent→条件边（路由表把 "tools"/END 映射到节点/终点）；
        4. compile() 做拓扑校验并产出可执行对象。

    为什么这么设计:
        节点函数是「无状态函数」、llm 经参数闭包注入，意味着同一张图定义可以
        用不同模型编译出多个实例（测试用 Mock、生产用真模型），互不干扰。
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: agent_node(state, llm))
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")  # 环的回边：观察喂回 LLM
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END},
    )
    return graph.compile()
