"""runtime —— LLM 注入点。

蓝图 §4 原则 3：LLM 一律通过本模块注入，图和节点不 import 任何具体模型。
测试期用 MockLLM（不花钱、可离线、可精确断言）；接真实模型时只改这里的注入，图不动。
这就是「依赖注入」在 Agent 项目里的形态：LLM 是被策略化替换的组件，不是硬编码的依赖。
"""

from typing import Protocol

from langchain_core.messages import AIMessage, AnyMessage


class LLMAdapter(Protocol):
    """LLM 的接口约定：图内所有节点只依赖这个协议，不依赖具体模型。

    为什么用 Protocol（结构化类型）而不是抽象基类：
        Protocol 只要「长得像」就算实现——MockLLM 不需要显式继承任何类，
        意味着第三方 SDK 的模型类只要包一层 invoke 方法就能接入，耦合度最低。
    """

    def invoke(self, messages: list[AnyMessage]) -> AIMessage:
        """输入完整对话轨迹，返回一条 AIMessage（可能带 tool_calls，也可能是纯文本收尾）。"""
        ...


class MockLLM:
    """脚本式假 LLM：按固定剧本行动，用来在「不花 token」的前提下验证 Agent Loop。

    剧本（两步）:
        第 1 步：轨迹里还没有任何工具观察（ToolMessage）→ 发起 parse_document 调用；
        第 2 步：看到了观察 → 产出纯文本收尾（不再带 tool_calls，条件边随即路由到 END）。

    具体做了什么（分步）:
        1. 检查 messages 里是否已存在 ToolMessage（即"是否已观察过"）；
        2. 未观察 → 返回带 tool_calls 的 AIMessage（tool_call_id 是协议要求的调用标识，
           后续 ToolMessage 必须用它回指，配对错了 LLM 侧会拒绝这条观察）；
        3. 已观察 → 返回纯文本 AIMessage，Loop 终止。

    为什么这么设计:
        Mock 的行为必须「与真实 LLM 足够像」才有测试价值——它复用真实协议的消息类型
        （AIMessage/tool_calls/ToolMessage），Phase 2 换真模型时图的代码一行不动。
        真实模型也会犯的错（调不存在的工具）由 tests 里另一个 Mock 变体覆盖。
    """

    def invoke(self, messages: list[AnyMessage]) -> AIMessage:
        has_observation = any(m.type == "tool" for m in messages)
        if not has_observation:
            return AIMessage(
                content="我需要先解析合同结构，了解条款组成。",
                tool_calls=[{"name": "parse_document", "args": {}, "id": "call_parse_1"}],
            )
        return AIMessage(
            content=(
                "合同结构解析完成：我已获得全部条款的编号、标题与内容预览。"
                "Phase 1 的任务到此为止——风险识别将在后续阶段引入。"
            )
        )
