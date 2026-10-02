"""runtime —— LLM 注入点。

蓝图 §4 原则 3：LLM 一律通过本模块注入，图和节点不 import 任何具体模型。
测试期用 MockLLM（不花钱、可离线、可精确断言）；接真实模型时只改这里的注入，图不动。
这就是「依赖注入」在 Agent 项目里的形态：LLM 是被策略化替换的组件，不是硬编码的依赖。

Phase 2 变化：MockLLM 从「固定剧本」升级为「观察驱动决策」——
每一步依据「最后一次工具调用 + 最新观察」决定下一步，与真实 LLM 读轨迹决策的行为同构。
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
    """观察驱动的 Mock：模拟真实 LLM「读轨迹 → 选工具 → 传参数」的决策方式。

    剧本（五步，每步的依据都是消息轨迹的内容）:
        1. 无工具调用历史 → parse_document（先拿合同结构）；
        2. 刚 parse 完 → read_clause(第六条)（选定「保密义务」条款细读）；
        3. 刚读完第六条 → clause_cross_ref(第六条)（正文含「见第十条」，先形式核对）；
        4. 刚核对完 → read_clause(第十条)（形式通过≠语义正确，跟过去验证）；
        5. 读过第十条 → 收尾：从两条观察里提取证据，报告引用错位。

    为什么每步都看轨迹而不是走计数器:
        真实 LLM 的决策依据就是消息轨迹本身；Mock 用规则模拟这一过程，
        换真模型时「读观察→选工具→传参数」由模型权重接管，图的代码一行不动。

    诚实标注（Mock 与真模型的能力差异）:
        第 2 步「选第六条」在 Mock 里是剧本指定的（预审目标：保密条款），
        真实模型会从 parse 观察的条款清单里**自行**挑选——这是 Mock 与真模型的本质差距，
        也是 Phase 10 评测要量化的东西。教学上必须说清，不能把剧本当成能力。
    """

    def invoke(self, messages: list[AnyMessage]) -> AIMessage:
        """按轨迹内容决策下一步。参数/返回值见 LLMAdapter 协议说明。"""
        # 决策依据一：我上一次调用了什么（找最后一条带 tool_calls 的 AI 消息）
        last_call = None
        for m in messages:
            if m.type == "ai" and getattr(m, "tool_calls", None):
                last_call = m.tool_calls[0]
        if last_call is None:
            return AIMessage(
                content="审查前先解析合同结构。",
                tool_calls=[{"name": "parse_document", "args": {}, "id": "call_1"}],
            )
        name = last_call["name"]
        number = (last_call.get("args") or {}).get("number")
        if name == "parse_document":
            return AIMessage(
                content="结构清单已拿到。先细读「保密义务」条款（第六条）。",
                tool_calls=[{"name": "read_clause", "args": {"number": "第六条"}, "id": "call_2"}],
            )
        if name == "read_clause" and number == "第六条":
            return AIMessage(
                content="第六条正文提到保密范围「见第十条」，先做引用的形式核对。",
                tool_calls=[{"name": "clause_cross_ref", "args": {"number": "第六条"}, "id": "call_3"}],
            )
        if name == "clause_cross_ref":
            return AIMessage(
                content="形式核对通过，但形式存在不代表语义正确——读取第十条验证。",
                tool_calls=[{"name": "read_clause", "args": {"number": "第十条"}, "id": "call_4"}],
            )
        return AIMessage(content=self._final_report(messages))

    def _final_report(self, messages: list[AnyMessage]) -> str:
        """从第六条/第十条两条 read 观察里抽取证据，生成收尾报告。

        参数:
            messages: 完整消息轨迹。

        返回:
            收尾文本：引用错位报告，或证据不足说明。

        具体做了什么（分步）:
            1. 在 tool 消息里找第六条观察（含「保密」关键词）与第十条观察（含「生效」关键词）；
            2. 两条都在 → 生成对比结论（引用错位）；
            3. 缺任何一条 → 明确说证据不足，不编造结论。

        为什么动态抽取而不是硬编码结论:
            让 Mock 的「结论」真正来自观察内容——与真实 LLM 从上下文归纳结论的行为同构。
            缺证据就报告不足，这条纪律比结论本身更重要（Phase 7 反思阶段会正式化它）。
        """
        clause6 = next(
            (
                m.content
                for m in messages
                if m.type == "tool" and "第六条" in str(m.content) and "保密" in str(m.content)
            ),
            "",
        )
        clause10 = next(
            (
                m.content
                for m in messages
                if m.type == "tool" and "第十条" in str(m.content) and "生效" in str(m.content)
            ),
            "",
        )
        if clause6 and clause10:
            return (
                "审查发现：第六条（保密义务）写明保密范围「见第十条」，"
                "但第十条实际内容是合同生效条款，与保密无关——引用错位。"
                "这条发现体现了工具与 LLM 的分工：clause_cross_ref 完成形式核对（引用存在），"
                "而「引用内容不匹配」这个语义判断只能由 LLM 完成。"
            )
        return "第六条引用的第十条内容未获取到，证据不足，无法完成对比。"
