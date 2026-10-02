"""Phase 1 验收测试：解析正确性 / Loop 终止形态 / 错误当观察。

运行: uv run pytest tests -q

测试隔离原则：本文件锁定 **Phase 1 定义的行为**，所以用一个「单步 Mock」（本地定义），
而不是复用 runtime 里不断演进的 MockLLM——否则每次升级 Mock 剧本，历史阶段的测试都会被动跟着改，
测试就失去了「行为锚点」的意义。后续阶段的 Mock 行为由各自阶段的测试文件锁定。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lexaudit.graph import build_graph
from lexaudit.tools.parse import parse_document

SAMPLE_TEXT = """第一条 合同期限
本合同期限为三年。

第二条 工作内容
乙方担任工程师。
"""


class SingleToolMockLLM:
    """Phase 1 行为的锚定 Mock：调一次 parse_document 后立即收尾。

    行为: 无工具观察 → 发 parse_document 调用；见到观察 → 纯文本收尾。
    存在意义: 用最小轨迹（一圈环）验证「Loop 能转起来、能停下来」这一 Phase 1 核心结论。
    """

    def invoke(self, messages):
        if not any(m.type == "tool" for m in messages):
            return AIMessage(
                content="我需要先解析合同结构，了解条款组成。",
                tool_calls=[{"name": "parse_document", "args": {}, "id": "call_parse_1"}],
            )
        return AIMessage(content="合同结构解析完成，Phase 1 的任务到此为止。")


class BadToolMockLLM:
    """Mock 的"坏行为"变体：故意调用一个不存在的工具，验证「错误当观察」护栏。

    与正常 Mock 的唯一区别: 第一步发出的 tool_call 名字是 "frobnicate"（未注册）。
    存在的意义: 真实 LLM 会犯这类错（幻觉工具名），图必须扛得住而不是崩溃。
    """

    def invoke(self, messages):
        if not any(m.type == "tool" for m in messages):
            return AIMessage(
                content="调用一个不存在的工具试试。",
                tool_calls=[{"name": "frobnicate", "args": {}, "id": "call_bad_1"}],
            )
        return AIMessage(content="好的，我注意到该工具不存在，收尾。")


def _initial_state(text: str) -> dict:
    """构造最小初始 State（测试辅助函数）。

    做了什么: 组装一条 HumanMessage + 合同原文。
    为什么单独抽出来: 多个 Loop 测试都要用同一形状的输入，复用避免抄写漂移。
    """
    return {
        "messages": [HumanMessage(content="请解析这份合同。")],
        "contract_text": text,
    }


class TestParseDocument:
    def test_basic_parsing(self):
        """编号行开新条、非编号行进正文：结构与内容都要对。"""
        doc = parse_document(SAMPLE_TEXT)
        assert len(doc.clauses) == 2
        assert doc.clauses[0].number == "第一条"
        assert doc.clauses[0].title == "合同期限"
        assert "三年" in doc.clauses[0].text
        assert doc.clauses[1].id == "c2"

    def test_preamble_is_captured(self):
        """Phase 2 修复了已知限制①：第一个编号行之前的文本进 preamble，不再被丢弃。

        这条测试是「限制变特性」的活记录——Phase 1 时它的断言是相反的（assert 被丢弃）。
        """
        doc = parse_document("甲方：某公司\n" + SAMPLE_TEXT)
        assert "甲方：某公司" in doc.preamble
        assert len(doc.clauses) == 2
        assert "甲方" not in "".join(c.text for c in doc.clauses)  # 前言不进条款正文


class TestAgentLoop:
    def test_loop_terminates_with_expected_trajectory(self):
        """Loop 必须以「human → ai(tool_call) → tool(观察) → ai(收尾)」的形态终止。

        断言的既是轨迹形态、也是终止性——若条件边写错（比如永远走 tools），
        图会一直转下去直到 recursion_limit 报错，本测试随即失败。
        """
        graph = build_graph(SingleToolMockLLM())
        result = graph.invoke(_initial_state(SAMPLE_TEXT))
        msgs = result["messages"]
        assert [m.type for m in msgs] == ["human", "ai", "tool", "ai"]
        assert msgs[1].tool_calls[0]["name"] == "parse_document"
        assert msgs[2].tool_call_id == msgs[1].tool_calls[0]["id"]
        assert msgs[3].tool_calls == []  # 收尾消息不再带调用 → 条件边导向 END

    def test_observation_contains_clause_structure(self):
        """观察（ToolMessage）必须是自解释的：含条数与编号，LLM 看了能继续决策。"""
        graph = build_graph(SingleToolMockLLM())
        result = graph.invoke(_initial_state(SAMPLE_TEXT))
        tool_msg = result["messages"][2]
        assert isinstance(tool_msg, ToolMessage)
        assert "2 个条款" in tool_msg.content
        assert "第一条" in tool_msg.content

    def test_unknown_tool_returns_error_as_observation(self):
        """LLM 调了不存在的工具：图不崩溃，错误文本作为观察返回，LLM 下一步收尾。

        这是「错误当观察」（蓝图 §7）的行为锚定，也是 Phase 7 失败恢复的地基。
        """
        graph = build_graph(BadToolMockLLM())
        result = graph.invoke(_initial_state(SAMPLE_TEXT))
        msgs = result["messages"]
        assert [m.type for m in msgs] == ["human", "ai", "tool", "ai"]
        assert "不存在" in msgs[2].content  # 观察是错误说明文本
        assert msgs[3].content.startswith("好的")  # "LLM"看到了错误并正常收尾


@pytest.mark.parametrize("number", ["第一条", "第五条", "第十条"])
def test_clause_number_regex_coverage(number):
    """参数化小测：常见中文数字条号（一/五/十）都能被正则识别为条款编号。

    注意 id 是「文档内位置」（单条文档恒为 c1），条号本身落在 number 字段——
    这两者是不同的概念，本测试锚定的是 number 的解析覆盖范围。
    """
    doc = parse_document(f"{number} 标题\n正文\n")
    assert doc.clauses[0].number == number
    assert doc.clauses[0].id == "c1"
