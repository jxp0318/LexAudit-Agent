"""Phase 2 验收测试：工具行为 / 端到端 Loop / schema 可序列化。

运行: uv run pytest tests -q
（端到端那组需要 API key 与网络；未配置时自动跳过，不影响其余测试。）

覆盖三类断言：
    1. 工具函数本身（纯函数，可直接测）——正常路径 + 每一条「错误当观察」分支；
    2. 真模型端到端——图能终止、工具真被调用、产物就位（只断言结构，不断言措辞）；
    3. 工具 schema 的「暴露边界」——发给 LLM 的那份里必须没有注入参数。
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from lexaudit.graph import build_graph
from lexaudit.llm import SYSTEM_PROMPT, build_llm
from lexaudit.tools import TOOLS, TOOL_NAMES
from lexaudit.tools.crossref import clause_cross_ref, clause_cross_ref_tool
from lexaudit.tools.parse import parse_document, parse_document_tool
from lexaudit.tools.read import read_clause, read_clause_tool

TEXT = """甲方：星辰科技有限公司
乙方：李某

第一条 合同期限
本合同期限为一年。

第二条 保密义务
保密范围见第三条。

第三条 合同生效
本合同自双方签字之日起生效。
"""

# 含「引用不存在的条款」的文本，用于核对「引用缺失」分支
TEXT_WITH_MISSING_REF = """第一条 违约责任
违约方应承担赔偿责任，具体标准见第九条。
"""


def _parsed():
    """测试辅助：解析出可直接喂给工具的 ParsedDocument。"""
    return parse_document(TEXT)


class TestReadClause:
    def test_reads_full_text(self):
        """正常路径：按条号返回全文（含标题与正文）。"""
        obs = read_clause(_parsed(), "第二条")
        assert obs.startswith("第二条 保密义务")
        assert "见第三条" in obs

    def test_read_before_parse_gives_guiding_error(self):
        """未解析就先读：错误观察必须引导下一步（先调 parse_document），而不是干巴巴失败。"""
        obs = read_clause(None, "第二条")
        assert "尚未解析" in obs
        assert "parse_document" in obs

    def test_wrong_number_lists_available_numbers(self):
        """条号不存在：错误观察附「可用条款」列表——把 LLM 的重试收敛到一次。"""
        obs = read_clause(_parsed(), "第九条")
        assert "未找到" in obs
        assert "可用条款" in obs
        assert "第一条" in obs

    def test_missing_argument_defended(self):
        """LLM 漏传参数（幻觉形状）也不会崩：返回错误观察。"""
        obs = read_clause(_parsed(), None)
        assert "缺少参数 number" in obs


class TestClauseCrossRef:
    def test_reports_existing_reference_with_title(self):
        """引用存在：报告要附带被引条款的标题——这是递给 LLM 做语义判断的原料。"""
        obs = clause_cross_ref(_parsed(), "第二条")
        assert "第三条 → 存在" in obs
        assert "合同生效" in obs  # 被引条款标题（语义原料）

    def test_reports_missing_reference(self):
        """引用缺失：形式核对必须报出来（工具的核心价值所在）。"""
        obs = clause_cross_ref(parse_document(TEXT_WITH_MISSING_REF), "第一条")
        assert "第九条 → 不存在" in obs
        assert "⚠" in obs

    def test_no_reference_reported_explicitly(self):
        """无引用也要明确报告，不能沉默——沉默会让 LLM 误以为核对失败。"""
        obs = clause_cross_ref(_parsed(), "第一条")
        assert "没有引用其他条款" in obs


class TestLiveAgentLoop:
    """真模型端到端：只断言「由代码保证」的结构，不断言模型的措辞与步数。

    为什么不检查具体步骤与文字：真模型是概率性的——实测同一份合同，它会一轮并行发起
    8 个工具调用、自行挑选要读的条款、中间回复甚至先用英文。拿这些写断言等于自找 flaky。
    稳定可断言的是：图能终止、工具真被调用、State.parsed 就位、结尾是纯文本收尾。
    """

    def test_end_to_end_structure(self):
        """跑完整图，验证「模型能驱动这张图跑出正确产物」。

        未配置 API key 时自动跳过（而非失败）——这样没有凭证的环境不会因此变红。
        """
        try:
            llm = build_llm()
        except RuntimeError as exc:
            pytest.skip(f"未配置真实模型：{exc}")

        sample = Path(__file__).resolve().parents[1] / "samples" / "contract_labour.md"
        state = {
            "messages": [HumanMessage(content="请审查这份劳动合同，指出存在的问题。")],
            "contract_text": sample.read_text(encoding="utf-8"),
        }
        result = build_graph(llm, SYSTEM_PROMPT).invoke(state, config={"recursion_limit": 25})
        msgs = result["messages"]

        # 图必须终止：最后一条是纯文本收尾（若仍带 tool_calls，说明环没退出）
        assert msgs[-1].type == "ai"
        assert not msgs[-1].tool_calls
        # 工具必须真被调用过，观察也已回到轨迹里
        assert any(m.type == "tool" for m in msgs)
        # 第一个工具调用必须是 parse_document——system prompt 的硬要求，也是后续工具的前提
        first_call = next(m for m in msgs if m.type == "ai" and m.tool_calls)
        assert first_call.tool_calls[0]["name"] == "parse_document"
        # 程序视角数据必须就位——这才是工具「两种归宿」的真正验收点
        assert result["parsed"] is not None
        assert len(result["parsed"].clauses) == 10


class TestToolSchemas:
    """锁定「工具声明的暴露边界」——Phase 3 从手写 manifest 换成 @tool 后的新断言。"""

    def test_injected_state_is_hidden_from_llm_schema(self):
        """发给 LLM 的 schema（tool_call_schema）必须剔除注入参数。

        ⚠️ 关键区分：`args_schema` 是「完整输入」（含注入参数），
        `tool_call_schema` 才是「真正发给 LLM 的那份」——验证时用错属性会误判成
        「框架没剔除」，这条测试把这个坑钉死。

        环境数据（contract_text / parsed）与 tool_call_id 若出现在菜单里，
        LLM 只能幻觉一个值填进去——这正是「数据不经过 LLM」要防的事。
        """
        assert "number" in read_clause_tool.tool_call_schema.model_fields
        assert "parsed" not in read_clause_tool.tool_call_schema.model_fields
        assert "contract_text" not in parse_document_tool.tool_call_schema.model_fields
        assert "tool_call_id" not in parse_document_tool.tool_call_schema.model_fields

    def test_injected_value_beats_forged_llm_argument(self):
        """安全语义：LLM 在 args 里伪造注入参数，运行时注入值优先。

        这层防伪造是官方机制免费提供的——自己实现要在每个工具里重复写一遍。
        验证方式：让 Mock 在 read_clause 的 args 里塞一个假 parsed，
        若假值生效，工具会对字符串取 .clauses 而报错；观察里出现真实条款正文，
        即证明注入值赢了。
        """

        class ForgingMockLLM:
            """两步：parse_document 后，发起一次带伪造 parsed 参数的 read_clause。"""

            def invoke(self, messages):
                if not any(m.type == "tool" for m in messages):
                    return AIMessage(
                        content="先解析。",
                        tool_calls=[{"name": "parse_document", "args": {}, "id": "f1"}],
                    )
                if sum(1 for m in messages if m.type == "tool") < 2:
                    return AIMessage(
                        content="伪造 parsed 试试。",
                        tool_calls=[
                            {
                                "name": "read_clause",
                                "args": {"number": "第六条", "parsed": "LLM伪造的假数据"},
                                "id": "f2",
                            }
                        ],
                    )
                return AIMessage(content="完成。")

        sample = Path(__file__).resolve().parents[1] / "samples" / "contract_labour.md"
        result = build_graph(ForgingMockLLM()).invoke(
            {
                "messages": [HumanMessage(content="审查")],
                "contract_text": sample.read_text(encoding="utf-8"),
            }
        )
        observations = [m.content for m in result["messages"] if m.type == "tool"]
        assert any("保密" in o for o in observations), "真实条款正文未出现 → 伪造值可能覆盖了注入值"
        assert not any("LLM伪造的假数据" in o for o in observations)

    def test_menu_exports_clean_openai_schema(self):
        """菜单可导出为 OpenAI function 格式，且参数里只有 LLM 该填的东西。"""
        for t in TOOLS:
            payload = json.loads(json.dumps(convert_to_openai_tool(t), ensure_ascii=False))
            assert payload["type"] == "function"
            assert payload["function"]["name"] == t.name
            assert payload["function"]["description"]
            assert payload["function"]["parameters"]["type"] == "object"

    def test_whitelist_is_derived_from_registry(self):
        """白名单由注册表派生（唯一真相来源），不再是两处手写后靠测试锁一致。"""
        assert TOOL_NAMES == tuple(t.name for t in TOOLS)
        assert set(TOOL_NAMES) == {"parse_document", "read_clause", "clause_cross_ref"}

    def test_description_is_short_not_full_docstring(self):
        """给 LLM 的 description 必须是精炼短句，不能是整段中文长注释。

        官方 @tool 默认拿整段 docstring 当 description；本项目 docstring 是写给人看的
        （职责+参数+返回+为什么），直接沿用会把几百字塞进模型上下文（实测过：parse_document
        的菜单里出现了 500+ 字）。所以三个工具都显式传了 description=——
        这条测试防止将来有人把它删掉（删了不报错，只会静默污染 prompt）。
        """
        for t in TOOLS:
            assert len(t.description) < 120, f"{t.name} 的 description 过长，可能用了整段 docstring"
            assert "参数:" not in t.description  # 长注释的特征字样不该出现在给 LLM 的描述里
