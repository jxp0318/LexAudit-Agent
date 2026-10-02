"""Phase 2 验收测试：工具行为 / 观察驱动轨迹 / schema 可序列化。

运行: uv run pytest tests -q

覆盖三类断言：
    1. 工具函数本身（纯函数，可直接测）——正常路径 + 每一条「错误当观察」分支；
    2. Loop 的五步观察驱动轨迹——Mock 每步决策都依赖轨迹内容；
    3. 工具 manifest 是可 JSON 序列化的合法声明（将来 bind_tools 的前提）。
"""

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from lexaudit.graph import build_graph
from lexaudit.runtime import MockLLM
from lexaudit.tools.crossref import TOOL_CLAUSE_CROSS_REF, clause_cross_ref
from lexaudit.tools.parse import TOOL_PARSE_DOCUMENT, parse_document
from lexaudit.tools.read import TOOL_READ_CLAUSE, read_clause

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


class TestObservationDrivenLoop:
    def test_five_step_trajectory_and_final_report(self):
        """Mock 的五步决策轨迹：每步依据轨迹内容，最终给出引用错位的结论。

        断言的是完整轨迹的**顺序**（这是「观察驱动」的行为证据）：
        parse → read(第六条) → cross_ref(第六条) → read(第十条) → 收尾。
        """
        sample = Path(__file__).resolve().parents[1] / "samples" / "contract_labour.md"
        state = {
            "messages": [HumanMessage(content="请审查这份劳动合同。")],
            "contract_text": sample.read_text(encoding="utf-8"),
        }
        result = build_graph(MockLLM()).invoke(state)
        msgs = result["messages"]

        # 轨迹形态：human, (ai, tool) × 4, ai
        assert [m.type for m in msgs] == ["human", "ai", "tool", "ai", "tool", "ai", "tool", "ai", "tool", "ai"]
        calls = [m.tool_calls[0] for m in msgs if m.type == "ai" and m.tool_calls]
        assert [c["name"] for c in calls] == [
            "parse_document",
            "read_clause",
            "clause_cross_ref",
            "read_clause",
        ]
        # 第 3 步的验证对象跟随引用链：读了第六条，再读第十条
        assert calls[1]["args"]["number"] == "第六条"
        assert calls[3]["args"]["number"] == "第十条"
        # 收尾结论来自观察内容（不是硬编码），且 State.parsed 已就位（程序视角数据）
        assert "引用错位" in msgs[-1].content
        assert result.get("parsed") is not None
        assert len(result["parsed"].clauses) == 10


class TestToolManifests:
    def test_manifests_are_json_serializable_function_call_format(self):
        """工具清单必须能序列化为 JSON——这是 bind_tools 发给真模型的前提条件。"""
        for manifest in (TOOL_PARSE_DOCUMENT, TOOL_READ_CLAUSE, TOOL_CLAUSE_CROSS_REF):
            dumped = json.dumps(manifest, ensure_ascii=False)
            payload = json.loads(dumped)
            assert payload["type"] == "function"
            assert payload["function"]["name"]
            assert payload["function"]["description"]
            assert payload["function"]["parameters"]["type"] == "object"

    def test_manifests_match_available_tools_whitelist(self):
        """白名单与 manifest 一致：防止「声明了但派发不了」或「能派发但没声明」的漂移。"""
        from lexaudit.graph import AVAILABLE_TOOLS

        declared = {
            TOOL_PARSE_DOCUMENT["function"]["name"],
            TOOL_READ_CLAUSE["function"]["name"],
            TOOL_CLAUSE_CROSS_REF["function"]["name"],
        }
        assert declared == set(AVAILABLE_TOOLS)
