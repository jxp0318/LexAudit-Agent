"""demo —— 可运行入口：跑通 Agent Loop，逐步打印执行轨迹。

运行: uv run python scripts/demo.py

设计意图（蓝图规则「优先让用户看到完整执行轨迹」）:
    用 stream(stream_mode="updates") 逐节点拿到增量更新——每走一个节点就能看到
    「谁在动、它产出什么」，而不是等全图跑完才给一个最终答案。
    调试 Agent 的第一手工具就是这条轨迹。

Phase 2 新增:
    开头打印三个工具的 manifest（JSON Schema）——这就是将来 bind_tools 发给真模型的「菜单」，
    现在先让用户看清 function calling 协议的原料长什么样。
"""

import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from lexaudit.graph import build_graph
from lexaudit.runtime import MockLLM
from lexaudit.tools.crossref import TOOL_CLAUSE_CROSS_REF
from lexaudit.tools.parse import TOOL_PARSE_DOCUMENT
from lexaudit.tools.read import TOOL_READ_CLAUSE

# 工具清单（发给 LLM 的「菜单」）：name + description + parameters(JSON Schema)
TOOL_MANIFESTS = [TOOL_PARSE_DOCUMENT, TOOL_READ_CLAUSE, TOOL_CLAUSE_CROSS_REF]

# 样例合同路径（从项目根定位，兼容任意工作目录执行）
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "contract_labour.md"


def summarize_message(msg) -> str:
    """把一条消息压缩成一行可读摘要（只用于终端展示，不进业务逻辑）。

    参数: msg —— langchain 消息对象（Human/AI/Tool 之一）。
    返回: 形如 "[ai] 我需要先解析合同结构… (tool_calls: parse_document)" 的单行文本。
    做了什么: 按消息类型取 content 前 50 字；AI 消息额外列出它发起的 tool_calls 名字。
    为什么: 轨迹是给人看的，全量原文会淹没关键信息。
    """
    kind = msg.type
    content = str(msg.content).replace("\n", " ")[:50]
    if getattr(msg, "tool_calls", None):
        calls = ", ".join(tc["name"] for tc in msg.tool_calls)
        return f"[{kind}] {content}…  (tool_calls: {calls})"
    return f"[{kind}] {content}…"


def main() -> None:
    """跑一次完整审查 Loop：读样例 → 建图 → 流式执行 → 打印轨迹与最终答复。

    具体做了什么（分步）:
        1. 读入样例合同全文；
        2. 构造初始 State：messages 只放一条 HumanMessage（用户的审查要求），
           contract_text 放合同原文（普通字段，一次性写入）；
        3. build_graph(MockLLM()) 编译出图——注意 MockLLM 是在这里注入的，
           换真模型只改这一行；
        4. stream 逐节点打印增量更新（执行轨迹）；
        5. 打印最终 AI 答复。

    为什么初始 State 不由图内节点写入:
        「用户说了什么」和「合同原文」是图的输入而非图的产物，
        由调用方放进初始 State 是最诚实的建模——图内不该有节点"凭空造输入"。
    """
    contract_text = SAMPLE.read_text(encoding="utf-8")
    print(f"已加载样例合同: {SAMPLE.name}（{len(contract_text)} 字符）\n")

    print("=== 工具菜单（将来 bind_tools 发给真模型的就是这份 JSON Schema）===")
    print(json.dumps(TOOL_MANIFESTS, ensure_ascii=False, indent=2))
    print()

    graph = build_graph(MockLLM())

    initial_state = {
        "messages": [HumanMessage(content="请解析这份劳动合同的结构，为后续审查做准备。")],
        "contract_text": contract_text,
    }

    print("=== 执行轨迹（stream_mode=updates，每块 = 一个节点刚跑完）===")
    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():
            print(f"\n-- 节点 [{node_name}] 输出:")
            for msg in update["messages"]:
                print("   " + summarize_message(msg))

    # stream 不返回最终 State，这里再 invoke 一次拿完整轨迹做收尾展示。
    # 为什么敢跑两次：本阶段的图是纯函数式的（无外部副作用），重复执行结果相同。
    final_state = graph.invoke(initial_state)
    print("\n=== 最终答复 ===")
    print(final_state["messages"][-1].content)
    parsed = final_state.get("parsed")
    if parsed is not None:
        print(f"\n[State.parsed] 已就位：{len(parsed.clauses)} 个条款（程序视角数据，供后续工具使用）")


if __name__ == "__main__":
    sys.exit(main())
