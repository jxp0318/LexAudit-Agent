"""demo —— 可运行入口：跑通 Agent Loop，逐步打印执行轨迹。

运行:
    uv run python scripts/demo.py

设计意图（蓝图规则「优先让用户看到完整执行轨迹」）:
    用 stream 逐节点拿到增量更新——每走一个节点就能看到「谁在动、它产出什么」，
    而不是等全图跑完才给一个最终答案。调试 Agent 的第一手工具就是这条轨迹。

为什么同时用 updates 和 values 两种流模式:
    updates 给「这个节点刚产出了什么」（轨迹展示），values 给「跑完后的完整 State」
    （取最终答复）。真 LLM 有成本且非确定，不能流一遍再 invoke 一遍——那是两次调用。
"""

import json
import os
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from lexaudit.graph import build_graph
from lexaudit.llm import PROVIDERS, SYSTEM_PROMPT, build_llm
from lexaudit.tools import TOOLS

# 样例合同路径（从项目根定位，兼容任意工作目录执行）
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "contract_labour.md"

# 单次运行的节点步数上限：模型失控时（反复调工具不收敛）用它兜底，而不是让脚本一直转
RECURSION_LIMIT = 25


def summarize_message(msg, width: int = 60) -> str:
    """把一条消息压缩成一行可读摘要（只用于终端展示，不进业务逻辑）。

    参数:
        msg: langchain 消息对象（Human / AI / Tool 之一）。
        width: content 截断长度。

    返回:
        单行文本；AI 消息额外列出它发起的 tool_calls 及参数——参数是观察模型决策的关键。
    """
    kind = msg.type
    content = str(msg.content).replace("\n", " ")[:width]
    calls = getattr(msg, "tool_calls", None)
    if calls:
        desc = "; ".join(
            f"{tc['name']}({json.dumps(tc.get('args') or {}, ensure_ascii=False)})" for tc in calls
        )
        return f"[{kind}] {content}…  → {desc}"
    return f"[{kind}] {content}…"


def main() -> int:
    """跑一次完整审查 Loop：读样例 → 建图 → 流式执行 → 打印轨迹与最终答复。

    返回:
        进程退出码（0 成功，1 配置或调用失败）。
    """
    try:
        llm = build_llm()
    except (ValueError, RuntimeError) as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 1

    provider = os.getenv("LLM_PROVIDER") or "deepseek"
    model = os.getenv("LLM_MODEL") or PROVIDERS[provider][1]

    contract_text = SAMPLE.read_text(encoding="utf-8")
    print(f"已加载样例合同: {SAMPLE.name}（{len(contract_text)} 字符）")
    print(f"LLM: {provider} / {model}\n")

    print("=== 工具菜单（由 @tool 自动导出，bind_tools 发给模型的就是这份）===")
    print(json.dumps([convert_to_openai_tool(t) for t in TOOLS], ensure_ascii=False, indent=2))
    print()

    graph = build_graph(llm, SYSTEM_PROMPT)

    initial_state = {
        "messages": [HumanMessage(content="请审查这份劳动合同，指出存在的问题。")],
        "contract_text": contract_text,
    }

    print("=== 执行轨迹（每块 = 一个节点刚跑完）===")
    started = time.perf_counter()
    final_state: dict = initial_state
    step = 0
    usage = {"in": 0, "out": 0}
    try:
        for mode, chunk in graph.stream(
            initial_state,
            stream_mode=["updates", "values"],
            config={"recursion_limit": RECURSION_LIMIT},
        ):
            if mode == "values":
                final_state = chunk
                continue
            for node_name, update in chunk.items():
                step += 1
                elapsed = time.perf_counter() - started
                print(f"\n-- 步骤 {step} · 节点 [{node_name}]（+{elapsed:.1f}s）")
                for msg in update.get("messages", []):
                    print("   " + summarize_message(msg))
                    meta = getattr(msg, "usage_metadata", None) or {}
                    usage["in"] += meta.get("input_tokens", 0)
                    usage["out"] += meta.get("output_tokens", 0)
    except Exception as exc:  # noqa: BLE001 —— 顶层入口：任何调用失败都要给用户可读信息
        print(f"\n[运行失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  已消耗的步骤见上方轨迹；检查网络 / API key / 余额后重试。", file=sys.stderr)
        return 1

    total = time.perf_counter() - started
    print("\n=== 最终答复 ===")
    print(final_state["messages"][-1].content)

    parsed = final_state.get("parsed")
    if parsed is not None:
        print(f"\n[State.parsed] 已就位：{len(parsed.clauses)} 个条款（程序视角数据，供后续工具使用）")
    else:
        print("\n[State.parsed] 未就位 —— 模型这一轮没调用 parse_document，后续工具将无法工作。")

    tool_calls = sum(1 for m in final_state["messages"] if getattr(m, "tool_calls", None))
    print(f"\n[统计] 节点步数 {step} · 发起工具调用的 AI 消息 {tool_calls} 条 · 总耗时 {total:.1f}s")
    if usage["in"] or usage["out"]:
        print(f"[统计] token 用量：输入 {usage['in']} / 输出 {usage['out']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
