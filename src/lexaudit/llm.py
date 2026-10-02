"""llm —— LLM 接入层（OpenAI 兼容协议）。

能接的所有服务（DeepSeek、通义千问、OpenAI、自建 vLLM…）都走同一套 chat/completions
协议，区别只在 base_url 与模型名——所以配置项就是几个环境变量，换厂商不改代码。

注入方式：build_llm() 的产物只有一个 invoke 方法，而 agent_node 收到的参数类型是
langchain 官方的 Runnable（它要求的也正是「有 invoke」）。所以接任何模型都不必改图，
测试里自备一个带 invoke 的替身也能直接注入——不需要自定义协议或基类。
"""

import os

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from lexaudit.tools import TOOLS

# 预置的 OpenAI 兼容服务：(base_url, 默认模型, 该厂商的 key 环境变量名)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "dashscope": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
        "DASHSCOPE_API_KEY",
    ),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
}

# 审查 Agent 的系统提示词：角色 + 工具用法 + 纪律。
# 刻意不写「该读哪一条」——选哪条条款是模型的决策，写死就把 Agent 退化成了固定流程。
SYSTEM_PROMPT = """你是合同审查助手，按需调用工具完成审查。

可用工具：
- parse_document：解析合同原文，得到当事人信息与全部条款的结构清单。审查任何合同前必须先调用。
- read_clause：按条号读取某条款的完整正文。
- clause_cross_ref：核对某条款正文里引用的其他条款是否存在。

工作方式：
1. 先调用 parse_document 掌握合同结构；
2. 从条款清单里挑出需要细审的条款（交叉引用、金额、期限等），按需调用 read_clause /
   clause_cross_ref 获取证据；
3. 给出审查发现：指出是哪一条、问题是什么、依据是什么。

纪律：证据不足就如实说明「证据不足」，不要编造条款内容或结论。"""


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> Runnable:
    """构造一个「已绑定工具」的真实 LLM，可直接注入 build_graph。

    参数:
        provider: 预置厂商名（见 PROVIDERS）；默认取环境变量 LLM_PROVIDER，再默认 deepseek。
        model: 模型名；默认取 LLM_MODEL，再取该厂商的默认模型。
        api_key: 显式 key；不传则取 LLM_API_KEY，再取该厂商的标准环境变量。
        temperature: 默认 0 —— 审查要稳定可复现，不是创作。
        timeout: 单次请求超时秒数。

    返回:
        Runnable：ChatOpenAI 经 bind_tools(TOOLS) 后的绑定对象，仍只有 invoke 一个入口。

    为什么必须 bind_tools：工具的 JSON Schema 靠这一步才发给模型，不绑定模型就不知道
    有工具可用。注入参数（ToolRuntime / InjectedState）会被框架自动从 schema 剔除，
    模型看不到它们。
    """
    load_dotenv()  # 读项目根 .env（若存在）；已设置的环境变量优先，不会被覆盖

    name = provider or os.getenv("LLM_PROVIDER") or "deepseek"
    if name not in PROVIDERS:
        raise ValueError(f"未知 provider: {name!r}，可选：{list(PROVIDERS)}")
    default_base, default_model, key_env = PROVIDERS[name]

    resolved_key = api_key or os.getenv("LLM_API_KEY") or os.getenv(key_env)
    if not resolved_key:
        raise RuntimeError(
            f"未找到 {name} 的 API key：请设置环境变量 {key_env}（或 LLM_API_KEY），"
            "或复制 .env.example 为 .env 后填写。"
        )

    chat = ChatOpenAI(
        model=model or os.getenv("LLM_MODEL") or default_model,
        base_url=os.getenv("LLM_BASE_URL") or default_base,
        api_key=resolved_key,
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
    )
    return chat.bind_tools(TOOLS)
