"""parse_document —— 解析工具：合同原文 → 结构化文档（当事人信息 + 条款列表）。

设计原则（蓝图 §7）：
    1. 工具是「纯函数」：不调 LLM、无副作用，同输入同输出。
    2. 结构化信息（条号、标题）用代码「确定性」获取，绝不让 LLM 从纯文本「概率性」地抽
       （旧版决策 D6）——能用代码拿到的结构绝不用 LLM。
    3. 环境数据做首参由 State 注入；本工具没有 LLM 决策参数（合同全文不是 LLM 的决策）。

工具结果的「两种归宿」（Phase 2 新增的核心概念）：
    - 给 LLM 看：观察文本（format_document_as_observation 的输出，进 messages）；
    - 给后续工具用：ParsedDocument 结构化对象（由 tools 节点写入 State.parsed）。
    同一次解析服务两类消费者：观察是消息、数据是 State——LLM 视角与程序视角分开。
"""

import re

from pydantic import BaseModel, Field

# 条款编号行的匹配模式：如「第一条 合同期限」「第十条、合同生效」。
# 中文数字覆盖常见写法，更大数字按需扩展（克制原则：现在加就是过度设计）。
_CLAUSE_RE = re.compile(r"^第[一二三四五六七八九十]{1,3}条[、\s]*(.*)$")


class Clause(BaseModel):
    """解析出的单个条款。

    字段说明：
        id: 条款在本次解析中的稳定标识（如 "c1"），供后续阶段的工具定位引用。
        number: 条款原文编号（如 "第一条"）。
        title: 条款标题（编号行里跟在条号后面的部分，可能为空字符串）。
        text: 条款正文（不含编号行本身；后续行按原样累积）。
    """

    id: str
    number: str
    title: str
    text: str


class ParsedDocument(BaseModel):
    """parse_document 的结构化输出（给「后续工具」消费的程序视角数据）。

    字段说明：
        preamble: 第一个条款编号行之前的全部文本（当事人信息等前言）。
        clauses: 按文档顺序排列的条款列表。

    为什么 preamble 单独存：Phase 1 曾把这段文本整体丢弃（已知限制①），
    本阶段修复——当事人信息是后续「主体一致性」审查的原料，丢失即不可审。
    """

    preamble: str
    clauses: list[Clause]


class ParseDocumentArgs(BaseModel):
    """parse_document 的参数 schema。

    刻意定义为空 schema（无参数）：LLM 调用这个工具时不需要传达任何决策信息，
    合同文本由 tools 节点执行时从 State.contract_text 读取。
    保留空类的原因：function calling 协议要求每个工具有明确的 JSON Schema（空对象）。
    """

    placeholder: str = Field(default="", description="保留字段，无实际用途")


# ---- 工具清单（manifest）：将来 bind_tools 发给真模型的「菜单」 ----
# 为什么手写而不用 @tool 装饰器：Phase 2 的教学目标是先看清 function calling 协议的原料
# ——name / description / parameters 就是工具声明的全部。@tool 只是把这三样从
# 函数签名和 docstring 自动推导出来的语法糖；先见过原料，再认识糖。
# MockLLM 不消费这份菜单（它是规则剧本），但接真模型时这份声明零改动可用。
TOOL_PARSE_DOCUMENT = {
    "type": "function",
    "function": {
        "name": "parse_document",
        "description": (
            "解析合同原文，返回当事人信息与全部条款的结构清单"
            "（编号/标题/正文预览）。审查任何合同前必须先调用本工具。"
        ),
        "parameters": ParseDocumentArgs.model_json_schema(),
    },
}


def parse_document(text: str) -> ParsedDocument:
    """把合同原文解析成结构化文档（纯函数，不调 LLM、无副作用）。

    参数:
        text: 合同原文（markdown 纯文本）。

    返回:
        ParsedDocument：preamble（条款编号行之前的全部文本）+ clauses（条款列表）。

    具体做了什么（分步）:
        1. 逐行扫描原文；
        2. 见到第一个「第X条」行之前，所有行累积进 preamble；
        3. 匹配到编号行时，关闭上一条、开新条款（id 按 c1/c2/... 递增分配）；
        4. 非编号行追加到「当前条款」的正文；
        5. 返回 ParsedDocument(preamble, clauses)。

    为什么这么设计:
        「两段式状态机」（先收前言、再收条款）比 Phase 1 版只多一个阶段变量，
        就把已知限制①修掉了——修复成本正好用来演示「限制是设计欠账，不是缺陷」。
    """
    preamble_lines: list[str] = []
    clauses: list[Clause] = []
    current: Clause | None = None
    for line in text.splitlines():
        matched = _CLAUSE_RE.match(line.strip())
        if matched:
            if current is not None:
                clauses.append(current)
            current = Clause(
                id=f"c{len(clauses) + 1}",
                number=line.strip().split()[0],  # 形如「第一条」
                title=matched.group(1).strip(),
                text="",
            )
        elif current is not None:
            current.text += line + "\n"
        else:
            # 还没遇到第一个条款编号行：这段是当事人信息等前言
            preamble_lines.append(line)
    if current is not None:
        clauses.append(current)
    return ParsedDocument(preamble="\n".join(preamble_lines).strip(), clauses=clauses)


def format_document_as_observation(doc: ParsedDocument) -> str:
    """把解析结果序列化为喂给 LLM 的「观察」文本（工具结果的字符串表示）。

    参数:
        doc: parse_document 的返回值。

    返回:
        适合放入 ToolMessage 的多行文本：当事人信息摘要 + 条款总数 + 逐条「编号 标题 | 预览」。

    具体做了什么（分步）:
        1. preamble 非空时给出当事人信息（换行压成单行预览）；
        2. 条款总数一行（全局概览）；
        3. 每条一行：编号、标题、正文前 60 个字符。

    为什么这么设计:
        观察（ToolMessage 内容）是 LLM 唯一能看到的工具结果，必须自解释；
        但全文塞进去会让上下文膨胀，所以给「结构概览 + 预览」——
        需要全文时 LLM 应调用 read_clause 按条号取，这就是「按需读取」的 context 管理。
    """
    lines: list[str] = []
    if doc.preamble:
        preview = doc.preamble.replace("\n", " ")[:80]
        lines.append(f"当事人信息：{preview}")
    lines.append(f"解析完成，共 {len(doc.clauses)} 个条款：")
    for c in doc.clauses:
        text_preview = c.text.strip().replace("\n", " ")[:60]
        lines.append(f"  {c.number} {c.title} | {text_preview}")
    return "\n".join(lines)
