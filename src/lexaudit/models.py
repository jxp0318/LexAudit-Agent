"""领域模型：被多个模块共享的纯数据契约。

独立成模块的原因在依赖方向——state 与 tools 都要用它，若留在 tools 里，
tools 又需要 import AgentState 时就会形成 `state ⇄ tools` 循环导入。
它不 import 任何框架组件，是全项目最稳定、依赖最少的一层。
"""

from pydantic import BaseModel


class Clause(BaseModel):
    """解析出的单个条款。

    id: 本次解析中的稳定标识（如 "c1"）——是「文档内位置」，不是「原文条号」。
    number: 条款原文编号（如 "第一条"）。
    title: 编号行里条号之后的部分，可能为空字符串。
    text: 条文正文，不含编号行本身。
    """

    id: str
    number: str
    title: str
    text: str


class ParsedDocument(BaseModel):
    """parse_document 的结构化输出（给后续工具消费的程序视角数据）。

    preamble: 第一个条款编号行之前的全部文本（当事人信息等前言）。
    clauses: 按文档顺序排列的条款列表。
    """

    preamble: str
    clauses: list[Clause]
