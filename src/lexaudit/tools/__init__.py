"""工具包：工具实现 + 工具注册表。

TOOLS 是「本项目有哪些工具」的唯一真相来源：图层用它给 ToolNode 建分发，
给真模型的「菜单」由它导出，TOOL_NAMES 白名单由它派生（规划期校验用）。
显式列出即白名单——只有出现在这里的工具才可能被 LLM 调用。
"""

from lexaudit.tools.crossref import clause_cross_ref, clause_cross_ref_tool
from lexaudit.models import ParsedDocument
from lexaudit.tools.parse import parse_document, parse_document_tool
from lexaudit.tools.read import read_clause, read_clause_tool

# 工具注册表：交给 ToolNode 执行、并用于导出「菜单」。
TOOLS = (parse_document_tool, read_clause_tool, clause_cross_ref_tool)

# 工具名白名单：🧱 规划期校验用——planner 生成的任务只能引用这里的名字。
# 为什么自建：官方 ToolNode / ValidationNode 只在「执行期」校验工具名，
# 而我们需要在「规划期」就拦掉引用了不存在工具的任务（框架不提供的那一环）。
TOOL_NAMES = tuple(t.name for t in TOOLS)

__all__ = [
    "TOOLS",
    "TOOL_NAMES",
    "ParsedDocument",
    "parse_document",
    "parse_document_tool",
    "read_clause",
    "read_clause_tool",
    "clause_cross_ref",
    "clause_cross_ref_tool",
]
