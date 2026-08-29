"""三个只读文件工具的模型可见 Schema。

Schema 在模型网关工厂中注册；本模块只负责定义，不连接模型或执行本地工具。
"""

from __future__ import annotations

from app.llm.contracts import LLMToolSchema
from app.tools.contracts import DEFAULT_TOOL_PATH


LIST_FILES_SCHEMA = LLMToolSchema(
    name="list_files",
    description=(
        "列出 Workspace 内目录的直接子项，不递归。结果按 Workspace 相对路径"
        "稳定排序，每行格式为 relative_path<TAB>file|directory|symlink|other。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "default": DEFAULT_TOOL_PATH,
                "description": "Workspace 内的目录路径，默认为当前 Workspace。",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)


READ_FILE_SCHEMA = LLMToolSchema(
    name="read_file",
    description=(
        "读取 Workspace 内一个 UTF-8 文本文件，返回保持原始换行的完整文本。"
        "不支持二进制文件或非 UTF-8 文件。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace 内要读取的文件路径。",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)


SEARCH_FILES_SCHEMA = LLMToolSchema(
    name="search_files",
    description=(
        "在 Workspace 内按字面文本递归搜索 UTF-8 文本文件，不使用正则表达式。"
        "同一匹配行只返回一次，结果按相对路径和行号稳定排序，每行格式为 "
        "relative_path:line_number:matching_text。没有匹配是正常结果。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "要查找的非空字面文本。",
            },
            "path": {
                "type": "string",
                "minLength": 1,
                "default": DEFAULT_TOOL_PATH,
                "description": "Workspace 内的搜索起点，默认为当前 Workspace。",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


READ_ONLY_FILE_TOOL_SCHEMAS = (
    LIST_FILES_SCHEMA,
    READ_FILE_SCHEMA,
    SEARCH_FILES_SCHEMA,
)
