"""当前七个编程工具的模型可见 Schema。

Schema 在模型网关工厂中注册；本模块只负责定义，不连接模型或执行本地工具。
"""

from __future__ import annotations

from app.llm.contracts import LLMToolSchema
from app.tools.command_contracts import DEFAULT_COMMAND_CWD
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


CREATE_FILE_SCHEMA = LLMToolSchema(
    name="create_file",
    description=(
        "在 Workspace 内创建一个新的 UTF-8 文本文件。目标必须不存在，工具不会"
        "覆盖已有文件；父目录必须已经存在。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace 内要创建的新文件路径。",
            },
            "content": {
                "type": "string",
                "description": "要写入新文件的完整 UTF-8 文本，可以为空。",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)


WRITE_FILE_SCHEMA = LLMToolSchema(
    name="write_file",
    description=(
        "使用给定 UTF-8 文本整体覆盖 Workspace 内的一个已有文本文件。调用前应先"
        "读取文件；这不是局部编辑。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace 内要整体覆盖的已有文件路径。",
            },
            "content": {
                "type": "string",
                "description": "覆盖后文件应具有的完整 UTF-8 文本，可以为空。",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)


EDIT_FILE_SCHEMA = LLMToolSchema(
    name="edit_file",
    description=(
        "在 Workspace 内已有 UTF-8 文本文件中执行一次精确文本替换。old_text 必须"
        "恰好出现一次；找不到或出现多次都会返回错误且不修改文件。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace 内要精确修改的已有文件路径。",
            },
            "old_text": {
                "type": "string",
                "minLength": 1,
                "description": "文件中必须恰好出现一次的原始文本。",
            },
            "new_text": {
                "type": "string",
                "description": "用于替换 old_text 的新文本，可以为空。",
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
)


RUN_COMMAND_SCHEMA = LLMToolSchema(
    name="run_command",
    description=(
        "在 Workspace 内指定工作目录执行一个受安全策略限制的本地命令，用于测试、"
        "构建和项目运行。command 必须按 argv 拆分，不支持 Shell 字符串、管道、"
        "重定向或后台执行。非零退出码、超时和拒绝结果都是需要分析的 Observation。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "minLength": 1,
                },
                "description": "非空 argv 数组，每个元素对应一个命令行参数。",
            },
            "cwd": {
                "type": "string",
                "minLength": 1,
                "default": DEFAULT_COMMAND_CWD,
                "description": "Workspace 内的相对工作目录，默认为 Workspace 根目录。",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
)


CODING_TOOL_SCHEMAS = (
    LIST_FILES_SCHEMA,
    READ_FILE_SCHEMA,
    SEARCH_FILES_SCHEMA,
    CREATE_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    EDIT_FILE_SCHEMA,
    RUN_COMMAND_SCHEMA,
)
