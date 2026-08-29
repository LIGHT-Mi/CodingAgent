"""可执行本地文件工具注册表。"""

from __future__ import annotations

from collections.abc import Iterable

from app.tools.contracts import (
    DEFAULT_READ_ONLY_FILE_TOOL_LIMITS,
    ReadOnlyFileToolLimits,
)
from app.tools.file_tools import (
    ListFilesTool,
    ReadFileTool,
    ReadOnlyFileTool,
    SearchFilesTool,
)


class FileToolRegistryError(ValueError):
    """文件工具注册表错误。"""


class DuplicateFileToolError(FileToolRegistryError):
    """同名文件工具被重复注册。"""


class FileToolNotFoundError(FileToolRegistryError):
    """请求的文件工具没有注册。"""


class FileToolRegistry:
    """按工具名映射本地执行器，与模型 ToolSchemaRegistry 相互独立。"""

    def __init__(self, tools: Iterable[ReadOnlyFileTool] = ()) -> None:
        self._tools: dict[str, ReadOnlyFileTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ReadOnlyFileTool) -> None:
        if not isinstance(tool, ReadOnlyFileTool):
            raise TypeError("tool must be a ReadOnlyFileTool")
        if tool.name in self._tools:
            raise DuplicateFileToolError(
                f"file tool {tool.name!r} is already registered"
            )
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> ReadOnlyFileTool | None:
        _require_tool_name(tool_name)
        return self._tools.get(tool_name)

    def require(self, tool_name: str) -> ReadOnlyFileTool:
        tool = self.get(tool_name)
        if tool is None:
            raise FileToolNotFoundError(
                f"file tool {tool_name!r} is not registered"
            )
        return tool

    def get_all(self) -> tuple[ReadOnlyFileTool, ...]:
        return tuple(self._tools.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def __contains__(self, tool_name: object) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def create_read_only_file_tool_registry(
    *,
    limits: ReadOnlyFileToolLimits = DEFAULT_READ_ONLY_FILE_TOOL_LIMITS,
) -> FileToolRegistry:
    """使用共享资源上限装配三个只读文件工具执行器。"""

    return FileToolRegistry(
        (
            ListFilesTool(limits),
            ReadFileTool(limits),
            SearchFilesTool(limits),
        )
    )


def _require_tool_name(tool_name: str) -> None:
    if not isinstance(tool_name, str):
        raise TypeError("tool_name must be a string")
    if not tool_name.strip():
        raise ValueError("tool_name must not be blank")
