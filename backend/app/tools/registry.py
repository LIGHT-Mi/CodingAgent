"""文件与命令执行器共用的本地工具注册表。"""

from __future__ import annotations

from collections.abc import Iterable

from app.tools.command_tool import RunCommandTool
from app.tools.contracts import DEFAULT_FILE_TOOL_LIMITS, FileToolLimits
from app.tools.file_tools import (
    CreateFileTool,
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from app.tools.local_tool import LocalTool


class LocalToolRegistryError(ValueError):
    """本地工具注册表错误。"""


class DuplicateLocalToolError(LocalToolRegistryError):
    """同名本地工具被重复注册。"""


class LocalToolNotFoundError(LocalToolRegistryError):
    """请求的本地工具没有注册。"""


class LocalToolRegistry:
    """按名称保存本地执行器，与模型 ToolSchemaRegistry 保持分离。"""

    def __init__(self, tools: Iterable[LocalTool] = ()) -> None:
        self._tools: dict[str, LocalTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: LocalTool) -> None:
        if not isinstance(tool, LocalTool):
            raise TypeError("tool must be a LocalTool")
        if tool.name in self._tools:
            raise DuplicateLocalToolError(
                f"local tool {tool.name!r} is already registered"
            )
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> LocalTool | None:
        _require_tool_name(tool_name)
        return self._tools.get(tool_name)

    def require(self, tool_name: str) -> LocalTool:
        tool = self.get(tool_name)
        if tool is None:
            raise LocalToolNotFoundError(
                f"local tool {tool_name!r} is not registered"
            )
        return tool

    def get_all(self) -> tuple[LocalTool, ...]:
        return tuple(self._tools.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def __contains__(self, tool_name: object) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def create_local_tool_registry(
    command_tool: RunCommandTool,
    *,
    limits: FileToolLimits = DEFAULT_FILE_TOOL_LIMITS,
) -> LocalToolRegistry:
    """装配当前六个文件工具和一个 run_command 执行器。"""

    if not isinstance(command_tool, RunCommandTool):
        raise TypeError("command_tool must be a RunCommandTool")
    return LocalToolRegistry(
        (
            ListFilesTool(limits),
            ReadFileTool(limits),
            SearchFilesTool(limits),
            CreateFileTool(limits),
            WriteFileTool(limits),
            EditFileTool(limits),
            command_tool,
        )
    )


def _require_tool_name(tool_name: str) -> None:
    if not isinstance(tool_name, str):
        raise TypeError("tool_name must be a string")
    if not tool_name.strip():
        raise ValueError("tool_name must not be blank")
