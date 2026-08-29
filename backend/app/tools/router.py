"""模型 Tool Call 到本地执行器的两阶段路由。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.agent.contracts import ToolCallRequest, ToolResult, ToolResultStatus
from app.tools.contracts import (
    ListFilesArguments,
    ReadFileArguments,
    SearchFilesArguments,
)
from app.tools.file_tools import FileToolArguments
from app.tools.path_guard import WorkspacePathError, WorkspacePathGuard
from app.tools.registry import FileToolRegistry


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    """已经通过工具查找、参数校验和 Workspace 路径校验的调用。"""

    tool_call_id: str
    tool_name: str
    workspace: Path
    arguments: FileToolArguments
    resolved_path: Path

    def __post_init__(self) -> None:
        _require_non_blank(self.tool_call_id, "tool_call_id")
        _require_non_blank(self.tool_name, "tool_name")
        if not isinstance(self.workspace, Path):
            raise TypeError("workspace must be a pathlib.Path")
        if not isinstance(self.resolved_path, Path):
            raise TypeError("resolved_path must be a pathlib.Path")
        if not isinstance(
            self.arguments,
            (ListFilesArguments, ReadFileArguments, SearchFilesArguments),
        ):
            raise TypeError("arguments must be validated file tool arguments")
        if not self.workspace.is_absolute() or not self.resolved_path.is_absolute():
            raise ValueError("prepared workspace paths must be absolute")
        if not self.resolved_path.is_relative_to(self.workspace):
            raise ValueError("resolved_path must be inside workspace")


PrepareToolResult = PreparedToolCall | ToolResult


class ToolRouter:
    """先准备 Tool Call，再单独启动真实文件操作。"""

    def __init__(
        self,
        registry: FileToolRegistry,
        path_guard: WorkspacePathGuard,
    ) -> None:
        if not isinstance(registry, FileToolRegistry):
            raise TypeError("registry must be a FileToolRegistry")
        if not isinstance(path_guard, WorkspacePathGuard):
            raise TypeError("path_guard must be a WorkspacePathGuard")
        self._registry = registry
        self._path_guard = path_guard

    def prepare(
        self,
        request: ToolCallRequest,
        workspace: str | Path,
    ) -> PrepareToolResult:
        """完成执行前检查；失败结果仍对应 PENDING ToolCall。"""

        if not isinstance(request, ToolCallRequest):
            raise TypeError("request must be a ToolCallRequest")

        tool = self._registry.get(request.tool_name)
        if tool is None:
            return _prepare_error_result(
                request,
                f"unknown tool: {request.tool_name}",
            )

        try:
            arguments = tool.arguments_type(**dict(request.arguments))
        except (TypeError, ValueError) as exc:
            return _prepare_error_result(
                request,
                f"invalid arguments for {request.tool_name}: {exc}",
            )

        try:
            resolved_path = tool.resolve_path(
                self._path_guard,
                workspace,
                arguments,
            )
        except WorkspacePathError as exc:
            return _prepare_error_result(
                request,
                str(exc),
                status=exc.status,
                metadata={"requested_path": arguments.path},
            )

        return PreparedToolCall(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            workspace=Path(workspace),
            arguments=arguments,
            resolved_path=resolved_path,
        )

    def execute(self, prepared_call: PreparedToolCall) -> ToolResult:
        """执行 PreparedToolCall；调用方可在此方法前标记 ToolCall RUNNING。"""

        if not isinstance(prepared_call, PreparedToolCall):
            raise TypeError("prepared_call must be a PreparedToolCall")
        tool = self._registry.get(prepared_call.tool_name)
        if tool is None:
            return ToolResult(
                tool_call_id=prepared_call.tool_call_id,
                tool_name=prepared_call.tool_name,
                status=ToolResultStatus.ERROR,
                error=(
                    "prepared tool is no longer registered: "
                    f"{prepared_call.tool_name}"
                ),
                metadata={},
            )
        return tool.execute(
            prepared_call.tool_call_id,
            prepared_call.workspace,
            prepared_call.arguments,
            prepared_call.resolved_path,
        )


def _prepare_error_result(
    request: ToolCallRequest,
    error: str,
    *,
    status: ToolResultStatus = ToolResultStatus.ERROR,
    metadata: dict[str, object] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id=request.tool_call_id,
        tool_name=request.tool_name,
        status=status,
        error=error,
        metadata={} if metadata is None else metadata,
    )


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
