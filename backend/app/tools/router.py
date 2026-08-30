"""模型 Tool Call 到文件或命令执行器的两阶段路由。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.agent.contracts import ToolCallRequest, ToolResult, ToolResultStatus
from app.tools.command_contracts import RunCommandArguments
from app.tools.command_policy import (
    CommandSafetyPolicy,
    CommandSafetyVerdict,
    build_rejected_command_result,
)
from app.tools.command_results import CommandResultBuilder
from app.tools.command_tool import RunCommandTool
from app.tools.contracts import (
    CreateFileArguments,
    EditFileArguments,
    FileToolArguments,
    ListFilesArguments,
    ReadFileArguments,
    SearchFilesArguments,
    WriteFileArguments,
)
from app.tools.file_tools import FileTool
from app.tools.path_guard import WorkspacePathError, WorkspacePathGuard
from app.tools.registry import LocalToolRegistry
from app.tools.working_directory_guard import (
    WorkingDirectoryError,
    WorkingDirectoryGuard,
)


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
    """所有已完成准备阶段的本地工具调用基类。"""

    tool_call_id: str
    tool_name: str
    workspace: Path

    def __post_init__(self) -> None:
        _require_non_blank(self.tool_call_id, "tool_call_id")
        _require_non_blank(self.tool_name, "tool_name")
        if not isinstance(self.workspace, Path):
            raise TypeError("workspace must be a pathlib.Path")
        if not self.workspace.is_absolute():
            raise ValueError("prepared workspace must be absolute")


@dataclass(frozen=True, slots=True)
class PreparedFileToolCall(PreparedToolCall):
    """已完成参数与 Workspace 文件路径校验的调用。"""

    arguments: FileToolArguments
    resolved_path: Path

    def __post_init__(self) -> None:
        super(PreparedFileToolCall, self).__post_init__()
        if not isinstance(
            self.arguments,
            (
                ListFilesArguments,
                ReadFileArguments,
                SearchFilesArguments,
                CreateFileArguments,
                WriteFileArguments,
                EditFileArguments,
            ),
        ):
            raise TypeError("arguments must be validated file tool arguments")
        if not isinstance(self.resolved_path, Path):
            raise TypeError("resolved_path must be a pathlib.Path")
        if not self.resolved_path.is_absolute():
            raise ValueError("resolved_path must be absolute")
        if not self.resolved_path.is_relative_to(self.workspace):
            raise ValueError("resolved_path must be inside workspace")


@dataclass(frozen=True, slots=True)
class PreparedCommandToolCall(PreparedToolCall):
    """已完成参数、cwd 和安全策略校验的命令调用。"""

    arguments: RunCommandArguments
    resolved_cwd: Path

    def __post_init__(self) -> None:
        super(PreparedCommandToolCall, self).__post_init__()
        if not isinstance(self.arguments, RunCommandArguments):
            raise TypeError("arguments must be RunCommandArguments")
        if not isinstance(self.resolved_cwd, Path):
            raise TypeError("resolved_cwd must be a pathlib.Path")
        if not self.resolved_cwd.is_absolute():
            raise ValueError("resolved_cwd must be absolute")
        if not self.resolved_cwd.is_relative_to(self.workspace):
            raise ValueError("resolved_cwd must be inside workspace")


PrepareToolResult = PreparedToolCall | ToolResult


class ToolRouter:
    """准备并执行文件和命令 Tool Call，不处理持久化生命周期。"""

    def __init__(
        self,
        registry: LocalToolRegistry,
        path_guard: WorkspacePathGuard,
        working_directory_guard: WorkingDirectoryGuard | None = None,
        command_safety_policy: CommandSafetyPolicy | None = None,
        command_result_builder: CommandResultBuilder | None = None,
    ) -> None:
        if not isinstance(registry, LocalToolRegistry):
            raise TypeError("registry must be a LocalToolRegistry")
        if not isinstance(path_guard, WorkspacePathGuard):
            raise TypeError("path_guard must be a WorkspacePathGuard")
        if working_directory_guard is None:
            working_directory_guard = WorkingDirectoryGuard()
        if not isinstance(working_directory_guard, WorkingDirectoryGuard):
            raise TypeError(
                "working_directory_guard must be a WorkingDirectoryGuard"
            )
        if command_safety_policy is None:
            command_safety_policy = CommandSafetyPolicy()
        if not isinstance(command_safety_policy, CommandSafetyPolicy):
            raise TypeError(
                "command_safety_policy must be a CommandSafetyPolicy"
            )
        if command_result_builder is None:
            command_result_builder = CommandResultBuilder()
        if not isinstance(command_result_builder, CommandResultBuilder):
            raise TypeError(
                "command_result_builder must be a CommandResultBuilder"
            )
        self._registry = registry
        self._path_guard = path_guard
        self._working_directory_guard = working_directory_guard
        self._command_safety_policy = command_safety_policy
        self._command_result_builder = command_result_builder

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
        if isinstance(tool, FileTool):
            return self._prepare_file_call(request, workspace, tool)
        if isinstance(tool, RunCommandTool):
            return self._prepare_command_call(request, workspace, tool)
        raise RuntimeError(
            f"unsupported registered local tool type: {type(tool).__name__}"
        )

    def execute(self, prepared_call: PreparedToolCall) -> ToolResult:
        """执行 PreparedToolCall；调用方在调用前将 ToolCall 标记为 RUNNING。"""

        if not isinstance(prepared_call, PreparedToolCall):
            raise TypeError("prepared_call must be a PreparedToolCall")
        tool = self._registry.get(prepared_call.tool_name)
        if tool is None:
            raise RuntimeError(
                "prepared local tool is no longer registered: "
                f"{prepared_call.tool_name}"
            )

        if isinstance(prepared_call, PreparedFileToolCall):
            if not isinstance(tool, FileTool):
                raise RuntimeError("prepared file call resolved to a non-file tool")
            return tool.execute(
                prepared_call.tool_call_id,
                prepared_call.workspace,
                prepared_call.arguments,
                prepared_call.resolved_path,
            )
        if isinstance(prepared_call, PreparedCommandToolCall):
            if not isinstance(tool, RunCommandTool):
                raise RuntimeError(
                    "prepared command call resolved to a non-command tool"
                )
            return tool.execute(
                prepared_call.tool_call_id,
                prepared_call.arguments,
                prepared_call.resolved_cwd,
            )
        raise RuntimeError(
            f"unsupported prepared call type: {type(prepared_call).__name__}"
        )

    def _prepare_file_call(
        self,
        request: ToolCallRequest,
        workspace: str | Path,
        tool: FileTool,
    ) -> PrepareToolResult:
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

        return PreparedFileToolCall(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            workspace=Path(workspace),
            arguments=arguments,
            resolved_path=resolved_path,
        )

    def _prepare_command_call(
        self,
        request: ToolCallRequest,
        workspace: str | Path,
        tool: RunCommandTool,
    ) -> PrepareToolResult:
        try:
            arguments = tool.arguments_type(**dict(request.arguments))
        except (TypeError, ValueError) as exc:
            requested_cwd = _extract_requested_cwd(request)
            return self._command_result_builder.build_error(
                request.tool_call_id,
                f"invalid arguments for {request.tool_name}: {exc}",
                argv=_extract_valid_argv(request),
                cwd=requested_cwd,
                metadata={"requested_cwd": requested_cwd},
            )

        try:
            resolved_cwd = self._working_directory_guard.resolve(
                workspace,
                arguments.cwd,
            )
        except WorkingDirectoryError as exc:
            builder = (
                self._command_result_builder.build_rejected
                if exc.status is ToolResultStatus.REJECTED
                else self._command_result_builder.build_error
            )
            return builder(
                request.tool_call_id,
                str(exc),
                argv=arguments.command,
                cwd=arguments.cwd,
                metadata={"requested_cwd": arguments.cwd},
            )

        decision = self._command_safety_policy.evaluate(arguments, resolved_cwd)
        if decision.verdict is CommandSafetyVerdict.REJECT:
            return build_rejected_command_result(
                request.tool_call_id,
                arguments,
                resolved_cwd,
                decision,
            )

        return PreparedCommandToolCall(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            workspace=Path(workspace),
            arguments=arguments,
            resolved_cwd=resolved_cwd,
        )


def _extract_valid_argv(request: ToolCallRequest) -> tuple[str, ...] | None:
    command = request.arguments.get("command")
    if not isinstance(command, (list, tuple)) or not command:
        return None
    if any(not isinstance(argument, str) for argument in command):
        return None
    return tuple(command)


def _extract_requested_cwd(request: ToolCallRequest) -> str | None:
    cwd = request.arguments.get("cwd", ".")
    return cwd if isinstance(cwd, str) and cwd.strip() else None


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
