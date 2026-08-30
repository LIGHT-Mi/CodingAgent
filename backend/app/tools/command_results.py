"""把命令准备及执行结果统一构造成模型可见的 ToolResult。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from app.agent.contracts import ToolResult, ToolResultStatus
from app.tools.command_contracts import RUN_COMMAND_TOOL_NAME
from app.tools.command_executor import CommandExecutionResult


class CommandResultBuilder:
    """集中维护命令 Observation 的文本和 metadata 结构。"""

    def build_execution(
        self,
        tool_call_id: str,
        execution: CommandExecutionResult,
    ) -> ToolResult:
        """正常退出和非零退出均为 COMPLETED，完成清理的超时为 TIMEOUT。"""

        _require_non_blank(tool_call_id, "tool_call_id")
        if not isinstance(execution, CommandExecutionResult):
            raise TypeError("execution must be CommandExecutionResult")

        status = (
            ToolResultStatus.TIMEOUT
            if execution.timed_out
            else ToolResultStatus.COMPLETED
        )
        error = (
            "command timed out after "
            f"{execution.timeout_seconds:g} seconds and its process group was "
            "terminated"
            if execution.timed_out
            else None
        )
        metadata = _execution_metadata(execution)
        content = _format_command_observation(
            argv=execution.argv,
            cwd=execution.cwd,
            status=status,
            exit_code=execution.exit_code,
            timed_out=execution.timed_out,
            timeout_seconds=execution.timeout_seconds,
            duration_seconds=execution.duration_seconds,
            stdout=execution.stdout.text,
            stderr=execution.stderr.text,
            stdout_truncated=execution.stdout.truncated,
            stderr_truncated=execution.stderr.truncated,
            termination_signal=execution.termination_signal,
            forced_termination=execution.forced_termination,
            error=error,
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=RUN_COMMAND_TOOL_NAME,
            status=status,
            content=content,
            error=error,
            metadata=metadata,
        )

    def build_error(
        self,
        tool_call_id: str,
        error: str,
        *,
        argv: tuple[str, ...] | None = None,
        cwd: str | Path | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ToolResult:
        """构造参数、目录或进程创建失败的普通 ERROR Observation。"""

        return self._build_pre_execution_result(
            tool_call_id,
            error,
            status=ToolResultStatus.ERROR,
            argv=argv,
            cwd=cwd,
            metadata=metadata,
        )

    def build_rejected(
        self,
        tool_call_id: str,
        error: str,
        *,
        argv: tuple[str, ...] | None = None,
        cwd: str | Path | None = None,
        details: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ToolResult:
        """构造安全策略或 Workspace 边界拒绝的普通 REJECTED Observation。"""

        return self._build_pre_execution_result(
            tool_call_id,
            error,
            status=ToolResultStatus.REJECTED,
            argv=argv,
            cwd=cwd,
            details=details,
            metadata=metadata,
        )

    def _build_pre_execution_result(
        self,
        tool_call_id: str,
        error: str,
        *,
        status: ToolResultStatus,
        argv: tuple[str, ...] | None,
        cwd: str | Path | None,
        details: str | None = None,
        metadata: Mapping[str, object] | None,
    ) -> ToolResult:
        _require_non_blank(tool_call_id, "tool_call_id")
        _require_non_blank(error, "error")
        normalized_argv = _validate_optional_argv(argv)
        normalized_cwd = _validate_optional_cwd(cwd)
        if details is not None:
            _require_non_blank(details, "details")

        base_metadata: dict[str, object] = {
            "status": status.value,
            "argv": list(normalized_argv) if normalized_argv is not None else None,
            "cwd": normalized_cwd,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "timeout_seconds": None,
            "duration_seconds": None,
            "stdout_original_byte_count": 0,
            "stdout_retained_byte_count": 0,
            "stdout_discarded_byte_count": 0,
            "stdout_truncated": False,
            "stderr_original_byte_count": 0,
            "stderr_retained_byte_count": 0,
            "stderr_discarded_byte_count": 0,
            "stderr_truncated": False,
            "termination_signal": None,
            "forced_termination": False,
        }
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping or None")
            for key, value in metadata.items():
                if key in base_metadata and base_metadata[key] != value:
                    raise ValueError(
                        f"metadata cannot override normalized command field {key!r}"
                    )
                base_metadata[key] = value

        content = _format_command_observation(
            argv=normalized_argv,
            cwd=normalized_cwd,
            status=status,
            exit_code=None,
            timed_out=False,
            timeout_seconds=None,
            duration_seconds=None,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            termination_signal=None,
            forced_termination=False,
            error=error,
            details=details,
        )
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=RUN_COMMAND_TOOL_NAME,
            status=status,
            content=content,
            error=error,
            metadata=base_metadata,
        )


def _execution_metadata(execution: CommandExecutionResult) -> dict[str, object]:
    return {
        "status": (
            ToolResultStatus.TIMEOUT.value
            if execution.timed_out
            else ToolResultStatus.COMPLETED.value
        ),
        "argv": list(execution.argv),
        "cwd": str(execution.cwd),
        "exit_code": execution.exit_code,
        "stdout": execution.stdout.text,
        "stderr": execution.stderr.text,
        "timeout": execution.timed_out,
        "timeout_seconds": execution.timeout_seconds,
        "duration_seconds": execution.duration_seconds,
        "stdout_original_byte_count": execution.stdout.original_byte_count,
        "stdout_retained_byte_count": execution.stdout.retained_byte_count,
        "stdout_discarded_byte_count": execution.stdout.discarded_byte_count,
        "stdout_truncated": execution.stdout.truncated,
        "stderr_original_byte_count": execution.stderr.original_byte_count,
        "stderr_retained_byte_count": execution.stderr.retained_byte_count,
        "stderr_discarded_byte_count": execution.stderr.discarded_byte_count,
        "stderr_truncated": execution.stderr.truncated,
        "termination_signal": execution.termination_signal,
        "forced_termination": execution.forced_termination,
    }


def _format_command_observation(
    *,
    argv: tuple[str, ...] | None,
    cwd: str | Path | None,
    status: ToolResultStatus,
    exit_code: int | None,
    timed_out: bool,
    timeout_seconds: float | None,
    duration_seconds: float | None,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
    termination_signal: int | None,
    forced_termination: bool,
    error: str | None,
    details: str | None = None,
) -> str:
    command_text = (
        json.dumps(argv, ensure_ascii=True, separators=(",", ":"))
        if argv is not None
        else "unavailable"
    )
    lines = [
        "Command observation",
        f"command: {command_text}",
        f"cwd: {cwd if cwd is not None else 'unavailable'}",
        f"status: {status.value}",
        f"exit_code: {exit_code if exit_code is not None else 'unavailable'}",
        f"timeout: {str(timed_out).lower()}",
        (
            f"timeout_seconds: {timeout_seconds:g}"
            if timeout_seconds is not None
            else "timeout_seconds: unavailable"
        ),
        (
            f"duration_seconds: {duration_seconds:.6f}"
            if duration_seconds is not None
            else "duration_seconds: unavailable"
        ),
        (
            f"termination_signal: {termination_signal}"
            if termination_signal is not None
            else "termination_signal: none"
        ),
        f"forced_termination: {str(forced_termination).lower()}",
        f"stdout_truncated: {str(stdout_truncated).lower()}",
        "stdout:",
        stdout,
        f"stderr_truncated: {str(stderr_truncated).lower()}",
        "stderr:",
        stderr,
    ]
    if details is not None:
        lines.extend(("details:", details))
    if error is not None:
        lines.append(f"error: {error}")
    return "\n".join(lines)


def _validate_optional_argv(
    argv: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if argv is None:
        return None
    if (
        not isinstance(argv, tuple)
        or not argv
        or any(not isinstance(argument, str) for argument in argv)
    ):
        raise TypeError("argv must be a non-empty tuple of strings or None")
    return argv


def _validate_optional_cwd(cwd: str | Path | None) -> str | None:
    if cwd is None:
        return None
    if not isinstance(cwd, (str, Path)):
        raise TypeError("cwd must be a string, pathlib.Path, or None")
    normalized = str(cwd)
    _require_non_blank(normalized, "cwd")
    return normalized


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
