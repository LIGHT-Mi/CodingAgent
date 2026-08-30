"""run_command 的执行边界。"""

from __future__ import annotations

from pathlib import Path

from app.agent.contracts import ToolResult
from app.tools.command_contracts import RUN_COMMAND_TOOL_NAME, RunCommandArguments
from app.tools.command_executor import CommandExecutor, CommandProcessStartError
from app.tools.command_results import CommandResultBuilder
from app.tools.local_tool import LocalTool


class RunCommandTool(LocalTool):
    """执行已准备命令，并只标准化可预期的进程创建失败。"""

    name = RUN_COMMAND_TOOL_NAME
    arguments_type = RunCommandArguments

    def __init__(
        self,
        executor: CommandExecutor,
        result_builder: CommandResultBuilder | None = None,
    ) -> None:
        if not isinstance(executor, CommandExecutor):
            raise TypeError("executor must be a CommandExecutor")
        if result_builder is None:
            result_builder = CommandResultBuilder()
        if not isinstance(result_builder, CommandResultBuilder):
            raise TypeError("result_builder must be a CommandResultBuilder")
        self._executor = executor
        self._result_builder = result_builder

    def execute(
        self,
        tool_call_id: str,
        arguments: RunCommandArguments,
        resolved_cwd: Path,
    ) -> ToolResult:
        """返回 Observation；内部执行和清理故障继续抛给 Agent Runtime。"""

        if not isinstance(tool_call_id, str):
            raise TypeError("tool_call_id must be a string")
        if not tool_call_id.strip():
            raise ValueError("tool_call_id must not be blank")
        if not isinstance(arguments, RunCommandArguments):
            raise TypeError("arguments must be RunCommandArguments")
        if not isinstance(resolved_cwd, Path):
            raise TypeError("resolved_cwd must be a pathlib.Path")

        try:
            execution = self._executor.execute(arguments, resolved_cwd)
        except CommandProcessStartError as exc:
            return self._result_builder.build_error(
                tool_call_id,
                str(exc),
                argv=arguments.command,
                cwd=resolved_cwd,
            )
        return self._result_builder.build_execution(tool_call_id, execution)
