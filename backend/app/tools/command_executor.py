"""POSIX 平台上的有界本地命令执行器。"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from app.tools.command_contracts import RunCommandArguments
from app.tools.command_environment import CommandEnvironmentBuilder
from app.tools.process_output import (
    CollectedProcessOutput,
    CollectedProcessStream,
    ProcessOutputCollectionError,
    ProcessOutputCollector,
)


class CommandExecutionError(RuntimeError):
    """命令执行基础设施发生无法直接表示为退出码的错误。"""


class CommandProcessStartError(CommandExecutionError):
    """操作系统未能创建命令子进程。"""


class CommandProcessTerminationError(CommandExecutionError):
    """超时或异常清理时无法确认整个命令进程组已经退出。"""


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    """一个已启动命令在进程及输出全部闭合后的原始执行结果。"""

    argv: tuple[str, ...]
    cwd: Path
    exit_code: int
    output: CollectedProcessOutput
    timed_out: bool
    timeout_seconds: float
    duration_seconds: float
    termination_signal: int | None = None
    forced_termination: bool = False

    @property
    def stdout(self) -> CollectedProcessStream:
        """返回 stdout 的有界文本和字节统计。"""

        return self.output.stdout

    @property
    def stderr(self) -> CollectedProcessStream:
        """返回 stderr 的有界文本和字节统计。"""

        return self.output.stderr

    def __post_init__(self) -> None:
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or any(not isinstance(argument, str) for argument in self.argv)
        ):
            raise TypeError("argv must be a non-empty tuple of strings")
        if not isinstance(self.cwd, Path):
            raise TypeError("cwd must be a pathlib.Path")
        if not self.cwd.is_absolute():
            raise ValueError("cwd must be absolute")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise TypeError("exit_code must be an integer")
        if not isinstance(self.output, CollectedProcessOutput):
            raise TypeError("output must be CollectedProcessOutput")
        if not isinstance(self.timed_out, bool):
            raise TypeError("timed_out must be a boolean")
        _require_positive_finite_number(
            self.timeout_seconds,
            "timeout_seconds",
        )
        _require_non_negative_finite_number(
            self.duration_seconds,
            "duration_seconds",
        )
        if self.termination_signal is not None:
            if (
                not isinstance(self.termination_signal, int)
                or isinstance(self.termination_signal, bool)
            ):
                raise TypeError("termination_signal must be an integer or None")
            if self.termination_signal <= 0:
                raise ValueError("termination_signal must be positive")
            if not self.timed_out:
                raise ValueError("termination_signal requires a timed out command")
        if not isinstance(self.forced_termination, bool):
            raise TypeError("forced_termination must be a boolean")
        if self.forced_termination:
            if not self.timed_out:
                raise ValueError("forced_termination requires a timed out command")
            if self.termination_signal != signal.SIGKILL:
                raise ValueError("forced_termination must use SIGKILL")


class CommandExecutor:
    """使用独立 POSIX session 执行命令并闭合其进程组与输出管道。"""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        termination_grace_seconds: float,
        max_output_bytes_per_stream: int,
        environment_builder: CommandEnvironmentBuilder | None = None,
    ) -> None:
        self._timeout_seconds = _require_positive_finite_number(
            timeout_seconds,
            "timeout_seconds",
        )
        self._termination_grace_seconds = _require_positive_finite_number(
            termination_grace_seconds,
            "termination_grace_seconds",
        )
        if environment_builder is None:
            environment_builder = CommandEnvironmentBuilder()
        if not isinstance(environment_builder, CommandEnvironmentBuilder):
            raise TypeError(
                "environment_builder must be CommandEnvironmentBuilder"
            )
        self._environment_builder = environment_builder

        # ProcessOutputCollector 负责该配置的整数、布尔值和最小值检查。
        ProcessOutputCollector(max_output_bytes_per_stream)
        self._max_output_bytes_per_stream = max_output_bytes_per_stream

    def execute(
        self,
        arguments: RunCommandArguments,
        resolved_cwd: Path,
    ) -> CommandExecutionResult:
        """执行已经通过参数、cwd 和安全策略检查的命令。"""

        if not isinstance(arguments, RunCommandArguments):
            raise TypeError("arguments must be RunCommandArguments")
        _require_absolute_path(resolved_cwd)
        if not resolved_cwd.is_dir():
            raise CommandProcessStartError(
                "command working directory no longer exists or is not a directory: "
                f"{resolved_cwd}"
            )

        started_at = time.monotonic()
        try:
            process = subprocess.Popen(
                arguments.command,
                cwd=str(resolved_cwd),
                env=dict(self._environment_builder.build()),
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            detail = str(exc).strip()
            suffix = f": {detail}" if detail else ""
            raise CommandProcessStartError(
                f"failed to start command {arguments.command[0]!r}{suffix}"
            ) from exc

        collector = ProcessOutputCollector(self._max_output_bytes_per_stream)
        try:
            collector.start(process.stdout, process.stderr)
        except Exception as exc:
            self._clean_up_started_process(process)
            _close_process_pipes(process)
            raise CommandExecutionError(
                "failed to start command output collection"
            ) from exc

        timed_out = False
        termination_signal: int | None = None
        forced_termination = False
        execution_error: Exception | None = None
        try:
            deadline = started_at + self._timeout_seconds
            try:
                _wait_for_process_group_until(process, deadline)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_signal, forced_termination = (
                    self._terminate_timed_out_process_group(process)
                )
        except Exception as exc:
            execution_error = exc
            try:
                self._clean_up_started_process(process)
            except Exception as cleanup_exc:
                execution_error = CommandProcessTerminationError(
                    "command execution failed and process-group cleanup also failed"
                )
                execution_error.__cause__ = cleanup_exc

        try:
            output = collector.finish()
        except ProcessOutputCollectionError as exc:
            if execution_error is None:
                execution_error = exc
        finally:
            _close_process_pipes(process)

        if execution_error is not None:
            if isinstance(execution_error, CommandExecutionError):
                raise execution_error
            raise CommandExecutionError("command execution failed") from execution_error

        exit_code = process.returncode
        if exit_code is None:
            raise CommandExecutionError(
                "command process has no exit code after output collection"
            )

        return CommandExecutionResult(
            argv=arguments.command,
            cwd=resolved_cwd,
            exit_code=exit_code,
            output=output,
            timed_out=timed_out,
            timeout_seconds=self._timeout_seconds,
            duration_seconds=time.monotonic() - started_at,
            termination_signal=termination_signal,
            forced_termination=forced_termination,
        )

    def _terminate_timed_out_process_group(
        self,
        process: subprocess.Popen[bytes],
    ) -> tuple[int | None, bool]:
        """先发送 SIGTERM，宽限期后仍存活则发送 SIGKILL。"""

        termination_signal = _signal_process_group(process, signal.SIGTERM)
        grace_deadline = time.monotonic() + self._termination_grace_seconds
        try:
            _wait_for_process_group_until(process, grace_deadline)
            return termination_signal, False
        except subprocess.TimeoutExpired:
            kill_signal = _signal_process_group(process, signal.SIGKILL)
            try:
                process.wait()
            except OSError as exc:
                raise CommandProcessTerminationError(
                    "failed to reap command after SIGKILL"
                ) from exc
            return kill_signal or termination_signal, kill_signal == signal.SIGKILL

    def _clean_up_started_process(
        self,
        process: subprocess.Popen[bytes],
    ) -> None:
        """内部错误发生后以同样的 TERM/KILL 顺序回收已经启动的进程。"""

        process.poll()
        if not _process_group_exists(process.pid):
            process.wait()
            return
        _signal_process_group(process, signal.SIGTERM)
        cleanup_deadline = time.monotonic() + self._termination_grace_seconds
        try:
            _wait_for_process_group_until(process, cleanup_deadline)
            return
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
        try:
            process.wait()
        except OSError as exc:
            raise CommandProcessTerminationError(
                "failed to reap command during process-group cleanup"
            ) from exc


def _wait_for_process_group_until(
    process: subprocess.Popen[bytes],
    deadline: float,
) -> int:
    """等待 session 中所有进程退出，而不只等待最初创建的进程。"""

    while True:
        process.poll()
        if not _process_group_exists(process.pid):
            return process.wait()

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout=0)
        time.sleep(min(0.01, remaining_seconds))


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(
    process: subprocess.Popen[bytes],
    requested_signal: int,
) -> int | None:
    """向 start_new_session 创建的整个进程组发送信号。"""

    try:
        os.killpg(process.pid, requested_signal)
    except ProcessLookupError:
        # 超时判定与发信号之间进程可能已经自然退出。
        return None
    except OSError as exc:
        raise CommandProcessTerminationError(
            f"failed to signal command process group with {requested_signal}"
        ) from exc
    return requested_signal


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _require_absolute_path(path: Path) -> None:
    if not isinstance(path, Path):
        raise TypeError("resolved_cwd must be a pathlib.Path")
    if not path.is_absolute():
        raise ValueError("resolved_cwd must be absolute")


def _require_positive_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be positive and finite")
    return normalized


def _require_non_negative_finite_number(
    value: float,
    field_name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{field_name} must be non-negative and finite")
