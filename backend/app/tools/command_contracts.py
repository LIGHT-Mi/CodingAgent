"""命令工具使用的纯 Python 参数契约。

本模块只定义并校验模型请求的命令参数，不访问文件系统、不执行子进程，也不依赖
数据库或 Agent Runtime。工作目录的真实路径与 Workspace 边界留给后续 Guard。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMMAND_CWD = "."
RUN_COMMAND_TOOL_NAME = "run_command"


@dataclass(frozen=True, slots=True)
class RunCommandArguments:
    """``run_command`` 的供应商无关参数。

    模型 JSON 中的 command 数组在 Python 边界通常表现为 list；构造后统一保存为
    tuple，确保后续安全校验和执行阶段接收到稳定、不可变的 argv。
    """

    command: tuple[str, ...]
    cwd: str = DEFAULT_COMMAND_CWD

    def __post_init__(self) -> None:
        command = _normalize_command(self.command)
        _validate_relative_cwd(self.cwd)
        object.__setattr__(self, "command", command)


def _normalize_command(command: object) -> tuple[str, ...]:
    if not isinstance(command, (list, tuple)):
        raise TypeError("command must be an array of strings")

    normalized = tuple(command)
    if not normalized:
        raise ValueError("command must contain at least one argument")

    for index, argument in enumerate(normalized):
        if not isinstance(argument, str):
            raise TypeError(f"command[{index}] must be a string")
        if not argument:
            raise ValueError(f"command[{index}] must not be empty")

    if not normalized[0].strip():
        raise ValueError("command[0] must not be blank")
    return normalized


def _validate_relative_cwd(cwd: object) -> None:
    if not isinstance(cwd, str):
        raise TypeError("cwd must be a string")
    if not cwd.strip():
        raise ValueError("cwd must not be blank")
    if Path(cwd).is_absolute():
        raise ValueError("cwd must be relative to the Task Workspace")
