"""命令工具工作目录的 Workspace 边界保护。"""

from __future__ import annotations

from pathlib import Path

from app.agent.contracts import ToolResultStatus


class WorkingDirectoryError(ValueError):
    """可以由命令工具路由转换成普通 ERROR ToolResult 的目录错误。"""

    status = ToolResultStatus.ERROR


class WorkingDirectoryRejectedError(WorkingDirectoryError):
    """工作目录的真实路径越出当前 Task Workspace。"""

    status = ToolResultStatus.REJECTED


class WorkingDirectoryNotFoundError(WorkingDirectoryError):
    """Workspace 内请求的工作目录不存在。"""


class WorkingDirectoryTypeError(WorkingDirectoryError):
    """Workspace 内请求的工作目录实际不是目录。"""


class WorkingDirectoryConfigurationError(RuntimeError):
    """Task 中保存的 Workspace 已失效或不再是规范绝对目录。"""


class WorkingDirectoryGuard:
    """解析命令 cwd，并保证其真实路径位于当前 Task Workspace 内。"""

    def resolve(
        self,
        workspace: str | Path,
        requested_cwd: str | Path,
    ) -> Path:
        """返回已有工作目录的规范绝对真实路径。

        相对 cwd 始终以 Task.workspace 为基准。先解析 ``..`` 和已有符号链接，
        再检查 Workspace 边界，最后检查存在性和目录类型；因此 Workspace 外请求
        统一被拒绝，也不会通过错误类型泄露外部路径是否存在。
        """

        workspace_path = self._require_canonical_workspace(workspace)
        raw_cwd = _require_path_input(
            requested_cwd,
            "cwd",
            WorkingDirectoryError,
        )
        candidate = (
            raw_cwd
            if raw_cwd.is_absolute()
            else workspace_path / raw_cwd
        )

        try:
            resolved_cwd = candidate.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkingDirectoryError(
                "working directory cannot be resolved: "
                f"{_format_path_for_error(requested_cwd)}"
            ) from exc

        if not resolved_cwd.is_relative_to(workspace_path):
            raise WorkingDirectoryRejectedError(
                "working directory is outside the workspace: "
                f"{_format_path_for_error(requested_cwd)}"
            )
        if not resolved_cwd.exists():
            raise WorkingDirectoryNotFoundError(
                "working directory does not exist: "
                f"{_format_path_for_error(requested_cwd)}"
            )
        if not resolved_cwd.is_dir():
            raise WorkingDirectoryTypeError(
                "working directory is not a directory: "
                f"{_format_path_for_error(requested_cwd)}"
            )
        return resolved_cwd

    @staticmethod
    def _require_canonical_workspace(workspace: str | Path) -> Path:
        workspace_path = _require_path_input(
            workspace,
            "workspace",
            WorkingDirectoryConfigurationError,
        )
        if not workspace_path.is_absolute():
            raise WorkingDirectoryConfigurationError(
                "task workspace must be a canonical absolute path"
            )
        try:
            resolved_workspace = workspace_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkingDirectoryConfigurationError(
                "task workspace cannot be resolved: "
                f"{_format_path_for_error(workspace_path)}"
            ) from exc
        if resolved_workspace != workspace_path:
            raise WorkingDirectoryConfigurationError(
                "task workspace must already be stored as its canonical path"
            )
        if not resolved_workspace.is_dir():
            raise WorkingDirectoryConfigurationError(
                "task workspace is not a directory: "
                f"{_format_path_for_error(resolved_workspace)}"
            )
        return resolved_workspace


def _require_path_input(
    value: str | Path,
    field_name: str,
    error_type: type[Exception],
) -> Path:
    if isinstance(value, str):
        if not value.strip():
            raise error_type(f"{field_name} path is required")
        return Path(value)
    if isinstance(value, Path):
        return value
    raise TypeError(f"{field_name} must be a string or pathlib.Path")


def _format_path_for_error(value: str | Path) -> str:
    """返回不含原始控制字符或孤立代理字符的路径表示。"""

    return repr(str(value))
