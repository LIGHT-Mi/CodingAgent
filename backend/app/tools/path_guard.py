"""单次文件工具调用使用的 Workspace 路径保护。"""

from __future__ import annotations

from pathlib import Path

from app.agent.contracts import ToolResultStatus


class WorkspacePathError(ValueError):
    """可以由文件工具转换成普通 ToolResult 的路径错误。"""

    status = ToolResultStatus.ERROR


class WorkspacePathRejectedError(WorkspacePathError):
    """目标真实路径越出当前 Task Workspace。"""

    status = ToolResultStatus.REJECTED


class WorkspacePathNotFoundError(WorkspacePathError):
    """Workspace 内请求的已有路径不存在。"""


class WorkspacePathTypeError(WorkspacePathError):
    """已有目标不是具体工具要求的文件类型。"""


class WorkspacePathConfigurationError(RuntimeError):
    """Task 中保存的 Workspace 已失效或不再是规范绝对目录。"""


class WorkspacePathGuard:
    """解析并限制模型在单次文件工具调用中请求的已有路径。

    本类不负责用户选择 Workspace 时的允许根目录校验，也不读取目标内容。当前
    只支持已经存在的目标；不存在目标及其父目录的写入校验留到第 5 步。
    """

    def resolve_existing(
        self,
        workspace: str | Path,
        requested_path: str | Path,
    ) -> Path:
        """返回 Workspace 内已有目标的规范绝对真实路径。

        相对路径始终以 Task.workspace 为基准。``resolve(strict=False)`` 会解析
        ``..`` 和已有符号链接；先针对解析后的路径做边界判断，再检查存在性，因而
        所有 Workspace 外请求都统一拒绝且不会泄露外部目标是否存在。
        """

        workspace_path = self._require_canonical_workspace(workspace)
        raw_requested_path = _require_path_input(
            requested_path,
            "requested_path",
            WorkspacePathError,
        )
        candidate = (
            raw_requested_path
            if raw_requested_path.is_absolute()
            else workspace_path / raw_requested_path
        )

        try:
            resolved_path = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise WorkspacePathError(
                f"requested path cannot be resolved: {requested_path}"
            ) from exc

        if not resolved_path.is_relative_to(workspace_path):
            raise WorkspacePathRejectedError(
                f"requested path is outside the workspace: {requested_path}"
            )
        if not resolved_path.exists():
            raise WorkspacePathNotFoundError(
                f"requested path does not exist: {requested_path}"
            )
        return resolved_path

    def resolve_existing_file(
        self,
        workspace: str | Path,
        requested_path: str | Path,
    ) -> Path:
        """解析已有路径，并确认具体工具请求的是普通文件。"""

        resolved_path = self.resolve_existing(workspace, requested_path)
        if not resolved_path.is_file():
            raise WorkspacePathTypeError(
                f"requested path is not a file: {requested_path}"
            )
        return resolved_path

    def resolve_existing_directory(
        self,
        workspace: str | Path,
        requested_path: str | Path,
    ) -> Path:
        """解析已有路径，并确认具体工具请求的是目录。"""

        resolved_path = self.resolve_existing(workspace, requested_path)
        if not resolved_path.is_dir():
            raise WorkspacePathTypeError(
                f"requested path is not a directory: {requested_path}"
            )
        return resolved_path

    @staticmethod
    def _require_canonical_workspace(workspace: str | Path) -> Path:
        workspace_path = _require_path_input(
            workspace,
            "workspace",
            WorkspacePathConfigurationError,
        )
        if not workspace_path.is_absolute():
            raise WorkspacePathConfigurationError(
                "task workspace must be a canonical absolute path"
            )
        try:
            resolved_workspace = workspace_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspacePathConfigurationError(
                f"task workspace cannot be resolved: {workspace_path}"
            ) from exc
        if resolved_workspace != workspace_path:
            raise WorkspacePathConfigurationError(
                "task workspace must already be stored as its canonical path"
            )
        if not resolved_workspace.is_dir():
            raise WorkspacePathConfigurationError(
                f"task workspace is not a directory: {resolved_workspace}"
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
