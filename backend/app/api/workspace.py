"""任务创建前使用的 Workspace 路径校验。"""

from __future__ import annotations

from pathlib import Path


class WorkspaceValidationError(ValueError):
    """用户提供的 Workspace 路径不符合安全边界。"""


class WorkspaceConfigurationError(ValueError):
    """允许的 Workspace 根目录配置无效。"""


class WorkspaceValidator:
    """将 Workspace 规范化，并限制在配置的允许根目录中。"""

    def __init__(self, allowed_root: str | Path) -> None:
        self.allowed_root = self._resolve_allowed_root(allowed_root)

    def validate(self, workspace: str | Path) -> Path:
        """返回通过存在性、目录类型和根目录边界校验的绝对路径。"""

        raw_workspace = _require_path_input(
            workspace,
            "workspace",
            WorkspaceValidationError,
        )
        try:
            resolved_workspace = raw_workspace.expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise WorkspaceValidationError(
                f"workspace path cannot be resolved: {raw_workspace}"
            ) from exc

        if not resolved_workspace.exists():
            raise WorkspaceValidationError(
                f"workspace path does not exist: {resolved_workspace}"
            )
        if not resolved_workspace.is_dir():
            raise WorkspaceValidationError(
                f"workspace path is not a directory: {resolved_workspace}"
            )
        if not resolved_workspace.is_relative_to(self.allowed_root):
            raise WorkspaceValidationError(
                "workspace path is outside the allowed root: "
                f"{resolved_workspace}"
            )
        return resolved_workspace

    @staticmethod
    def _resolve_allowed_root(allowed_root: str | Path) -> Path:
        raw_root = _require_path_input(
            allowed_root,
            "allowed_root",
            WorkspaceConfigurationError,
        )
        try:
            resolved_root = raw_root.expanduser().resolve()
        except (OSError, RuntimeError) as exc:
            raise WorkspaceConfigurationError(
                f"allowed workspace root cannot be resolved: {raw_root}"
            ) from exc

        if not resolved_root.exists():
            raise WorkspaceConfigurationError(
                f"allowed workspace root does not exist: {resolved_root}"
            )
        if not resolved_root.is_dir():
            raise WorkspaceConfigurationError(
                f"allowed workspace root is not a directory: {resolved_root}"
            )
        return resolved_root


def _require_path_input(
    value: str | Path,
    field_name: str,
    error_type: type[ValueError],
) -> Path:
    if isinstance(value, str):
        if not value.strip():
            raise error_type(f"{field_name} path is required")
        return Path(value)
    if isinstance(value, Path):
        return value
    raise TypeError(f"{field_name} must be a string or pathlib.Path")
