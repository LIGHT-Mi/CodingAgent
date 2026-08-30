"""为本地命令构造不包含服务端凭据的最小子进程环境。"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


DEFAULT_COMMAND_LOCALE = "C.UTF-8"
_LOCALE_KEYS = ("LANG", "LC_ALL", "LC_CTYPE")


class CommandEnvironmentBuilder:
    """从父进程环境的明确允许集中构造只读子进程环境。"""

    def __init__(
        self,
        source_environment: Mapping[str, str] | None = None,
    ) -> None:
        self._include_current_python_directory = source_environment is None
        source = os.environ if source_environment is None else source_environment
        if not isinstance(source, Mapping):
            raise TypeError("source_environment must be a mapping")
        self._source_environment = dict(source)

    def build(self) -> Mapping[str, str]:
        """返回 PATH、locale 和临时目录组成的只读环境快照。

        方法不接受模型参数；API Key、数据库连接、HOME、PYTHONPATH 以及其他未在
        允许集中的父进程变量不会进入返回值。
        """

        command_path = self._read_optional("PATH") or os.defpath
        if self._include_current_python_directory:
            python_directory = str(Path(sys.executable).parent)
            path_entries = command_path.split(os.pathsep)
            if python_directory not in path_entries:
                command_path = os.pathsep.join((python_directory, command_path))

        environment = {
            "PATH": command_path,
            "TMPDIR": str(self._resolve_temporary_directory()),
        }

        copied_locale = False
        for key in _LOCALE_KEYS:
            value = self._read_optional(key)
            if value is None:
                continue
            environment[key] = value
            copied_locale = True
        if not copied_locale:
            environment["LANG"] = DEFAULT_COMMAND_LOCALE

        return MappingProxyType(environment)

    def _resolve_temporary_directory(self) -> Path:
        configured = self._read_optional("TMPDIR")
        if configured is not None:
            try:
                configured_path = Path(configured).resolve(strict=True)
            except (OSError, RuntimeError, ValueError):
                configured_path = None
            if configured_path is not None and configured_path.is_dir():
                return configured_path

        fallback = Path(tempfile.gettempdir()).resolve(strict=True)
        if not fallback.is_dir():
            raise RuntimeError("system temporary directory is not a directory")
        return fallback

    def _read_optional(self, key: str) -> str | None:
        value = self._source_environment.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"environment variable {key} must be a string")
        if not value.strip():
            return None
        return value
