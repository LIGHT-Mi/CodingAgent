"""三个只读文件工具的本地执行实现。"""

from __future__ import annotations

import os
import stat
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from app.agent.contracts import ToolResult, ToolResultStatus
from app.tools.contracts import (
    DEFAULT_READ_ONLY_FILE_TOOL_LIMITS,
    FileEntryType,
    ListedFile,
    ListFilesArguments,
    ReadFileArguments,
    ReadOnlyFileToolLimits,
    SearchFilesArguments,
    SearchMatch,
    UnsupportedTextFileError,
    decode_utf8_text,
    format_list_files_result,
    format_search_files_result,
)
from app.tools.path_guard import (
    WorkspacePathGuard,
    WorkspacePathTypeError,
)


FileToolArguments = ListFilesArguments | ReadFileArguments | SearchFilesArguments


class FileToolResourceLimitError(ValueError):
    """单个文件超过第 4 步固定的读取上限。"""


class ReadOnlyFileTool(ABC):
    """具体只读文件工具共用的执行与错误标准化边界。"""

    name: ClassVar[str]
    arguments_type: ClassVar[type[FileToolArguments]]

    def __init__(
        self,
        limits: ReadOnlyFileToolLimits = DEFAULT_READ_ONLY_FILE_TOOL_LIMITS,
    ) -> None:
        if not isinstance(limits, ReadOnlyFileToolLimits):
            raise TypeError("limits must be ReadOnlyFileToolLimits")
        self._limits = limits

    @abstractmethod
    def resolve_path(
        self,
        path_guard: WorkspacePathGuard,
        workspace: str | Path,
        arguments: FileToolArguments,
    ) -> Path:
        """在真实文件操作前解析并检查本工具的目标路径。"""

    def execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
    ) -> ToolResult:
        """执行已经完成参数及路径检查的文件调用。"""

        if not isinstance(arguments, self.arguments_type):
            return self._error_result(
                tool_call_id,
                f"arguments must be {self.arguments_type.__name__}",
            )
        if not isinstance(workspace, Path) or not isinstance(resolved_path, Path):
            raise TypeError("workspace and resolved_path must be pathlib.Path values")

        metadata: dict[str, object] = {
            "requested_path": arguments.path,
            "resolved_path": str(resolved_path),
            "limit_reached": False,
        }
        try:
            return self._execute(
                tool_call_id,
                workspace,
                arguments,
                resolved_path,
                metadata,
            )
        except FileToolResourceLimitError as exc:
            metadata["limit_reached"] = True
            return self._error_result(
                tool_call_id,
                str(exc),
                metadata=metadata,
            )
        except UnsupportedTextFileError as exc:
            return self._error_result(
                tool_call_id,
                str(exc),
                metadata=metadata,
            )
        except OSError as exc:
            return self._error_result(
                tool_call_id,
                f"file operation failed: {exc}",
                metadata=metadata,
            )

    @abstractmethod
    def _execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
        metadata: dict[str, object],
    ) -> ToolResult:
        """由具体工具完成只读操作。"""

    def _completed_result(
        self,
        tool_call_id: str,
        content: str,
        metadata: dict[str, object],
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=self.name,
            status=ToolResultStatus.COMPLETED,
            content=content,
            metadata=metadata,
        )

    def _error_result(
        self,
        tool_call_id: str,
        error: str,
        *,
        status: ToolResultStatus = ToolResultStatus.ERROR,
        metadata: dict[str, object] | None = None,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=self.name,
            status=status,
            error=error,
            metadata={} if metadata is None else metadata,
        )

    def _read_utf8_file(self, path: Path) -> tuple[str, int]:
        file_size = path.stat().st_size
        if file_size > self._limits.max_file_bytes:
            raise FileToolResourceLimitError(
                "file exceeds the maximum readable size of "
                f"{self._limits.max_file_bytes} bytes: {path}"
            )
        data = path.read_bytes()
        if len(data) > self._limits.max_file_bytes:
            raise FileToolResourceLimitError(
                "file exceeds the maximum readable size of "
                f"{self._limits.max_file_bytes} bytes: {path}"
            )
        return decode_utf8_text(data), len(data)


class ListFilesTool(ReadOnlyFileTool):
    """稳定列出一个目录的直接子项，不递归。"""

    name = "list_files"
    arguments_type = ListFilesArguments

    def resolve_path(
        self,
        path_guard: WorkspacePathGuard,
        workspace: str | Path,
        arguments: FileToolArguments,
    ) -> Path:
        assert isinstance(arguments, ListFilesArguments)
        return path_guard.resolve_existing_directory(workspace, arguments.path)

    def _execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
        metadata: dict[str, object],
    ) -> ToolResult:
        assert isinstance(arguments, ListFilesArguments)
        entries = tuple(
            ListedFile(
                relative_path=entry.relative_to(workspace).as_posix(),
                entry_type=_classify_directory_entry(entry),
            )
            for entry in resolved_path.iterdir()
        )
        metadata["entry_count"] = len(entries)
        return self._completed_result(
            tool_call_id,
            format_list_files_result(entries),
            metadata,
        )


class ReadFileTool(ReadOnlyFileTool):
    """读取一个大小受限的 UTF-8 文本文件。"""

    name = "read_file"
    arguments_type = ReadFileArguments

    def resolve_path(
        self,
        path_guard: WorkspacePathGuard,
        workspace: str | Path,
        arguments: FileToolArguments,
    ) -> Path:
        assert isinstance(arguments, ReadFileArguments)
        return path_guard.resolve_existing_file(workspace, arguments.path)

    def _execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
        metadata: dict[str, object],
    ) -> ToolResult:
        assert isinstance(arguments, ReadFileArguments)
        content, byte_count = self._read_utf8_file(resolved_path)
        metadata.update(
            {
                "byte_count": byte_count,
                "character_count": len(content),
            }
        )
        return self._completed_result(tool_call_id, content, metadata)


class SearchFilesTool(ReadOnlyFileTool):
    """在一个已有文件或目录中递归执行大小受限的字面文本搜索。"""

    name = "search_files"
    arguments_type = SearchFilesArguments

    def resolve_path(
        self,
        path_guard: WorkspacePathGuard,
        workspace: str | Path,
        arguments: FileToolArguments,
    ) -> Path:
        assert isinstance(arguments, SearchFilesArguments)
        search_root = path_guard.resolve_existing(workspace, arguments.path)
        if not search_root.is_file() and not search_root.is_dir():
            raise WorkspacePathTypeError(
                f"requested path is not a file or directory: {arguments.path}"
            )
        return search_root

    def _execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
        metadata: dict[str, object],
    ) -> ToolResult:
        assert isinstance(arguments, SearchFilesArguments)
        matches: list[SearchMatch] = []
        searched_file_count = 0
        limit_reached = False

        for candidate in _iter_search_candidates(resolved_path):
            if searched_file_count >= self._limits.max_search_files:
                limit_reached = True
                break
            if candidate.is_symlink():
                continue
            if not stat.S_ISREG(candidate.stat(follow_symlinks=False).st_mode):
                continue
            content, _ = self._read_utf8_file(candidate)
            searched_file_count += 1
            relative_path = candidate.relative_to(workspace).as_posix()

            for line_number, line in enumerate(content.splitlines(), start=1):
                if arguments.query not in line:
                    continue
                matches.append(
                    SearchMatch(
                        relative_path=relative_path,
                        line_number=line_number,
                        text=line,
                    )
                )
                if len(matches) >= self._limits.max_search_matches:
                    limit_reached = True
                    break
            if limit_reached:
                break

        metadata.update(
            {
                "searched_file_count": searched_file_count,
                "match_count": len(matches),
                "limit_reached": limit_reached,
            }
        )
        return self._completed_result(
            tool_call_id,
            format_search_files_result(matches),
            metadata,
        )


def _classify_directory_entry(path: Path) -> FileEntryType:
    if path.is_symlink():
        return FileEntryType.SYMLINK
    if path.is_file():
        return FileEntryType.FILE
    if path.is_dir():
        return FileEntryType.DIRECTORY
    return FileEntryType.OTHER


def _iter_search_candidates(search_root: Path):
    if search_root.is_file():
        yield search_root
        return

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directory_names, file_names in os.walk(
        search_root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        directory_path = Path(directory)
        for file_name in file_names:
            yield directory_path / file_name
