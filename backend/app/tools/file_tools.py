"""当前文件工具的本地执行实现。"""

from __future__ import annotations

import os
import stat
import tempfile
from abc import ABC, abstractmethod
from difflib import unified_diff
from pathlib import Path
from typing import ClassVar

from app.agent.contracts import ToolResult, ToolResultStatus
from app.tools.contracts import (
    DEFAULT_FILE_TOOL_LIMITS,
    CreateFileArguments,
    EditFileArguments,
    FileEntryType,
    FileToolArguments,
    FileToolLimits,
    ListedFile,
    ListFilesArguments,
    ReadFileArguments,
    SearchFilesArguments,
    SearchMatch,
    UnsupportedTextFileError,
    WriteFileArguments,
    decode_utf8_text,
    format_list_files_result,
    format_search_files_result,
)
from app.tools.path_guard import (
    WorkspacePathGuard,
    WorkspacePathTypeError,
)
from app.tools.local_tool import LocalTool


class FileToolResourceLimitError(ValueError):
    """单个文件超过文件工具固定的资源保护上限。"""


class FileTool(LocalTool, ABC):
    """具体文件工具共用的执行与错误标准化边界。"""

    name: ClassVar[str]
    arguments_type: ClassVar[type[FileToolArguments]]

    def __init__(
        self,
        limits: FileToolLimits = DEFAULT_FILE_TOOL_LIMITS,
    ) -> None:
        if not isinstance(limits, FileToolLimits):
            raise TypeError("limits must be FileToolLimits")
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
        """由具体工具完成文件操作。"""

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

    def _completed_modification_result(
        self,
        tool_call_id: str,
        *,
        heading: str,
        relative_path: str,
        before_content: str,
        after_content: str,
        before_byte_count: int,
        after_byte_count: int,
        changed: bool,
        replacement_count: int,
        metadata: dict[str, object],
        from_file: str,
        to_file: str,
    ) -> ToolResult:
        """统一生成模型可见的文件修改摘要及结构化 metadata。"""

        diff_content, diff_truncated = _build_limited_unified_diff(
            before_content,
            after_content,
            from_file=from_file,
            to_file=to_file,
            max_lines=self._limits.max_diff_lines,
            max_characters=self._limits.max_diff_characters,
        )
        metadata.update(
            {
                "operation": self.name,
                "relative_path": relative_path,
                "changed": changed,
                "before_byte_count": before_byte_count,
                "after_byte_count": after_byte_count,
                "before_character_count": len(before_content),
                "after_character_count": len(after_content),
                "replacement_count": replacement_count,
                "diff_truncated": diff_truncated,
            }
        )

        summary_lines = [
            f"{heading} {relative_path}",
        ]
        if replacement_count:
            summary_lines.append(
                f"{replacement_count} exact replacement"
                if replacement_count == 1
                else f"{replacement_count} exact replacements"
            )
        summary_lines.extend(
            (
                f"changed: {str(changed).lower()}",
                f"characters: {len(before_content)} → {len(after_content)}",
                f"bytes: {before_byte_count} → {after_byte_count}",
            )
        )
        if diff_content:
            summary_lines.extend(("diff:", diff_content))

        return self._completed_result(
            tool_call_id,
            "\n".join(summary_lines),
            metadata,
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

    def _encode_utf8_content(self, content: str) -> bytes:
        try:
            encoded_content = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise UnsupportedTextFileError(
                "file content cannot be encoded as UTF-8"
            ) from exc
        if len(encoded_content) > self._limits.max_file_bytes:
            raise FileToolResourceLimitError(
                "file content exceeds the maximum writable size of "
                f"{self._limits.max_file_bytes} bytes"
            )
        return encoded_content

    @staticmethod
    def _atomic_replace_file(path: Path, content: bytes) -> None:
        """在目标同目录写临时文件，并以原子替换更新已有文件。"""

        original_mode = stat.S_IMODE(path.stat().st_mode)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                file_descriptor = -1
                written_byte_count = stream.write(content)
                if written_byte_count != len(content):
                    raise OSError(
                        "temporary file wrote fewer bytes than expected: "
                        f"{written_byte_count} of {len(content)}"
                    )
                stream.flush()
            temporary_path.chmod(original_mode)
            os.replace(temporary_path, path)
        except Exception:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
            raise


class ListFilesTool(FileTool):
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


class ReadFileTool(FileTool):
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


class SearchFilesTool(FileTool):
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


class CreateFileTool(FileTool):
    """在 Workspace 内独占创建一个新的 UTF-8 文本文件。"""

    name = "create_file"
    arguments_type = CreateFileArguments

    def resolve_path(
        self,
        path_guard: WorkspacePathGuard,
        workspace: str | Path,
        arguments: FileToolArguments,
    ) -> Path:
        assert isinstance(arguments, CreateFileArguments)
        return path_guard.resolve_new_file_target(workspace, arguments.path)

    def _execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
        metadata: dict[str, object],
    ) -> ToolResult:
        assert isinstance(arguments, CreateFileArguments)
        encoded_content = self._encode_utf8_content(arguments.content)

        with resolved_path.open("xb") as stream:
            written_byte_count = stream.write(encoded_content)
            if written_byte_count != len(encoded_content):
                raise OSError(
                    "file creation wrote fewer bytes than expected: "
                    f"{written_byte_count} of {len(encoded_content)}"
                )

        relative_path = resolved_path.relative_to(workspace).as_posix()
        byte_count = len(encoded_content)
        return self._completed_modification_result(
            tool_call_id,
            heading="Created",
            relative_path=relative_path,
            before_content="",
            after_content=arguments.content,
            before_byte_count=0,
            after_byte_count=byte_count,
            changed=True,
            replacement_count=0,
            metadata=metadata,
            from_file="/dev/null",
            to_file=f"b/{relative_path}",
        )


class WriteFileTool(FileTool):
    """整体覆盖 Workspace 内一个已有 UTF-8 文本文件。"""

    name = "write_file"
    arguments_type = WriteFileArguments

    def resolve_path(
        self,
        path_guard: WorkspacePathGuard,
        workspace: str | Path,
        arguments: FileToolArguments,
    ) -> Path:
        assert isinstance(arguments, WriteFileArguments)
        return path_guard.resolve_existing_file(workspace, arguments.path)

    def _execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
        metadata: dict[str, object],
    ) -> ToolResult:
        assert isinstance(arguments, WriteFileArguments)
        previous_content, before_byte_count = self._read_utf8_file(resolved_path)
        encoded_content = self._encode_utf8_content(arguments.content)
        previous_bytes = previous_content.encode("utf-8")
        changed = previous_bytes != encoded_content

        if changed:
            self._atomic_replace_file(resolved_path, encoded_content)

        relative_path = resolved_path.relative_to(workspace).as_posix()
        after_byte_count = len(encoded_content)
        return self._completed_modification_result(
            tool_call_id,
            heading="Wrote",
            relative_path=relative_path,
            before_content=previous_content,
            after_content=arguments.content,
            before_byte_count=before_byte_count,
            after_byte_count=after_byte_count,
            changed=changed,
            replacement_count=0,
            metadata=metadata,
            from_file=f"a/{relative_path}",
            to_file=f"b/{relative_path}",
        )


class EditFileTool(FileTool):
    """精确替换已有 UTF-8 文本文件中唯一一处文本。"""

    name = "edit_file"
    arguments_type = EditFileArguments

    def resolve_path(
        self,
        path_guard: WorkspacePathGuard,
        workspace: str | Path,
        arguments: FileToolArguments,
    ) -> Path:
        assert isinstance(arguments, EditFileArguments)
        return path_guard.resolve_existing_file(workspace, arguments.path)

    def _execute(
        self,
        tool_call_id: str,
        workspace: Path,
        arguments: FileToolArguments,
        resolved_path: Path,
        metadata: dict[str, object],
    ) -> ToolResult:
        assert isinstance(arguments, EditFileArguments)
        previous_content, before_byte_count = self._read_utf8_file(resolved_path)
        match_offsets = _find_exact_match_offsets(
            previous_content,
            arguments.old_text,
        )
        relative_path = resolved_path.relative_to(workspace).as_posix()
        metadata.update(
            {
                "operation": self.name,
                "relative_path": relative_path,
                "match_count": len(match_offsets),
            }
        )

        if not match_offsets:
            return self._error_result(
                tool_call_id,
                f"old_text was not found in {relative_path}",
                metadata=metadata,
            )
        if len(match_offsets) > 1:
            return self._error_result(
                tool_call_id,
                (
                    f"old_text matched {len(match_offsets)} locations in "
                    f"{relative_path}; exactly one match is required"
                ),
                metadata=metadata,
            )

        match_offset = match_offsets[0]
        updated_content = (
            previous_content[:match_offset]
            + arguments.new_text
            + previous_content[match_offset + len(arguments.old_text) :]
        )
        encoded_content = self._encode_utf8_content(updated_content)
        self._atomic_replace_file(resolved_path, encoded_content)

        after_byte_count = len(encoded_content)
        return self._completed_modification_result(
            tool_call_id,
            heading="Edited",
            relative_path=relative_path,
            before_content=previous_content,
            after_content=updated_content,
            before_byte_count=before_byte_count,
            after_byte_count=after_byte_count,
            changed=True,
            replacement_count=1,
            metadata=metadata,
            from_file=f"a/{relative_path}",
            to_file=f"b/{relative_path}",
        )


def _find_exact_match_offsets(text: str, query: str) -> tuple[int, ...]:
    """返回所有精确匹配起点，包括彼此重叠的匹配。"""

    offsets: list[int] = []
    search_start = 0
    while True:
        offset = text.find(query, search_start)
        if offset < 0:
            return tuple(offsets)
        offsets.append(offset)
        search_start = offset + 1


def _build_limited_unified_diff(
    before_content: str,
    after_content: str,
    *,
    from_file: str,
    to_file: str,
    max_lines: int,
    max_characters: int,
) -> tuple[str, bool]:
    """生成受固定行数及字符数保护的 unified diff。"""

    diff_lines = tuple(
        unified_diff(
            before_content.splitlines(),
            after_content.splitlines(),
            fromfile=from_file,
            tofile=to_file,
            lineterm="",
        )
    )
    if not diff_lines:
        return "", False

    full_diff = "\n".join(diff_lines)
    if len(diff_lines) <= max_lines and len(full_diff) <= max_characters:
        return full_diff, False

    truncation_marker = "... diff truncated ..."
    visible_lines = diff_lines[: max(max_lines - 1, 0)]
    visible_diff = "\n".join(visible_lines)
    available_characters = max_characters - len(truncation_marker)
    if visible_diff and available_characters > 1:
        visible_diff = visible_diff[: available_characters - 1].rstrip()
        if visible_diff:
            return f"{visible_diff}\n{truncation_marker}", True
    return truncation_marker[:max_characters], True


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
