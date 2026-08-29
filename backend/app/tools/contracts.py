"""只读文件工具使用的纯 Python 语义契约。

本模块只固定工具参数、文本结果格式和资源保护上限，不访问文件系统，也不依赖
数据库或 Agent Runtime。路径边界校验、参数路由和真实文件执行由后续模块负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


DEFAULT_TOOL_PATH = "."


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_positive_integer(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class ReadOnlyFileToolLimits:
    """第 4 步使用的固定资源保护上限。

    这些限制用于阻止一次本地工具调用读取过多数据，不是第 7 步的模型上下文
    字符预算或 Tool Result 截断策略。
    """

    max_file_bytes: int = 1024 * 1024
    max_search_files: int = 1000
    max_search_matches: int = 200

    def __post_init__(self) -> None:
        _require_positive_integer(self.max_file_bytes, "max_file_bytes")
        _require_positive_integer(self.max_search_files, "max_search_files")
        _require_positive_integer(self.max_search_matches, "max_search_matches")


DEFAULT_READ_ONLY_FILE_TOOL_LIMITS = ReadOnlyFileToolLimits()


@dataclass(frozen=True, slots=True)
class ListFilesArguments:
    """``list_files`` 参数；只列出目标目录的直接子项。"""

    path: str = DEFAULT_TOOL_PATH

    def __post_init__(self) -> None:
        _require_non_blank(self.path, "path")


@dataclass(frozen=True, slots=True)
class ReadFileArguments:
    """``read_file`` 参数；path 必须由调用方显式提供。"""

    path: str

    def __post_init__(self) -> None:
        _require_non_blank(self.path, "path")


@dataclass(frozen=True, slots=True)
class SearchFilesArguments:
    """``search_files`` 参数；query 是普通字面文本，不是正则表达式。"""

    query: str
    path: str = DEFAULT_TOOL_PATH

    def __post_init__(self) -> None:
        _require_non_blank(self.query, "query")
        _require_non_blank(self.path, "path")


class FileEntryType(str, Enum):
    """``list_files`` 可报告的直接子项类型。"""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ListedFile:
    """一个待格式化的目录直接子项。

    ``relative_path`` 使用相对于 Workspace 的 POSIX 路径，避免向模型暴露宿主机
    绝对路径，并保证不同操作系统上的结果格式一致。
    """

    relative_path: str
    entry_type: FileEntryType

    def __post_init__(self) -> None:
        _require_non_blank(self.relative_path, "relative_path")
        if not isinstance(self.entry_type, FileEntryType):
            raise TypeError("entry_type must be a FileEntryType")


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """``search_files`` 找到的一个匹配行。

    同一行即使多次包含 query 也只产生一个 SearchMatch。行号从 1 开始，text
    不包含行尾换行符。
    """

    relative_path: str
    line_number: int
    text: str

    def __post_init__(self) -> None:
        _require_non_blank(self.relative_path, "relative_path")
        _require_positive_integer(self.line_number, "line_number")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if "\n" in self.text or "\r" in self.text:
            raise ValueError("text must not contain line-ending characters")


def format_list_files_result(entries: Iterable[ListedFile]) -> str:
    """按路径稳定排序，并格式化为 ``relative_path<TAB>type``。

    空目录返回空字符串；执行器应把它保存为 ``COMPLETED ToolResult``。
    """

    normalized_entries = tuple(entries)
    if any(not isinstance(entry, ListedFile) for entry in normalized_entries):
        raise TypeError("entries must contain only ListedFile values")
    ordered_entries = sorted(
        normalized_entries,
        key=lambda entry: entry.relative_path,
    )
    return "\n".join(
        f"{entry.relative_path}\t{entry.entry_type.value}"
        for entry in ordered_entries
    )


def format_search_files_result(matches: Iterable[SearchMatch]) -> str:
    """按路径、行号稳定排序，并格式化为 ``path:line:text``。

    没有匹配时返回空字符串；执行器应把它保存为 ``COMPLETED ToolResult``。
    """

    normalized_matches = tuple(matches)
    if any(not isinstance(match, SearchMatch) for match in normalized_matches):
        raise TypeError("matches must contain only SearchMatch values")
    ordered_matches = sorted(
        normalized_matches,
        key=lambda match: (match.relative_path, match.line_number),
    )
    return "\n".join(
        f"{match.relative_path}:{match.line_number}:{match.text}"
        for match in ordered_matches
    )


class UnsupportedTextFileError(ValueError):
    """文件不是本阶段支持的 UTF-8 纯文本。"""


def decode_utf8_text(data: bytes) -> str:
    """严格解码 UTF-8 文本，并把包含 NUL 的内容识别为二进制。

    后续文件工具执行器会把本函数抛出的 ``UnsupportedTextFileError`` 转换成普通
    ``ERROR ToolResult``，而不是让异常终止 Task。
    """

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if b"\x00" in data:
        raise UnsupportedTextFileError("binary files are not supported")
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise UnsupportedTextFileError("file content is not valid UTF-8") from exc
