"""并发排空并有界保留子进程 stdout 与 stderr。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, Thread
from typing import BinaryIO


DEFAULT_PROCESS_READ_CHUNK_BYTES = 8192


class ProcessOutputCollectionError(RuntimeError):
    """读取子进程输出流时发生无法标准化的异常。"""


@dataclass(frozen=True, slots=True)
class CollectedProcessStream:
    """一个输出流经有界头尾保留后的结果。"""

    text: str
    original_byte_count: int
    retained_byte_count: int
    discarded_byte_count: int
    truncated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        for field_name in (
            "original_byte_count",
            "retained_byte_count",
            "discarded_byte_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")
        if self.retained_byte_count + self.discarded_byte_count != (
            self.original_byte_count
        ):
            raise ValueError("retained and discarded byte counts must be complete")
        if self.truncated != (self.discarded_byte_count > 0):
            raise ValueError("truncated must match discarded_byte_count")


@dataclass(frozen=True, slots=True)
class CollectedProcessOutput:
    """一次命令的 stdout 与 stderr 收集结果。"""

    stdout: CollectedProcessStream
    stderr: CollectedProcessStream

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, CollectedProcessStream):
            raise TypeError("stdout must be a CollectedProcessStream")
        if not isinstance(self.stderr, CollectedProcessStream):
            raise TypeError("stderr must be a CollectedProcessStream")


class ProcessOutputCollector:
    """使用两个线程持续读取输出，每个流只保留固定字节数的头尾。"""

    def __init__(
        self,
        max_bytes_per_stream: int,
        *,
        read_chunk_bytes: int = DEFAULT_PROCESS_READ_CHUNK_BYTES,
    ) -> None:
        _require_integer_at_least(
            max_bytes_per_stream,
            "max_bytes_per_stream",
            minimum=2,
        )
        _require_integer_at_least(
            read_chunk_bytes,
            "read_chunk_bytes",
            minimum=1,
        )
        self._max_bytes_per_stream = max_bytes_per_stream
        self._read_chunk_bytes = read_chunk_bytes
        self._lock = Lock()
        self._threads: tuple[Thread, ...] = ()
        self._buffers: dict[str, _HeadTailByteBuffer] = {}
        self._errors: list[tuple[str, Exception]] = []
        self._started = False
        self._finished_result: CollectedProcessOutput | None = None

    def start(
        self,
        stdout: BinaryIO | None,
        stderr: BinaryIO | None,
    ) -> None:
        """启动两个读取线程；必须在等待子进程退出之前调用。"""

        if self._started:
            raise RuntimeError("process output collection has already started")
        _require_binary_stream_or_none(stdout, "stdout")
        _require_binary_stream_or_none(stderr, "stderr")

        self._started = True
        self._buffers = {
            "stdout": _HeadTailByteBuffer(self._max_bytes_per_stream),
            "stderr": _HeadTailByteBuffer(self._max_bytes_per_stream),
        }
        streams = (("stdout", stdout), ("stderr", stderr))
        self._threads = tuple(
            Thread(
                target=self._drain_stream,
                args=(stream_name, stream),
                name=f"process-{stream_name}-collector",
            )
            for stream_name, stream in streams
            if stream is not None
        )
        for thread in self._threads:
            thread.start()

    def finish(self) -> CollectedProcessOutput:
        """等待两个流到达 EOF，并返回稳定的文本与统计信息。"""

        if not self._started:
            raise RuntimeError("process output collection has not started")
        if self._finished_result is not None:
            return self._finished_result

        for thread in self._threads:
            thread.join()

        if self._errors:
            stream_name, error = self._errors[0]
            message = str(error).strip()
            detail = f": {message}" if message else ""
            raise ProcessOutputCollectionError(
                f"failed to read process {stream_name}{detail}"
            ) from error

        result = CollectedProcessOutput(
            stdout=self._buffers["stdout"].build_result("stdout"),
            stderr=self._buffers["stderr"].build_result("stderr"),
        )
        self._finished_result = result
        return result

    def _drain_stream(self, stream_name: str, stream: BinaryIO) -> None:
        buffer = self._buffers[stream_name]
        try:
            while True:
                chunk = stream.read(self._read_chunk_bytes)
                if not chunk:
                    return
                if not isinstance(chunk, bytes):
                    raise TypeError(
                        f"{stream_name} must return bytes, got "
                        f"{type(chunk).__name__}"
                    )
                buffer.append(chunk)
        except Exception as exc:
            with self._lock:
                self._errors.append((stream_name, exc))


class _HeadTailByteBuffer:
    """保留原始字节流开头和结尾的固定大小缓冲区。"""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._head_limit = (limit + 1) // 2
        self._tail_limit = limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._original_byte_count = 0

    def append(self, chunk: bytes) -> None:
        self._original_byte_count += len(chunk)
        head_remaining = self._head_limit - len(self._head)
        if head_remaining > 0:
            self._head.extend(chunk[:head_remaining])
            chunk = chunk[head_remaining:]
        if not chunk or self._tail_limit == 0:
            return
        self._tail.extend(chunk)
        if len(self._tail) > self._tail_limit:
            del self._tail[:-self._tail_limit]

    def build_result(self, stream_name: str) -> CollectedProcessStream:
        retained = bytes(self._head) + bytes(self._tail)
        retained_byte_count = len(retained)
        discarded_byte_count = self._original_byte_count - retained_byte_count
        truncated = discarded_byte_count > 0

        if truncated:
            head_text = bytes(self._head).decode("utf-8", errors="replace")
            tail_text = bytes(self._tail).decode("utf-8", errors="replace")
            marker = (
                "\n[... "
                f"{stream_name} truncated: {discarded_byte_count} bytes omitted; "
                "showing beginning and end ...]\n"
            )
            text = head_text + marker + tail_text
        else:
            text = retained.decode("utf-8", errors="replace")

        return CollectedProcessStream(
            text=text,
            original_byte_count=self._original_byte_count,
            retained_byte_count=retained_byte_count,
            discarded_byte_count=discarded_byte_count,
            truncated=truncated,
        )


def _require_binary_stream_or_none(
    stream: BinaryIO | None,
    field_name: str,
) -> None:
    if stream is not None and not callable(getattr(stream, "read", None)):
        raise TypeError(f"{field_name} must be a readable binary stream or None")


def _require_integer_at_least(
    value: int,
    field_name: str,
    *,
    minimum: int,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
