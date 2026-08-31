"""AgentRuntime 使用的线程安全协作式取消令牌。"""

from __future__ import annotations

import math
import threading


class CancellationToken:
    """保存第一次取消原因，并允许等待过程被取消事件提前唤醒。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def cancel(self, reason: str) -> None:
        """请求取消；重复调用不会覆盖第一次取消原因。"""

        normalized_reason = _normalize_reason(reason)
        with self._lock:
            if self._event.is_set():
                return
            self._reason = normalized_reason
            self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout_seconds: float) -> bool:
        """等待超时或取消；取消已经发生或期间发生时返回 True。"""

        return self._event.wait(_normalize_timeout(timeout_seconds))


def _normalize_reason(reason: str) -> str:
    if not isinstance(reason, str):
        raise TypeError("reason must be a string")
    normalized = reason.strip()
    if not normalized:
        raise ValueError("reason must not be blank")
    return normalized


def _normalize_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        (int, float),
    ):
        raise TypeError("timeout_seconds must be a number")
    normalized = float(timeout_seconds)
    if not math.isfinite(normalized):
        raise ValueError("timeout_seconds must be finite")
    if normalized < 0:
        raise ValueError(
            "timeout_seconds must be greater than or equal to zero"
        )
    return normalized
