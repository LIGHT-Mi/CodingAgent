"""Runtime 执行重试等待时使用的可注入标准库边界。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Protocol


class CancellationWaitHandle(Protocol):
    """取消令牌提供给重试等待器的最小等待接口。"""

    def wait(self, timeout_seconds: float) -> bool:
        """等待超时或取消；取消时返回 True。"""


WaitFunction = Callable[[float], object]


class RetryWaiter:
    """执行 RuntimePolicy 已决定的等待，不计算退避秒数。"""

    def __init__(self, wait_function: WaitFunction | None = None) -> None:
        if wait_function is None:
            wait_function = time.sleep
        if not callable(wait_function):
            raise TypeError("wait_function must be callable")
        self._wait_function = wait_function

    def wait(
        self,
        seconds: float,
        cancellation_token: CancellationWaitHandle | None = None,
    ) -> bool:
        """等待指定秒数；传入取消句柄时允许等待被提前唤醒。"""

        normalized_seconds = _normalize_wait_seconds(seconds)
        if cancellation_token is None:
            self._wait_function(normalized_seconds)
            return False

        wait = getattr(cancellation_token, "wait", None)
        if not callable(wait):
            raise TypeError("cancellation_token must provide wait(seconds)")
        cancelled = wait(normalized_seconds)
        if not isinstance(cancelled, bool):
            raise TypeError("cancellation_token.wait() must return a boolean")
        return cancelled


def _normalize_wait_seconds(seconds: float) -> float:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError("seconds must be a number")
    normalized = float(seconds)
    if not math.isfinite(normalized):
        raise ValueError("seconds must be finite")
    if normalized < 0:
        raise ValueError("seconds must be greater than or equal to zero")
    return normalized
