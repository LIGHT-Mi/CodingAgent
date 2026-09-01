"""单进程内唤醒等待用户批准的 Agent Runtime。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from app.agent.cancellation import CancellationToken


class ApprovalWaitOutcome(str, Enum):
    NOTIFIED = "NOTIFIED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class _ApprovalWaitHandle:
    event: threading.Event


class CommandApprovalCoordinator:
    """维护 request_id 到等待事件的映射；数据库仍是决定状态的事实来源。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handles: dict[str, _ApprovalWaitHandle] = {}

    def register(self, request_id: str) -> None:
        request_id = _require_non_blank(request_id, "request_id")
        with self._lock:
            if request_id in self._handles:
                raise ValueError(
                    f"approval request {request_id} is already registered"
                )
            self._handles[request_id] = _ApprovalWaitHandle(threading.Event())

    def is_registered(self, request_id: str) -> bool:
        request_id = _require_non_blank(request_id, "request_id")
        with self._lock:
            return request_id in self._handles

    def notify(self, request_id: str) -> bool:
        request_id = _require_non_blank(request_id, "request_id")
        with self._lock:
            handle = self._handles.get(request_id)
            if handle is None:
                return False
            handle.event.set()
            return True

    def wait(
        self,
        request_id: str,
        expires_at: datetime,
        cancellation_token: CancellationToken,
    ) -> ApprovalWaitOutcome:
        request_id = _require_non_blank(request_id, "request_id")
        if not isinstance(expires_at, datetime) or expires_at.tzinfo is None:
            raise TypeError("expires_at must be a timezone-aware datetime")
        if not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be a CancellationToken")
        with self._lock:
            handle = self._handles.get(request_id)
        if handle is None:
            raise ValueError(f"approval request {request_id} is not registered")

        while True:
            if cancellation_token.is_cancelled():
                return ApprovalWaitOutcome.CANCELLED
            remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                return ApprovalWaitOutcome.EXPIRED
            if handle.event.wait(min(remaining, 0.1)):
                return ApprovalWaitOutcome.NOTIFIED

    def unregister(self, request_id: str) -> bool:
        request_id = _require_non_blank(request_id, "request_id")
        with self._lock:
            return self._handles.pop(request_id, None) is not None


def _require_non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized
