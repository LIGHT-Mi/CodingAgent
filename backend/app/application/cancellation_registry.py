"""单进程后台 Task 的协作式取消令牌注册表。"""

from __future__ import annotations

import threading
from enum import Enum

from app.agent import CancellationToken


class CancellationTokenAlreadyRegisteredError(RuntimeError):
    """同一 Task 已经注册了一个仍有效的取消令牌。"""


class CancellationRequestStatus(str, Enum):
    """注册表处理一次取消请求的结果。"""

    REQUESTED = "REQUESTED"
    ALREADY_REQUESTED = "ALREADY_REQUESTED"
    NOT_REGISTERED = "NOT_REGISTERED"


class CancellationTokenRegistry:
    """线程安全地维护当前进程内的 ``task_id -> token`` 映射。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: dict[str, CancellationToken] = {}

    @property
    def registered_task_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._tokens))

    def register(self, task_id: str, token: CancellationToken) -> None:
        """在任务提交时注册令牌；同一 Task 不允许覆盖已有令牌。"""

        normalized_task_id = _require_non_blank(task_id, "task_id")
        if not isinstance(token, CancellationToken):
            raise TypeError("token must be a CancellationToken")

        with self._lock:
            if normalized_task_id in self._tokens:
                raise CancellationTokenAlreadyRegisteredError(
                    f"Task {normalized_task_id} already has a cancellation token"
                )
            self._tokens[normalized_task_id] = token

    def request_cancel(
        self,
        task_id: str,
        reason: str,
    ) -> CancellationRequestStatus:
        """请求取消已注册 Task，并区分首次请求和重复请求。"""

        normalized_task_id = _require_non_blank(task_id, "task_id")
        normalized_reason = _require_non_blank(reason, "reason")

        with self._lock:
            token = self._tokens.get(normalized_task_id)
            if token is None:
                return CancellationRequestStatus.NOT_REGISTERED
            if token.is_cancelled():
                return CancellationRequestStatus.ALREADY_REQUESTED
            token.cancel(normalized_reason)
            return CancellationRequestStatus.REQUESTED

    def unregister(self, task_id: str, token: CancellationToken) -> bool:
        """仅在令牌身份匹配时移除映射，防止陈旧回调误删新令牌。"""

        normalized_task_id = _require_non_blank(task_id, "task_id")
        if not isinstance(token, CancellationToken):
            raise TypeError("token must be a CancellationToken")

        with self._lock:
            current = self._tokens.get(normalized_task_id)
            if current is not token:
                return False
            del self._tokens[normalized_task_id]
            return True


def _require_non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized
