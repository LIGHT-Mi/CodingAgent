"""在单进程线程池中执行已创建 Task 的应用边界。"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from app.agent import AgentResult, CancellationToken, TaskStatus
from app.application.cancellation_registry import (
    CancellationRequestStatus,
    CancellationTokenRegistry,
)
from app.application.factory import ApplicationFactory


DEFAULT_TASK_RUNNER_MAX_WORKERS = 4
DEFAULT_CANCELLATION_REASON = "USER_CANCELLED"


class TaskRunnerError(RuntimeError):
    """TaskRunner 无法接受或管理任务。"""


class TaskRunnerShutdownError(TaskRunnerError):
    """TaskRunner 关闭后仍尝试提交任务。"""


class TaskAlreadySubmittedError(TaskRunnerError):
    """同一 Task 已经在当前 TaskRunner 中等待或运行。"""


class TaskNotPendingError(TaskRunnerError):
    """待提交 Task 不存在或不是 PENDING。"""


class TaskCancellationOutcome(str, Enum):
    """TaskRunner 对取消请求给出的应用层结果。"""

    REQUESTED = "REQUESTED"
    ALREADY_REQUESTED = "ALREADY_REQUESTED"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_FINISHED = "TASK_FINISHED"
    TASK_NOT_ACTIVE = "TASK_NOT_ACTIVE"


@dataclass(frozen=True, slots=True)
class TaskCancellationResult:
    """取消 API 可据此区分任务不存在、终态和请求幂等性。"""

    task_id: str
    outcome: TaskCancellationOutcome
    task_status: TaskStatus | None = None

    @property
    def cancellation_requested(self) -> bool:
        return self.outcome in {
            TaskCancellationOutcome.REQUESTED,
            TaskCancellationOutcome.ALREADY_REQUESTED,
        }


@dataclass(frozen=True, slots=True)
class _RunningTask:
    """进程内正在等待或执行的 Task 句柄。"""

    future: Future[AgentResult]


class TaskRunner:
    """为每个后台 Task 创建独立数据库 Session 和应用组件。"""

    def __init__(
        self,
        application_factory: ApplicationFactory,
        *,
        max_workers: int = DEFAULT_TASK_RUNNER_MAX_WORKERS,
        cancellation_registry: CancellationTokenRegistry | None = None,
    ) -> None:
        if not isinstance(application_factory, ApplicationFactory):
            raise TypeError("application_factory must be an ApplicationFactory")
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        if cancellation_registry is not None and not isinstance(
            cancellation_registry,
            CancellationTokenRegistry,
        ):
            raise TypeError(
                "cancellation_registry must be a CancellationTokenRegistry"
            )

        self._application_factory = application_factory
        self._cancellation_registry = (
            cancellation_registry or CancellationTokenRegistry()
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="coding-agent-task",
        )
        self._lock = threading.Lock()
        self._running_tasks: dict[str, _RunningTask] = {}
        self._shutdown = False

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._running_tasks))

    @property
    def cancellation_registry(self) -> CancellationTokenRegistry:
        """返回当前 Runner 使用的进程内注册表。"""

        return self._cancellation_registry

    def submit(self, task_id: str) -> Future[AgentResult]:
        """提交已持久化的 PENDING Task，立即返回 Future。"""

        normalized_task_id = _require_non_blank(task_id, "task_id")
        self._require_accepting_submission(normalized_task_id)
        self._require_pending_task(normalized_task_id)

        with self._lock:
            if self._shutdown:
                raise TaskRunnerShutdownError("TaskRunner has been shut down")
            if normalized_task_id in self._running_tasks:
                raise TaskAlreadySubmittedError(
                    f"Task {normalized_task_id} is already submitted"
                )

            cancellation_token = CancellationToken()
            self._cancellation_registry.register(
                normalized_task_id,
                cancellation_token,
            )
            try:
                future = self._executor.submit(
                    self._execute_task,
                    normalized_task_id,
                    cancellation_token,
                )
            except BaseException:
                self._cancellation_registry.unregister(
                    normalized_task_id,
                    cancellation_token,
                )
                raise
            self._running_tasks[normalized_task_id] = _RunningTask(future)

        future.add_done_callback(
            lambda completed,
            current_task_id=normalized_task_id,
            current_token=cancellation_token: (
                self._release_completed_task(
                    current_task_id,
                    current_token,
                    completed,
                )
            )
        )
        return future

    def cancel(
        self,
        task_id: str,
        reason: str = DEFAULT_CANCELLATION_REASON,
    ) -> TaskCancellationResult:
        """请求取消 Task，并返回可供 HTTP 层映射的结构化结果。"""

        normalized_task_id = _require_non_blank(task_id, "task_id")
        normalized_reason = _require_non_blank(reason, "reason")
        with self._lock:
            running_task = self._running_tasks.get(normalized_task_id)
            if running_task is not None and not running_task.future.done():
                request_status = self._cancellation_registry.request_cancel(
                    normalized_task_id,
                    normalized_reason,
                )
                if request_status is CancellationRequestStatus.REQUESTED:
                    return TaskCancellationResult(
                        task_id=normalized_task_id,
                        outcome=TaskCancellationOutcome.REQUESTED,
                    )
                if request_status is CancellationRequestStatus.ALREADY_REQUESTED:
                    return TaskCancellationResult(
                        task_id=normalized_task_id,
                        outcome=TaskCancellationOutcome.ALREADY_REQUESTED,
                    )

        return self._classify_inactive_cancellation(normalized_task_id)

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_running: bool = True,
    ) -> None:
        """停止接收新 Task，并可选请求取消已提交 Task。"""

        if not isinstance(wait, bool):
            raise TypeError("wait must be a boolean")
        if not isinstance(cancel_running, bool):
            raise TypeError("cancel_running must be a boolean")

        with self._lock:
            self._shutdown = True
            active_task_ids = tuple(self._running_tasks)

        if cancel_running:
            for task_id in active_task_ids:
                self._cancellation_registry.request_cancel(
                    task_id,
                    DEFAULT_CANCELLATION_REASON
                )

        # 不取消还未开始的 Future；它们需要进入 Runtime，
        # 消费已设置的 CancellationToken 并持久化 CANCELLED。
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _require_accepting_submission(self, task_id: str) -> None:
        with self._lock:
            if self._shutdown:
                raise TaskRunnerShutdownError("TaskRunner has been shut down")
            if task_id in self._running_tasks:
                raise TaskAlreadySubmittedError(
                    f"Task {task_id} is already submitted"
                )

    def _require_pending_task(self, task_id: str) -> None:
        with self._application_factory.create_db_session() as db:
            persistence = self._application_factory.create_persistence_service(db)
            task = persistence.get_task(task_id)
            if task is None:
                raise TaskNotPendingError(f"Task {task_id} was not found")
            if task.status != TaskStatus.PENDING.value:
                raise TaskNotPendingError(
                    f"Task {task_id} is {task.status}; expected PENDING"
                )

    def _execute_task(
        self,
        task_id: str,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        with self._application_factory.create_db_session() as db:
            persistence = self._application_factory.create_persistence_service(db)
            task_service = self._application_factory.create_task_service(
                persistence
            )
            return task_service.execute_task(task_id, cancellation_token)

    def _release_completed_task(
        self,
        task_id: str,
        cancellation_token: CancellationToken,
        completed_future: Future[AgentResult],
    ) -> None:
        with self._lock:
            current = self._running_tasks.get(task_id)
            if current is not None and current.future is completed_future:
                del self._running_tasks[task_id]
                self._cancellation_registry.unregister(
                    task_id,
                    cancellation_token,
                )

    def _classify_inactive_cancellation(
        self,
        task_id: str,
    ) -> TaskCancellationResult:
        with self._application_factory.create_db_session() as db:
            persistence = self._application_factory.create_persistence_service(db)
            task = persistence.get_task(task_id)
            if task is None:
                return TaskCancellationResult(
                    task_id=task_id,
                    outcome=TaskCancellationOutcome.TASK_NOT_FOUND,
                )

            task_status = TaskStatus(task.status)
            if task_status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TERMINATED,
            }:
                return TaskCancellationResult(
                    task_id=task_id,
                    outcome=TaskCancellationOutcome.TASK_FINISHED,
                    task_status=task_status,
                )
            return TaskCancellationResult(
                task_id=task_id,
                outcome=TaskCancellationOutcome.TASK_NOT_ACTIVE,
                task_status=task_status,
            )


def _require_non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized
