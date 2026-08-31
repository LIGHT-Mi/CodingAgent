"""应用组件的统一装配边界。"""

from app.application.cancellation_registry import (
    CancellationRequestStatus,
    CancellationTokenAlreadyRegisteredError,
    CancellationTokenRegistry,
)
from app.application.factory import (
    ApplicationFactory,
    LLMGatewayFactory,
    SessionFactory,
)
from app.application.task_runner import (
    DEFAULT_CANCELLATION_REASON,
    DEFAULT_TASK_RUNNER_MAX_WORKERS,
    TaskAlreadySubmittedError,
    TaskCancellationOutcome,
    TaskCancellationResult,
    TaskNotPendingError,
    TaskRunner,
    TaskRunnerError,
    TaskRunnerShutdownError,
)

__all__ = [
    "ApplicationFactory",
    "CancellationRequestStatus",
    "CancellationTokenAlreadyRegisteredError",
    "CancellationTokenRegistry",
    "LLMGatewayFactory",
    "SessionFactory",
    "DEFAULT_CANCELLATION_REASON",
    "DEFAULT_TASK_RUNNER_MAX_WORKERS",
    "TaskAlreadySubmittedError",
    "TaskCancellationOutcome",
    "TaskCancellationResult",
    "TaskNotPendingError",
    "TaskRunner",
    "TaskRunnerError",
    "TaskRunnerShutdownError",
]
