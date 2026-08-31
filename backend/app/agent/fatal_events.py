"""把 Runtime 各执行边界的未恢复异常统一转换为 RuntimeEvent。"""

from __future__ import annotations

from enum import Enum

from app.agent.contracts import RuntimeEvent, RuntimeEventType
from app.context.manager import ContextHistoryError
from app.db.persistence import PersistenceServiceError


class RuntimeFailureBoundary(str, Enum):
    """异常离开哪个运行边界时被 AgentRuntime 捕获。"""

    CONTEXT_MANAGER = "context_manager"
    LLM_GATEWAY = "llm_gateway"
    TOOL_ROUTER = "tool_router"
    RETRY_WAITER = "retry_waiter"
    RUNTIME_POLICY = "runtime_policy"
    PERSISTENCE_SERVICE = "persistence_service"
    AGENT_RUNTIME = "agent_runtime"


def build_fatal_runtime_event(
    error: Exception,
    boundary: RuntimeFailureBoundary,
) -> RuntimeEvent:
    """根据异常真实类型和捕获边界构造稳定的 Fatal RuntimeEvent。"""

    if not isinstance(error, Exception):
        raise TypeError("error must be an Exception")
    if not isinstance(boundary, RuntimeFailureBoundary):
        raise TypeError("boundary must be a RuntimeFailureBoundary")

    if isinstance(error, ContextHistoryError):
        event_type = RuntimeEventType.AGENT_STATE_CORRUPTED
        source = RuntimeFailureBoundary.CONTEXT_MANAGER.value
    elif isinstance(error, PersistenceServiceError) or (
        boundary is RuntimeFailureBoundary.PERSISTENCE_SERVICE
    ):
        event_type = RuntimeEventType.FATAL_SYSTEM_ERROR
        source = RuntimeFailureBoundary.PERSISTENCE_SERVICE.value
    elif boundary is RuntimeFailureBoundary.TOOL_ROUTER:
        event_type = RuntimeEventType.FATAL_TOOL_ERROR
        source = RuntimeFailureBoundary.TOOL_ROUTER.value
    else:
        event_type = RuntimeEventType.FATAL_SYSTEM_ERROR
        source = boundary.value

    return RuntimeEvent(
        event_type=event_type,
        source=source,
        message=_exception_message(error),
        details={
            "error_type": type(error).__name__,
            "boundary": boundary.value,
        },
    )


def _exception_message(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return message
    return f"{type(error).__name__} occurred without an error message"
