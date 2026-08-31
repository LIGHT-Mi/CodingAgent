"""Web API 的 HTTP 契约与后续路由入口。"""

from app.web.contracts import (
    API_TASK_CANCEL_PATH,
    API_TASK_MESSAGES_PATH,
    API_TASK_STEPS_PATH,
    API_TASK_TOOL_CALLS_PATH,
    API_TASKS_PATH,
    API_TASK_PATH,
    AgentStepResponse,
    CancelTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    ErrorResponse,
    MessageResponse,
    TaskResponse,
    ToolCallResponse,
)
from app.web.query_service import TaskQueryService

__all__ = [
    "API_TASK_CANCEL_PATH",
    "API_TASK_MESSAGES_PATH",
    "API_TASK_STEPS_PATH",
    "API_TASK_TOOL_CALLS_PATH",
    "API_TASKS_PATH",
    "API_TASK_PATH",
    "AgentStepResponse",
    "CancelTaskResponse",
    "CreateTaskRequest",
    "CreateTaskResponse",
    "ErrorResponse",
    "MessageResponse",
    "TaskResponse",
    "TaskQueryService",
    "ToolCallResponse",
]
