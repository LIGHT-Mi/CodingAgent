"""Web 客户端与应用服务之间的稳定 HTTP 数据契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.contracts import (
    AgentStepStatus,
    MessageRole,
    MessageType,
    TaskStatus,
    ToolCallStatus,
)
from app.application import TaskCancellationOutcome


API_TASK_PATH = "/api/tasks/{task_id}"
API_TASK_STEPS_PATH = "/api/tasks/{task_id}/steps"
API_TASK_MESSAGES_PATH = "/api/tasks/{task_id}/messages"
API_TASK_TOOL_CALLS_PATH = "/api/tasks/{task_id}/tool-calls"
API_TASK_CANCEL_PATH = "/api/tasks/{task_id}/cancel"
API_TASK_SNAPSHOT_PATH = "/api/tasks/{task_id}/snapshot"
API_SESSIONS_PATH = "/api/sessions"
API_SESSION_PATH = "/api/sessions/{session_id}"
API_SESSION_TASKS_PATH = "/api/sessions/{session_id}/tasks"


class WebContract(BaseModel):
    """HTTP 边界共用的严格 Pydantic 基类。"""

    model_config = ConfigDict(extra="forbid")


class TaskCreationRequest(WebContract):
    prompt: str = Field(min_length=1)
    workspace: str = Field(min_length=1)

    @field_validator("prompt", "workspace")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class CreateSessionRequest(TaskCreationRequest):
    """创建 Session 及第一轮 Task 的请求。"""


class CreateSessionTaskRequest(TaskCreationRequest):
    """在已有 Session 中创建后续 Task 的请求。"""


class CreateSessionResponse(WebContract):
    session_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    status: Literal[TaskStatus.PENDING] = TaskStatus.PENDING


class CreateSessionTaskResponse(WebContract):
    session_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    status: Literal[TaskStatus.PENDING] = TaskStatus.PENDING


class SessionSummaryResponse(WebContract):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    latest_task_id: str = Field(min_length=1)
    latest_task_status: TaskStatus
    latest_workspace: str = Field(min_length=1)


class TaskResponse(WebContract):
    id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    original_prompt: str
    workspace: str
    status: TaskStatus
    final_answer: str | None = None
    error: str | None = None
    termination_reason: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AgentStepResponse(WebContract):
    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    step_number: int = Field(ge=0)
    status: AgentStepStatus
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class MessageResponse(WebContract):
    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    tool_call_id: str | None = None
    sequence: int = Field(ge=0)
    role: MessageRole
    message_type: MessageType
    content: str
    created_at: datetime


class ToolCallResponse(WebContract):
    id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    assistant_message_id: str = Field(min_length=1)
    call_index: int = Field(ge=0)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    status: ToolCallStatus
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    result: str | None = None
    result_metadata: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class CancelTaskResponse(WebContract):
    task_id: str = Field(min_length=1)
    status: TaskStatus
    cancellation_requested: bool
    outcome: TaskCancellationOutcome


class TaskSnapshotResponse(WebContract):
    task: TaskResponse
    steps: list[AgentStepResponse]
    messages: list[MessageResponse]
    tool_calls: list[ToolCallResponse]


class ErrorResponse(WebContract):
    detail: str = Field(min_length=1)
