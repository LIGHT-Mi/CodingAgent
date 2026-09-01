"""Coding Agent 的最小 HTTP 路由。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.approval.service import (
    CommandApprovalFingerprintMismatchError,
    CommandApprovalNotActiveError,
    CommandApprovalNotFoundError,
)
from app.api.conversation_service import (
    ConversationService,
    ConversationTaskSubmissionError,
)
from app.application import (
    TaskCancellationOutcome,
    TaskRunner,
)
from app.db.persistence import (
    InvalidStateTransitionError,
    PersistenceService,
)
from app.web.contracts import (
    API_SESSIONS_PATH,
    API_COMMAND_APPROVAL_DECISION_PATH,
    API_SESSION_PATH,
    API_SESSION_TASKS_PATH,
    API_TASK_CANCEL_PATH,
    API_TASK_COMMAND_APPROVALS_PATH,
    API_TASK_MESSAGES_PATH,
    API_TASK_PATH,
    API_TASK_SNAPSHOT_PATH,
    API_TASK_STEPS_PATH,
    API_TASK_TOOL_CALLS_PATH,
    AgentStepResponse,
    CancelTaskResponse,
    CommandApprovalDecisionRequest,
    CommandApprovalResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    CreateSessionTaskRequest,
    CreateSessionTaskResponse,
    MessageResponse,
    SessionSummaryResponse,
    TaskSnapshotResponse,
    TaskResponse,
    ToolCallResponse,
)
from app.web.query_service import ConversationQueryService, TaskQueryService


router = APIRouter()


def get_persistence(request: Request) -> Iterator[PersistenceService]:
    """为每个 HTTP 请求创建并关闭独立数据库 Session。"""

    application_factory = request.app.state.application_factory
    with application_factory.create_db_session() as db:
        yield application_factory.create_persistence_service(db)


PersistenceDependency = Annotated[
    PersistenceService,
    Depends(get_persistence),
]


def _create_conversation_service(
    request: Request,
    persistence: PersistenceService,
) -> ConversationService:
    return request.app.state.application_factory.create_conversation_service(
        persistence,
        request.app.state.task_runner,
    )


@router.post(
    API_SESSIONS_PATH,
    response_model=CreateSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_session(
    payload: CreateSessionRequest,
    request: Request,
    persistence: PersistenceDependency,
) -> CreateSessionResponse:
    conversation_service = _create_conversation_service(request, persistence)
    try:
        created = conversation_service.create_conversation(
            payload.prompt,
            payload.workspace,
        )
    except ConversationTaskSubmissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task could not be scheduled",
        ) from exc
    return CreateSessionResponse(
        session_id=created.session_id,
        task_id=created.task_id,
        title=created.title,
    )


@router.get(
    API_SESSIONS_PATH,
    response_model=list[SessionSummaryResponse],
)
def list_sessions(
    request: Request,
    persistence: PersistenceDependency,
) -> list[SessionSummaryResponse]:
    conversation_service = _create_conversation_service(request, persistence)
    return ConversationQueryService(
        conversation_service
    ).list_conversations()


@router.get(API_SESSION_PATH, response_model=SessionSummaryResponse)
def get_session(
    session_id: str,
    request: Request,
    persistence: PersistenceDependency,
) -> SessionSummaryResponse:
    conversation_service = _create_conversation_service(request, persistence)
    return ConversationQueryService(conversation_service).get_conversation(
        session_id
    )


@router.get(API_SESSION_TASKS_PATH, response_model=list[TaskResponse])
def list_session_tasks(
    session_id: str,
    request: Request,
    persistence: PersistenceDependency,
) -> list[TaskResponse]:
    conversation_service = _create_conversation_service(request, persistence)
    return ConversationQueryService(conversation_service).list_tasks(
        session_id
    )


@router.post(
    API_SESSION_TASKS_PATH,
    response_model=CreateSessionTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_session_task(
    session_id: str,
    payload: CreateSessionTaskRequest,
    request: Request,
    persistence: PersistenceDependency,
) -> CreateSessionTaskResponse:
    conversation_service = _create_conversation_service(request, persistence)
    try:
        task_id = conversation_service.create_task(
            session_id,
            payload.prompt,
            payload.workspace,
        )
    except InvalidStateTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session already has an active task",
        ) from exc
    except ConversationTaskSubmissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task could not be scheduled",
        ) from exc
    return CreateSessionTaskResponse(
        session_id=session_id,
        task_id=task_id,
    )


@router.get(API_TASK_PATH, response_model=TaskResponse)
def get_task(
    task_id: str,
    persistence: PersistenceDependency,
) -> TaskResponse:
    return TaskQueryService(persistence).get_task(task_id)


@router.get(API_TASK_SNAPSHOT_PATH, response_model=TaskSnapshotResponse)
def get_task_snapshot(
    task_id: str,
    persistence: PersistenceDependency,
) -> TaskSnapshotResponse:
    return TaskQueryService(persistence).get_snapshot(task_id)


@router.get(API_TASK_STEPS_PATH, response_model=list[AgentStepResponse])
def get_task_steps(
    task_id: str,
    persistence: PersistenceDependency,
) -> list[AgentStepResponse]:
    return TaskQueryService(persistence).get_steps(task_id)


@router.get(API_TASK_MESSAGES_PATH, response_model=list[MessageResponse])
def get_task_messages(
    task_id: str,
    persistence: PersistenceDependency,
) -> list[MessageResponse]:
    return TaskQueryService(persistence).get_messages(task_id)


@router.get(API_TASK_TOOL_CALLS_PATH, response_model=list[ToolCallResponse])
def get_task_tool_calls(
    task_id: str,
    persistence: PersistenceDependency,
) -> list[ToolCallResponse]:
    return TaskQueryService(persistence).get_tool_calls(task_id)


@router.get(
    API_TASK_COMMAND_APPROVALS_PATH,
    response_model=list[CommandApprovalResponse],
)
def get_task_command_approvals(
    task_id: str,
    persistence: PersistenceDependency,
) -> list[CommandApprovalResponse]:
    return TaskQueryService(persistence).get_command_approvals(task_id)


@router.post(
    API_COMMAND_APPROVAL_DECISION_PATH,
    response_model=CommandApprovalResponse,
)
def decide_command_approval(
    task_id: str,
    approval_id: str,
    payload: CommandApprovalDecisionRequest,
    request: Request,
    persistence: PersistenceDependency,
) -> CommandApprovalResponse:
    service = request.app.state.application_factory.create_command_approval_service(
        persistence
    )
    try:
        approval = service.decide(
            task_id=task_id,
            request_id=approval_id,
            decision=payload.decision,
            command_fingerprint=payload.command_fingerprint,
        )
    except CommandApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Command approval request was not found",
        ) from exc
    except CommandApprovalFingerprintMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Command approval fingerprint does not match",
        ) from exc
    except CommandApprovalNotActiveError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Command approval request is no longer active",
        ) from exc
    approvals = TaskQueryService(persistence).get_command_approvals(task_id)
    return next(item for item in approvals if item.id == approval.id)


@router.post(API_TASK_CANCEL_PATH, response_model=CancelTaskResponse)
def cancel_task(
    task_id: str,
    request: Request,
    persistence: PersistenceDependency,
) -> CancelTaskResponse:
    task_runner: TaskRunner = request.app.state.task_runner
    cancellation = task_runner.cancel(task_id)
    if cancellation.outcome is TaskCancellationOutcome.TASK_NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task was not found",
        )
    if cancellation.outcome is TaskCancellationOutcome.TASK_NOT_ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task is not active in this server process",
        )

    task = TaskQueryService(persistence).get_task(task_id)
    return CancelTaskResponse(
        task_id=task_id,
        status=task.status,
        cancellation_requested=cancellation.cancellation_requested,
        outcome=cancellation.outcome,
    )
