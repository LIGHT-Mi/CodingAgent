"""Coding Agent 的最小 HTTP 路由。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.application import (
    TaskCancellationOutcome,
    TaskRunner,
    TaskRunnerError,
)
from app.db.persistence import PersistenceService
from app.web.contracts import (
    API_TASK_CANCEL_PATH,
    API_TASK_MESSAGES_PATH,
    API_TASK_PATH,
    API_TASK_STEPS_PATH,
    API_TASK_TOOL_CALLS_PATH,
    API_TASKS_PATH,
    AgentStepResponse,
    CancelTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    MessageResponse,
    TaskResponse,
    ToolCallResponse,
)
from app.web.query_service import TaskQueryService


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


@router.post(
    API_TASKS_PATH,
    response_model=CreateTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_task(
    payload: CreateTaskRequest,
    request: Request,
    persistence: PersistenceDependency,
) -> CreateTaskResponse:
    task_service = request.app.state.application_factory.create_task_service(
        persistence
    )
    task_id = task_service.create_task(payload.prompt, payload.workspace)
    task_runner: TaskRunner = request.app.state.task_runner
    try:
        task_runner.submit(task_id)
    except TaskRunnerError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task could not be scheduled",
        ) from exc
    return CreateTaskResponse(task_id=task_id)


@router.get(API_TASK_PATH, response_model=TaskResponse)
def get_task(
    task_id: str,
    persistence: PersistenceDependency,
) -> TaskResponse:
    return TaskQueryService(persistence).get_task(task_id)


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
