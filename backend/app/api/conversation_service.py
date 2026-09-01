"""多轮 Session 与 Task 创建、查询及后台提交的应用服务。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from app.agent.contracts import TaskStatus
from app.api.session_title import generate_session_title
from app.api.task_validation import validate_task_prompt
from app.api.workspace import WorkspaceValidator
from app.db.models.session_record import CodingSession
from app.db.models.task import Task
from app.db.persistence import PersistenceService, RecordNotFoundError


TASK_SUBMISSION_FAILURE_ERROR = "Background task scheduling failed"


class TaskSubmitter(Protocol):
    """ConversationService 所需的最小后台任务提交能力。"""

    def submit(self, task_id: str) -> object:
        """接受一个已持久化的 PENDING Task。"""


class ConversationTaskSubmissionError(RuntimeError):
    """Task 已创建，但后台执行边界未能接受它。"""

    def __init__(self, task_id: str, *, task_closed: bool) -> None:
        super().__init__(f"Task {task_id} could not be scheduled")
        self.task_id = task_id
        self.task_closed = task_closed


class ConversationNotFoundError(RecordNotFoundError):
    """请求的多轮会话不存在。"""


@dataclass(frozen=True, slots=True)
class ConversationCreationResult:
    """创建首轮会话后供应用入口使用的稳定标识。"""

    session_id: str
    task_id: str
    title: str


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """不暴露 ORM 对象的会话查询结果。"""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    latest_task_id: str
    latest_task_status: TaskStatus
    latest_workspace: str


@dataclass(frozen=True, slots=True)
class ConversationTaskRecord:
    """不暴露 ORM 对象的会话内 Task 查询结果。"""

    id: str
    session_id: str
    original_prompt: str
    workspace: str
    status: TaskStatus
    final_answer: str | None
    error: str | None
    termination_reason: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ConversationService:
    """管理多轮会话，并保证新建 Task 被提交或明确闭合。"""

    def __init__(
        self,
        persistence: PersistenceService,
        workspace_validator: WorkspaceValidator,
        task_submitter: TaskSubmitter,
    ) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        if not isinstance(workspace_validator, WorkspaceValidator):
            raise TypeError(
                "workspace_validator must be a WorkspaceValidator"
            )
        if not callable(getattr(task_submitter, "submit", None)):
            raise TypeError("task_submitter must provide submit(task_id)")
        self._persistence = persistence
        self._workspace_validator = workspace_validator
        self._task_submitter = task_submitter

    def create_conversation(
        self,
        prompt: str,
        workspace: str | Path,
    ) -> ConversationCreationResult:
        """原子创建 Session 和首个 Task，然后提交后台执行。"""

        validate_task_prompt(prompt)
        validated_workspace = self._workspace_validator.validate(workspace)
        coding_session, task = self._persistence.create_session_with_task(
            title=generate_session_title(prompt),
            original_prompt=prompt,
            workspace=str(validated_workspace),
        )
        self._submit_or_close(task.id)
        return ConversationCreationResult(
            session_id=coding_session.id,
            task_id=task.id,
            title=coding_session.title,
        )

    def create_task(
        self,
        session_id: str,
        prompt: str,
        workspace: str | Path,
    ) -> str:
        """在没有活动 Task 的已有 Session 中创建并提交下一轮。"""

        validate_task_prompt(prompt)
        validated_workspace = self._workspace_validator.validate(workspace)
        try:
            task = self._persistence.create_task_in_session(
                session_id=session_id,
                original_prompt=prompt,
                workspace=str(validated_workspace),
            )
        except RecordNotFoundError as exc:
            raise ConversationNotFoundError(
                f"Session {session_id} was not found"
            ) from exc
        self._submit_or_close(task.id)
        return task.id

    def list_conversations(self) -> tuple[ConversationRecord, ...]:
        """按最近更新时间从新到旧返回会话列表。"""

        return tuple(
            self._to_conversation_record(coding_session)
            for coding_session in self._persistence.list_sessions()
        )

    def get_conversation(self, session_id: str) -> ConversationRecord:
        """返回指定会话及其最新 Task 摘要。"""

        coding_session = self._persistence.get_session(session_id)
        if coding_session is None:
            raise ConversationNotFoundError(
                f"Session {session_id} was not found"
            )
        return self._to_conversation_record(coding_session)

    def list_tasks(
        self,
        session_id: str,
    ) -> tuple[ConversationTaskRecord, ...]:
        """按创建时间从旧到新返回会话内所有 Task。"""

        try:
            tasks = self._persistence.load_session_tasks(session_id)
        except RecordNotFoundError as exc:
            raise ConversationNotFoundError(
                f"Session {session_id} was not found"
            ) from exc
        return tuple(_to_task_record(task) for task in tasks)

    def _submit_or_close(self, task_id: str) -> None:
        try:
            self._task_submitter.submit(task_id)
        except Exception as submission_error:
            try:
                self._persistence.fail_pending_task(
                    task_id,
                    TASK_SUBMISSION_FAILURE_ERROR,
                )
            except Exception as closing_error:
                raise ConversationTaskSubmissionError(
                    task_id,
                    task_closed=False,
                ) from closing_error
            raise ConversationTaskSubmissionError(
                task_id,
                task_closed=True,
            ) from submission_error

    def _to_conversation_record(
        self,
        coding_session: CodingSession,
    ) -> ConversationRecord:
        latest_task = self._persistence.get_latest_session_task(
            coding_session.id
        )
        if latest_task is None:
            raise ConversationNotFoundError(
                f"Session {coding_session.id} has no Task"
            )
        return ConversationRecord(
            id=coding_session.id,
            title=coding_session.title,
            created_at=coding_session.created_at,
            updated_at=coding_session.updated_at,
            latest_task_id=latest_task.id,
            latest_task_status=TaskStatus(latest_task.status),
            latest_workspace=latest_task.workspace,
        )


def _to_task_record(task: Task) -> ConversationTaskRecord:
    return ConversationTaskRecord(
        id=task.id,
        session_id=task.session_id,
        original_prompt=task.original_prompt,
        workspace=task.workspace,
        status=TaskStatus(task.status),
        final_answer=task.final_answer,
        error=task.error,
        termination_reason=task.termination_reason,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )
