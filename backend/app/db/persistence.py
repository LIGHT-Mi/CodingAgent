"""Agent 运行记录的统一持久化服务。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.agent.contracts import (
    AgentResult,
    AgentStepStatus,
    MessageRole,
    MessageType,
    TaskStatus,
    ToolCallsAction,
    ToolCallStatus,
    ToolResult,
    ToolResultStatus,
)
from app.db.models.agent_step import AgentStep
from app.db.models.message import Message
from app.db.models.session_record import (
    SESSION_TITLE_MAX_LENGTH,
    CodingSession,
)
from app.db.models.task import Task
from app.db.models.tool_call import ToolCall


class PersistenceServiceError(RuntimeError):
    """持久化服务无法完成请求时的基础异常。"""


class RecordNotFoundError(PersistenceServiceError):
    """请求操作的数据库记录不存在。"""


class InvalidStateTransitionError(PersistenceServiceError):
    """请求的生命周期状态流转不合法。"""


class PersistenceValidationError(PersistenceServiceError):
    """传给持久化服务的数据不符合业务约束。"""


class PersistenceService:
    """封装 Session、Task、Step、Message 和 ToolCall 的持久化操作。

    每个写方法对应一个完整的业务状态变化，并在方法结束时提交。发生校验错误、
    状态错误或数据库错误时会回滚当前事务。
    """

    def __init__(self, db: Session) -> None:
        if not isinstance(db, Session):
            raise TypeError("db must be a sqlalchemy.orm.Session")
        self._db = db

    def create_session_with_task(
        self,
        title: str,
        original_prompt: str,
        workspace: str,
    ) -> tuple[CodingSession, Task]:
        """在同一事务中创建 Session 及其第一个 PENDING Task。"""

        _require_non_blank(title, "title")
        if len(title) > SESSION_TITLE_MAX_LENGTH:
            raise PersistenceValidationError(
                f"title must not exceed {SESSION_TITLE_MAX_LENGTH} characters"
            )
        _require_non_blank(original_prompt, "original_prompt")
        _require_non_blank(workspace, "workspace")

        with self._write_transaction("create session with first task"):
            coding_session = CodingSession(title=title)
            task = Task(
                session=coding_session,
                original_prompt=original_prompt,
                workspace=workspace,
                status=TaskStatus.PENDING.value,
            )
            self._db.add_all((coding_session, task))
            self._db.flush()
        return coding_session, task

    def get_session(self, session_id: str) -> CodingSession | None:
        """按 ID 查询用户工作会话，不存在时返回 None。"""

        _require_non_blank(session_id, "session_id")
        return self._read_one(CodingSession, session_id, "load session")

    def list_sessions(self) -> list[CodingSession]:
        """按最近更新时间从新到旧加载全部 Session。"""

        return self._read_many(
            select(CodingSession).order_by(
                CodingSession.updated_at.desc(),
                CodingSession.id.asc(),
            ),
            "list sessions",
        )

    def create_task_in_session(
        self,
        session_id: str,
        original_prompt: str,
        workspace: str,
    ) -> Task:
        """锁定已有 Session，并在没有活动 Task 时创建下一轮 Task。"""

        _require_non_blank(session_id, "session_id")
        _require_non_blank(original_prompt, "original_prompt")
        _require_non_blank(workspace, "workspace")

        with self._write_transaction("create task in session"):
            coding_session = self._db.scalar(
                select(CodingSession)
                .where(CodingSession.id == session_id)
                .with_for_update()
            )
            if coding_session is None:
                raise RecordNotFoundError(
                    f"Session {session_id} was not found"
                )
            active_task_id = self._db.scalar(
                select(Task.id)
                .where(
                    Task.session_id == session_id,
                    Task.status.in_(
                        (
                            TaskStatus.PENDING.value,
                            TaskStatus.RUNNING.value,
                        )
                    ),
                )
                .limit(1)
            )
            if active_task_id is not None:
                raise InvalidStateTransitionError(
                    f"Session {session_id} already has an active Task "
                    f"{active_task_id}"
                )
            task = Task(
                session=coding_session,
                original_prompt=original_prompt,
                workspace=workspace,
                status=TaskStatus.PENDING.value,
            )
            coding_session.updated_at = _utc_now()
            self._db.add(task)
            self._db.flush()
        return task

    def load_session_tasks(self, session_id: str) -> list[Task]:
        """按创建时间从旧到新加载 Session 的全部 Task。"""

        _require_non_blank(session_id, "session_id")
        self._require_session_for_read(session_id)
        return self._read_many(
            select(Task)
            .where(Task.session_id == session_id)
            .order_by(Task.created_at.asc(), Task.id.asc()),
            "load session tasks",
        )

    def get_latest_session_task(self, session_id: str) -> Task | None:
        """返回 Session 最近创建的 Task；无 Task 时返回 None。"""

        _require_non_blank(session_id, "session_id")
        self._require_session_for_read(session_id)
        try:
            return self._db.scalar(
                select(Task)
                .where(Task.session_id == session_id)
                .order_by(Task.created_at.desc(), Task.id.desc())
                .limit(1)
            )
        except SQLAlchemyError as exc:
            raise PersistenceServiceError(
                "load latest session task failed"
            ) from exc

    def has_active_session_task(self, session_id: str) -> bool:
        """判断 Session 是否包含 PENDING 或 RUNNING Task。"""

        _require_non_blank(session_id, "session_id")
        self._require_session_for_read(session_id)
        try:
            active_task_id = self._db.scalar(
                select(Task.id)
                .where(
                    Task.session_id == session_id,
                    Task.status.in_(
                        (
                            TaskStatus.PENDING.value,
                            TaskStatus.RUNNING.value,
                        )
                    ),
                )
                .limit(1)
            )
        except SQLAlchemyError as exc:
            raise PersistenceServiceError(
                "check active session task failed"
            ) from exc
        return active_task_id is not None

    def get_task(self, task_id: str) -> Task | None:
        """按 ID 查询任务，不存在时返回 None。"""

        _require_non_blank(task_id, "task_id")
        return self._read_one(Task, task_id, "load task")

    def start_task(self, task_id: str) -> Task:
        """将任务从 PENDING 更新为 RUNNING。"""

        _require_non_blank(task_id, "task_id")
        with self._write_transaction("start task"):
            task = self._require_task(task_id)
            self._require_status(
                "Task",
                task.id,
                task.status,
                TaskStatus.PENDING.value,
            )
            now = _utc_now()
            task.status = TaskStatus.RUNNING.value
            task.started_at = now
            task.session.updated_at = now
        return task

    def fail_pending_task(self, task_id: str, error: str) -> Task:
        """将后台提交失败的 PENDING Task 直接闭合为 FAILED。"""

        _require_non_blank(task_id, "task_id")
        _require_non_blank(error, "error")
        with self._write_transaction("fail pending task"):
            task = self._require_task(task_id)
            self._require_status(
                "Task",
                task.id,
                task.status,
                TaskStatus.PENDING.value,
            )
            now = _utc_now()
            task.status = TaskStatus.FAILED.value
            task.error = error
            task.finished_at = now
            task.session.updated_at = now
        return task

    def finish_task(self, task_id: str, result: AgentResult) -> Task:
        """根据 AgentResult 将 RUNNING Task 更新为对应终态。"""

        _require_non_blank(task_id, "task_id")
        if not isinstance(result, AgentResult):
            raise TypeError("result must be an AgentResult")

        with self._write_transaction("finish task"):
            task = self._require_task(task_id)
            self._require_status(
                "Task",
                task.id,
                task.status,
                TaskStatus.RUNNING.value,
            )
            now = _utc_now()
            task.status = result.status.value
            task.final_answer = result.final_answer
            task.error = result.error
            task.termination_reason = result.termination_reason
            task.finished_at = now
            task.session.updated_at = now
        return task

    def create_agent_step(self, task_id: str, step_number: int) -> AgentStep:
        """为 RUNNING Task 创建一条 RUNNING AgentStep。"""

        _require_non_blank(task_id, "task_id")
        if not isinstance(step_number, int):
            raise TypeError("step_number must be an integer")
        if step_number < 0:
            raise PersistenceValidationError(
                "step_number must be greater than or equal to zero"
            )

        with self._write_transaction("create agent step"):
            task = self._require_task(task_id)
            self._require_status(
                "Task",
                task.id,
                task.status,
                TaskStatus.RUNNING.value,
            )
            existing_step = self._db.scalar(
                select(AgentStep).where(
                    AgentStep.task_id == task_id,
                    AgentStep.step_number == step_number,
                )
            )
            if existing_step is not None:
                raise PersistenceValidationError(
                    f"AgentStep {step_number} already exists for Task {task_id}"
                )

            step = AgentStep(
                task_id=task.id,
                step_number=step_number,
                status=AgentStepStatus.RUNNING.value,
            )
            self._db.add(step)
            self._db.flush()
        return step

    def finish_agent_step(
        self,
        step_id: str,
        status: AgentStepStatus,
        error: str | None = None,
    ) -> AgentStep:
        """将 RUNNING AgentStep 更新为 COMPLETED、FAILED 或 INTERRUPTED。"""

        _require_non_blank(step_id, "step_id")
        if not isinstance(status, AgentStepStatus):
            raise TypeError("status must be an AgentStepStatus")
        if status is AgentStepStatus.RUNNING:
            raise PersistenceValidationError("finish status must be terminal")
        if status is AgentStepStatus.FAILED:
            if error is None:
                raise PersistenceValidationError(
                    "a failed AgentStep must contain an error"
                )
            _require_non_blank(error, "error")
        elif error is not None:
            raise PersistenceValidationError(
                "only a failed AgentStep can contain an error"
            )

        with self._write_transaction("finish agent step"):
            step = self._require_agent_step(step_id)
            self._require_status(
                "AgentStep",
                step.id,
                step.status,
                AgentStepStatus.RUNNING.value,
            )
            step.status = status.value
            step.error = error
            step.finished_at = _utc_now()
        return step

    def save_assistant_message(
        self,
        task_id: str,
        step_id: str,
        content: str,
        message_type: MessageType,
    ) -> Message:
        """保存普通或最终 Assistant Message，并集中分配 sequence。"""

        _require_non_blank(task_id, "task_id")
        _require_non_blank(step_id, "step_id")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        if not isinstance(message_type, MessageType):
            raise TypeError("message_type must be a MessageType")
        if message_type is MessageType.TOOL_RESULT:
            raise PersistenceValidationError(
                "save_assistant_message cannot save a TOOL_RESULT message"
            )
        if message_type is MessageType.FINAL:
            _require_non_blank(content, "content")

        with self._write_transaction("save assistant message"):
            self._require_running_task_and_step(task_id, step_id)
            message = Message(
                task_id=task_id,
                step_id=step_id,
                sequence=self._next_message_sequence(task_id),
                role=MessageRole.ASSISTANT.value,
                message_type=message_type.value,
                content=content,
            )
            self._db.add(message)
            self._db.flush()
        return message

    def save_tool_calls_action(
        self,
        task_id: str,
        step_id: str,
        action: ToolCallsAction,
    ) -> tuple[Message, tuple[ToolCall, ...]]:
        """原子保存 ToolCalls 来源消息以及其中全部 ToolCall。"""

        _require_non_blank(task_id, "task_id")
        _require_non_blank(step_id, "step_id")
        if not isinstance(action, ToolCallsAction):
            raise TypeError("action must be a ToolCallsAction")

        with self._write_transaction("save tool calls action"):
            self._require_running_task_and_step(task_id, step_id)
            assistant_message = Message(
                task_id=task_id,
                step_id=step_id,
                sequence=self._next_message_sequence(task_id),
                role=MessageRole.ASSISTANT.value,
                message_type=MessageType.TEXT.value,
                content=action.content or "",
            )
            self._db.add(assistant_message)
            self._db.flush()

            tool_calls = tuple(
                ToolCall(
                    step_id=step_id,
                    assistant_message_id=assistant_message.id,
                    provider_call_id=request.tool_call_id,
                    call_index=request.call_index,
                    tool_name=request.tool_name,
                    arguments=dict(request.arguments),
                    status=ToolCallStatus.PENDING.value,
                )
                for request in action.tool_calls
            )
            self._db.add_all(tool_calls)
            self._db.flush()
        return assistant_message, tool_calls

    def start_tool_call(self, tool_call_id: str) -> ToolCall:
        """将 ToolCall 从 PENDING 更新为 RUNNING。"""

        _require_non_blank(tool_call_id, "tool_call_id")
        with self._write_transaction("start tool call"):
            tool_call = self._require_tool_call(tool_call_id)
            self._require_status(
                "ToolCall",
                tool_call.id,
                tool_call.status,
                ToolCallStatus.PENDING.value,
            )
            tool_call.status = ToolCallStatus.RUNNING.value
            tool_call.started_at = _utc_now()
        return tool_call

    def save_tool_result(
        self,
        tool_call_record_id: str,
        result: ToolResult,
    ) -> tuple[ToolCall, Message]:
        """原子更新 ToolCall 终态并保存对应的 TOOL_RESULT Message。"""

        _require_non_blank(tool_call_record_id, "tool_call_record_id")
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")

        with self._write_transaction("save tool result"):
            tool_call = self._require_tool_call(tool_call_record_id)
            self._validate_tool_result_identity(tool_call, result)
            self._validate_tool_result_transition(tool_call, result.status)

            metadata = dict(result.metadata)
            tool_call.status = result.status.value
            tool_call.exit_code = _optional_int(metadata, "exit_code")
            tool_call.stdout = _optional_str(metadata, "stdout")
            tool_call.stderr = _optional_str(metadata, "stderr")
            tool_call.result = result.content
            tool_call.result_metadata = metadata
            tool_call.error = result.error
            tool_call.finished_at = _utc_now()

            result_message = Message(
                task_id=tool_call.step.task_id,
                step_id=tool_call.step_id,
                tool_call_id=tool_call.id,
                sequence=self._next_message_sequence(tool_call.step.task_id),
                role=MessageRole.TOOL.value,
                message_type=MessageType.TOOL_RESULT.value,
                content=result.content if result.content is not None else result.error or "",
            )
            self._db.add(result_message)
            self._db.flush()
        return tool_call, result_message

    def fail_open_tool_calls(
        self,
        step_id: str,
        error: str,
    ) -> tuple[ToolCall, ...]:
        """将 Step 中全部未终态 ToolCall 原子关闭为 ERROR 并保存结果消息。"""

        _require_non_blank(step_id, "step_id")
        _require_non_blank(error, "error")

        with self._write_transaction("fail open tool calls"):
            step = self._require_agent_step(step_id)
            self._require_running_task_and_step(step.task_id, step.id)
            open_tool_calls = tuple(
                self._db.scalars(
                    select(ToolCall)
                    .where(
                        ToolCall.step_id == step.id,
                        ToolCall.status.in_(
                            (
                                ToolCallStatus.PENDING.value,
                                ToolCallStatus.RUNNING.value,
                            )
                        ),
                    )
                    .order_by(ToolCall.call_index.asc())
                ).all()
            )
            if not open_tool_calls:
                return ()

            next_sequence = self._next_message_sequence(step.task_id)
            finished_at = _utc_now()
            result_messages: list[Message] = []
            for sequence_offset, tool_call in enumerate(open_tool_calls):
                tool_call.status = ToolCallStatus.ERROR.value
                tool_call.exit_code = None
                tool_call.stdout = None
                tool_call.stderr = None
                tool_call.result = None
                tool_call.result_metadata = {"fatal": True}
                tool_call.error = error
                tool_call.finished_at = finished_at
                result_messages.append(
                    Message(
                        task_id=step.task_id,
                        step_id=step.id,
                        tool_call_id=tool_call.id,
                        sequence=next_sequence + sequence_offset,
                        role=MessageRole.TOOL.value,
                        message_type=MessageType.TOOL_RESULT.value,
                        content=error,
                    )
                )

            self._db.add_all(result_messages)
            self._db.flush()
        return open_tool_calls

    def interrupt_open_tool_calls(
        self,
        step_id: str,
        reason: str,
    ) -> tuple[ToolCall, ...]:
        """因 Task 中断将未终态 ToolCall 闭合为带中断标记的 ERROR。"""

        _require_non_blank(step_id, "step_id")
        _require_non_blank(reason, "reason")
        error = f"ToolCall interrupted: {reason}"

        with self._write_transaction("interrupt open tool calls"):
            step = self._require_agent_step(step_id)
            self._require_running_task_and_step(step.task_id, step.id)
            open_tool_calls = tuple(
                self._db.scalars(
                    select(ToolCall)
                    .where(
                        ToolCall.step_id == step.id,
                        ToolCall.status.in_(
                            (
                                ToolCallStatus.PENDING.value,
                                ToolCallStatus.RUNNING.value,
                            )
                        ),
                    )
                    .order_by(ToolCall.call_index.asc())
                ).all()
            )
            if not open_tool_calls:
                return ()

            next_sequence = self._next_message_sequence(step.task_id)
            finished_at = _utc_now()
            result_messages: list[Message] = []
            for sequence_offset, tool_call in enumerate(open_tool_calls):
                tool_call.status = ToolCallStatus.ERROR.value
                tool_call.exit_code = None
                tool_call.stdout = None
                tool_call.stderr = None
                tool_call.result = None
                tool_call.result_metadata = {
                    "interrupted": True,
                    "reason": reason,
                }
                tool_call.error = error
                tool_call.finished_at = finished_at
                result_messages.append(
                    Message(
                        task_id=step.task_id,
                        step_id=step.id,
                        tool_call_id=tool_call.id,
                        sequence=next_sequence + sequence_offset,
                        role=MessageRole.TOOL.value,
                        message_type=MessageType.TOOL_RESULT.value,
                        content=error,
                    )
                )

            self._db.add_all(result_messages)
            self._db.flush()
        return open_tool_calls

    def load_agent_steps(self, task_id: str) -> list[AgentStep]:
        """按 step_number 加载任务的全部 AgentStep。"""

        _require_non_blank(task_id, "task_id")
        self._require_task_for_read(task_id)
        return self._read_many(
            select(AgentStep)
            .where(AgentStep.task_id == task_id)
            .order_by(AgentStep.step_number.asc()),
            "load agent steps",
        )

    def load_messages(self, task_id: str) -> list[Message]:
        """按 sequence 加载任务的完整消息历史。"""

        _require_non_blank(task_id, "task_id")
        self._require_task_for_read(task_id)
        return self._read_many(
            select(Message)
            .where(Message.task_id == task_id)
            .order_by(Message.sequence.asc()),
            "load messages",
        )

    def load_tool_calls(self, task_id: str) -> list[ToolCall]:
        """按 Step 和 call_index 加载任务的全部 ToolCall。"""

        _require_non_blank(task_id, "task_id")
        self._require_task_for_read(task_id)
        return self._read_many(
            select(ToolCall)
            .join(AgentStep, ToolCall.step_id == AgentStep.id)
            .where(AgentStep.task_id == task_id)
            .order_by(AgentStep.step_number.asc(), ToolCall.call_index.asc()),
            "load tool calls",
        )

    def _next_message_sequence(self, task_id: str) -> int:
        latest_sequence = self._db.scalar(
            select(func.max(Message.sequence)).where(Message.task_id == task_id)
        )
        return 0 if latest_sequence is None else latest_sequence + 1

    def _require_running_task_and_step(self, task_id: str, step_id: str) -> None:
        task = self._require_task(task_id)
        self._require_status(
            "Task",
            task.id,
            task.status,
            TaskStatus.RUNNING.value,
        )
        step = self._require_agent_step(step_id)
        if step.task_id != task.id:
            raise PersistenceValidationError(
                f"AgentStep {step_id} does not belong to Task {task_id}"
            )
        self._require_status(
            "AgentStep",
            step.id,
            step.status,
            AgentStepStatus.RUNNING.value,
        )

    def _validate_tool_result_identity(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:
        if result.tool_call_id != tool_call.provider_call_id:
            raise PersistenceValidationError(
                "ToolResult tool_call_id does not match provider_call_id"
            )
        if result.tool_name != tool_call.tool_name:
            raise PersistenceValidationError(
                "ToolResult tool_name does not match the persisted ToolCall"
            )

    def _validate_tool_result_transition(
        self,
        tool_call: ToolCall,
        result_status: ToolResultStatus,
    ) -> None:
        current_status = ToolCallStatus(tool_call.status)
        allowed_results = {
            ToolCallStatus.PENDING: {
                ToolResultStatus.ERROR,
                ToolResultStatus.REJECTED,
            },
            ToolCallStatus.RUNNING: {
                ToolResultStatus.COMPLETED,
                ToolResultStatus.ERROR,
                ToolResultStatus.TIMEOUT,
            },
        }
        if result_status not in allowed_results.get(current_status, set()):
            raise InvalidStateTransitionError(
                f"ToolCall {tool_call.id} cannot transition from "
                f"{current_status.value} to {result_status.value}"
            )

    def _require_session(self, session_id: str) -> CodingSession:
        coding_session = self._db.get(CodingSession, session_id)
        if coding_session is None:
            raise RecordNotFoundError(f"Session {session_id} was not found")
        return coding_session

    def _require_task(self, task_id: str) -> Task:
        task = self._db.get(Task, task_id)
        if task is None:
            raise RecordNotFoundError(f"Task {task_id} was not found")
        return task

    def _require_agent_step(self, step_id: str) -> AgentStep:
        step = self._db.get(AgentStep, step_id)
        if step is None:
            raise RecordNotFoundError(f"AgentStep {step_id} was not found")
        return step

    def _require_tool_call(self, tool_call_id: str) -> ToolCall:
        tool_call = self._db.get(ToolCall, tool_call_id)
        if tool_call is None:
            raise RecordNotFoundError(f"ToolCall {tool_call_id} was not found")
        return tool_call

    def _require_task_for_read(self, task_id: str) -> None:
        try:
            self._require_task(task_id)
        except SQLAlchemyError as exc:
            raise PersistenceServiceError("load task failed") from exc

    def _require_session_for_read(self, session_id: str) -> None:
        try:
            self._require_session(session_id)
        except SQLAlchemyError as exc:
            raise PersistenceServiceError("load session failed") from exc

    def _read_one(self, model: type, record_id: str, operation: str):
        try:
            return self._db.get(model, record_id)
        except SQLAlchemyError as exc:
            raise PersistenceServiceError(f"{operation} failed") from exc

    def _read_many(self, statement, operation: str) -> list:
        try:
            return list(self._db.scalars(statement).all())
        except SQLAlchemyError as exc:
            raise PersistenceServiceError(f"{operation} failed") from exc

    @contextmanager
    def _write_transaction(self, operation: str) -> Iterator[None]:
        try:
            yield
            self._db.commit()
        except SQLAlchemyError as exc:
            self._db.rollback()
            raise PersistenceServiceError(f"{operation} failed") from exc
        except Exception:
            self._db.rollback()
            raise

    @staticmethod
    def _require_status(
        record_type: str,
        record_id: str,
        actual_status: str,
        required_status: str,
    ) -> None:
        if actual_status != required_status:
            raise InvalidStateTransitionError(
                f"{record_type} {record_id} is {actual_status}; "
                f"expected {required_status}"
            )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise PersistenceValidationError(f"{field_name} must not be blank")


def _optional_int(metadata: dict, key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceValidationError(f"metadata.{key} must be an integer or None")
    return value


def _optional_str(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PersistenceValidationError(f"metadata.{key} must be a string or None")
    return value
