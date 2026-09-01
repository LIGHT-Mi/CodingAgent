"""将持久化记录显式转换为面向 Web 客户端的响应结构。"""

from __future__ import annotations

from app.agent import (
    AgentStepStatus,
    MessageRole,
    MessageType,
    TaskStatus,
    ToolCallStatus,
)
from app.api.conversation_service import (
    ConversationRecord,
    ConversationService,
    ConversationTaskRecord,
)
from app.db.persistence import PersistenceService, RecordNotFoundError
from app.web.contracts import (
    AgentStepResponse,
    CommandApprovalResponse,
    MessageResponse,
    SessionSummaryResponse,
    TaskSnapshotResponse,
    TaskResponse,
    ToolCallResponse,
)


class TaskQueryService:
    """查询 Task 运行快照；不向 HTTP 层返回 ORM 对象。"""

    def __init__(self, persistence: PersistenceService) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        self._persistence = persistence

    def get_task(self, task_id: str) -> TaskResponse:
        task = self._persistence.get_task(task_id)
        if task is None:
            raise RecordNotFoundError(f"Task {task_id} was not found")
        return TaskResponse(
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

    def get_steps(self, task_id: str) -> list[AgentStepResponse]:
        return [
            AgentStepResponse(
                id=step.id,
                task_id=step.task_id,
                step_number=step.step_number,
                status=AgentStepStatus(step.status),
                error=step.error,
                started_at=step.started_at,
                finished_at=step.finished_at,
            )
            for step in self._persistence.load_agent_steps(task_id)
        ]

    def get_messages(self, task_id: str) -> list[MessageResponse]:
        return [
            MessageResponse(
                id=message.id,
                task_id=message.task_id,
                step_id=message.step_id,
                tool_call_id=message.tool_call_id,
                sequence=message.sequence,
                role=MessageRole(message.role),
                message_type=MessageType(message.message_type),
                content=message.content,
                created_at=message.created_at,
            )
            for message in self._persistence.load_messages(task_id)
        ]

    def get_tool_calls(self, task_id: str) -> list[ToolCallResponse]:
        return [
            ToolCallResponse(
                id=tool_call.id,
                step_id=tool_call.step_id,
                assistant_message_id=tool_call.assistant_message_id,
                call_index=tool_call.call_index,
                tool_name=tool_call.tool_name,
                arguments=dict(tool_call.arguments),
                status=ToolCallStatus(tool_call.status),
                exit_code=tool_call.exit_code,
                stdout=tool_call.stdout,
                stderr=tool_call.stderr,
                result=tool_call.result,
                result_metadata=(
                    None
                    if tool_call.result_metadata is None
                    else dict(tool_call.result_metadata)
                ),
                error=tool_call.error,
                started_at=tool_call.started_at,
                finished_at=tool_call.finished_at,
            )
            for tool_call in self._persistence.load_tool_calls(task_id)
        ]

    def get_command_approvals(
        self,
        task_id: str,
    ) -> list[CommandApprovalResponse]:
        return [
            CommandApprovalResponse(
                id=approval.id,
                task_id=approval.task_id,
                step_id=approval.step_id,
                tool_call_id=approval.tool_call_id,
                status=approval.status,
                command=list(approval.command),
                cwd=approval.cwd,
                command_fingerprint=approval.command_fingerprint,
                rule_id=approval.rule_id,
                risk_level=approval.risk_level,
                reason=approval.reason,
                resolution_reason=approval.resolution_reason,
                created_at=approval.created_at,
                expires_at=approval.expires_at,
                decided_at=approval.decided_at,
                consumed_at=approval.consumed_at,
            )
            for approval in self._persistence.load_command_approval_requests(
                task_id
            )
        ]

    def get_snapshot(self, task_id: str) -> TaskSnapshotResponse:
        """聚合一个 Task 的状态与完整有序执行历史。"""

        return TaskSnapshotResponse(
            task=self.get_task(task_id),
            steps=self.get_steps(task_id),
            messages=self.get_messages(task_id),
            tool_calls=self.get_tool_calls(task_id),
            command_approvals=self.get_command_approvals(task_id),
        )


class ConversationQueryService:
    """把 ConversationService 结果转换为明确的 Web DTO。"""

    def __init__(self, conversation_service: ConversationService) -> None:
        if not isinstance(conversation_service, ConversationService):
            raise TypeError(
                "conversation_service must be a ConversationService"
            )
        self._conversation_service = conversation_service

    def list_conversations(self) -> list[SessionSummaryResponse]:
        return [
            _to_session_summary(record)
            for record in self._conversation_service.list_conversations()
        ]

    def get_conversation(
        self,
        session_id: str,
    ) -> SessionSummaryResponse:
        return _to_session_summary(
            self._conversation_service.get_conversation(session_id)
        )

    def list_tasks(self, session_id: str) -> list[TaskResponse]:
        return [
            _to_task_response(task)
            for task in self._conversation_service.list_tasks(session_id)
        ]


def _to_session_summary(
    conversation: ConversationRecord,
) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        latest_task_id=conversation.latest_task_id,
        latest_task_status=conversation.latest_task_status,
        latest_workspace=conversation.latest_workspace,
    )


def _to_task_response(task: ConversationTaskRecord) -> TaskResponse:
    return TaskResponse(
        id=task.id,
        session_id=task.session_id,
        original_prompt=task.original_prompt,
        workspace=task.workspace,
        status=task.status,
        final_answer=task.final_answer,
        error=task.error,
        termination_reason=task.termination_reason,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )
