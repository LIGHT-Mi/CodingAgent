"""执行只读工具调用并持续调用模型的基础 Agent Loop。"""

from __future__ import annotations

from app.agent.contracts import (
    AgentResult,
    AgentStepStatus,
    FinalAction,
    InvalidAction,
    MessageType,
    RuntimeEvent,
    TaskStatus,
    ToolCallsAction,
    ToolResult,
)
from app.context.manager import ContextManager
from app.db.models.tool_call import ToolCall
from app.db.persistence import PersistenceService
from app.llm.gateway import LLMGateway, LLMGatewayResult
from app.tools.router import PreparedToolCall, ToolRouter


DEFAULT_MAX_AGENT_STEPS = 8
MAX_STEPS_TERMINATION_REASON = "MAX_STEPS"


class AgentRuntime:
    """执行受最大步数限制的基础模型—工具循环。"""

    def __init__(
        self,
        persistence: PersistenceService,
        context_manager: ContextManager,
        llm_gateway: LLMGateway,
        tool_router: ToolRouter,
        *,
        max_agent_steps: int = DEFAULT_MAX_AGENT_STEPS,
    ) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        if not isinstance(context_manager, ContextManager):
            raise TypeError("context_manager must be a ContextManager")
        if not isinstance(llm_gateway, LLMGateway):
            raise TypeError("llm_gateway must be an LLMGateway")
        if not isinstance(tool_router, ToolRouter):
            raise TypeError("tool_router must be a ToolRouter")
        if isinstance(max_agent_steps, bool) or not isinstance(
            max_agent_steps,
            int,
        ):
            raise TypeError("max_agent_steps must be an integer")
        if max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be greater than zero")
        self._persistence = persistence
        self._context_manager = context_manager
        self._llm_gateway = llm_gateway
        self._tool_router = tool_router
        self._max_agent_steps = max_agent_steps

    def run(self, task_id: str) -> AgentResult:
        """循环执行模型动作，直到最终回答、失败或达到最大步数。"""

        task = self._persistence.get_task(task_id)
        if task is None:
            return AgentResult(
                status=TaskStatus.FAILED,
                error=f"Task {task_id} was not found",
            )

        for step_number in range(self._max_agent_steps):
            step = self._persistence.create_agent_step(task_id, step_number)
            try:
                context = self._context_manager.build(task_id)
                gateway_result = self._llm_gateway.invoke(context)
            except Exception as exc:
                return self._fail_step(
                    step.id,
                    _exception_failure_message(exc),
                )

            if isinstance(gateway_result, FinalAction):
                return self._complete_step(task_id, step.id, gateway_result)

            if isinstance(gateway_result, ToolCallsAction):
                try:
                    self._execute_tool_calls(
                        task_id,
                        step.id,
                        task.workspace,
                        gateway_result,
                    )
                except Exception as exc:
                    error = _exception_failure_message(exc)
                    self._persistence.fail_open_tool_calls(step.id, error)
                    return self._fail_step(step.id, error)
                self._persistence.finish_agent_step(
                    step.id,
                    AgentStepStatus.COMPLETED,
                )
                continue

            return self._fail_step(
                step.id,
                _gateway_failure_message(gateway_result),
            )

        return AgentResult(
            status=TaskStatus.TERMINATED,
            termination_reason=MAX_STEPS_TERMINATION_REASON,
        )

    def _execute_tool_calls(
        self,
        task_id: str,
        step_id: str,
        workspace: str,
        action: ToolCallsAction,
    ) -> None:
        _, records = self._persistence.save_tool_calls_action(
            task_id,
            step_id,
            action,
        )
        records_by_provider_id = _index_tool_call_records(records)

        for request in sorted(
            action.tool_calls,
            key=lambda tool_call: tool_call.call_index,
        ):
            record = records_by_provider_id[request.tool_call_id]
            prepared = self._tool_router.prepare(request, workspace)
            if isinstance(prepared, ToolResult):
                self._persistence.save_tool_result(record.id, prepared)
                continue

            assert isinstance(prepared, PreparedToolCall)
            self._persistence.start_tool_call(record.id)
            result = self._tool_router.execute(prepared)
            self._persistence.save_tool_result(record.id, result)

    def _complete_step(
        self,
        task_id: str,
        step_id: str,
        action: FinalAction,
    ) -> AgentResult:
        self._persistence.save_assistant_message(
            task_id,
            step_id,
            action.content,
            MessageType.FINAL,
        )
        self._persistence.finish_agent_step(
            step_id,
            AgentStepStatus.COMPLETED,
        )
        return AgentResult(
            status=TaskStatus.COMPLETED,
            final_answer=action.content,
        )

    def _fail_step(self, step_id: str, error: str) -> AgentResult:
        self._persistence.finish_agent_step(
            step_id,
            AgentStepStatus.FAILED,
            error=error,
        )
        return AgentResult(
            status=TaskStatus.FAILED,
            error=error,
        )


def _index_tool_call_records(
    records: tuple[ToolCall, ...],
) -> dict[str, ToolCall]:
    records_by_provider_id = {
        record.provider_call_id: record for record in records
    }
    if len(records_by_provider_id) != len(records):
        raise RuntimeError("persisted ToolCall provider_call_id values are not unique")
    return records_by_provider_id


def _gateway_failure_message(result: LLMGatewayResult) -> str:
    if isinstance(result, InvalidAction):
        return f"Invalid model action: {result.reason}"
    if isinstance(result, RuntimeEvent):
        return (
            f"Model runtime event {result.event_type.value}: "
            f"{result.message}"
        )
    return f"Unsupported model gateway result: {type(result).__name__}"


def _exception_failure_message(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return f"Agent runtime failed with {type(error).__name__}: {message}"
    return f"Agent runtime failed with {type(error).__name__}"
