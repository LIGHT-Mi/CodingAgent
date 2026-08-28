"""当前阶段只执行一轮模型调用的最薄 Agent Runtime。"""

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
)
from app.context.manager import ContextManager
from app.db.persistence import PersistenceService
from app.llm.gateway import LLMGateway, LLMGatewayResult


class AgentRuntime:
    """执行单个 AgentStep，并为所有结果关闭步骤生命周期。"""

    def __init__(
        self,
        persistence: PersistenceService,
        context_manager: ContextManager,
        llm_gateway: LLMGateway,
    ) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        if not isinstance(context_manager, ContextManager):
            raise TypeError("context_manager must be a ContextManager")
        if not isinstance(llm_gateway, LLMGateway):
            raise TypeError("llm_gateway must be an LLMGateway")
        self._persistence = persistence
        self._context_manager = context_manager
        self._llm_gateway = llm_gateway

    def run(self, task_id: str) -> AgentResult:
        """执行 step_number=0 的一次模型调用并返回 Task 终态结果。"""

        step = self._persistence.create_agent_step(task_id, step_number=0)
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
        return self._fail_step(
            step.id,
            _gateway_failure_message(gateway_result),
        )

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


def _gateway_failure_message(result: LLMGatewayResult) -> str:
    if isinstance(result, ToolCallsAction):
        return "Tool calls are unsupported by the current single-step runtime"
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
