"""执行文件工具调用并持续调用模型的基础 Agent Loop。"""

from __future__ import annotations

from dataclasses import replace

from app.agent.cancellation import CancellationToken
from app.agent.contracts import (
    AgentResult,
    AgentStepStatus,
    FinalAction,
    InvalidAction,
    MessageType,
    RuntimeDecision,
    RuntimeDecisionType,
    RuntimeEvent,
    RuntimeEventType,
    TaskStatus,
    ToolCallsAction,
    ToolResult,
)
from app.agent.fatal_events import (
    RuntimeFailureBoundary,
    build_fatal_runtime_event,
)
from app.agent.loop_fingerprint import build_loop_fingerprint
from app.agent.runtime_policy import (
    RuntimePolicy,
    USER_CANCELLED_TERMINATION_REASON,
)
from app.agent.runtime_policy_contracts import RuntimeState
from app.agent.retry_waiter import RetryWaiter
from app.context.manager import ContextManager
from app.db.models.tool_call import ToolCall
from app.db.persistence import PersistenceService
from app.llm.gateway import LLMGateway, LLMGatewayResult
from app.tools.router import PreparedToolCall, ToolRouter


class AgentRuntime:
    """执行由 RuntimePolicy 控制终止条件的模型—工具循环。"""

    def __init__(
        self,
        persistence: PersistenceService,
        context_manager: ContextManager,
        llm_gateway: LLMGateway,
        tool_router: ToolRouter,
        runtime_policy: RuntimePolicy,
        retry_waiter: RetryWaiter,
    ) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        if not isinstance(context_manager, ContextManager):
            raise TypeError("context_manager must be a ContextManager")
        if not isinstance(llm_gateway, LLMGateway):
            raise TypeError("llm_gateway must be an LLMGateway")
        if not isinstance(tool_router, ToolRouter):
            raise TypeError("tool_router must be a ToolRouter")
        if not isinstance(runtime_policy, RuntimePolicy):
            raise TypeError("runtime_policy must be a RuntimePolicy")
        if not isinstance(retry_waiter, RetryWaiter):
            raise TypeError("retry_waiter must be a RetryWaiter")
        self._persistence = persistence
        self._context_manager = context_manager
        self._llm_gateway = llm_gateway
        self._tool_router = tool_router
        self._runtime_policy = runtime_policy
        self._retry_waiter = retry_waiter

    def run(
        self,
        task_id: str,
        cancellation_token: CancellationToken | None = None,
    ) -> AgentResult:
        """循环执行模型动作，直到最终回答、失败或达到最大步数。"""

        cancellation_wait_token = cancellation_token
        if cancellation_token is None:
            cancellation_token = CancellationToken()
        if not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be a CancellationToken")

        initial_state = RuntimeState()
        try:
            task = self._persistence.get_task(task_id)
        except Exception as exc:
            return self._fail_without_step_from_exception(
                exc,
                RuntimeFailureBoundary.PERSISTENCE_SERVICE,
                initial_state,
            )
        if task is None:
            return AgentResult(
                status=TaskStatus.FAILED,
                error=f"Task {task_id} was not found",
            )

        state = RuntimeState()
        while True:
            if cancellation_token.is_cancelled():
                state = replace(state, cancel_requested=True)
            try:
                before_step_decision = self._runtime_policy.evaluate(state)
            except Exception as exc:
                return self._fail_without_step_from_exception(
                    exc,
                    RuntimeFailureBoundary.RUNTIME_POLICY,
                    state,
                )

            decision_result = self._apply_runtime_decision(
                before_step_decision,
                state,
            )
            if decision_result is not None:
                return decision_result

            step_number = state.next_step_number

            try:
                step = self._persistence.create_agent_step(
                    task_id,
                    step_number,
                )
            except Exception as exc:
                return self._fail_without_step_from_exception(
                    exc,
                    RuntimeFailureBoundary.PERSISTENCE_SERVICE,
                    state,
                )

            try:
                context_result = self._context_manager.build(task_id)
            except Exception as exc:
                return self._fail_step_from_exception(
                    step.id,
                    exc,
                    RuntimeFailureBoundary.CONTEXT_MANAGER,
                    state,
                )
            cancelled = self._cancel_if_requested(
                cancellation_token,
                state,
                step_id=step.id,
            )
            if cancelled is not None:
                return cancelled
            if isinstance(context_result, RuntimeEvent):
                return self._fail_step_from_event(
                    step.id,
                    context_result,
                    state,
                )

            llm_retry_count = 0
            while True:
                retry_state = replace(
                    state,
                    llm_retry_count=llm_retry_count,
                )
                cancelled = self._cancel_if_requested(
                    cancellation_token,
                    retry_state,
                    step_id=step.id,
                )
                if cancelled is not None:
                    return cancelled

                try:
                    gateway_result = self._llm_gateway.invoke(context_result)
                except Exception as exc:
                    cancelled = self._cancel_if_requested(
                        cancellation_token,
                        retry_state,
                        step_id=step.id,
                    )
                    if cancelled is not None:
                        return cancelled
                    return self._fail_step_from_exception(
                        step.id,
                        exc,
                        RuntimeFailureBoundary.LLM_GATEWAY,
                        retry_state,
                    )

                cancelled = self._cancel_if_requested(
                    cancellation_token,
                    retry_state,
                    step_id=step.id,
                )
                if cancelled is not None:
                    return cancelled

                event = _llm_runtime_event(gateway_result)
                if event is None:
                    break

                try:
                    decision = self._runtime_policy.evaluate(
                        retry_state,
                        event,
                    )
                except Exception as exc:
                    return self._fail_step_from_exception(
                        step.id,
                        exc,
                        RuntimeFailureBoundary.RUNTIME_POLICY,
                        retry_state,
                    )

                decision_result = self._apply_runtime_decision(
                    decision,
                    retry_state,
                    step_id=step.id,
                )
                if decision_result is not None:
                    return decision_result

                if decision.decision is RuntimeDecisionType.RETRY:
                    llm_retry_count += 1
                    try:
                        wait_cancelled = self._retry_waiter.wait(
                            decision.retry_after_seconds,
                            cancellation_wait_token,
                        )
                    except Exception as exc:
                        return self._fail_step_from_exception(
                            step.id,
                            exc,
                            RuntimeFailureBoundary.RETRY_WAITER,
                            retry_state,
                        )
                    if wait_cancelled or cancellation_token.is_cancelled():
                        cancelled = self._cancel_if_requested(
                            cancellation_token,
                            replace(
                                state,
                                llm_retry_count=llm_retry_count,
                            ),
                            step_id=step.id,
                        )
                        if cancelled is not None:
                            return cancelled
                    continue
                return self._decision_mapping_failure(
                    decision,
                    step_id=step.id,
                )

            if isinstance(gateway_result, FinalAction):
                try:
                    return self._complete_step(
                        task_id,
                        step.id,
                        gateway_result,
                    )
                except Exception as exc:
                    return self._fail_step_from_exception(
                        step.id,
                        exc,
                        RuntimeFailureBoundary.PERSISTENCE_SERVICE,
                        state,
                    )

            if isinstance(gateway_result, ToolCallsAction):
                try:
                    tool_results = self._execute_tool_calls(
                        task_id,
                        step.id,
                        task.workspace,
                        gateway_result,
                        cancellation_token,
                    )
                except Exception as exc:
                    cancelled = self._cancel_if_requested(
                        cancellation_token,
                        state,
                        step_id=step.id,
                        close_open_tool_calls=True,
                    )
                    if cancelled is not None:
                        return cancelled
                    return self._fail_step_from_exception(
                        step.id,
                        exc,
                        RuntimeFailureBoundary.TOOL_ROUTER,
                        state,
                        close_open_tool_calls=True,
                    )
                if (
                    tool_results is None
                    or cancellation_token.is_cancelled()
                ):
                    cancelled = self._cancel_if_requested(
                        cancellation_token,
                        state,
                        step_id=step.id,
                        close_open_tool_calls=True,
                    )
                    if cancelled is not None:
                        return cancelled
                    return self._fail_step_from_exception(
                        step.id,
                        RuntimeError(
                            "Tool execution reported cancellation without a "
                            "cancelled token"
                        ),
                        RuntimeFailureBoundary.TOOL_ROUTER,
                        state,
                        close_open_tool_calls=True,
                    )
                try:
                    loop_fingerprint = build_loop_fingerprint(
                        gateway_result.tool_calls,
                        tool_results,
                    )
                    next_state = _advance_after_tool_step(
                        state,
                        loop_fingerprint,
                    )
                except Exception as exc:
                    return self._fail_step_from_exception(
                        step.id,
                        exc,
                        RuntimeFailureBoundary.AGENT_RUNTIME,
                        state,
                    )
                try:
                    self._persistence.finish_agent_step(
                        step.id,
                        AgentStepStatus.COMPLETED,
                    )
                except Exception as exc:
                    return self._fail_step_from_exception(
                        step.id,
                        exc,
                        RuntimeFailureBoundary.PERSISTENCE_SERVICE,
                        state,
                    )
                state = next_state
                continue

            return self._fail_step_from_exception(
                step.id,
                RuntimeError(
                    "LLMGateway returned unsupported result: "
                    f"{type(gateway_result).__name__}"
                ),
                RuntimeFailureBoundary.LLM_GATEWAY,
                state,
            )

    def _execute_tool_calls(
        self,
        task_id: str,
        step_id: str,
        workspace: str,
        action: ToolCallsAction,
        cancellation_token: CancellationToken,
    ) -> tuple[ToolResult, ...] | None:
        """持久化全部标准 ToolResult；只有抛出的异常才进入 Fatal Policy。"""

        if cancellation_token.is_cancelled():
            return None

        _, records = self._persistence.save_tool_calls_action(
            task_id,
            step_id,
            action,
        )
        records_by_provider_id = _index_tool_call_records(records)
        persisted_results: list[ToolResult] = []

        for request in sorted(
            action.tool_calls,
            key=lambda tool_call: tool_call.call_index,
        ):
            if cancellation_token.is_cancelled():
                return None

            record = records_by_provider_id[request.tool_call_id]
            prepared = self._tool_router.prepare(request, workspace)
            if cancellation_token.is_cancelled():
                return None
            if isinstance(prepared, ToolResult):
                self._persistence.save_tool_result(record.id, prepared)
                persisted_results.append(prepared)
                if cancellation_token.is_cancelled():
                    return None
                continue

            assert isinstance(prepared, PreparedToolCall)
            self._persistence.start_tool_call(record.id)
            if cancellation_token.is_cancelled():
                return None
            result = self._tool_router.execute(prepared)
            self._persistence.save_tool_result(record.id, result)
            persisted_results.append(result)
            if cancellation_token.is_cancelled():
                return None

        return tuple(persisted_results)

    def _cancel_if_requested(
        self,
        cancellation_token: CancellationToken,
        state: RuntimeState,
        *,
        step_id: str | None = None,
        close_open_tool_calls: bool = False,
    ) -> AgentResult | None:
        if not cancellation_token.is_cancelled():
            return None

        cancellation_event = RuntimeEvent(
            event_type=RuntimeEventType.USER_CANCELLED,
            source="cancellation_token",
            message=cancellation_token.reason or "User requested cancellation",
            details={"reason": USER_CANCELLED_TERMINATION_REASON},
        )
        cancellation_state = replace(state, cancel_requested=True)
        try:
            decision = self._runtime_policy.evaluate(
                cancellation_state,
                cancellation_event,
            )
        except Exception as exc:
            if step_id is None:
                return self._fail_without_step_from_exception(
                    exc,
                    RuntimeFailureBoundary.RUNTIME_POLICY,
                    state,
                )
            return self._fail_step_from_exception(
                step_id,
                exc,
                RuntimeFailureBoundary.RUNTIME_POLICY,
                state,
                close_open_tool_calls=close_open_tool_calls,
            )

        decision_result = self._apply_runtime_decision(
            decision,
            cancellation_state,
            step_id=step_id,
            close_open_tool_calls=close_open_tool_calls,
        )
        if decision_result is not None:
            return decision_result
        return self._decision_mapping_failure(
            decision,
            step_id=step_id,
            close_open_tool_calls=close_open_tool_calls,
        )

    def _apply_runtime_decision(
        self,
        decision: RuntimeDecision,
        state: RuntimeState,
        *,
        step_id: str | None = None,
        close_open_tool_calls: bool = False,
    ) -> AgentResult | None:
        """把 RuntimePolicy 的纯数据决策统一映射到持久化生命周期。"""

        if not isinstance(decision, RuntimeDecision):
            raise TypeError("decision must be a RuntimeDecision")
        if not isinstance(state, RuntimeState):
            raise TypeError("state must be a RuntimeState")

        if decision.decision is RuntimeDecisionType.CONTINUE:
            if step_id is None:
                return None
            return self._decision_mapping_failure(
                decision,
                step_id=step_id,
                close_open_tool_calls=close_open_tool_calls,
            )

        if decision.decision is RuntimeDecisionType.RETRY:
            if step_id is not None:
                return None
            return self._decision_mapping_failure(decision)

        reason = decision.reason or (
            f"Runtime decision {decision.decision.value} has no reason"
        )
        if decision.decision is RuntimeDecisionType.FAILED:
            if step_id is None:
                return AgentResult(status=TaskStatus.FAILED, error=reason)
            if close_open_tool_calls:
                reason = self._close_open_tool_calls_best_effort(
                    step_id,
                    reason,
                )
            return self._fail_step_best_effort(step_id, reason)

        if decision.decision not in {
            RuntimeDecisionType.CANCELLED,
            RuntimeDecisionType.TERMINATED,
        }:
            return self._decision_mapping_failure(
                decision,
                step_id=step_id,
                close_open_tool_calls=close_open_tool_calls,
            )

        task_status = (
            TaskStatus.CANCELLED
            if decision.decision is RuntimeDecisionType.CANCELLED
            else TaskStatus.TERMINATED
        )
        if step_id is not None:
            try:
                if close_open_tool_calls:
                    self._persistence.interrupt_open_tool_calls(
                        step_id,
                        reason,
                    )
                self._persistence.finish_agent_step(
                    step_id,
                    AgentStepStatus.INTERRUPTED,
                )
            except Exception as exc:
                return self._fail_step_from_exception(
                    step_id,
                    exc,
                    RuntimeFailureBoundary.PERSISTENCE_SERVICE,
                    state,
                    close_open_tool_calls=close_open_tool_calls,
                )
        return AgentResult(
            status=task_status,
            termination_reason=reason,
        )

    def _decision_mapping_failure(
        self,
        decision: RuntimeDecision,
        *,
        step_id: str | None = None,
        close_open_tool_calls: bool = False,
    ) -> AgentResult:
        failure = (
            "RuntimeDecision is invalid for the current lifecycle: "
            f"decision={decision.decision.value}, "
            f"step={'present' if step_id is not None else 'absent'}"
        )
        if step_id is None:
            return AgentResult(status=TaskStatus.FAILED, error=failure)
        if close_open_tool_calls:
            failure = self._close_open_tool_calls_best_effort(
                step_id,
                failure,
            )
        return self._fail_step_best_effort(step_id, failure)

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

    def _fail_without_step_from_exception(
        self,
        error: Exception,
        boundary: RuntimeFailureBoundary,
        state: RuntimeState,
    ) -> AgentResult:
        event = build_fatal_runtime_event(error, boundary)
        decision = self._evaluate_runtime_event(event, state)
        result = self._apply_runtime_decision(
            decision,
            state,
        )
        if result is not None:
            return result
        return self._decision_mapping_failure(decision)

    def _fail_step_from_exception(
        self,
        step_id: str,
        error: Exception,
        boundary: RuntimeFailureBoundary,
        state: RuntimeState,
        *,
        close_open_tool_calls: bool = False,
    ) -> AgentResult:
        event = build_fatal_runtime_event(error, boundary)
        return self._fail_step_from_event(
            step_id,
            event,
            state,
            close_open_tool_calls=close_open_tool_calls,
        )

    def _fail_step_from_event(
        self,
        step_id: str,
        event: RuntimeEvent,
        state: RuntimeState,
        *,
        close_open_tool_calls: bool = False,
    ) -> AgentResult:
        decision = self._evaluate_runtime_event(event, state)
        result = self._apply_runtime_decision(
            decision,
            state,
            step_id=step_id,
            close_open_tool_calls=close_open_tool_calls,
        )
        if result is not None:
            return result
        return self._decision_mapping_failure(
            decision,
            step_id=step_id,
            close_open_tool_calls=close_open_tool_calls,
        )

    def _evaluate_runtime_event(
        self,
        event: RuntimeEvent,
        state: RuntimeState,
    ) -> RuntimeDecision:
        try:
            return self._runtime_policy.evaluate(state, event)
        except Exception as exc:
            policy_event = build_fatal_runtime_event(
                exc,
                RuntimeFailureBoundary.RUNTIME_POLICY,
            )
            return RuntimeDecision(
                RuntimeDecisionType.FAILED,
                reason=_runtime_event_failure_message(policy_event),
            )

    def _close_open_tool_calls_best_effort(
        self,
        step_id: str,
        failure: str,
    ) -> str:
        try:
            self._persistence.fail_open_tool_calls(step_id, failure)
        except Exception as exc:
            persistence_event = build_fatal_runtime_event(
                exc,
                RuntimeFailureBoundary.PERSISTENCE_SERVICE,
            )
            return (
                f"{failure}; unable to close open ToolCalls: "
                f"{_runtime_event_failure_message(persistence_event)}"
            )
        return failure

    def _fail_step_best_effort(
        self,
        step_id: str,
        failure: str,
    ) -> AgentResult:
        try:
            self._persistence.finish_agent_step(
                step_id,
                AgentStepStatus.FAILED,
                error=failure,
            )
        except Exception as exc:
            persistence_event = build_fatal_runtime_event(
                exc,
                RuntimeFailureBoundary.PERSISTENCE_SERVICE,
            )
            failure = (
                f"{failure}; unable to persist AgentStep failure: "
                f"{_runtime_event_failure_message(persistence_event)}"
            )
        return AgentResult(status=TaskStatus.FAILED, error=failure)


def _index_tool_call_records(
    records: tuple[ToolCall, ...],
) -> dict[str, ToolCall]:
    records_by_provider_id = {
        record.provider_call_id: record for record in records
    }
    if len(records_by_provider_id) != len(records):
        raise RuntimeError("persisted ToolCall provider_call_id values are not unique")
    return records_by_provider_id


def _advance_after_tool_step(
    state: RuntimeState,
    loop_fingerprint: str,
) -> RuntimeState:
    if not isinstance(state, RuntimeState):
        raise TypeError("state must be a RuntimeState")
    if not isinstance(loop_fingerprint, str):
        raise TypeError("loop_fingerprint must be a string")
    if not loop_fingerprint.strip():
        raise ValueError("loop_fingerprint must not be blank")

    consecutive_loop_count = (
        state.consecutive_loop_count + 1
        if state.last_loop_fingerprint == loop_fingerprint
        else 1
    )
    return RuntimeState(
        next_step_number=state.next_step_number + 1,
        last_loop_fingerprint=loop_fingerprint,
        consecutive_loop_count=consecutive_loop_count,
    )


def _invalid_action_event(action: InvalidAction) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.INVALID_ACTION,
        source="action_parser",
        message=action.reason,
    )


def _llm_runtime_event(result: LLMGatewayResult) -> RuntimeEvent | None:
    if isinstance(result, InvalidAction):
        return _invalid_action_event(result)
    if isinstance(result, RuntimeEvent):
        return result
    return None


def _runtime_event_failure_message(event: RuntimeEvent) -> str:
    return (
        f"Runtime event {event.event_type.value} from {event.source}: "
        f"{event.message}"
    )
