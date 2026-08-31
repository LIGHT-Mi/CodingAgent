"""把运行状态与运行事件映射为纯数据决策。"""

from __future__ import annotations

import math

from app.agent.contracts import (
    RuntimeDecision,
    RuntimeDecisionType,
    RuntimeEvent,
    RuntimeEventType,
)
from app.agent.runtime_policy_contracts import (
    RuntimePolicyConfig,
    RuntimeState,
)


MAX_STEPS_TERMINATION_REASON = "MAX_STEPS"
LOOP_DETECTED_TERMINATION_REASON = "LOOP_DETECTED"
USER_CANCELLED_TERMINATION_REASON = "USER_CANCELLED"

_RETRYABLE_EVENT_TYPES = frozenset(
    {
        RuntimeEventType.LLM_TIMEOUT,
        RuntimeEventType.LLM_RATE_LIMIT,
        RuntimeEventType.LLM_NETWORK_ERROR,
        RuntimeEventType.INVALID_ACTION,
    }
)

_TERMINAL_FAILURE_EVENT_TYPES = frozenset(
    {
        RuntimeEventType.CONTEXT_OVERFLOW,
        RuntimeEventType.FATAL_TOOL_ERROR,
        RuntimeEventType.FATAL_SYSTEM_ERROR,
        RuntimeEventType.INFRASTRUCTURE_ERROR,
        RuntimeEventType.AGENT_STATE_CORRUPTED,
    }
)


class RuntimePolicy:
    """只根据不可变状态快照和可选事件返回运行决定。"""

    def __init__(self, config: RuntimePolicyConfig) -> None:
        if not isinstance(config, RuntimePolicyConfig):
            raise TypeError("config must be a RuntimePolicyConfig")
        self._config = config

    @property
    def config(self) -> RuntimePolicyConfig:
        return self._config

    def evaluate(
        self,
        state: RuntimeState,
        event: RuntimeEvent | None = None,
    ) -> RuntimeDecision:
        """按固定优先级判断继续、重试、失败、取消或规则终止。"""

        if not isinstance(state, RuntimeState):
            raise TypeError("state must be a RuntimeState")
        if event is not None and not isinstance(event, RuntimeEvent):
            raise TypeError("event must be a RuntimeEvent or None")

        if state.cancel_requested or (
            event is not None
            and event.event_type is RuntimeEventType.USER_CANCELLED
        ):
            return RuntimeDecision(
                RuntimeDecisionType.CANCELLED,
                reason=USER_CANCELLED_TERMINATION_REASON,
            )

        if (
            state.consecutive_loop_count
            >= self._config.loop_repeat_threshold
        ):
            return RuntimeDecision(
                RuntimeDecisionType.TERMINATED,
                reason=LOOP_DETECTED_TERMINATION_REASON,
            )

        if state.next_step_number >= self._config.max_agent_steps:
            return RuntimeDecision(
                RuntimeDecisionType.TERMINATED,
                reason=MAX_STEPS_TERMINATION_REASON,
            )

        if event is None:
            return RuntimeDecision(RuntimeDecisionType.CONTINUE)

        if event.event_type in _RETRYABLE_EVENT_TYPES:
            return self._evaluate_retryable_event(state, event)

        if event.event_type in _TERMINAL_FAILURE_EVENT_TYPES:
            return RuntimeDecision(
                RuntimeDecisionType.FAILED,
                reason=_format_event_reason(event),
            )

        return RuntimeDecision(
            RuntimeDecisionType.FAILED,
            reason=(
                "Unsupported runtime event failed closed: "
                f"{event.event_type.value}"
            ),
        )

    def _evaluate_retryable_event(
        self,
        state: RuntimeState,
        event: RuntimeEvent,
    ) -> RuntimeDecision:
        if state.llm_retry_count < self._config.max_llm_retries:
            return RuntimeDecision(
                RuntimeDecisionType.RETRY,
                reason=_format_event_reason(event),
                retry_after_seconds=self._retry_delay_seconds(state, event),
            )
        return RuntimeDecision(
            RuntimeDecisionType.FAILED,
            reason=(
                f"{event.event_type.value} exhausted "
                f"{self._config.max_llm_retries} LLM retries: "
                f"{event.message}"
            ),
        )

    def _retry_delay_seconds(
        self,
        state: RuntimeState,
        event: RuntimeEvent,
    ) -> float:
        retry_after_seconds = _rate_limit_retry_after_seconds(event)
        if retry_after_seconds is not None:
            return min(
                retry_after_seconds,
                self._config.retry_max_seconds,
            )

        base_seconds = self._config.retry_base_seconds
        maximum_seconds = self._config.retry_max_seconds
        if base_seconds == 0 or maximum_seconds == 0:
            return 0.0
        try:
            delay_seconds = math.ldexp(
                base_seconds,
                state.llm_retry_count,
            )
        except OverflowError:
            return maximum_seconds
        return min(delay_seconds, maximum_seconds)


def _format_event_reason(event: RuntimeEvent) -> str:
    return (
        f"Runtime event {event.event_type.value} from "
        f"{event.source}: {event.message}"
    )


def _rate_limit_retry_after_seconds(event: RuntimeEvent) -> float | None:
    if event.event_type is not RuntimeEventType.LLM_RATE_LIMIT:
        return None
    value = event.details.get("retry_after_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        return None
    return normalized
