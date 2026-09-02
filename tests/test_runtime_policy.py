import unittest

from app.agent import (
    RuntimeDecisionType,
    RuntimeEvent,
    RuntimeEventType,
    RuntimePolicy,
    RuntimePolicyConfig,
    RuntimeState,
)
from app.agent.runtime_policy import (
    LOOP_DETECTED_TERMINATION_REASON,
    MAX_STEPS_TERMINATION_REASON,
    USER_CANCELLED_TERMINATION_REASON,
)


def make_event(
    event_type: RuntimeEventType,
    details=None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type,
        message=f"event {event_type.value}",
        source="unit_test",
        details={} if details is None else details,
    )


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimePolicyConfig(
            max_agent_steps=8,
            max_llm_retries=2,
            retry_base_seconds=1,
            retry_max_seconds=8,
            loop_repeat_threshold=3,
        )
        self.policy = RuntimePolicy(self.config)

    def test_normal_state_continues(self) -> None:
        decision = self.policy.evaluate(RuntimeState())

        self.assertEqual(decision.decision, RuntimeDecisionType.CONTINUE)
        self.assertIsNone(decision.reason)

    def test_retryable_events_retry_before_limit(self) -> None:
        retryable_types = (
            RuntimeEventType.LLM_TIMEOUT,
            RuntimeEventType.LLM_RATE_LIMIT,
            RuntimeEventType.LLM_NETWORK_ERROR,
            RuntimeEventType.INVALID_ACTION,
        )

        for event_type in retryable_types:
            with self.subTest(event_type=event_type):
                decision = self.policy.evaluate(
                    RuntimeState(llm_retry_count=1),
                    make_event(event_type),
                )

                self.assertEqual(decision.decision, RuntimeDecisionType.RETRY)
                self.assertEqual(decision.retry_after_seconds, 2.0)
                self.assertIn(event_type.value, decision.reason)

    def test_exponential_backoff_is_deterministic_and_capped(self) -> None:
        policy = RuntimePolicy(
            RuntimePolicyConfig(
                max_llm_retries=6,
                retry_base_seconds=1,
                retry_max_seconds=8,
            )
        )

        delays = [
            policy.evaluate(
                RuntimeState(llm_retry_count=retry_count),
                make_event(RuntimeEventType.LLM_TIMEOUT),
            ).retry_after_seconds
            for retry_count in range(6)
        ]

        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0, 8.0, 8.0])
        repeated = policy.evaluate(
            RuntimeState(llm_retry_count=2),
            make_event(RuntimeEventType.LLM_TIMEOUT),
        )
        self.assertEqual(repeated.retry_after_seconds, 4.0)

    def test_rate_limit_retry_after_is_preferred_and_capped(self) -> None:
        preferred = self.policy.evaluate(
            RuntimeState(llm_retry_count=0),
            make_event(
                RuntimeEventType.LLM_RATE_LIMIT,
                {"retry_after_seconds": 3.5},
            ),
        )
        capped = self.policy.evaluate(
            RuntimeState(llm_retry_count=0),
            make_event(
                RuntimeEventType.LLM_RATE_LIMIT,
                {"retry_after_seconds": 30},
            ),
        )

        self.assertEqual(preferred.retry_after_seconds, 3.5)
        self.assertEqual(capped.retry_after_seconds, 8.0)

    def test_invalid_rate_limit_retry_after_uses_exponential_backoff(self) -> None:
        for invalid_value in (True, -1, float("nan"), "3"):
            with self.subTest(invalid_value=invalid_value):
                decision = self.policy.evaluate(
                    RuntimeState(llm_retry_count=1),
                    make_event(
                        RuntimeEventType.LLM_RATE_LIMIT,
                        {"retry_after_seconds": invalid_value},
                    ),
                )

                self.assertEqual(decision.retry_after_seconds, 2.0)

    def test_retryable_events_fail_when_retries_are_exhausted(self) -> None:
        retryable_types = (
            RuntimeEventType.LLM_TIMEOUT,
            RuntimeEventType.LLM_RATE_LIMIT,
            RuntimeEventType.LLM_NETWORK_ERROR,
            RuntimeEventType.INVALID_ACTION,
        )

        for event_type in retryable_types:
            with self.subTest(event_type=event_type):
                decision = self.policy.evaluate(
                    RuntimeState(llm_retry_count=2),
                    make_event(event_type),
                )

                self.assertEqual(decision.decision, RuntimeDecisionType.FAILED)
                self.assertIn("exhausted 2 LLM retries", decision.reason)

    def test_zero_retry_configuration_fails_first_retryable_event(self) -> None:
        policy = RuntimePolicy(
            RuntimePolicyConfig(max_llm_retries=0),
        )

        decision = policy.evaluate(
            RuntimeState(),
            make_event(RuntimeEventType.LLM_TIMEOUT),
        )

        self.assertEqual(decision.decision, RuntimeDecisionType.FAILED)

    def test_terminal_events_fail_without_retry(self) -> None:
        terminal_types = (
            RuntimeEventType.CONTEXT_OVERFLOW,
            RuntimeEventType.FATAL_TOOL_ERROR,
            RuntimeEventType.FATAL_SYSTEM_ERROR,
            RuntimeEventType.INFRASTRUCTURE_ERROR,
            RuntimeEventType.AGENT_STATE_CORRUPTED,
        )

        for event_type in terminal_types:
            with self.subTest(event_type=event_type):
                decision = self.policy.evaluate(
                    RuntimeState(),
                    make_event(event_type),
                )

                self.assertEqual(decision.decision, RuntimeDecisionType.FAILED)
                self.assertEqual(decision.retry_after_seconds, 0)
                self.assertIn(event_type.value, decision.reason)

    def test_every_runtime_event_is_handled_fail_closed(self) -> None:
        for event_type in RuntimeEventType:
            with self.subTest(event_type=event_type):
                decision = self.policy.evaluate(
                    RuntimeState(),
                    make_event(event_type),
                )

                self.assertNotEqual(
                    decision.decision,
                    RuntimeDecisionType.CONTINUE,
                )

    def test_user_cancel_state_or_event_cancels(self) -> None:
        decisions = (
            self.policy.evaluate(RuntimeState(cancel_requested=True)),
            self.policy.evaluate(
                RuntimeState(),
                make_event(RuntimeEventType.USER_CANCELLED),
            ),
        )

        for decision in decisions:
            self.assertEqual(
                decision.decision,
                RuntimeDecisionType.CANCELLED,
            )
            self.assertEqual(
                decision.reason,
                USER_CANCELLED_TERMINATION_REASON,
            )

    def test_maximum_steps_terminates_without_event(self) -> None:
        decision = self.policy.evaluate(RuntimeState(next_step_number=8))

        self.assertEqual(decision.decision, RuntimeDecisionType.TERMINATED)
        self.assertEqual(decision.reason, MAX_STEPS_TERMINATION_REASON)

    def test_repeated_loop_terminates_without_event(self) -> None:
        decision = self.policy.evaluate(
            RuntimeState(
                last_loop_fingerprint="same-interaction",
                consecutive_loop_count=3,
            )
        )

        self.assertEqual(decision.decision, RuntimeDecisionType.TERMINATED)
        self.assertEqual(
            decision.reason,
            LOOP_DETECTED_TERMINATION_REASON,
        )

    def test_control_condition_priority_is_cancel_loop_then_max_steps(self) -> None:
        loop_and_max = RuntimeState(
            next_step_number=8,
            last_loop_fingerprint="same-interaction",
            consecutive_loop_count=3,
        )
        cancelled_loop_and_max = RuntimeState(
            next_step_number=8,
            cancel_requested=True,
            last_loop_fingerprint="same-interaction",
            consecutive_loop_count=3,
        )

        self.assertEqual(
            self.policy.evaluate(loop_and_max).reason,
            LOOP_DETECTED_TERMINATION_REASON,
        )
        self.assertEqual(
            self.policy.evaluate(cancelled_loop_and_max).reason,
            USER_CANCELLED_TERMINATION_REASON,
        )

    def test_policy_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            RuntimePolicy(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.policy.evaluate(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.policy.evaluate(RuntimeState(), object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
