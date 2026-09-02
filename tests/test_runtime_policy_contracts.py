import unittest
from dataclasses import FrozenInstanceError

from pydantic import ValidationError

from app.agent import RuntimePolicyConfig, RuntimeState
from app.core.config import Settings


class RuntimePolicyConfigTests(unittest.TestCase):
    def test_defaults_match_application_plan(self) -> None:
        config = RuntimePolicyConfig()

        self.assertEqual(config.max_agent_steps, 8)
        self.assertEqual(config.max_llm_retries, 2)
        self.assertEqual(config.retry_base_seconds, 1.0)
        self.assertEqual(config.retry_max_seconds, 8.0)
        self.assertEqual(config.loop_repeat_threshold, 3)

    def test_accepts_small_injected_test_configuration(self) -> None:
        config = RuntimePolicyConfig(
            max_agent_steps=2,
            max_llm_retries=0,
            retry_base_seconds=0,
            retry_max_seconds=0,
            loop_repeat_threshold=2,
        )

        self.assertEqual(config.max_llm_retries, 0)
        self.assertEqual(config.retry_base_seconds, 0.0)

    def test_rejects_invalid_limits(self) -> None:
        invalid_values = (
            {"max_agent_steps": 0},
            {"max_llm_retries": -1},
            {"retry_base_seconds": -1},
            {"retry_max_seconds": float("inf")},
            {"retry_base_seconds": 2, "retry_max_seconds": 1},
            {"loop_repeat_threshold": 1},
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RuntimePolicyConfig(**values)

    def test_rejects_boolean_numeric_values(self) -> None:
        for field_name in (
            "max_agent_steps",
            "max_llm_retries",
            "retry_base_seconds",
            "retry_max_seconds",
            "loop_repeat_threshold",
        ):
            with self.subTest(field_name=field_name), self.assertRaises(TypeError):
                RuntimePolicyConfig(**{field_name: True})


class RuntimeStateTests(unittest.TestCase):
    def test_initial_state_has_no_retry_cancel_or_loop_history(self) -> None:
        state = RuntimeState()

        self.assertEqual(state.next_step_number, 0)
        self.assertEqual(state.llm_retry_count, 0)
        self.assertFalse(state.cancel_requested)
        self.assertIsNone(state.last_loop_fingerprint)
        self.assertEqual(state.consecutive_loop_count, 0)

    def test_accepts_active_runtime_state(self) -> None:
        state = RuntimeState(
            next_step_number=4,
            llm_retry_count=1,
            cancel_requested=True,
            last_loop_fingerprint="sha256:fingerprint",
            consecutive_loop_count=2,
        )

        self.assertEqual(state.next_step_number, 4)
        self.assertEqual(state.llm_retry_count, 1)
        self.assertTrue(state.cancel_requested)
        self.assertEqual(state.consecutive_loop_count, 2)

    def test_loop_fingerprint_and_count_are_consistent(self) -> None:
        invalid_values = (
            {
                "last_loop_fingerprint": None,
                "consecutive_loop_count": 1,
            },
            {
                "last_loop_fingerprint": "fingerprint",
                "consecutive_loop_count": 0,
            },
            {
                "last_loop_fingerprint": "  ",
                "consecutive_loop_count": 1,
            },
        )
        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RuntimeState(**values)

    def test_rejects_invalid_state_types_and_values(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeState(next_step_number=-1)
        with self.assertRaises(ValueError):
            RuntimeState(llm_retry_count=-1)
        with self.assertRaises(TypeError):
            RuntimeState(cancel_requested=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RuntimeState(next_step_number=True)

    def test_contracts_are_frozen(self) -> None:
        config = RuntimePolicyConfig()
        state = RuntimeState()

        with self.assertRaises(FrozenInstanceError):
            config.max_agent_steps = 1  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            state.next_step_number = 1  # type: ignore[misc]


class RuntimePolicySettingsTests(unittest.TestCase):
    def test_defaults_match_runtime_policy_contract(self) -> None:
        fields = Settings.model_fields

        self.assertEqual(fields["MAX_AGENT_STEPS"].default, 8)
        self.assertEqual(fields["MAX_LLM_RETRIES"].default, 2)
        self.assertEqual(fields["LLM_RETRY_BASE_SECONDS"].default, 1.0)
        self.assertEqual(fields["LLM_RETRY_MAX_SECONDS"].default, 8.0)
        self.assertEqual(fields["AGENT_LOOP_REPEAT_THRESHOLD"].default, 3)

    def test_values_are_loaded_and_typed(self) -> None:
        configured = Settings(
            DATABASE_URL="sqlite://",
            MAX_AGENT_STEPS="4",
            MAX_LLM_RETRIES="1",
            LLM_RETRY_BASE_SECONDS="0.5",
            LLM_RETRY_MAX_SECONDS="3",
            AGENT_LOOP_REPEAT_THRESHOLD="2",
            _env_file=None,
        )

        self.assertEqual(configured.MAX_AGENT_STEPS, 4)
        self.assertEqual(configured.MAX_LLM_RETRIES, 1)
        self.assertEqual(configured.LLM_RETRY_BASE_SECONDS, 0.5)
        self.assertEqual(configured.LLM_RETRY_MAX_SECONDS, 3.0)
        self.assertEqual(configured.AGENT_LOOP_REPEAT_THRESHOLD, 2)

    def test_field_limits_are_validated(self) -> None:
        invalid_values = (
            {"MAX_AGENT_STEPS": 0},
            {"MAX_LLM_RETRIES": -1},
            {"LLM_RETRY_BASE_SECONDS": -1},
            {"LLM_RETRY_MAX_SECONDS": -1},
            {"AGENT_LOOP_REPEAT_THRESHOLD": 1},
        )

        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Settings(
                    DATABASE_URL="sqlite://",
                    _env_file=None,
                    **values,
                )


if __name__ == "__main__":
    unittest.main()
