import unittest
from dataclasses import FrozenInstanceError

from pydantic import ValidationError

from app.context import ContextLimits
from app.core.config import Settings


class ContextLimitsTests(unittest.TestCase):
    def test_accepts_small_injected_character_limits(self) -> None:
        limits = ContextLimits(
            max_context_characters=500,
            max_tool_result_characters=100,
        )

        self.assertEqual(limits.max_context_characters, 500)
        self.assertEqual(limits.max_tool_result_characters, 100)

    def test_is_an_immutable_value_object(self) -> None:
        limits = ContextLimits(500, 100)

        with self.assertRaises(FrozenInstanceError):
            limits.max_context_characters = 600  # type: ignore[misc]

    def test_requires_positive_integer_limits(self) -> None:
        invalid_values = (0, -1, 1.5, True, "100")

        for field_name in (
            "max_context_characters",
            "max_tool_result_characters",
        ):
            for invalid_value in invalid_values:
                with self.subTest(
                    field_name=field_name,
                    invalid_value=invalid_value,
                ):
                    values = {
                        "max_context_characters": 500,
                        "max_tool_result_characters": 100,
                    }
                    values[field_name] = invalid_value
                    with self.assertRaises((TypeError, ValueError)):
                        ContextLimits(**values)  # type: ignore[arg-type]


class ContextBudgetConfigurationTests(unittest.TestCase):
    def test_defaults_match_the_context_budget_contract(self) -> None:
        fields = Settings.model_fields

        self.assertEqual(fields["MAX_LLM_CONTEXT_CHARACTERS"].default, 60_000)
        self.assertEqual(
            fields["MAX_CONTEXT_TOOL_RESULT_CHARACTERS"].default,
            12_000,
        )

    def test_values_are_loaded_and_typed(self) -> None:
        configured = Settings(
            DATABASE_URL="sqlite://",
            MAX_LLM_CONTEXT_CHARACTERS="500",
            MAX_CONTEXT_TOOL_RESULT_CHARACTERS="100",
            _env_file=None,
        )

        self.assertEqual(configured.MAX_LLM_CONTEXT_CHARACTERS, 500)
        self.assertEqual(configured.MAX_CONTEXT_TOOL_RESULT_CHARACTERS, 100)

    def test_values_must_be_positive(self) -> None:
        for values in (
            {"MAX_LLM_CONTEXT_CHARACTERS": 0},
            {"MAX_CONTEXT_TOOL_RESULT_CHARACTERS": 0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    Settings(
                        DATABASE_URL="sqlite://",
                        _env_file=None,
                        **values,
                    )


if __name__ == "__main__":
    unittest.main()
