import unittest

from pydantic import ValidationError

from app.core.config import Settings


class CommandResourceConfigurationTests(unittest.TestCase):
    def test_command_resource_defaults_match_execution_plan(self) -> None:
        fields = Settings.model_fields

        self.assertEqual(fields["COMMAND_TIMEOUT_SECONDS"].default, 30.0)
        self.assertEqual(
            fields["MAX_COMMAND_OUTPUT_BYTES_PER_STREAM"].default,
            65_536,
        )
        self.assertEqual(
            fields["COMMAND_TERMINATION_GRACE_SECONDS"].default,
            2.0,
        )

    def test_command_resource_values_are_loaded_and_typed(self) -> None:
        configured = Settings(
            DATABASE_URL="sqlite://",
            COMMAND_TIMEOUT_SECONDS="12.5",
            MAX_COMMAND_OUTPUT_BYTES_PER_STREAM="4096",
            COMMAND_TERMINATION_GRACE_SECONDS="0.5",
            _env_file=None,
        )

        self.assertEqual(configured.COMMAND_TIMEOUT_SECONDS, 12.5)
        self.assertEqual(
            configured.MAX_COMMAND_OUTPUT_BYTES_PER_STREAM,
            4096,
        )
        self.assertEqual(configured.COMMAND_TERMINATION_GRACE_SECONDS, 0.5)

    def test_command_resource_values_must_be_positive(self) -> None:
        invalid_values = (
            {"COMMAND_TIMEOUT_SECONDS": 0},
            {"MAX_COMMAND_OUTPUT_BYTES_PER_STREAM": 0},
            {"MAX_COMMAND_OUTPUT_BYTES_PER_STREAM": 1},
            {"COMMAND_TERMINATION_GRACE_SECONDS": 0},
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValidationError):
                    Settings(
                        DATABASE_URL="sqlite://",
                        _env_file=None,
                        **values,
                    )


if __name__ == "__main__":
    unittest.main()
