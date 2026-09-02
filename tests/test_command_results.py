import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent import ToolResultStatus
from app.tools import (
    CommandExecutionError,
    CommandExecutor,
    CommandResultBuilder,
    RunCommandArguments,
    RunCommandTool,
)


class CommandResultSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temporary_directory.name).resolve(strict=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _tool(
        self,
        *,
        timeout: float = 2,
        grace: float = 0.1,
        output_limit: int = 1024,
    ) -> RunCommandTool:
        return RunCommandTool(
            CommandExecutor(
                timeout_seconds=timeout,
                termination_grace_seconds=grace,
                max_output_bytes_per_stream=output_limit,
            )
        )

    def test_nonzero_exit_code_is_completed_observation(self) -> None:
        result = self._tool().execute(
            "call-nonzero",
            RunCommandArguments(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; print('1 failed'); "
                        "print('trace', file=sys.stderr); sys.exit(3)"
                    ),
                ]
            ),
            self.cwd,
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["exit_code"], 3)
        self.assertEqual(result.metadata["stdout"], "1 failed\n")
        self.assertEqual(result.metadata["stderr"], "trace\n")
        self.assertFalse(result.metadata["timeout"])
        content = result.content or ""
        self.assertIn("status: COMPLETED", content)
        self.assertIn("exit_code: 3", content)
        self.assertIn("stdout:\n1 failed", content)
        self.assertIn("stderr:\ntrace", content)

    def test_zero_exit_and_truncated_output_have_complete_metadata(self) -> None:
        result = self._tool(output_limit=16).execute(
            "call-truncated",
            RunCommandArguments(
                [sys.executable, "-c", "print('BEGIN' + 'x' * 100 + 'END')"]
            ),
            self.cwd,
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.metadata["exit_code"], 0)
        self.assertTrue(result.metadata["stdout_truncated"])
        self.assertGreater(result.metadata["stdout_discarded_byte_count"], 0)
        self.assertIn("stdout truncated", str(result.metadata["stdout"]))
        self.assertIn("stdout_truncated: true", result.content or "")
        self.assertIn("stderr_truncated: false", result.content or "")

    def test_timeout_is_observation_after_process_group_cleanup(self) -> None:
        result = self._tool(timeout=0.15, grace=0.1).execute(
            "call-timeout",
            RunCommandArguments(
                [sys.executable, "-c", "import time; time.sleep(30)"]
            ),
            self.cwd,
        )

        self.assertEqual(result.status, ToolResultStatus.TIMEOUT)
        self.assertIsNotNone(result.error)
        self.assertTrue(result.metadata["timeout"])
        self.assertEqual(result.metadata["timeout_seconds"], 0.15)
        self.assertIn(
            result.metadata["termination_signal"],
            (signal.SIGTERM, signal.SIGKILL),
        )
        self.assertIn("status: TIMEOUT", result.content or "")
        self.assertIn("timeout: true", result.content or "")

    def test_process_start_failure_is_error_observation(self) -> None:
        result = self._tool().execute(
            "call-start-error",
            RunCommandArguments(["not-a-real-executable-785422"]),
            self.cwd,
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.metadata["exit_code"])
        self.assertEqual(result.metadata["stdout"], "")
        self.assertEqual(result.metadata["stderr"], "")
        self.assertIn("status: ERROR", result.content or "")
        self.assertIn("exit_code: unavailable", result.content or "")

    def test_non_utf8_stdout_and_stderr_are_completed_observations(self) -> None:
        result = self._tool().execute(
            "call-non-utf8",
            RunCommandArguments(
                [
                    sys.executable,
                    "-c",
                    "import os; os.write(1, b'out\\xff'); os.write(2, b'err\\xfe')",
                ]
            ),
            self.cwd,
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.metadata["stdout"], "out\ufffd")
        self.assertEqual(result.metadata["stderr"], "err\ufffd")
        self.assertIn("stdout:\nout\ufffd", result.content or "")
        self.assertIn("stderr:\nerr\ufffd", result.content or "")

    def test_parameter_and_path_errors_share_the_observation_format(self) -> None:
        builder = CommandResultBuilder()
        argument_error = builder.build_error(
            "call-arguments",
            "command must contain at least one argument",
        )
        path_rejection = builder.build_rejected(
            "call-cwd",
            "working directory is outside the workspace",
            argv=("pytest",),
            cwd="../outside",
        )

        self.assertEqual(argument_error.status, ToolResultStatus.ERROR)
        self.assertEqual(path_rejection.status, ToolResultStatus.REJECTED)
        self.assertIn("command: unavailable", argument_error.content or "")
        self.assertIn("status: REJECTED", path_rejection.content or "")
        self.assertEqual(path_rejection.metadata["argv"], ["pytest"])

    def test_internal_execution_error_is_not_downgraded_to_observation(self) -> None:
        tool = self._tool()
        with patch.object(
            CommandExecutor,
            "execute",
            side_effect=CommandExecutionError("output collector failed"),
        ):
            with self.assertRaises(CommandExecutionError):
                tool.execute(
                    "call-fatal",
                    RunCommandArguments([sys.executable, "--version"]),
                    self.cwd,
                )


if __name__ == "__main__":
    unittest.main()
