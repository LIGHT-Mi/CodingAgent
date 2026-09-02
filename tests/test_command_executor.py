import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

from app.tools import (
    CommandEnvironmentBuilder,
    CommandExecutionResult,
    CommandExecutor,
    CommandProcessStartError,
    RunCommandArguments,
)


class CommandExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temporary_directory.name).resolve(strict=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _executor(
        self,
        *,
        timeout: float = 2,
        grace: float = 0.2,
        output_limit: int = 1024,
        source_environment: dict[str, str] | None = None,
    ) -> CommandExecutor:
        return CommandExecutor(
            timeout_seconds=timeout,
            termination_grace_seconds=grace,
            max_output_bytes_per_stream=output_limit,
            environment_builder=CommandEnvironmentBuilder(source_environment),
        )

    def test_executes_without_shell_and_collects_both_streams(self) -> None:
        arguments = RunCommandArguments(
            [
                sys.executable,
                "-c",
                (
                    "import os,sys; "
                    "sys.stdout.write('out:' + os.getcwd()); "
                    "sys.stderr.write('err')"
                ),
            ]
        )

        result = self._executor().execute(arguments, self.cwd)

        self.assertIsInstance(result, CommandExecutionResult)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.text, f"out:{self.cwd}")
        self.assertEqual(result.stderr.text, "err")
        self.assertFalse(result.timed_out)
        self.assertIsNone(result.termination_signal)
        self.assertFalse(result.forced_termination)

    def test_uses_devnull_stdin_and_controlled_environment(self) -> None:
        result = self._executor(
            source_environment={
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "DEEPSEEK_API_KEY": "must-not-leak",
                "DATABASE_URL": "must-not-leak",
            }
        ).execute(
            RunCommandArguments(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,sys; "
                        "data=sys.stdin.read(); "
                        "print(len(data), "
                        "'DEEPSEEK_API_KEY' in os.environ, "
                        "'DATABASE_URL' in os.environ)"
                    ),
                ]
            ),
            self.cwd,
        )

        self.assertEqual(result.stdout.text.strip(), "0 False False")

    def test_large_stdout_and_stderr_are_drained_without_deadlock(self) -> None:
        result = self._executor(output_limit=128).execute(
            RunCommandArguments(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os; "
                        "os.write(1, b'A' * 200000); "
                        "os.write(2, b'B' * 200000)"
                    ),
                ]
            ),
            self.cwd,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.original_byte_count, 200000)
        self.assertEqual(result.stderr.original_byte_count, 200000)
        self.assertTrue(result.stdout.truncated)
        self.assertTrue(result.stderr.truncated)

    def test_timeout_terminates_the_entire_process_group(self) -> None:
        child_pid_file = self.cwd / "child.pid"
        parent_code = (
            "import pathlib,subprocess,sys; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid))"
        )

        result = self._executor(timeout=0.4, grace=0.2).execute(
            RunCommandArguments([sys.executable, "-c", parent_code]),
            self.cwd,
        )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 0)
        self.assertIn(result.termination_signal, (signal.SIGTERM, signal.SIGKILL))
        self.assertTrue(child_pid_file.exists())
        child_pid = int(child_pid_file.read_text())
        self._assert_process_gone(child_pid)

    def test_escalates_to_sigkill_after_grace_period(self) -> None:
        result = self._executor(timeout=0.2, grace=0.1).execute(
            RunCommandArguments(
                [
                    sys.executable,
                    "-c",
                    (
                        "import signal,time; "
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        "time.sleep(30)"
                    ),
                ]
            ),
            self.cwd,
        )

        self.assertTrue(result.timed_out)
        self.assertTrue(result.forced_termination)
        self.assertEqual(result.termination_signal, signal.SIGKILL)
        self.assertEqual(result.exit_code, -signal.SIGKILL)

    def test_wraps_process_creation_failures(self) -> None:
        with self.assertRaises(CommandProcessStartError) as raised:
            self._executor().execute(
                RunCommandArguments(["definitely-not-a-real-command-32947"]),
                self.cwd,
            )
        self.assertIn("failed to start command", str(raised.exception))

    def test_validates_executor_limits_and_resolved_cwd(self) -> None:
        invalid_numbers = (0, -1, float("inf"), float("nan"), True)
        for invalid in invalid_numbers:
            with self.subTest(timeout=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    CommandExecutor(
                        timeout_seconds=invalid,
                        termination_grace_seconds=1,
                        max_output_bytes_per_stream=16,
                    )

        with self.assertRaises(ValueError):
            self._executor().execute(
                RunCommandArguments([sys.executable, "--version"]),
                Path("."),
            )

    def _assert_process_gone(self, process_id: int) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                return
            time.sleep(0.02)
        self.fail(f"child process {process_id} is still alive")


if __name__ == "__main__":
    unittest.main()
