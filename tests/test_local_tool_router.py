import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from app.agent import ToolCallRequest, ToolResult, ToolResultStatus
from app.tools import (
    CommandExecutionError,
    CommandEnvironmentBuilder,
    CommandExecutor,
    LocalToolRegistry,
    PreparedCommandToolCall,
    RunCommandArguments,
    RunCommandTool,
    ToolRouter,
    WorkspacePathGuard,
    create_local_tool_registry,
)


class LocalToolRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve(strict=True)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.executor = CommandExecutor(
            timeout_seconds=0.2,
            termination_grace_seconds=0.1,
            max_output_bytes_per_stream=1024,
            environment_builder=CommandEnvironmentBuilder(
                {
                    "PATH": str(Path(sys.executable).parent),
                    "LANG": "C.UTF-8",
                }
            ),
        )
        self.command_tool = RunCommandTool(self.executor)
        self.registry = create_local_tool_registry(self.command_tool)
        self.router = ToolRouter(self.registry, WorkspacePathGuard())

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def request(self, **arguments: object) -> ToolCallRequest:
        return ToolCallRequest(
            tool_call_id="call-command",
            tool_name="run_command",
            arguments=arguments,
        )

    def test_registry_contains_six_file_tools_and_run_command(self) -> None:
        self.assertIsInstance(self.registry, LocalToolRegistry)
        self.assertEqual(
            self.registry.names(),
            (
                "list_files",
                "read_file",
                "search_files",
                "create_file",
                "write_file",
                "edit_file",
                "run_command",
            ),
        )
        self.assertIs(self.registry.require("run_command"), self.command_tool)

    def test_prepare_validates_command_without_starting_process(self) -> None:
        source = self.workspace / "main.py"
        source.write_text("print('ok')\n", encoding="utf-8")

        with patch.object(CommandExecutor, "execute") as execute:
            prepared = self.router.prepare(
                self.request(command=["python", "main.py"], cwd="."),
                self.workspace,
            )

        execute.assert_not_called()
        self.assertIsInstance(prepared, PreparedCommandToolCall)
        assert isinstance(prepared, PreparedCommandToolCall)
        self.assertEqual(
            prepared.arguments,
            RunCommandArguments(("python", "main.py")),
        )
        self.assertEqual(prepared.resolved_cwd, self.workspace)

    def test_strict_argument_errors_remain_pending_phase_observations(self) -> None:
        requests = (
            self.request(),
            self.request(command="python main.py"),
            self.request(command=[]),
            self.request(command=["python", "main.py"], unknown=True),
        )

        with patch.object(CommandExecutor, "execute") as execute:
            for request in requests:
                with self.subTest(arguments=request.arguments):
                    result = self.router.prepare(request, self.workspace)
                    self.assertIsInstance(result, ToolResult)
                    assert isinstance(result, ToolResult)
                    self.assertEqual(result.status, ToolResultStatus.ERROR)
                    self.assertIn("status: ERROR", result.content or "")
        execute.assert_not_called()

    def test_cwd_errors_and_escape_have_correct_status_without_execution(self) -> None:
        file_path = self.workspace / "file.txt"
        file_path.write_text("content", encoding="utf-8")
        cases = (
            ("missing", ToolResultStatus.ERROR),
            ("file.txt", ToolResultStatus.ERROR),
            ("../outside", ToolResultStatus.REJECTED),
            (str(self.outside), ToolResultStatus.ERROR),
        )

        with patch.object(CommandExecutor, "execute") as execute:
            for cwd, expected_status in cases:
                with self.subTest(cwd=cwd):
                    result = self.router.prepare(
                        self.request(command=["pytest"], cwd=cwd),
                        self.workspace,
                    )
                    self.assertIsInstance(result, ToolResult)
                    assert isinstance(result, ToolResult)
                    self.assertEqual(result.status, expected_status)
                    self.assertEqual(result.metadata["requested_cwd"], cwd)
        execute.assert_not_called()

    def test_safety_rejection_is_observation_and_does_not_execute(self) -> None:
        with patch.object(CommandExecutor, "execute") as execute:
            result = self.router.prepare(
                self.request(command=["sudo", "rm", "file.txt"]),
                self.workspace,
            )

        execute.assert_not_called()
        self.assertIsInstance(result, ToolResult)
        assert isinstance(result, ToolResult)
        self.assertEqual(result.status, ToolResultStatus.REJECTED)
        self.assertEqual(
            result.metadata["rule_id"],
            "PERMANENTLY_BLOCKED_EXECUTABLE",
        )

    def test_execute_nonzero_exit_is_completed_and_timeout_is_timeout(self) -> None:
        failure_script = self.workspace / "failure.py"
        failure_script.write_text(
            "import sys\nprint('failed')\nsys.exit(7)\n",
            encoding="utf-8",
        )
        timeout_script = self.workspace / "timeout.py"
        timeout_script.write_text(
            "import time\ntime.sleep(30)\n",
            encoding="utf-8",
        )

        failed = self._prepare_and_execute(["python", "failure.py"])
        timed_out = self._prepare_and_execute(["python", "timeout.py"])

        self.assertEqual(failed.status, ToolResultStatus.COMPLETED)
        self.assertEqual(failed.metadata["exit_code"], 7)
        self.assertEqual(timed_out.status, ToolResultStatus.TIMEOUT)
        self.assertTrue(timed_out.metadata["timeout"])

    def test_internal_execution_error_is_propagated_for_runtime_to_close(self) -> None:
        source = self.workspace / "main.py"
        source.write_text("print('ok')\n", encoding="utf-8")
        prepared = self.router.prepare(
            self.request(command=["python", "main.py"]),
            self.workspace,
        )
        assert isinstance(prepared, PreparedCommandToolCall)

        with patch.object(
            CommandExecutor,
            "execute",
            side_effect=CommandExecutionError("collector failed"),
        ):
            with self.assertRaises(CommandExecutionError):
                self.router.execute(prepared)

    def _prepare_and_execute(self, command: list[str]) -> ToolResult:
        prepared = self.router.prepare(
            self.request(command=command),
            self.workspace,
        )
        self.assertIsInstance(prepared, PreparedCommandToolCall)
        assert isinstance(prepared, PreparedCommandToolCall)
        return self.router.execute(prepared)


if __name__ == "__main__":
    unittest.main()
