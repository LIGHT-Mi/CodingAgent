import json
import sys
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.agent import (
    AgentStepStatus,
    MessageType,
    RetryWaiter,
    RuntimePolicy,
    RuntimePolicyConfig,
    TaskStatus,
    ToolCallStatus,
)
from app.agent.runtime import AgentRuntime
from app.api.task_service import TaskService
from app.api.workspace import WorkspaceValidator
from app.context import ContextLimits, ContextManager
from app.db.base import Base
from app.db.models.task import Task
from app.db.persistence import PersistenceService
from app.llm import ModelConfig, ToolSchemaRegistry
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.tools import (
    CODING_TOOL_SCHEMAS,
    CommandEnvironmentBuilder,
    CommandExecutor,
    CommandResultBuilder,
    CommandSafetyPolicy,
    RunCommandTool,
    ToolRouter,
    WorkingDirectoryGuard,
    WorkspacePathGuard,
    create_local_tool_registry,
)

from test_deepseek_client import FakeHTTPResponse


class CommandVerticalAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.allowed_root = Path(self.temporary_directory.name).resolve(strict=True)
        self.workspace = self.allowed_root / "defect-project"
        self.workspace.mkdir()
        self.original_source = (
            "def add(left: int, right: int) -> int:\n"
            "    return left - right\n"
        )
        self.fixed_source = self.original_source.replace(
            "return left - right",
            "return left + right  # fixed",
        )
        (self.workspace / "calculator.py").write_text(
            self.original_source,
            encoding="utf-8",
        )
        self.test_source = (
            "import unittest\n\n"
            "from calculator import add\n\n\n"
            "class CalculatorTests(unittest.TestCase):\n"
            "    def test_adds_two_numbers(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        (self.workspace / "test_calculator.py").write_text(
            self.test_source,
            encoding="utf-8",
        )

        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        testing_session = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.db: Session = testing_session()
        self.addCleanup(self.db.close)
        self.persistence = PersistenceService(self.db)
        self.payloads: list[dict] = []

    def test_complete_programming_loop_from_failure_to_passing_tests(self) -> None:
        task_service = self._task_service()

        result = task_service.run_task_and_wait(
            "检查这个缺陷项目，运行测试，修复加法函数并验证测试通过。",
            self.workspace,
        )

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "已修复 add，并确认测试通过。")
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.COMPLETED.value)
        self.assertEqual(task.final_answer, result.final_answer)
        self.assertEqual(
            (self.workspace / "calculator.py").read_text(encoding="utf-8"),
            self.fixed_source,
        )
        self.assertEqual(
            (self.workspace / "test_calculator.py").read_text(encoding="utf-8"),
            self.test_source,
        )

        steps = self.persistence.load_agent_steps(task.id)
        self.assertEqual([step.step_number for step in steps], list(range(6)))
        self.assertTrue(
            all(step.status == AgentStepStatus.COMPLETED.value for step in steps)
        )

        tool_calls = self.persistence.load_tool_calls(task.id)
        self.assertEqual(
            [tool_call.tool_name for tool_call in tool_calls],
            [
                "list_files",
                "read_file",
                "read_file",
                "run_command",
                "edit_file",
                "read_file",
                "run_command",
            ],
        )
        self.assertTrue(
            all(call.status == ToolCallStatus.COMPLETED.value for call in tool_calls)
        )
        command_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call.tool_name == "run_command"
        ]
        self.assertEqual(
            [call.exit_code for call in command_calls],
            [1, 0],
            [call.stderr for call in command_calls],
        )
        self.assertIn("FAILED", command_calls[0].stderr or "")
        self.assertIn("OK", command_calls[1].stderr or "")
        self.assertTrue(
            all(call.result_message is not None for call in tool_calls)
        )

        messages = self.persistence.load_messages(task.id)
        tool_result_messages = [
            message
            for message in messages
            if message.message_type == MessageType.TOOL_RESULT.value
        ]
        final_messages = [
            message
            for message in messages
            if message.message_type == MessageType.FINAL.value
        ]
        self.assertEqual(len(tool_result_messages), len(tool_calls))
        self.assertEqual(len(final_messages), 1)
        self.assertEqual(final_messages[0].content, task.final_answer)
        self.assertEqual(
            [message.sequence for message in messages],
            list(range(len(messages))),
        )

        failed_command_observation = self.payloads[2]["messages"][-1]["content"]
        self.assertIn("status: COMPLETED", failed_command_observation)
        self.assertIn("exit_code: 1", failed_command_observation)
        self.assertIn("FAILED", failed_command_observation)
        verified_file_observation = self.payloads[4]["messages"][-1]["content"]
        self.assertIn("return left + right  # fixed", verified_file_observation)
        passed_command_observation = self.payloads[5]["messages"][-1]["content"]
        self.assertIn("exit_code: 0", passed_command_observation)
        self.assertIn("OK", passed_command_observation)

    def _task_service(self) -> TaskService:
        persistence = self.persistence
        context_manager = ContextManager(
            persistence,
            ContextLimits(60_000, 12_000),
        )
        gateway = LLMGateway(
            DeepSeekClient(api_key="offline-test-key", open_url=self._open_url),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
        )
        result_builder = CommandResultBuilder()
        command_tool = RunCommandTool(
            CommandExecutor(
                timeout_seconds=5,
                termination_grace_seconds=0.1,
                max_output_bytes_per_stream=4096,
                environment_builder=CommandEnvironmentBuilder(
                    {
                        "PATH": str(Path(sys.executable).parent),
                        "LANG": "C.UTF-8",
                    }
                ),
            ),
            result_builder,
        )
        router = ToolRouter(
            create_local_tool_registry(command_tool),
            WorkspacePathGuard(),
            WorkingDirectoryGuard(),
            CommandSafetyPolicy(),
            result_builder,
        )
        runtime = AgentRuntime(
            persistence,
            context_manager,
            gateway,
            router,
            RuntimePolicy(RuntimePolicyConfig(max_agent_steps=8)),
            RetryWaiter(lambda seconds: None),
        )
        return TaskService(
            persistence,
            WorkspaceValidator(self.allowed_root),
            runtime,
        )

    def _open_url(self, request, *, timeout):
        self.payloads.append(json.loads(request.data.decode("utf-8")))
        response_number = len(self.payloads)
        if response_number == 1:
            return self._tool_calls_response(
                (
                    ("call-list", "list_files", {"path": "."}),
                    ("call-read-source", "read_file", {"path": "calculator.py"}),
                    (
                        "call-read-test",
                        "read_file",
                        {"path": "test_calculator.py"},
                    ),
                )
            )
        if response_number == 2:
            return self._tool_calls_response(
                (
                    (
                        "call-test-failing",
                        "run_command",
                        {
                            "command": [
                                "python",
                                "-m",
                                "unittest",
                                "discover",
                                "-s",
                                ".",
                                "-p",
                                "test_calculator.py",
                            ],
                            "cwd": ".",
                        },
                    ),
                )
            )
        if response_number == 3:
            return self._tool_calls_response(
                (
                    (
                        "call-edit",
                        "edit_file",
                        {
                            "path": "calculator.py",
                            "old_text": "return left - right",
                            "new_text": "return left + right  # fixed",
                        },
                    ),
                )
            )
        if response_number == 4:
            return self._tool_calls_response(
                (
                    (
                        "call-read-fixed",
                        "read_file",
                        {"path": "calculator.py"},
                    ),
                )
            )
        if response_number == 5:
            return self._tool_calls_response(
                (
                    (
                        "call-test-passing",
                        "run_command",
                        {
                            "command": [
                                "python",
                                "-m",
                                "unittest",
                                "discover",
                                "-s",
                                ".",
                                "-p",
                                "test_calculator.py",
                            ],
                            "cwd": ".",
                        },
                    ),
                )
            )
        if response_number == 6:
            return self._final_response("已修复 add，并确认测试通过。")
        raise AssertionError("offline model was called more times than expected")

    @staticmethod
    def _tool_calls_response(calls) -> FakeHTTPResponse:
        return FakeHTTPResponse(
            json.dumps(
                {
                    "id": "offline-tool-response",
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                    for call_id, tool_name, arguments in calls
                                ],
                            },
                        }
                    ],
                }
            ).encode("utf-8")
        )

    @staticmethod
    def _final_response(content: str) -> FakeHTTPResponse:
        return FakeHTTPResponse(
            json.dumps(
                {
                    "id": "offline-final-response",
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                }
            ).encode("utf-8")
        )


if __name__ == "__main__":
    unittest.main()
