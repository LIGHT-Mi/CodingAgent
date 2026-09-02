import io
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.agent import AgentStepStatus, CancellationToken, RuntimePolicyConfig
from app.agent.contracts import MessageType, TaskStatus, ToolCallStatus
from app.api.task_service import TaskService
from app.agent.runtime_policy import MAX_STEPS_TERMINATION_REASON
from app.cli import main
from app.context import ContextLimits
from app.db.base import Base
from app.db.models.task import Task
from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry
from app.tools import CODING_TOOL_SCHEMAS

from test_deepseek_client import FakeHTTPResponse


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.allowed_root = Path(self.temporary_directory.name) / "allowed"
        self.allowed_root.mkdir()
        self.workspace = self.allowed_root / "project"
        self.workspace.mkdir()

        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_print_final_answer_and_return_zero(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main(
            ["--workspace", str(self.workspace), "解释列表和元组。"],
            session_factory=self.session_factory,
            llm_gateway_factory=lambda: self._gateway_with_final_answer(
                "列表可变，元组不可变。"
            ),
            allowed_workspace_root=self.allowed_root,
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "列表可变，元组不可变。\n")
        self.assertEqual(stderr.getvalue(), "")
        with self.session_factory() as db:
            task = db.scalars(select(Task)).one()
            self.assertEqual(task.status, TaskStatus.COMPLETED.value)
            self.assertEqual(task.final_answer, stdout.getvalue().strip())
            self.assertEqual(len(task.messages), 1)
            self.assertEqual(
                task.messages[0].message_type,
                MessageType.FINAL.value,
            )
            self.assertEqual(task.messages[0].content, task.final_answer)

    def test_write_failure_to_stderr_and_return_nonzero(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        wait_calls = []

        def immediate_wait(token, timeout_seconds):
            wait_calls.append(timeout_seconds)
            return False

        with patch.object(
            CancellationToken,
            "wait",
            autospec=True,
            side_effect=immediate_wait,
        ):
            exit_code = main(
                ["--workspace", str(self.workspace), "触发无效响应"],
                session_factory=self.session_factory,
                llm_gateway_factory=self._gateway_with_invalid_response,
                allowed_workspace_root=self.allowed_root,
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(wait_calls, [1.0, 2.0])
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("任务执行失败", stderr.getvalue())
        with self.session_factory() as db:
            task = db.scalars(select(Task)).one()
            self.assertEqual(task.status, TaskStatus.FAILED.value)
            self.assertEqual(task.agent_steps[0].status, "FAILED")

    def test_assembles_runtime_policy_settings_and_cancellation_token(
        self,
    ) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        captured = {}
        run_task = TaskService.run_task_and_wait

        def capture_run(service, prompt, workspace, cancellation_token=None):
            captured["cancellation_token"] = cancellation_token
            return run_task(service, prompt, workspace, cancellation_token)

        with (
            patch("app.cli.settings.MAX_AGENT_STEPS", 4),
            patch("app.cli.settings.MAX_LLM_RETRIES", 1),
            patch("app.cli.settings.LLM_RETRY_BASE_SECONDS", 0.5),
            patch("app.cli.settings.LLM_RETRY_MAX_SECONDS", 3.0),
            patch("app.cli.settings.AGENT_LOOP_REPEAT_THRESHOLD", 2),
            patch(
                "app.application.factory.RuntimePolicyConfig",
                wraps=RuntimePolicyConfig,
            ) as config_type,
            patch(
                "app.application.factory.TaskService.run_task_and_wait",
                new=capture_run,
            ),
        ):
            exit_code = main(
                ["--workspace", str(self.workspace), "检查运行策略装配。"],
                session_factory=self.session_factory,
                llm_gateway_factory=lambda: self._gateway_with_final_answer(
                    "装配完成。"
                ),
                allowed_workspace_root=self.allowed_root,
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 0)
        config_type.assert_called_once_with(
            max_agent_steps=4,
            max_llm_retries=1,
            retry_base_seconds=0.5,
            retry_max_seconds=3.0,
            loop_repeat_threshold=2,
        )
        self.assertIsInstance(
            captured["cancellation_token"],
            CancellationToken,
        )

    def test_ctrl_c_requests_cooperative_cancellation_and_restores_handler(
        self,
    ) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        previous_handler = signal.getsignal(signal.SIGINT)

        def open_url(request, *, timeout):
            del request, timeout
            signal.raise_signal(signal.SIGINT)
            return FakeHTTPResponse(
                json.dumps(
                    {
                        "id": "response-after-sigint",
                        "model": "deepseek-v4-flash",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "不应保存的最终答案",
                                },
                            }
                        ],
                    }
                ).encode("utf-8")
            )

        exit_code = main(
            ["--workspace", str(self.workspace), "运行期间取消"],
            session_factory=self.session_factory,
            llm_gateway_factory=lambda: self._gateway(open_url),
            allowed_workspace_root=self.allowed_root,
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("任务已取消：USER_CANCELLED", stderr.getvalue())
        self.assertIs(signal.getsignal(signal.SIGINT), previous_handler)
        with self.session_factory() as db:
            task = db.scalars(select(Task)).one()
            self.assertEqual(task.status, TaskStatus.CANCELLED.value)
            self.assertEqual(task.termination_reason, "USER_CANCELLED")
            self.assertIsNotNone(task.finished_at)
            self.assertEqual(len(task.agent_steps), 1)
            self.assertEqual(
                task.agent_steps[0].status,
                AgentStepStatus.INTERRUPTED.value,
            )
            self.assertIsNotNone(task.agent_steps[0].finished_at)
            self.assertEqual(task.messages, [])

    def test_uses_context_limits_from_settings(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        model_called = False

        def open_url(request, *, timeout):
            nonlocal model_called
            model_called = True
            return FakeHTTPResponse(b"{}")

        with (
            patch("app.cli.settings.MAX_LLM_CONTEXT_CHARACTERS", 1),
            patch(
                "app.cli.settings.MAX_CONTEXT_TOOL_RESULT_CHARACTERS",
                100,
            ),
            patch(
                "app.application.factory.ContextLimits",
                wraps=ContextLimits,
            ) as limits_type,
        ):
            exit_code = main(
                ["--workspace", str(self.workspace), "基础上下文会超过预算"],
                session_factory=self.session_factory,
                llm_gateway_factory=lambda: self._gateway(open_url),
                allowed_workspace_root=self.allowed_root,
                stdout=stdout,
                stderr=stderr,
            )

        limits_type.assert_called_once_with(
            max_context_characters=1,
            max_tool_result_characters=100,
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("CONTEXT_OVERFLOW", stderr.getvalue())
        self.assertFalse(model_called)
        with self.session_factory() as db:
            task = db.scalars(select(Task)).one()
            self.assertEqual(task.status, TaskStatus.FAILED.value)
            self.assertEqual(task.agent_steps[0].status, "FAILED")
            self.assertEqual(task.messages, [])
            self.assertEqual(task.agent_steps[0].tool_calls, [])

    def test_print_termination_reason_and_return_nonzero_at_max_steps(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            return self._tool_calls_response(f"call-list-{response_number}")

        with patch("app.cli.settings.MAX_AGENT_STEPS", 2):
            exit_code = main(
                ["--workspace", str(self.workspace), "持续查看目录"],
                session_factory=self.session_factory,
                llm_gateway_factory=lambda: self._gateway(open_url),
                allowed_workspace_root=self.allowed_root,
                stdout=stdout,
                stderr=stderr,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(MAX_STEPS_TERMINATION_REASON, stderr.getvalue())
        with self.session_factory() as db:
            task = db.scalars(select(Task)).one()
            self.assertEqual(task.status, TaskStatus.TERMINATED.value)
            self.assertEqual(
                task.termination_reason,
                MAX_STEPS_TERMINATION_REASON,
            )
            self.assertEqual(len(task.agent_steps), 2)
            self.assertTrue(
                all(step.status == "COMPLETED" for step in task.agent_steps)
            )
            self.assertTrue(
                all(
                    tool_call.status == ToolCallStatus.COMPLETED.value
                    for step in task.agent_steps
                    for tool_call in step.tool_calls
                )
            )

    def test_execute_create_then_read_with_full_cli_tool_registry(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            if response_number == 1:
                return self._tool_calls_response(
                    "call-create",
                    "create_file",
                    {"path": "generated.txt", "content": "generated\n"},
                )
            if response_number == 2:
                payload = json.loads(request.data)
                tool_messages = [
                    message
                    for message in payload["messages"]
                    if message["role"] == "tool"
                ]
                self.assertIn("Created generated.txt", tool_messages[-1]["content"])
                return self._tool_calls_response(
                    "call-read",
                    "read_file",
                    {"path": "generated.txt"},
                )
            return FakeHTTPResponse(
                json.dumps(
                    {
                        "id": "response-final",
                        "model": "deepseek-v4-flash",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "已创建并读取验证 generated.txt。",
                                },
                            }
                        ],
                    }
                ).encode("utf-8")
            )

        exit_code = main(
            ["--workspace", str(self.workspace), "创建文件并读取验证。"],
            session_factory=self.session_factory,
            llm_gateway_factory=lambda: self._gateway(open_url),
            allowed_workspace_root=self.allowed_root,
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            (self.workspace / "generated.txt").read_text(encoding="utf-8"),
            "generated\n",
        )
        with self.session_factory() as db:
            task = db.scalars(select(Task)).one()
            tool_calls = [
                tool_call
                for step in task.agent_steps
                for tool_call in step.tool_calls
            ]
            self.assertEqual(
                [tool_call.tool_name for tool_call in tool_calls],
                ["create_file", "read_file"],
            )
            self.assertTrue(
                all(
                    tool_call.status == ToolCallStatus.COMPLETED.value
                    for tool_call in tool_calls
                )
            )
            self.assertEqual(task.final_answer, stdout.getvalue().strip())

    @staticmethod
    def _gateway_with_final_answer(answer: str) -> LLMGateway:
        def open_url(request, *, timeout):
            return FakeHTTPResponse(
                json.dumps(
                    {
                        "id": "response-final",
                        "model": "deepseek-v4-flash",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": answer,
                                },
                            }
                        ],
                    }
                ).encode("utf-8")
            )

        return CLITests._gateway(open_url)

    @staticmethod
    def _gateway_with_invalid_response() -> LLMGateway:
        return CLITests._gateway(
            lambda request, timeout: FakeHTTPResponse(b"{}")
        )

    @staticmethod
    def _gateway(open_url) -> LLMGateway:
        return LLMGateway(
            DeepSeekClient(api_key="secret", open_url=open_url),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
        )

    @staticmethod
    def _tool_calls_response(
        call_id: str,
        tool_name: str = "list_files",
        arguments: dict | None = None,
    ) -> FakeHTTPResponse:
        resolved_arguments = {"path": "."} if arguments is None else arguments
        return FakeHTTPResponse(
            json.dumps(
                {
                    "id": "response-tools",
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
                                            "arguments": json.dumps(
                                                resolved_arguments,
                                                ensure_ascii=False,
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            ).encode("utf-8")
        )


if __name__ == "__main__":
    unittest.main()
