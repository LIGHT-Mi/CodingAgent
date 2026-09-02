import json
import sys
import tempfile
import unittest
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, call, patch
from urllib.error import HTTPError, URLError

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent.contracts import (
    AgentStepStatus,
    InvalidAction,
    MessageRole,
    MessageType,
    RuntimeEventType,
    TaskStatus,
    ToolCallStatus,
)
from app.agent import (
    CancellationToken,
    RetryWaiter,
    RuntimeDecision,
    RuntimeDecisionType,
    RuntimePolicy,
    RuntimePolicyConfig,
    RuntimeState,
)
from app.agent.runtime import (
    AgentRuntime,
    _invalid_action_event,
)
from app.agent.runtime_policy import (
    LOOP_DETECTED_TERMINATION_REASON,
    MAX_STEPS_TERMINATION_REASON,
)
from app.context import ContextLimits
from app.context import ContextHistoryError
from app.context.manager import ContextManager
from app.db.base import Base
from app.db.persistence import PersistenceService, PersistenceServiceError
from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry
from app.tools import (
    CommandEnvironmentBuilder,
    CommandExecutor,
    CODING_TOOL_SCHEMAS,
    RunCommandTool,
    ToolRouter,
    WorkspacePathGuard,
    create_local_tool_registry,
)

from test_deepseek_client import FakeHTTPResponse


def create_test_local_tool_registry(*, command_timeout: float = 2):
    return create_local_tool_registry(
        RunCommandTool(
            CommandExecutor(
                timeout_seconds=command_timeout,
                termination_grace_seconds=0.1,
                max_output_bytes_per_stream=1024,
                environment_builder=CommandEnvironmentBuilder(
                    {
                        "PATH": str(Path(sys.executable).parent),
                        "LANG": "C.UTF-8",
                    }
                ),
            )
        )
    )


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.temporary_directory.name).resolve()
        (self.workspace / "main.py").write_text(
            "print('hello')\n",
            encoding="utf-8",
        )

        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        testing_session = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.db: Session = testing_session()
        self.persistence = PersistenceService(self.db)
        self.context_manager = ContextManager(
            self.persistence,
            ContextLimits(60_000, 12_000),
        )

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_invalid_action_is_normalized_to_runtime_event(self) -> None:
        event = _invalid_action_event(
            InvalidAction(reason="response contains no usable action")
        )

        self.assertEqual(event.event_type, RuntimeEventType.INVALID_ACTION)
        self.assertEqual(event.source, "action_parser")
        self.assertEqual(event.message, "response contains no usable action")
        self.assertEqual(event.details, {})

    def test_complete_single_step_from_final_action(self) -> None:
        task_id = self._create_running_task("解释依赖注入。")
        captured = {}

        def open_url(request, *, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return self._final_response("依赖注入用于从外部提供依赖。")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "依赖注入用于从外部提供依赖。")
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].step_number, 0)
        self.assertEqual(steps[0].status, AgentStepStatus.COMPLETED.value)
        self.assertIsNotNone(steps[0].finished_at)
        messages = self.persistence.load_messages(task_id)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, MessageRole.ASSISTANT.value)
        self.assertEqual(messages[0].message_type, MessageType.FINAL.value)
        self.assertEqual(messages[0].content, result.final_answer)
        self.assertEqual(captured["payload"]["messages"][1]["content"], "解释依赖注入。")
        self.assertEqual(
            self.persistence.get_task(task_id).status,
            TaskStatus.RUNNING.value,
        )

    def test_execute_tool_calls_then_send_full_history_and_finish(self) -> None:
        task_id = self._create_running_task("先查看目录，再读取 main.py。")
        payloads = []
        tool_call_transitions = []

        def open_url(request, *, timeout):
            payloads.append(json.loads(request.data.decode("utf-8")))
            if len(payloads) == 1:
                return self._tool_calls_response(
                    (
                        ("call-list", "list_files", {"path": "."}),
                        ("call-read", "read_file", {"path": "main.py"}),
                    )
                )
            return self._final_response("main.py 会输出 hello。")

        save_tool_calls_action = self.persistence.save_tool_calls_action
        start_tool_call = self.persistence.start_tool_call
        save_tool_result = self.persistence.save_tool_result

        def record_pending_calls(*args, **kwargs):
            message, records = save_tool_calls_action(*args, **kwargs)
            tool_call_transitions.extend(
                (record.provider_call_id, record.status) for record in records
            )
            return message, records

        def record_running_call(*args, **kwargs):
            record = start_tool_call(*args, **kwargs)
            tool_call_transitions.append((record.provider_call_id, record.status))
            return record

        def record_finished_call(*args, **kwargs):
            record, message = save_tool_result(*args, **kwargs)
            tool_call_transitions.append((record.provider_call_id, record.status))
            return record, message

        with (
            patch.object(
                self.persistence,
                "save_tool_calls_action",
                side_effect=record_pending_calls,
            ),
            patch.object(
                self.persistence,
                "start_tool_call",
                side_effect=record_running_call,
            ),
            patch.object(
                self.persistence,
                "save_tool_result",
                side_effect=record_finished_call,
            ),
        ):
            result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "main.py 会输出 hello。")
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual([step.step_number for step in steps], [0, 1])
        self.assertTrue(
            all(step.status == AgentStepStatus.COMPLETED.value for step in steps)
        )
        self.assertTrue(all(step.finished_at is not None for step in steps))

        tool_calls = self.persistence.load_tool_calls(task_id)
        self.assertEqual(
            [tool_call.provider_call_id for tool_call in tool_calls],
            ["call-list", "call-read"],
        )
        self.assertEqual([tool_call.call_index for tool_call in tool_calls], [0, 1])
        self.assertTrue(
            all(
                tool_call.status == ToolCallStatus.COMPLETED.value
                for tool_call in tool_calls
            )
        )
        self.assertTrue(all(tool_call.started_at is not None for tool_call in tool_calls))
        self.assertTrue(all(tool_call.finished_at is not None for tool_call in tool_calls))
        self.assertEqual(
            tool_call_transitions,
            [
                ("call-list", ToolCallStatus.PENDING.value),
                ("call-read", ToolCallStatus.PENDING.value),
                ("call-list", ToolCallStatus.RUNNING.value),
                ("call-list", ToolCallStatus.COMPLETED.value),
                ("call-read", ToolCallStatus.RUNNING.value),
                ("call-read", ToolCallStatus.COMPLETED.value),
            ],
        )

        messages = self.persistence.load_messages(task_id)
        self.assertEqual([message.sequence for message in messages], [0, 1, 2, 3])
        self.assertEqual(
            [message.message_type for message in messages],
            [
                MessageType.TEXT.value,
                MessageType.TOOL_RESULT.value,
                MessageType.TOOL_RESULT.value,
                MessageType.FINAL.value,
            ],
        )
        second_messages = payloads[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_messages],
            ["system", "user", "assistant", "tool", "tool"],
        )
        self.assertEqual(
            [call["id"] for call in second_messages[2]["tool_calls"]],
            ["call-list", "call-read"],
        )
        self.assertEqual(second_messages[3]["tool_call_id"], "call-list")
        self.assertEqual(second_messages[4]["tool_call_id"], "call-read")
        self.assertIn("print('hello')", second_messages[4]["content"])

    def test_tool_error_is_observation_and_model_can_recover(self) -> None:
        task_id = self._create_running_task("读取一个可能不存在的文件。")
        payloads = []

        def open_url(request, *, timeout):
            payloads.append(json.loads(request.data.decode("utf-8")))
            if len(payloads) == 1:
                return self._tool_calls_response(
                    (("call-missing", "read_file", {"path": "missing.py"}),)
                )
            return self._final_response("该文件不存在。")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "该文件不存在。")
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual(
            [step.status for step in steps],
            [AgentStepStatus.COMPLETED.value, AgentStepStatus.COMPLETED.value],
        )
        tool_call = self.persistence.load_tool_calls(task_id)[0]
        self.assertEqual(tool_call.status, ToolCallStatus.ERROR.value)
        self.assertIsNone(tool_call.started_at)
        self.assertIsNotNone(tool_call.finished_at)
        self.assertIn("does not exist", tool_call.error)
        self.assertEqual(payloads[1]["messages"][-1]["role"], "tool")
        self.assertIn("does not exist", payloads[1]["messages"][-1]["content"])

    def test_all_standard_tool_outcomes_bypass_runtime_event_policy(self) -> None:
        original_main = (self.workspace / "main.py").read_text(encoding="utf-8")
        (self.workspace / "timeout_observation.py").write_text(
            "import time\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        (self.workspace / "failure_observation.py").write_text(
            "import sys\nprint('failed')\nsys.exit(1)\n",
            encoding="utf-8",
        )
        outside_file = self.workspace.parent / (
            f"{self.workspace.name}-outside-observation.txt"
        )
        outside_file.write_text("outside\n", encoding="utf-8")
        self.addCleanup(outside_file.unlink, missing_ok=True)
        task_id = self._create_running_task("验证全部普通工具 Observation。")
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            if response_number == 1:
                return self._tool_calls_response(
                    (
                        (
                            "call-missing",
                            "read_file",
                            {"path": "missing.txt"},
                        ),
                        (
                            "call-arguments",
                            "read_file",
                            {"path": "main.py", "unknown": True},
                        ),
                        (
                            "call-outside",
                            "read_file",
                            {"path": str(outside_file)},
                        ),
                        (
                            "call-rejected-command",
                            "run_command",
                            {
                                "command": ["sudo", "rm", "main.py"],
                                "cwd": ".",
                            },
                        ),
                        (
                            "call-timeout",
                            "run_command",
                            {
                                "command": [
                                    "python",
                                    "timeout_observation.py",
                                ],
                                "cwd": ".",
                            },
                        ),
                        (
                            "call-exit-one",
                            "run_command",
                            {
                                "command": [
                                    "python",
                                    "failure_observation.py",
                                ],
                                "cwd": ".",
                            },
                        ),
                        (
                            "call-no-search-results",
                            "search_files",
                            {
                                "query": "definitely-no-matching-text",
                                "path": ".",
                            },
                        ),
                        (
                            "call-edit-no-match",
                            "edit_file",
                            {
                                "path": "main.py",
                                "old_text": "text that is not present",
                                "new_text": "replacement",
                            },
                        ),
                    )
                )
            return self._final_response("已分析全部普通 Observation。")

        runtime = self._runtime(open_url, command_timeout=0.05)
        policy = runtime._runtime_policy
        with patch.object(
            policy,
            "evaluate",
            wraps=policy.evaluate,
        ) as evaluate:
            result = runtime.run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(response_number, 2)
        calls = self.persistence.load_tool_calls(task_id)
        self.assertEqual(
            [tool_call.status for tool_call in calls],
            [
                ToolCallStatus.ERROR.value,
                ToolCallStatus.ERROR.value,
                ToolCallStatus.REJECTED.value,
                ToolCallStatus.REJECTED.value,
                ToolCallStatus.TIMEOUT.value,
                ToolCallStatus.COMPLETED.value,
                ToolCallStatus.COMPLETED.value,
                ToolCallStatus.ERROR.value,
            ],
        )
        self.assertEqual(calls[5].exit_code, 1)
        self.assertEqual(calls[5].status, ToolCallStatus.COMPLETED.value)
        self.assertEqual(
            (self.workspace / "main.py").read_text(encoding="utf-8"),
            original_main,
        )
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual([step.step_number for step in steps], [0, 1])
        self.assertTrue(
            all(
                step.status == AgentStepStatus.COMPLETED.value
                for step in steps
            )
        )
        messages = self.persistence.load_messages(task_id)
        result_messages = [
            message
            for message in messages
            if message.message_type == MessageType.TOOL_RESULT.value
        ]
        self.assertEqual(len(result_messages), len(calls))
        self.assertEqual([message.sequence for message in messages], list(range(10)))
        for policy_call in evaluate.call_args_list:
            positional = policy_call.args
            event = (
                positional[1]
                if len(positional) > 1
                else policy_call.kwargs.get("event")
            )
            self.assertIsNone(event)

    def test_nonzero_command_observation_keeps_runtime_loop_running(self) -> None:
        (self.workspace / "failure.py").write_text(
            "import sys\nprint('test failed')\nsys.exit(5)\n",
            encoding="utf-8",
        )
        task_id = self._create_running_task("运行失败测试并解释结果。")
        payloads = []

        def open_url(request, *, timeout):
            payloads.append(json.loads(request.data.decode("utf-8")))
            if len(payloads) == 1:
                return self._tool_calls_response(
                    (
                        (
                            "call-command",
                            "run_command",
                            {"command": ["python", "failure.py"], "cwd": "."},
                        ),
                    )
                )
            return self._final_response("测试失败，退出码为 5。")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        command_call = self.persistence.load_tool_calls(task_id)[0]
        self.assertEqual(command_call.status, ToolCallStatus.COMPLETED.value)
        self.assertEqual(command_call.exit_code, 5)
        self.assertEqual(command_call.stdout, "test failed\n")
        self.assertIsNotNone(command_call.started_at)
        self.assertIsNotNone(command_call.finished_at)
        self.assertEqual(
            [step.status for step in self.persistence.load_agent_steps(task_id)],
            [AgentStepStatus.COMPLETED.value, AgentStepStatus.COMPLETED.value],
        )
        self.assertIn("status: COMPLETED", payloads[1]["messages"][-1]["content"])
        self.assertIn("exit_code: 5", payloads[1]["messages"][-1]["content"])

    def test_command_prepare_observations_are_persisted_and_loop_continues(
        self,
    ) -> None:
        task_id = self._create_running_task("修正命令参数、目录和安全错误。")
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            if response_number == 1:
                return self._tool_calls_response(
                    (
                        ("call-argument", "run_command", {}),
                        (
                            "call-directory",
                            "run_command",
                            {"command": ["pytest"], "cwd": "missing"},
                        ),
                        (
                            "call-path-rejected",
                            "run_command",
                            {"command": ["pytest"], "cwd": "../outside"},
                        ),
                        (
                            "call-policy-rejected",
                            "run_command",
                            {"command": ["sudo", "rm", "file.txt"]},
                        ),
                    )
                )
            return self._final_response("已分析全部命令 Observation。")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        calls = self.persistence.load_tool_calls(task_id)
        self.assertEqual(
            [call.status for call in calls],
            [
                ToolCallStatus.ERROR.value,
                ToolCallStatus.ERROR.value,
                ToolCallStatus.REJECTED.value,
                ToolCallStatus.REJECTED.value,
            ],
        )
        self.assertTrue(all(call.started_at is None for call in calls))
        self.assertTrue(all(call.finished_at is not None for call in calls))
        self.assertTrue(all(call.result_message is not None for call in calls))
        self.assertEqual(
            [step.status for step in self.persistence.load_agent_steps(task_id)],
            [AgentStepStatus.COMPLETED.value, AgentStepStatus.COMPLETED.value],
        )

    def test_command_timeout_is_persisted_and_loop_continues(self) -> None:
        (self.workspace / "timeout.py").write_text(
            "import time\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        task_id = self._create_running_task("运行一个会超时的测试。")
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            if response_number == 1:
                return self._tool_calls_response(
                    (
                        (
                            "call-timeout",
                            "run_command",
                            {"command": ["python", "timeout.py"]},
                        ),
                    )
                )
            return self._final_response("命令已经超时并被终止。")

        result = self._runtime(
            open_url,
            command_timeout=0.15,
        ).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        command_call = self.persistence.load_tool_calls(task_id)[0]
        self.assertEqual(command_call.status, ToolCallStatus.TIMEOUT.value)
        self.assertIsNotNone(command_call.started_at)
        self.assertIsNotNone(command_call.finished_at)
        self.assertTrue(command_call.result_metadata["timeout"])
        self.assertIsNotNone(command_call.result_message)
        self.assertTrue(
            all(
                step.status == AgentStepStatus.COMPLETED.value
                for step in self.persistence.load_agent_steps(task_id)
            )
        )

    def test_invalid_path_and_unicode_content_remain_observations(self) -> None:
        task_id = self._create_running_task("修正非法文件工具参数。")
        payloads = []

        def open_url(request, *, timeout):
            payloads.append(json.loads(request.data.decode("utf-8")))
            if len(payloads) == 1:
                return self._tool_calls_response(
                    (
                        (
                            "call-null-path",
                            "create_file",
                            {"path": "bad\x00name", "content": "content"},
                        ),
                        (
                            "call-surrogate-content",
                            "create_file",
                            {"path": "new.txt", "content": "\ud800"},
                        ),
                    )
                )
            return self._final_response("已根据参数错误完成修正。")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "已根据参数错误完成修正。")
        self.assertEqual(
            [
                tool_call.status
                for tool_call in self.persistence.load_tool_calls(task_id)
            ],
            [ToolCallStatus.ERROR.value, ToolCallStatus.ERROR.value],
        )
        self.assertTrue(
            all(
                step.status == AgentStepStatus.COMPLETED.value
                for step in self.persistence.load_agent_steps(task_id)
            )
        )
        self.assertFalse((self.workspace / "new.txt").exists())
        second_messages = payloads[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_messages[-2:]],
            ["tool", "tool"],
        )
        self.assertTrue(
            all(message["content"] for message in second_messages[-2:])
        )

    def test_unknown_tool_and_workspace_rejection_remain_observations(self) -> None:
        task_id = self._create_running_task("修正不合法的工具调用。")
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            if response_number == 1:
                return self._tool_calls_response(
                    (("call-unknown", "delete_file", {"path": "main.py"}),)
                )
            if response_number == 2:
                return self._tool_calls_response(
                    (("call-outside", "read_file", {"path": "../outside.py"}),)
                )
            return self._final_response("已根据工具错误修正。")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(
            [tool_call.status for tool_call in self.persistence.load_tool_calls(task_id)],
            [ToolCallStatus.ERROR.value, ToolCallStatus.REJECTED.value],
        )
        self.assertTrue(
            all(
                step.status == AgentStepStatus.COMPLETED.value
                for step in self.persistence.load_agent_steps(task_id)
            )
        )

    def test_invalid_action_and_runtime_event_fail_current_step(self) -> None:
        cases = (
            (
                "invalid_action",
                lambda request, timeout: FakeHTTPResponse(
                    json.dumps(
                        {
                            "id": "response-invalid",
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": "   ",
                                    },
                                }
                            ],
                        }
                    ).encode("utf-8")
                ),
                "INVALID_ACTION exhausted 2 LLM retries",
            ),
            (
                "runtime_event",
                lambda request, timeout: (_ for _ in ()).throw(
                    URLError("offline")
                ),
                "LLM_NETWORK_ERROR",
            ),
        )

        for name, open_url, expected_error in cases:
            with self.subTest(name=name):
                task_id = self._create_running_task(f"任务 {name}")

                result = self._runtime(open_url).run(task_id)

                self.assertEqual(result.status, TaskStatus.FAILED)
                self.assertIn(expected_error, result.error)
                steps = self.persistence.load_agent_steps(task_id)
                self.assertEqual(len(steps), 1)
                self.assertEqual(steps[0].status, AgentStepStatus.FAILED.value)
                self.assertEqual(steps[0].error, result.error)
                self.assertIsNotNone(steps[0].finished_at)
                self.assertEqual(self.persistence.load_messages(task_id), [])
                self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_invalid_action_retries_same_context_and_step_then_completes(self) -> None:
        task_id = self._create_running_task("先返回非法动作，再正常完成。")
        payloads = []

        def open_url(request, *, timeout):
            payloads.append(json.loads(request.data.decode("utf-8")))
            if len(payloads) == 1:
                return FakeHTTPResponse(
                    json.dumps(
                        {
                            "id": "response-invalid",
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": "   ",
                                    },
                                }
                            ],
                        }
                    ).encode("utf-8")
                )
            return self._final_response("重试后完成。")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "重试后完成。")
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0], payloads[1])
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].status, AgentStepStatus.COMPLETED.value)
        messages = self.persistence.load_messages(task_id)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].sequence, 0)
        self.assertEqual(messages[0].message_type, MessageType.FINAL.value)
        self.assertEqual(messages[0].content, "重试后完成。")
        self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_invalid_action_exhaustion_uses_configured_retry_count(self) -> None:
        task_id = self._create_running_task("持续返回非法动作。")
        attempt_count = 0

        def open_url(request, *, timeout):
            nonlocal attempt_count
            attempt_count += 1
            return FakeHTTPResponse(
                json.dumps(
                    {
                        "id": f"response-invalid-{attempt_count}",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                },
                            }
                        ],
                    }
                ).encode("utf-8")
            )

        result = self._runtime(
            open_url,
            max_llm_retries=1,
        ).run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertEqual(attempt_count, 2)
        self.assertIn("INVALID_ACTION exhausted 1 LLM retries", result.error)
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].status, AgentStepStatus.FAILED.value)
        self.assertEqual(self.persistence.load_messages(task_id), [])
        self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_retryable_gateway_events_retry_within_current_step(self) -> None:
        def timeout_failure(request):
            raise TimeoutError("timed out")

        def network_failure(request):
            raise URLError("offline")

        def rate_limit_failure(request):
            headers = Message()
            headers["Retry-After"] = "3"
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                headers,
                BytesIO(b'{"error":{"message":"slow down"}}'),
            )

        cases = (
            ("timeout", timeout_failure),
            ("network", network_failure),
            ("rate_limit", rate_limit_failure),
        )
        for name, first_failure in cases:
            with self.subTest(name=name):
                task_id = self._create_running_task(f"retry {name}")
                payloads = []

                def open_url(request, *, timeout):
                    payloads.append(json.loads(request.data.decode("utf-8")))
                    if len(payloads) == 1:
                        first_failure(request)
                    return self._final_response(f"recovered from {name}")

                wait_function = Mock()
                result = self._runtime(
                    open_url,
                    max_llm_retries=1,
                    retry_waiter=RetryWaiter(wait_function),
                ).run(task_id)

                self.assertEqual(result.status, TaskStatus.COMPLETED)
                self.assertEqual(len(payloads), 2)
                self.assertEqual(payloads[0], payloads[1])
                expected_delay = 3.0 if name == "rate_limit" else 1.0
                wait_function.assert_called_once_with(expected_delay)
                steps = self.persistence.load_agent_steps(task_id)
                self.assertEqual(len(steps), 1)
                self.assertEqual(
                    steps[0].status,
                    AgentStepStatus.COMPLETED.value,
                )
                messages = self.persistence.load_messages(task_id)
                self.assertEqual(len(messages), 1)
                self.assertEqual(
                    messages[0].message_type,
                    MessageType.FINAL.value,
                )
                self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_retry_count_resets_for_next_step(self) -> None:
        task_id = self._create_running_task("每个 Step 独立维护重试计数。")
        attempt_count = 0
        payloads = []

        def open_url(request, *, timeout):
            nonlocal attempt_count
            attempt_count += 1
            payloads.append(json.loads(request.data.decode("utf-8")))
            if attempt_count in {1, 4}:
                raise TimeoutError("temporary timeout")
            if attempt_count == 2:
                raise URLError("temporary network error")
            if attempt_count == 3:
                return self._tool_calls_response(
                    (("call-list", "list_files", {"path": "."}),)
                )
            return self._final_response("两个 Step 均在重试后成功。")

        wait_function = Mock()
        result = self._runtime(
            open_url,
            max_llm_retries=2,
            retry_waiter=RetryWaiter(wait_function),
        ).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(attempt_count, 5)
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])
        self.assertEqual(payloads[3], payloads[4])
        self.assertNotEqual(payloads[2], payloads[3])
        self.assertEqual(
            wait_function.call_args_list,
            [call(1.0), call(2.0), call(1.0)],
        )
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual([step.step_number for step in steps], [0, 1])
        self.assertTrue(
            all(
                step.status == AgentStepStatus.COMPLETED.value
                for step in steps
            )
        )
        self.assertEqual(len(self.persistence.load_tool_calls(task_id)), 1)
        self.assertEqual(
            [message.sequence for message in self.persistence.load_messages(task_id)],
            [0, 1, 2],
        )

    def test_retry_wait_failure_closes_current_step(self) -> None:
        task_id = self._create_running_task("重试等待器异常。")

        def open_url(request, *, timeout):
            raise TimeoutError("temporary timeout")

        result = self._runtime(
            open_url,
            retry_waiter=RetryWaiter(
                Mock(side_effect=RuntimeError("wait failed"))
            ),
        ).run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("wait failed", result.error)
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].status, AgentStepStatus.FAILED.value)
        self.assertIsNotNone(steps[0].finished_at)
        self.assertEqual(self.persistence.load_messages(task_id), [])
        self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_terminate_at_max_steps_without_unfinished_records(self) -> None:
        task_id = self._create_running_task("持续查看目录。")
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            return self._tool_calls_response(
                (
                    (
                        f"call-list-{response_number}",
                        "list_files",
                        {"path": "."},
                    ),
                )
            )

        result = self._runtime(open_url, max_agent_steps=2).run(task_id)

        self.assertEqual(result.status, TaskStatus.TERMINATED)
        self.assertEqual(result.termination_reason, MAX_STEPS_TERMINATION_REASON)
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual([step.step_number for step in steps], [0, 1])
        self.assertTrue(
            all(step.status == AgentStepStatus.COMPLETED.value for step in steps)
        )
        tool_calls = self.persistence.load_tool_calls(task_id)
        self.assertEqual(len(tool_calls), 2)
        self.assertTrue(
            all(
                tool_call.status == ToolCallStatus.COMPLETED.value
                for tool_call in tool_calls
            )
        )
        self.assertTrue(all(tool_call.finished_at is not None for tool_call in tool_calls))
        self.assertFalse(
            any(
                tool_call.status
                in {ToolCallStatus.PENDING.value, ToolCallStatus.RUNNING.value}
                for tool_call in tool_calls
            )
        )
        self.assertFalse(
            any(
                message.message_type == MessageType.FINAL.value
                for message in self.persistence.load_messages(task_id)
            )
        )

    def test_terminate_repeated_tool_interactions_before_next_step(self) -> None:
        task_id = self._create_running_task("持续重复查看相同目录。")
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            return self._tool_calls_response(
                (
                    (
                        f"provider-list-{response_number}",
                        "list_files",
                        {"path": "."},
                    ),
                )
            )

        result = self._runtime(open_url, max_agent_steps=8).run(task_id)

        self.assertEqual(result.status, TaskStatus.TERMINATED)
        self.assertEqual(
            result.termination_reason,
            LOOP_DETECTED_TERMINATION_REASON,
        )
        self.assertEqual(response_number, 3)
        steps = self.persistence.load_agent_steps(task_id)
        self.assertEqual([step.step_number for step in steps], [0, 1, 2])
        self.assertTrue(
            all(step.status == AgentStepStatus.COMPLETED.value for step in steps)
        )
        tool_calls = self.persistence.load_tool_calls(task_id)
        self.assertEqual(len(tool_calls), 3)
        self.assertTrue(
            all(
                tool_call.status == ToolCallStatus.COMPLETED.value
                for tool_call in tool_calls
            )
        )

    def test_different_interaction_resets_consecutive_loop_count(self) -> None:
        task_id = self._create_running_task("重复中途改变工具调用。")
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            if response_number in {1, 2, 4, 5}:
                return self._tool_calls_response(
                    (
                        (
                            f"provider-list-{response_number}",
                            "list_files",
                            {"path": "."},
                        ),
                    )
                )
            if response_number == 3:
                return self._tool_calls_response(
                    (
                        (
                            "provider-read",
                            "read_file",
                            {"path": "main.py"},
                        ),
                    )
                )
            return self._final_response("重复计数已经重置。")

        result = self._runtime(open_url, max_agent_steps=8).run(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "重复计数已经重置。")
        self.assertEqual(response_number, 6)
        self.assertEqual(
            [
                step.step_number
                for step in self.persistence.load_agent_steps(task_id)
            ],
            [0, 1, 2, 3, 4, 5],
        )

    def test_close_step_when_context_or_gateway_raises(self) -> None:
        task_id = self._create_running_task("触发本地异常")
        runtime = self._runtime(
            lambda request, timeout: FakeHTTPResponse(b"{}")
        )

        with patch.object(
            self.context_manager,
            "build",
            side_effect=RuntimeError("context failed"),
        ):
            result = runtime.run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("context failed", result.error)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)
        self.assertIsNotNone(step.finished_at)

    def test_context_history_corruption_is_classified_before_failure(self) -> None:
        task_id = self._create_running_task("损坏的工具历史")
        runtime = self._runtime(
            lambda request, timeout: self._final_response("不应调用模型")
        )

        with patch.object(
            self.context_manager,
            "build",
            side_effect=ContextHistoryError("orphan tool result"),
        ):
            result = runtime.run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("AGENT_STATE_CORRUPTED", result.error)
        self.assertIn("from context_manager", result.error)
        self.assertIn("orphan tool result", result.error)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)
        self.assertEqual(step.error, result.error)

    def test_persistence_exception_is_classified_by_real_error_type(self) -> None:
        task_id = self._create_running_task("持久化读取失败")
        runtime = self._runtime(
            lambda request, timeout: self._final_response("不应调用模型")
        )

        with patch.object(
            self.context_manager,
            "build",
            side_effect=PersistenceServiceError("load messages failed"),
        ):
            result = runtime.run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("FATAL_SYSTEM_ERROR", result.error)
        self.assertIn("from persistence_service", result.error)
        self.assertIn("load messages failed", result.error)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)

    def test_infrastructure_event_remains_infrastructure_error(self) -> None:
        task_id = self._create_running_task("供应商返回非法 JSON")
        attempt_count = 0

        def open_url(request, *, timeout):
            nonlocal attempt_count
            attempt_count += 1
            return FakeHTTPResponse(b"not-json")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertEqual(attempt_count, 1)
        self.assertIn("INFRASTRUCTURE_ERROR", result.error)
        self.assertIn("from deepseek_client", result.error)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)

    def test_database_write_failure_is_reported_without_false_closure(self) -> None:
        task_id = self._create_running_task("数据库无法写入 Step 终态")
        runtime = self._runtime(
            lambda request, timeout: self._final_response("不应调用模型")
        )

        with (
            patch.object(
                self.context_manager,
                "build",
                side_effect=RuntimeError("context failed"),
            ),
            patch.object(
                self.persistence,
                "finish_agent_step",
                side_effect=PersistenceServiceError("database unavailable"),
            ),
        ):
            result = runtime.run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("context failed", result.error)
        self.assertIn("unable to persist AgentStep failure", result.error)
        self.assertIn("from persistence_service", result.error)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.RUNNING.value)

    def test_context_overflow_event_fails_step_without_calling_model(self) -> None:
        task_id = self._create_running_task("原始任务本身超过上下文预算")
        self.context_manager = ContextManager(
            self.persistence,
            ContextLimits(1, 100),
        )
        model_called = False

        def open_url(request, *, timeout):
            nonlocal model_called
            model_called = True
            return self._final_response("不应调用模型")

        result = self._runtime(open_url).run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("CONTEXT_OVERFLOW", result.error)
        self.assertFalse(model_called)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)
        self.assertIsNotNone(step.finished_at)
        self.assertEqual(self.persistence.load_messages(task_id), [])
        self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_runtime_decision_mapping_without_current_step(self) -> None:
        runtime = self._runtime(
            lambda request, timeout: self._final_response("unused")
        )
        state = RuntimeState()

        self.assertIsNone(
            runtime._apply_runtime_decision(
                RuntimeDecision(RuntimeDecisionType.CONTINUE),
                state,
            )
        )
        failed = runtime._apply_runtime_decision(
            RuntimeDecision(RuntimeDecisionType.FAILED, reason="fatal"),
            state,
        )
        cancelled = runtime._apply_runtime_decision(
            RuntimeDecision(
                RuntimeDecisionType.CANCELLED,
                reason="USER_CANCELLED",
            ),
            state,
        )
        terminated = runtime._apply_runtime_decision(
            RuntimeDecision(
                RuntimeDecisionType.TERMINATED,
                reason="MAX_STEPS",
            ),
            state,
        )

        self.assertEqual(failed.status, TaskStatus.FAILED)
        self.assertEqual(failed.error, "fatal")
        self.assertEqual(cancelled.status, TaskStatus.CANCELLED)
        self.assertEqual(cancelled.termination_reason, "USER_CANCELLED")
        self.assertEqual(terminated.status, TaskStatus.TERMINATED)
        self.assertEqual(terminated.termination_reason, "MAX_STEPS")

    def test_runtime_decision_mapping_with_current_step(self) -> None:
        runtime = self._runtime(
            lambda request, timeout: self._final_response("unused")
        )

        retry_task_id = self._create_running_task("retry")
        retry_step = self.persistence.create_agent_step(retry_task_id, 0)
        retry_result = runtime._apply_runtime_decision(
            RuntimeDecision(RuntimeDecisionType.RETRY, reason="retry llm"),
            RuntimeState(),
            step_id=retry_step.id,
        )
        self.assertIsNone(retry_result)
        self.db.refresh(retry_step)
        self.assertEqual(retry_step.status, AgentStepStatus.RUNNING.value)
        self.persistence.finish_agent_step(
            retry_step.id,
            AgentStepStatus.COMPLETED,
        )

        cases = (
            (
                RuntimeDecision(RuntimeDecisionType.FAILED, reason="fatal"),
                TaskStatus.FAILED,
                AgentStepStatus.FAILED,
            ),
            (
                RuntimeDecision(
                    RuntimeDecisionType.CANCELLED,
                    reason="USER_CANCELLED",
                ),
                TaskStatus.CANCELLED,
                AgentStepStatus.INTERRUPTED,
            ),
            (
                RuntimeDecision(
                    RuntimeDecisionType.TERMINATED,
                    reason="LOOP_DETECTED",
                ),
                TaskStatus.TERMINATED,
                AgentStepStatus.INTERRUPTED,
            ),
        )
        for decision, task_status, step_status in cases:
            with self.subTest(decision=decision.decision):
                task_id = self._create_running_task(
                    f"decision {decision.decision.value}"
                )
                step = self.persistence.create_agent_step(task_id, 0)

                result = runtime._apply_runtime_decision(
                    decision,
                    RuntimeState(),
                    step_id=step.id,
                )

                self.assertEqual(result.status, task_status)
                self.db.refresh(step)
                self.assertEqual(step.status, step_status.value)
                self.assertIsNotNone(step.finished_at)

    def test_cancellation_before_first_step_creates_no_step(self) -> None:
        task_id = self._create_running_task("取消任务。")
        token = CancellationToken()
        token.cancel("user pressed cancel")
        model_called = False

        def open_url(request, *, timeout):
            nonlocal model_called
            model_called = True
            return self._final_response("不应调用模型")

        result = self._runtime(open_url).run(task_id, token)

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        self.assertEqual(result.termination_reason, "USER_CANCELLED")
        self.assertFalse(model_called)
        self.assertEqual(self.persistence.load_agent_steps(task_id), [])
        self.assertEqual(self.persistence.load_messages(task_id), [])
        self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_cancellation_after_context_interrupts_current_step(self) -> None:
        task_id = self._create_running_task("构造上下文后取消。")
        token = CancellationToken()
        original_build = self.context_manager.build

        def build_then_cancel(current_task_id):
            context = original_build(current_task_id)
            token.cancel("cancel after context")
            return context

        model_called = False

        def open_url(request, *, timeout):
            nonlocal model_called
            model_called = True
            return self._final_response("不应调用模型")

        with patch.object(
            self.context_manager,
            "build",
            side_effect=build_then_cancel,
        ):
            result = self._runtime(open_url).run(task_id, token)

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        self.assertFalse(model_called)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.INTERRUPTED.value)
        self.assertIsNotNone(step.finished_at)
        self.assertEqual(self.persistence.load_messages(task_id), [])

    def test_cancellation_after_llm_discards_action_and_interrupts_step(self) -> None:
        task_id = self._create_running_task("模型返回时取消。")
        token = CancellationToken()

        def open_url(request, *, timeout):
            token.cancel("cancel after llm")
            return self._final_response("不应保存的最终答案")

        result = self._runtime(open_url).run(task_id, token)

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.INTERRUPTED.value)
        self.assertEqual(self.persistence.load_messages(task_id), [])
        self.assertEqual(self.persistence.load_tool_calls(task_id), [])

    def test_cancellation_interrupts_retry_backoff_without_new_attempt(self) -> None:
        class CancelDuringWaitToken(CancellationToken):
            def wait(self, timeout_seconds: float) -> bool:
                self.cancel("cancel during backoff")
                return super().wait(0)

        task_id = self._create_running_task("重试等待时取消。")
        token = CancelDuringWaitToken()
        attempt_count = 0

        def open_url(request, *, timeout):
            nonlocal attempt_count
            attempt_count += 1
            raise TimeoutError("temporary timeout")

        result = self._runtime(open_url).run(task_id, token)

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        self.assertEqual(attempt_count, 1)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.INTERRUPTED.value)
        self.assertEqual(self.persistence.load_messages(task_id), [])

    def test_tool_cancellation_closes_open_calls_and_keeps_completed_write(
        self,
    ) -> None:
        task_id = self._create_running_task("创建文件后取消。")
        token = CancellationToken()

        def open_url(request, *, timeout):
            return self._tool_calls_response(
                (
                    (
                        "call-create",
                        "create_file",
                        {"path": "created.txt", "content": "kept\n"},
                    ),
                    ("call-read", "read_file", {"path": "main.py"}),
                )
            )

        runtime = self._runtime(open_url)
        execute = runtime._tool_router.execute
        execution_count = 0

        def execute_then_cancel(prepared):
            nonlocal execution_count
            execution_count += 1
            result = execute(prepared)
            token.cancel("cancel after first tool")
            return result

        with patch.object(
            runtime._tool_router,
            "execute",
            side_effect=execute_then_cancel,
        ):
            result = runtime.run(task_id, token)

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        self.assertEqual(execution_count, 1)
        self.assertEqual(
            (self.workspace / "created.txt").read_text(encoding="utf-8"),
            "kept\n",
        )
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.INTERRUPTED.value)
        tool_calls = self.persistence.load_tool_calls(task_id)
        self.assertEqual(
            [tool_call.status for tool_call in tool_calls],
            [ToolCallStatus.COMPLETED.value, ToolCallStatus.ERROR.value],
        )
        self.assertNotIn("interrupted", tool_calls[0].result_metadata)
        self.assertEqual(
            tool_calls[1].result_metadata,
            {"interrupted": True, "reason": "USER_CANCELLED"},
        )
        self.assertIsNone(tool_calls[1].started_at)
        self.assertIsNotNone(tool_calls[1].finished_at)
        messages = self.persistence.load_messages(task_id)
        self.assertEqual([message.sequence for message in messages], [0, 1, 2])
        self.assertEqual(
            [message.message_type for message in messages],
            [
                MessageType.TEXT.value,
                MessageType.TOOL_RESULT.value,
                MessageType.TOOL_RESULT.value,
            ],
        )

    def test_cancellation_after_tool_start_prevents_executor_call(self) -> None:
        task_id = self._create_running_task("工具启动后取消。")
        token = CancellationToken()

        def open_url(request, *, timeout):
            return self._tool_calls_response(
                (("call-read", "read_file", {"path": "main.py"}),)
            )

        runtime = self._runtime(open_url)
        start_tool_call = self.persistence.start_tool_call

        def start_then_cancel(tool_call_id):
            record = start_tool_call(tool_call_id)
            token.cancel("cancel before executor")
            return record

        with (
            patch.object(
                self.persistence,
                "start_tool_call",
                side_effect=start_then_cancel,
            ),
            patch.object(runtime._tool_router, "execute") as execute,
        ):
            result = runtime.run(task_id, token)

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        execute.assert_not_called()
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.INTERRUPTED.value)
        tool_call = self.persistence.load_tool_calls(task_id)[0]
        self.assertEqual(tool_call.status, ToolCallStatus.ERROR.value)
        self.assertIsNotNone(tool_call.started_at)
        self.assertIsNotNone(tool_call.finished_at)
        self.assertEqual(
            tool_call.result_metadata,
            {"interrupted": True, "reason": "USER_CANCELLED"},
        )

    def test_fatal_tool_exception_closes_all_tool_calls_and_results(self) -> None:
        task_id = self._create_running_task("读取多个文件。")

        def open_url(request, *, timeout):
            return self._tool_calls_response(
                (
                    ("call-running", "read_file", {"path": "main.py"}),
                    ("call-pending", "list_files", {"path": "."}),
                )
            )

        runtime = self._runtime(open_url)
        with patch.object(
            ToolRouter,
            "execute",
            side_effect=RuntimeError("fatal tool error"),
        ):
            result = runtime.run(task_id)

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("fatal tool error", result.error)
        self.assertIn("FATAL_TOOL_ERROR", result.error)
        self.assertIn("from tool_router", result.error)
        step = self.persistence.load_agent_steps(task_id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)
        self.assertIsNotNone(step.finished_at)

        tool_calls = self.persistence.load_tool_calls(task_id)
        self.assertEqual(len(tool_calls), 2)
        self.assertTrue(
            all(call.status == ToolCallStatus.ERROR.value for call in tool_calls)
        )
        self.assertIsNotNone(tool_calls[0].started_at)
        self.assertIsNone(tool_calls[1].started_at)
        self.assertTrue(all(call.finished_at is not None for call in tool_calls))
        self.assertTrue(
            all(call.result_metadata == {"fatal": True} for call in tool_calls)
        )

        messages = self.persistence.load_messages(task_id)
        result_messages = [
            message
            for message in messages
            if message.message_type == MessageType.TOOL_RESULT.value
        ]
        self.assertEqual(len(result_messages), len(tool_calls))
        self.assertEqual(
            {message.tool_call_id for message in result_messages},
            {call.id for call in tool_calls},
        )
        self.assertEqual([message.sequence for message in messages], [0, 1, 2])

    def _create_running_task(self, prompt: str) -> str:
        _, task = self.persistence.create_session_with_task(
            title=prompt,
            original_prompt=prompt,
            workspace=str(self.workspace),
        )
        self.persistence.start_task(task.id)
        return task.id

    def _runtime(
        self,
        open_url,
        *,
        max_agent_steps: int = 8,
        max_llm_retries: int = 2,
        command_timeout: float = 2,
        retry_waiter: RetryWaiter | None = None,
    ) -> AgentRuntime:
        gateway = LLMGateway(
            DeepSeekClient(api_key="secret", open_url=open_url),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
        )
        return AgentRuntime(
            self.persistence,
            self.context_manager,
            gateway,
            ToolRouter(
                create_test_local_tool_registry(
                    command_timeout=command_timeout,
                ),
                WorkspacePathGuard(),
            ),
            RuntimePolicy(
                RuntimePolicyConfig(
                    max_agent_steps=max_agent_steps,
                    max_llm_retries=max_llm_retries,
                )
            ),
            (
                RetryWaiter(lambda seconds: None)
                if retry_waiter is None
                else retry_waiter
            ),
        )

    @staticmethod
    def _final_response(content: str) -> FakeHTTPResponse:
        return FakeHTTPResponse(
            json.dumps(
                {
                    "id": "response-final",
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

    @staticmethod
    def _tool_calls_response(calls) -> FakeHTTPResponse:
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


if __name__ == "__main__":
    unittest.main()
