import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.agent.contracts import (
    AgentStepStatus,
    MessageRole,
    MessageType,
    TaskStatus,
    ToolCallStatus,
)
from app.agent import (
    CancellationToken,
    RetryWaiter,
    RuntimePolicy,
    RuntimePolicyConfig,
)
from app.agent.runtime import AgentRuntime
from app.agent.runtime_policy import (
    LOOP_DETECTED_TERMINATION_REASON,
    MAX_STEPS_TERMINATION_REASON,
)
from app.api.session_title import generate_session_title
from app.api.task_service import TaskService
from app.api.task_validation import TaskPromptValidationError
from app.api.workspace import WorkspaceValidationError, WorkspaceValidator
from app.context import ContextLimits
from app.context.manager import ContextManager
from app.db.base import Base
from app.db.models.session_record import CodingSession
from app.db.models.task import Task
from app.db.persistence import PersistenceService
from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry
from app.tools import (
    CommandExecutor,
    CODING_TOOL_SCHEMAS,
    RunCommandTool,
    ToolRouter,
    WorkspacePathGuard,
    create_local_tool_registry,
)

from test_deepseek_client import FakeHTTPResponse


def create_test_local_tool_registry():
    return create_local_tool_registry(
        RunCommandTool(
            CommandExecutor(
                timeout_seconds=2,
                termination_grace_seconds=0.1,
                max_output_bytes_per_stream=1024,
            )
        )
    )


class TaskServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.allowed_root = Path(self.temporary_directory.name) / "allowed"
        self.allowed_root.mkdir()
        self.workspace = self.allowed_root / "project"
        self.workspace.mkdir()

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

    def test_manage_complete_task_lifecycle(self) -> None:
        captured = {}
        task_transitions = []
        step_transitions = []

        def open_url(request, *, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return self._final_response("任务已经完成。")

        service = self._task_service(open_url)
        unnormalized_workspace = self.workspace / ".." / self.workspace.name
        create_session_with_task = self.persistence.create_session_with_task
        start_task = self.persistence.start_task
        finish_task = self.persistence.finish_task
        create_agent_step = self.persistence.create_agent_step
        finish_agent_step = self.persistence.finish_agent_step

        def record_create_session_with_task(*args, **kwargs):
            coding_session, task = create_session_with_task(*args, **kwargs)
            task_transitions.append(task.status)
            return coding_session, task

        def record_start_task(*args, **kwargs):
            task = start_task(*args, **kwargs)
            task_transitions.append(task.status)
            return task

        def record_finish_task(*args, **kwargs):
            task = finish_task(*args, **kwargs)
            task_transitions.append(task.status)
            return task

        def record_create_agent_step(*args, **kwargs):
            step = create_agent_step(*args, **kwargs)
            step_transitions.append(step.status)
            return step

        def record_finish_agent_step(*args, **kwargs):
            step = finish_agent_step(*args, **kwargs)
            step_transitions.append(step.status)
            return step

        with (
            patch.object(
                self.persistence,
                "create_session_with_task",
                side_effect=record_create_session_with_task,
            ),
            patch.object(
                self.persistence,
                "start_task",
                side_effect=record_start_task,
            ),
            patch.object(
                self.persistence,
                "finish_task",
                side_effect=record_finish_task,
            ),
            patch.object(
                self.persistence,
                "create_agent_step",
                side_effect=record_create_agent_step,
            ),
            patch.object(
                self.persistence,
                "finish_agent_step",
                side_effect=record_finish_agent_step,
            ),
        ):
            result = service.run_task_and_wait(
                "  请解释这个项目。  ",
                unnormalized_workspace,
            )

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.final_answer, "任务已经完成。")
        self.assertEqual(
            task_transitions,
            [
                TaskStatus.PENDING.value,
                TaskStatus.RUNNING.value,
                TaskStatus.COMPLETED.value,
            ],
        )
        self.assertEqual(
            step_transitions,
            [
                AgentStepStatus.RUNNING.value,
                AgentStepStatus.COMPLETED.value,
            ],
        )
        sessions = self.db.scalars(select(CodingSession)).all()
        tasks = self.db.scalars(select(Task)).all()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(sessions[0].title, "请解释这个项目。")
        task = tasks[0]
        self.assertEqual(task.session_id, sessions[0].id)
        self.assertEqual(task.original_prompt, "  请解释这个项目。  ")
        self.assertEqual(task.workspace, str(self.workspace.resolve()))
        self.assertEqual(task.status, TaskStatus.COMPLETED.value)
        self.assertEqual(task.final_answer, result.final_answer)
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.finished_at)
        steps = self.persistence.load_agent_steps(task.id)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].status, AgentStepStatus.COMPLETED.value)
        messages = self.persistence.load_messages(task.id)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, MessageRole.ASSISTANT.value)
        self.assertEqual(messages[0].message_type, MessageType.FINAL.value)
        self.assertEqual(messages[0].sequence, 0)
        self.assertEqual(
            captured["payload"]["messages"][1]["content"],
            "  请解释这个项目。  ",
        )
        self.assertEqual(task.final_answer, messages[0].content)
        self.assertEqual(self.persistence.load_tool_calls(task.id), [])

    def test_invalid_input_creates_no_session_or_task(self) -> None:
        service = self._task_service(
            lambda request, timeout: self._final_response("不会调用")
        )
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()

        cases = (
            (TaskPromptValidationError, "   ", self.workspace),
            (TypeError, object(), self.workspace),
            (WorkspaceValidationError, "有效任务", outside),
        )
        for expected_error, prompt, workspace in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaises(expected_error):
                    service.run_task_and_wait(prompt, workspace)

                self.assertEqual(self._count(CodingSession), 0)
                self.assertEqual(self._count(Task), 0)

    def test_generate_session_title_uses_first_prompt_deterministically(
        self,
    ) -> None:
        self.assertEqual(
            generate_session_title("  修复\n  用户   登录问题  "),
            "修复 用户 登录问题",
        )
        exact_length = "一" * 30
        self.assertEqual(generate_session_title(exact_length), exact_length)
        self.assertEqual(
            generate_session_title(exact_length + "二"),
            exact_length + "…",
        )

    def test_create_and_execute_task_are_separate_lifecycle_boundaries(
        self,
    ) -> None:
        service = self._task_service(
            lambda request, timeout: self._final_response("分段执行完成。")
        )

        task_id = service.create_task("分段执行任务", self.workspace)

        task = self.persistence.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.status, TaskStatus.PENDING.value)
        self.assertIsNone(task.started_at)
        self.assertIsNone(task.finished_at)
        self.assertEqual(self.persistence.load_agent_steps(task_id), [])

        result = service.execute_task(task_id)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.db.refresh(task)
        self.assertEqual(task.status, TaskStatus.COMPLETED.value)
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.finished_at)
        self.assertEqual(task.final_answer, "分段执行完成。")

    def test_run_task_and_wait_reuses_create_and_execute_boundaries(self) -> None:
        service = self._task_service(
            lambda request, timeout: self._final_response("复用边界完成。")
        )

        with (
            patch.object(
                service,
                "create_task",
                wraps=service.create_task,
            ) as create_task,
            patch.object(
                service,
                "execute_task",
                wraps=service.execute_task,
            ) as execute_task,
        ):
            result = service.run_task_and_wait("复用边界", self.workspace)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        create_task.assert_called_once_with("复用边界", self.workspace)
        task_id = self.db.scalars(select(Task.id)).one()
        execute_task.assert_called_once_with(task_id, None)

    def test_finish_task_as_failed_when_runtime_returns_failure(self) -> None:
        def open_url(request, *, timeout):
            raise URLError("offline")

        result = self._task_service(open_url).run_task_and_wait(
            "调用模型",
            self.workspace,
        )

        self.assertEqual(result.status, TaskStatus.FAILED)
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.FAILED.value)
        self.assertEqual(task.error, result.error)
        self.assertIsNotNone(task.finished_at)
        step = self.persistence.load_agent_steps(task.id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)

    def test_pre_cancelled_task_finishes_cancelled_without_step(self) -> None:
        token = CancellationToken()
        token.cancel("user cancelled before runtime")
        model_called = False

        def open_url(request, *, timeout):
            nonlocal model_called
            model_called = True
            return self._final_response("不应调用模型")

        result = self._task_service(open_url).run_task_and_wait(
            "取消这个任务",
            self.workspace,
            token,
        )

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        self.assertEqual(result.termination_reason, "USER_CANCELLED")
        self.assertFalse(model_called)
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.CANCELLED.value)
        self.assertEqual(task.termination_reason, "USER_CANCELLED")
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.finished_at)
        self.assertEqual(self.persistence.load_agent_steps(task.id), [])
        self.assertEqual(self.persistence.load_messages(task.id), [])
        self.assertEqual(self.persistence.load_tool_calls(task.id), [])

    def test_cancel_after_tool_calls_are_saved_closes_task_and_all_calls(
        self,
    ) -> None:
        token = CancellationToken()

        def open_url(request, *, timeout):
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
                                            "id": "call-list",
                                            "type": "function",
                                            "function": {
                                                "name": "list_files",
                                                "arguments": json.dumps(
                                                    {"path": "."}
                                                ),
                                            },
                                        },
                                        {
                                            "id": "call-read",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": json.dumps(
                                                    {"path": "main.py"}
                                                ),
                                            },
                                        },
                                    ],
                                },
                            }
                        ],
                    }
                ).encode("utf-8")
            )

        save_tool_calls_action = self.persistence.save_tool_calls_action

        def save_then_cancel(*args, **kwargs):
            saved = save_tool_calls_action(*args, **kwargs)
            token.cancel("cancel after tool calls were saved")
            return saved

        with patch.object(
            self.persistence,
            "save_tool_calls_action",
            side_effect=save_then_cancel,
        ):
            result = self._task_service(open_url).run_task_and_wait(
                "保存工具调用后取消",
                self.workspace,
                token,
            )

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.CANCELLED.value)
        self.assertEqual(task.termination_reason, "USER_CANCELLED")
        self.assertIsNotNone(task.finished_at)
        step = self.persistence.load_agent_steps(task.id)[0]
        self.assertEqual(step.status, AgentStepStatus.INTERRUPTED.value)
        self.assertIsNotNone(step.finished_at)
        tool_calls = self.persistence.load_tool_calls(task.id)
        self.assertEqual(len(tool_calls), 2)
        self.assertTrue(
            all(
                tool_call.status == ToolCallStatus.ERROR.value
                and tool_call.result_metadata
                == {"interrupted": True, "reason": "USER_CANCELLED"}
                and tool_call.finished_at is not None
                for tool_call in tool_calls
            )
        )
        result_messages = [
            message
            for message in self.persistence.load_messages(task.id)
            if message.message_type == MessageType.TOOL_RESULT.value
        ]
        self.assertEqual(len(result_messages), len(tool_calls))

    def test_context_overflow_closes_task_and_step_without_calling_model(self) -> None:
        self.context_manager = ContextManager(
            self.persistence,
            ContextLimits(1, 100),
        )
        model_called = False

        def open_url(request, *, timeout):
            nonlocal model_called
            model_called = True
            return self._final_response("不应调用模型")

        result = self._task_service(open_url).run_task_and_wait(
            "这个任务的基础上下文会超过一个字符",
            self.workspace,
        )

        self.assertEqual(result.status, TaskStatus.FAILED)
        self.assertIn("CONTEXT_OVERFLOW", result.error)
        self.assertFalse(model_called)
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.FAILED.value)
        self.assertEqual(task.error, result.error)
        self.assertIsNotNone(task.finished_at)
        step = self.persistence.load_agent_steps(task.id)[0]
        self.assertEqual(step.status, AgentStepStatus.FAILED.value)
        self.assertIsNotNone(step.finished_at)
        self.assertEqual(self.persistence.load_messages(task.id), [])
        self.assertEqual(self.persistence.load_tool_calls(task.id), [])

    def test_complete_task_after_read_only_tool_round(self) -> None:
        (self.workspace / "main.py").write_text(
            "print('hello')\n",
            encoding="utf-8",
        )
        payloads = []

        def open_url(request, *, timeout):
            payloads.append(json.loads(request.data.decode("utf-8")))
            if len(payloads) == 1:
                return self._tool_calls_response(
                    "call-read",
                    "read_file",
                    {"path": "main.py"},
                )
            return self._final_response("main.py 会输出 hello。")

        result = self._task_service(open_url).run_task_and_wait(
            "读取 main.py 并说明作用",
            self.workspace,
        )

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.COMPLETED.value)
        steps = self.persistence.load_agent_steps(task.id)
        self.assertEqual([step.step_number for step in steps], [0, 1])
        self.assertTrue(
            all(step.status == AgentStepStatus.COMPLETED.value for step in steps)
        )
        tool_call = self.persistence.load_tool_calls(task.id)[0]
        self.assertEqual(tool_call.status, ToolCallStatus.COMPLETED.value)
        messages = self.persistence.load_messages(task.id)
        self.assertEqual([message.sequence for message in messages], [0, 1, 2])
        self.assertEqual(
            [message.message_type for message in messages],
            [
                MessageType.TEXT.value,
                MessageType.TOOL_RESULT.value,
                MessageType.FINAL.value,
            ],
        )
        final_message = messages[-1]
        self.assertEqual(task.final_answer, result.final_answer)
        self.assertEqual(task.final_answer, final_message.content)
        self.assertEqual(payloads[1]["messages"][-1]["tool_call_id"], "call-read")
        self.assertIn("print('hello')", payloads[1]["messages"][-1]["content"])

    def test_finish_task_as_terminated_at_max_steps(self) -> None:
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            return self._tool_calls_response(
                f"call-list-{response_number}",
                "list_files",
                {"path": "."},
            )

        result = self._task_service(
            open_url,
            max_agent_steps=2,
        ).run_task_and_wait(
            "持续查看目录",
            self.workspace,
        )

        self.assertEqual(result.status, TaskStatus.TERMINATED)
        self.assertEqual(result.termination_reason, MAX_STEPS_TERMINATION_REASON)
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.TERMINATED.value)
        self.assertEqual(
            task.termination_reason,
            MAX_STEPS_TERMINATION_REASON,
        )
        self.assertIsNotNone(task.finished_at)
        self.assertEqual(len(self.persistence.load_agent_steps(task.id)), 2)
        self.assertTrue(
            all(
                step.status == AgentStepStatus.COMPLETED.value
                for step in self.persistence.load_agent_steps(task.id)
            )
        )
        tool_calls = self.persistence.load_tool_calls(task.id)
        self.assertEqual(len(tool_calls), 2)
        self.assertTrue(
            all(
                tool_call.status == ToolCallStatus.COMPLETED.value
                for tool_call in tool_calls
            )
        )
        self.assertNotIn(
            task.status,
            {TaskStatus.PENDING.value, TaskStatus.RUNNING.value},
        )
        self.assertFalse(
            any(
                step.status == AgentStepStatus.RUNNING.value
                for step in self.persistence.load_agent_steps(task.id)
            )
        )
        self.assertFalse(
            any(
                tool_call.status
                in {ToolCallStatus.PENDING.value, ToolCallStatus.RUNNING.value}
                for tool_call in tool_calls
            )
        )

    def test_finish_task_as_terminated_when_loop_is_detected(self) -> None:
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            response_number += 1
            return self._tool_calls_response(
                f"call-list-{response_number}",
                "list_files",
                {"path": "."},
            )

        result = self._task_service(open_url).run_task_and_wait(
            "持续重复查看目录",
            self.workspace,
        )

        self.assertEqual(result.status, TaskStatus.TERMINATED)
        self.assertEqual(
            result.termination_reason,
            LOOP_DETECTED_TERMINATION_REASON,
        )
        self.assertEqual(response_number, 3)
        task = self.db.scalars(select(Task)).one()
        self.assertEqual(task.status, TaskStatus.TERMINATED.value)
        self.assertEqual(
            task.termination_reason,
            LOOP_DETECTED_TERMINATION_REASON,
        )
        self.assertIsNotNone(task.finished_at)
        self.assertTrue(
            all(
                step.status == AgentStepStatus.COMPLETED.value
                for step in self.persistence.load_agent_steps(task.id)
            )
        )

    def test_create_a_new_session_for_each_run(self) -> None:
        service = self._task_service(
            lambda request, timeout: self._final_response("完成")
        )

        first_result = service.run_task_and_wait("任务一", self.workspace)
        second_result = service.run_task_and_wait("任务二", self.workspace)

        self.assertEqual(first_result.status, TaskStatus.COMPLETED)
        self.assertEqual(second_result.status, TaskStatus.COMPLETED)
        self.assertEqual(self._count(CodingSession), 2)
        self.assertEqual(self._count(Task), 2)
        session_ids = self.db.scalars(select(Task.session_id)).all()
        self.assertEqual(len(set(session_ids)), 2)

    def _task_service(self, open_url, *, max_agent_steps: int = 8) -> TaskService:
        gateway = LLMGateway(
            DeepSeekClient(api_key="secret", open_url=open_url),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
        )
        runtime = AgentRuntime(
            self.persistence,
            self.context_manager,
            gateway,
            ToolRouter(
                create_test_local_tool_registry(),
                WorkspacePathGuard(),
            ),
            RuntimePolicy(
                RuntimePolicyConfig(max_agent_steps=max_agent_steps)
            ),
            RetryWaiter(lambda seconds: None),
        )
        return TaskService(
            self.persistence,
            WorkspaceValidator(self.allowed_root),
            runtime,
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
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                        }
                    ],
                }
            ).encode("utf-8")
        )

    @staticmethod
    def _tool_calls_response(
        call_id: str,
        tool_name: str,
        arguments: dict,
    ) -> FakeHTTPResponse:
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
                                ],
                            },
                        }
                    ],
                }
            ).encode("utf-8")
        )

    def _count(self, model: type) -> int:
        return self.db.scalar(select(func.count()).select_from(model))


if __name__ == "__main__":
    unittest.main()
