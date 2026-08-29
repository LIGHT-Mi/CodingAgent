import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent.contracts import (
    AgentResult,
    AgentStepStatus,
    MessageRole,
    MessageType,
    TaskStatus,
    ToolCallRequest,
    ToolCallsAction,
    ToolCallStatus,
    ToolResult,
    ToolResultStatus,
)
from app.db.base import Base
from app.db.persistence import (
    InvalidStateTransitionError,
    PersistenceService,
    PersistenceValidationError,
    RecordNotFoundError,
)


class PersistenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        testing_session = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.db: Session = testing_session()
        self.persistence = PersistenceService(self.db)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_persist_complete_task_history_in_stable_order(self) -> None:
        coding_session = self.persistence.create_session()
        self.assertEqual(
            self.persistence.get_session(coding_session.id).id,
            coding_session.id,
        )

        task = self.persistence.create_task(
            session_id=coding_session.id,
            original_prompt="修复失败的测试",
            workspace="/workspace/demo",
        )
        self.assertEqual(task.status, TaskStatus.PENDING.value)

        self.persistence.start_task(task.id)
        first_step = self.persistence.create_agent_step(task.id, step_number=0)

        action = ToolCallsAction(
            content="先运行测试，再读取相关文件。",
            tool_calls=(
                ToolCallRequest(
                    tool_call_id="provider-command-1",
                    tool_name="run_command",
                    arguments={"command": "pytest"},
                    call_index=0,
                ),
                ToolCallRequest(
                    tool_call_id="provider-read-1",
                    tool_name="read_file",
                    arguments={"path": "src/main.py"},
                    call_index=1,
                ),
            ),
        )
        assistant_message, tool_calls = self.persistence.save_tool_calls_action(
            task.id,
            first_step.id,
            action,
        )

        self.assertEqual(assistant_message.sequence, 0)
        self.assertEqual(assistant_message.role, MessageRole.ASSISTANT.value)
        self.assertEqual(
            [tool_call.provider_call_id for tool_call in tool_calls],
            ["provider-command-1", "provider-read-1"],
        )

        command_call = self.persistence.start_tool_call(tool_calls[0].id)
        self.assertEqual(command_call.status, ToolCallStatus.RUNNING.value)
        command_call, command_message = self.persistence.save_tool_result(
            command_call.id,
            ToolResult(
                tool_call_id="provider-command-1",
                tool_name="run_command",
                status=ToolResultStatus.COMPLETED,
                content="测试执行完毕，但有一个失败用例。",
                metadata={
                    "exit_code": 1,
                    "stdout": "1 failed",
                    "stderr": "",
                    "duration_ms": 125,
                },
            ),
        )

        self.assertEqual(command_call.status, ToolCallStatus.COMPLETED.value)
        self.assertEqual(command_call.exit_code, 1)
        self.assertEqual(command_call.result_metadata["duration_ms"], 125)
        self.assertEqual(command_message.sequence, 1)
        self.assertEqual(command_message.role, MessageRole.TOOL.value)
        self.assertEqual(command_message.tool_call_id, command_call.id)

        rejected_call, rejected_message = self.persistence.save_tool_result(
            tool_calls[1].id,
            ToolResult(
                tool_call_id="provider-read-1",
                tool_name="read_file",
                status=ToolResultStatus.REJECTED,
                error="目标路径位于 Workspace 之外。",
                metadata={"path": "../outside.py"},
            ),
        )
        self.assertEqual(rejected_call.status, ToolCallStatus.REJECTED.value)
        self.assertIsNone(rejected_call.started_at)
        self.assertEqual(rejected_message.sequence, 2)
        self.assertEqual(rejected_message.content, "目标路径位于 Workspace 之外。")

        self.persistence.finish_agent_step(
            first_step.id,
            AgentStepStatus.COMPLETED,
        )
        second_step = self.persistence.create_agent_step(task.id, step_number=1)
        final_message = self.persistence.save_assistant_message(
            task.id,
            second_step.id,
            "已定位测试失败原因。",
            MessageType.FINAL,
        )
        self.assertEqual(final_message.sequence, 3)
        self.persistence.finish_agent_step(
            second_step.id,
            AgentStepStatus.COMPLETED,
        )
        completed_task = self.persistence.finish_task(
            task.id,
            AgentResult(
                status=TaskStatus.COMPLETED,
                final_answer="已定位测试失败原因。",
            ),
        )

        self.assertEqual(completed_task.status, TaskStatus.COMPLETED.value)
        self.assertIsNotNone(completed_task.finished_at)
        self.assertEqual(
            [step.step_number for step in self.persistence.load_agent_steps(task.id)],
            [0, 1],
        )
        messages = self.persistence.load_messages(task.id)
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
        self.assertEqual(
            [call.call_index for call in self.persistence.load_tool_calls(task.id)],
            [0, 1],
        )

    def test_finish_task_supports_every_terminal_result(self) -> None:
        terminal_results = (
            AgentResult(TaskStatus.COMPLETED, final_answer="完成"),
            AgentResult(TaskStatus.FAILED, error="不可恢复错误"),
            AgentResult(TaskStatus.CANCELLED, termination_reason="用户取消"),
            AgentResult(TaskStatus.TERMINATED, termination_reason="MAX_STEPS"),
        )

        for index, result in enumerate(terminal_results):
            with self.subTest(status=result.status):
                coding_session = self.persistence.create_session()
                task = self.persistence.create_task(
                    coding_session.id,
                    original_prompt=f"任务 {index}",
                    workspace=f"/workspace/{index}",
                )
                self.persistence.start_task(task.id)
                finished_task = self.persistence.finish_task(task.id, result)

                self.assertEqual(finished_task.status, result.status.value)
                self.assertEqual(finished_task.final_answer, result.final_answer)
                self.assertEqual(finished_task.error, result.error)
                self.assertEqual(
                    finished_task.termination_reason,
                    result.termination_reason,
                )

    def test_finish_agent_step_supports_every_terminal_status(self) -> None:
        terminal_steps = (
            (AgentStepStatus.COMPLETED, None),
            (AgentStepStatus.FAILED, "模型响应无法解析"),
            (AgentStepStatus.INTERRUPTED, None),
        )

        for index, (status, error) in enumerate(terminal_steps):
            with self.subTest(status=status):
                task, step = self._create_running_task_with_step(
                    prompt=f"Step 终态 {index}"
                )
                finished_step = self.persistence.finish_agent_step(
                    step.id,
                    status,
                    error=error,
                )

                self.assertEqual(finished_step.status, status.value)
                self.assertEqual(finished_step.error, error)
                self.assertIsNotNone(finished_step.finished_at)

    def test_persist_every_valid_tool_result_transition(self) -> None:
        task, step = self._create_running_task_with_step()
        requests = tuple(
            ToolCallRequest(
                tool_call_id=f"provider-call-{index}",
                tool_name="run_command",
                arguments={"command": "pytest"},
                call_index=index,
            )
            for index in range(5)
        )
        _, tool_calls = self.persistence.save_tool_calls_action(
            task.id,
            step.id,
            ToolCallsAction(tool_calls=requests),
        )

        pending_results = (
            ToolResultStatus.ERROR,
            ToolResultStatus.REJECTED,
        )
        for tool_call, status in zip(tool_calls[:2], pending_results, strict=True):
            persisted_call, _ = self.persistence.save_tool_result(
                tool_call.id,
                ToolResult(
                    tool_call_id=tool_call.provider_call_id,
                    tool_name=tool_call.tool_name,
                    status=status,
                    error=f"工具结果：{status.value}",
                ),
            )
            self.assertEqual(persisted_call.status, status.value)

        running_results = (
            ToolResultStatus.COMPLETED,
            ToolResultStatus.ERROR,
            ToolResultStatus.TIMEOUT,
        )
        for tool_call, status in zip(tool_calls[2:], running_results, strict=True):
            self.persistence.start_tool_call(tool_call.id)
            result = (
                ToolResult(
                    tool_call_id=tool_call.provider_call_id,
                    tool_name=tool_call.tool_name,
                    status=status,
                    content="命令已执行",
                    metadata={"exit_code": 0, "stdout": "ok", "stderr": ""},
                )
                if status is ToolResultStatus.COMPLETED
                else ToolResult(
                    tool_call_id=tool_call.provider_call_id,
                    tool_name=tool_call.tool_name,
                    status=status,
                    error=f"工具结果：{status.value}",
                )
            )
            persisted_call, _ = self.persistence.save_tool_result(
                tool_call.id,
                result,
            )
            self.assertEqual(persisted_call.status, status.value)

    def test_fail_open_tool_calls_closes_running_and_pending_calls(self) -> None:
        task, step = self._create_running_task_with_step()
        _, tool_calls = self.persistence.save_tool_calls_action(
            task.id,
            step.id,
            ToolCallsAction(
                tool_calls=(
                    ToolCallRequest(
                        tool_call_id="provider-running",
                        tool_name="read_file",
                        arguments={"path": "main.py"},
                        call_index=0,
                    ),
                    ToolCallRequest(
                        tool_call_id="provider-pending",
                        tool_name="list_files",
                        arguments={"path": "."},
                        call_index=1,
                    ),
                )
            ),
        )
        self.persistence.start_tool_call(tool_calls[0].id)

        failed_calls = self.persistence.fail_open_tool_calls(
            step.id,
            "Agent runtime failed with RuntimeError: tool crashed",
        )

        self.assertEqual(
            [call.provider_call_id for call in failed_calls],
            ["provider-running", "provider-pending"],
        )
        persisted_calls = self.persistence.load_tool_calls(task.id)
        self.assertTrue(
            all(call.status == ToolCallStatus.ERROR.value for call in persisted_calls)
        )
        self.assertIsNotNone(persisted_calls[0].started_at)
        self.assertIsNone(persisted_calls[1].started_at)
        self.assertTrue(all(call.finished_at is not None for call in persisted_calls))
        self.assertTrue(
            all(call.result_metadata == {"fatal": True} for call in persisted_calls)
        )
        messages = self.persistence.load_messages(task.id)
        self.assertEqual([message.sequence for message in messages], [0, 1, 2])
        self.assertEqual(
            [message.message_type for message in messages],
            [
                MessageType.TEXT.value,
                MessageType.TOOL_RESULT.value,
                MessageType.TOOL_RESULT.value,
            ],
        )
        self.assertEqual(
            [message.tool_call_id for message in messages[1:]],
            [persisted_calls[0].id, persisted_calls[1].id],
        )

    def test_reject_invalid_lifecycle_transitions(self) -> None:
        coding_session = self.persistence.create_session()
        task = self.persistence.create_task(
            coding_session.id,
            original_prompt="检查状态流转",
            workspace="/workspace/demo",
        )

        with self.assertRaises(InvalidStateTransitionError):
            self.persistence.create_agent_step(task.id, step_number=0)

        self.persistence.start_task(task.id)
        with self.assertRaises(InvalidStateTransitionError):
            self.persistence.start_task(task.id)

        step = self.persistence.create_agent_step(task.id, step_number=0)
        with self.assertRaises(PersistenceValidationError):
            self.persistence.create_agent_step(task.id, step_number=0)

        with self.assertRaises(PersistenceValidationError):
            self.persistence.finish_agent_step(
                step.id,
                AgentStepStatus.FAILED,
            )

        self.persistence.finish_agent_step(step.id, AgentStepStatus.COMPLETED)
        with self.assertRaises(InvalidStateTransitionError):
            self.persistence.finish_agent_step(step.id, AgentStepStatus.COMPLETED)

    def test_reject_mismatched_and_invalid_tool_results(self) -> None:
        task, step = self._create_running_task_with_step()
        _, tool_calls = self.persistence.save_tool_calls_action(
            task.id,
            step.id,
            ToolCallsAction(
                tool_calls=(
                    ToolCallRequest(
                        tool_call_id="provider-read-1",
                        tool_name="read_file",
                        arguments={"path": "main.py"},
                    ),
                )
            ),
        )
        tool_call = tool_calls[0]

        with self.assertRaises(PersistenceValidationError):
            self.persistence.save_tool_result(
                tool_call.id,
                ToolResult(
                    tool_call_id="wrong-provider-id",
                    tool_name="read_file",
                    status=ToolResultStatus.ERROR,
                    error="参数错误",
                ),
            )

        with self.assertRaises(InvalidStateTransitionError):
            self.persistence.save_tool_result(
                tool_call.id,
                ToolResult(
                    tool_call_id="provider-read-1",
                    tool_name="read_file",
                    status=ToolResultStatus.COMPLETED,
                    content="文件内容",
                ),
            )

        self.persistence.start_tool_call(tool_call.id)
        with self.assertRaises(PersistenceValidationError):
            self.persistence.save_tool_result(
                tool_call.id,
                ToolResult(
                    tool_call_id="provider-read-1",
                    tool_name="read_file",
                    status=ToolResultStatus.COMPLETED,
                    content="文件内容",
                    metadata={"stdout": 123},
                ),
            )

        self.db.refresh(tool_call)
        self.assertEqual(tool_call.status, ToolCallStatus.RUNNING.value)
        self.assertIsNone(tool_call.result_message)

    def test_missing_records_raise_clear_errors(self) -> None:
        with self.assertRaises(RecordNotFoundError):
            self.persistence.create_task(
                "missing-session",
                original_prompt="任务",
                workspace="/workspace/demo",
            )
        with self.assertRaises(RecordNotFoundError):
            self.persistence.load_messages("missing-task")

    def _create_running_task_with_step(self, prompt: str = "读取文件"):
        coding_session = self.persistence.create_session()
        task = self.persistence.create_task(
            coding_session.id,
            original_prompt=prompt,
            workspace="/workspace/demo",
        )
        self.persistence.start_task(task.id)
        step = self.persistence.create_agent_step(task.id, step_number=0)
        return task, step


if __name__ == "__main__":
    unittest.main()
