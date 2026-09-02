import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent import (
    AgentResult,
    AgentStepStatus,
    MessageType,
    TaskStatus,
    ToolCallRequest,
    ToolCallsAction,
    ToolResult,
    ToolResultStatus,
)
from app.context import (
    CODING_AGENT_SYSTEM_PROMPT,
    ContextHistoryError,
    ContextLimits,
    ContextManager,
    ContextTaskNotFoundError,
    ConversationTurnBlock,
    InteractionBlock,
)
from app.context.manager import (
    CANCELLED_TASK_SUMMARY,
    FAILED_TASK_SUMMARY_FALLBACK,
    FAILED_TASK_SUMMARY_PREFIX,
    TERMINATED_TASK_SUMMARY_FALLBACK,
    TERMINATED_TASK_SUMMARY_PREFIX,
    _restore_conversation_turn_blocks,
    _restore_interaction_blocks,
)
from app.db.base import Base
from app.db.models.message import Message
from app.db.models.task import Task
from app.db.models.tool_call import ToolCall
from app.db.persistence import PersistenceService
from app.llm.contracts import LLMMessage, LLMMessageRole


def stored_assistant(
    message_id: str,
    sequence: int,
    *,
    step_id: str = "step-0",
    content: str = "调用工具",
) -> Message:
    return Message(
        id=message_id,
        task_id="task-0",
        step_id=step_id,
        sequence=sequence,
        role="ASSISTANT",
        message_type="TEXT",
        content=content,
    )


def stored_tool_result(
    message_id: str,
    sequence: int,
    tool_call_id: str | None,
    *,
    step_id: str = "step-0",
) -> Message:
    return Message(
        id=message_id,
        task_id="task-0",
        step_id=step_id,
        tool_call_id=tool_call_id,
        sequence=sequence,
        role="TOOL",
        message_type="TOOL_RESULT",
        content=f"result-{message_id}",
    )


def stored_tool_call(
    tool_call_id: str,
    assistant_message_id: str,
    call_index: int,
    *,
    step_id: str = "step-0",
) -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        step_id=step_id,
        assistant_message_id=assistant_message_id,
        provider_call_id=f"provider-{tool_call_id}",
        call_index=call_index,
        tool_name="read_file",
        arguments={"path": f"{tool_call_id}.txt"},
        status="COMPLETED",
    )


class ContextManagerTests(unittest.TestCase):
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
        self.manager = ContextManager(
            self.persistence,
            ContextLimits(60_000, 12_000),
        )

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_build_system_and_original_prompt_when_history_is_empty(self) -> None:
        _, task = self.persistence.create_session_with_task(
            title="解释 Python 列表和元组的区别。",
            original_prompt="解释 Python 列表和元组的区别。",
            workspace="/workspace/demo",
        )

        with (
            patch.object(
                self.persistence,
                "load_messages",
                wraps=self.persistence.load_messages,
            ) as load_messages,
            patch.object(
                self.persistence,
                "load_tool_calls",
                wraps=self.persistence.load_tool_calls,
            ) as load_tool_calls,
        ):
            context = self.manager.build(task.id)

        load_messages.assert_called_once_with(task.id)
        load_tool_calls.assert_called_once_with(task.id)
        self.assertEqual(len(context.messages), 2)
        self.assertEqual(
            [message.role for message in context.messages],
            [LLMMessageRole.SYSTEM, LLMMessageRole.USER],
        )
        self.assertEqual(context.messages[0].content, CODING_AGENT_SYSTEM_PROMPT)
        self.assertEqual(context.messages[1].content, task.original_prompt)
        for required_text in (
            "list_files",
            "read_file",
            "search_files",
            "create_file",
            "write_file",
            "edit_file",
            "run_command",
            "不得虚构",
            "修改已有文件前",
            "再次使用 read_file 验证",
            "根据 Observation 修正",
            "非零 exit code",
            "timeout",
            "不等于任务失败",
            "当前仍不能删除文件",
            "不得声称未实际执行的命令已经成功",
            "不得声称未验证的修改已经成功",
            "历史 Assistant 最终回答只是会话摘要",
            "不是新的工具证据",
            "不得从摘要推导或补写",
            "必须先调用工具重新验证",
            "不得根据常见项目结构或命名习惯猜测",
        ):
            self.assertIn(required_text, CODING_AGENT_SYSTEM_PROMPT)

    def test_restore_complete_tool_history_with_provider_call_ids(self) -> None:
        _, task = self.persistence.create_session_with_task(
            title="读取项目文件后进行说明。",
            original_prompt="读取项目文件后进行说明。",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(task.id)
        step = self.persistence.create_agent_step(task.id, step_number=0)
        action = ToolCallsAction(
            content="我先查看目录和文件。",
            tool_calls=(
                ToolCallRequest(
                    tool_call_id="provider-read",
                    tool_name="read_file",
                    arguments={"path": "README.md"},
                    call_index=1,
                ),
                ToolCallRequest(
                    tool_call_id="provider-list",
                    tool_name="list_files",
                    arguments={"path": "."},
                    call_index=0,
                ),
            ),
        )
        _, records = self.persistence.save_tool_calls_action(
            task.id,
            step.id,
            action,
        )
        records_by_provider_id = {
            record.provider_call_id: record for record in records
        }
        list_record = records_by_provider_id["provider-list"]
        read_record = records_by_provider_id["provider-read"]

        self.persistence.start_tool_call(list_record.id)
        self.persistence.save_tool_result(
            list_record.id,
            ToolResult(
                tool_call_id="provider-list",
                tool_name="list_files",
                status=ToolResultStatus.COMPLETED,
                content="README.md\tfile",
            ),
        )
        self.persistence.save_tool_result(
            read_record.id,
            ToolResult(
                tool_call_id="provider-read",
                tool_name="read_file",
                status=ToolResultStatus.ERROR,
                error="file is not valid UTF-8",
            ),
        )

        context = self.manager.build(task.id)

        self.assertEqual(
            [message.role for message in context.messages],
            [
                LLMMessageRole.SYSTEM,
                LLMMessageRole.USER,
                LLMMessageRole.ASSISTANT,
                LLMMessageRole.TOOL,
                LLMMessageRole.TOOL,
            ],
        )
        assistant_message = context.messages[2]
        self.assertEqual(assistant_message.content, action.content)
        self.assertEqual(
            [call.tool_call_id for call in assistant_message.tool_calls],
            ["provider-list", "provider-read"],
        )
        self.assertEqual(
            [call.call_index for call in assistant_message.tool_calls],
            [0, 1],
        )
        self.assertEqual(
            assistant_message.tool_calls[0].arguments_json,
            '{"path":"."}',
        )
        self.assertEqual(context.messages[3].content, "README.md\tfile")
        self.assertEqual(context.messages[3].tool_call_id, "provider-list")
        self.assertEqual(
            context.messages[4].content,
            "file is not valid UTF-8",
        )
        self.assertEqual(context.messages[4].tool_call_id, "provider-read")
        self.assertNotEqual(context.messages[3].tool_call_id, list_record.id)
        self.assertNotEqual(context.messages[4].tool_call_id, read_record.id)

    def test_restore_prior_tasks_as_compact_conversation_turns_only(self) -> None:
        coding_session, historical_task = (
            self.persistence.create_session_with_task(
                title="检查历史文件",
                original_prompt="读取旧文件并总结",
                workspace="/workspace/demo",
            )
        )
        self.persistence.start_task(historical_task.id)
        tool_step = self.persistence.create_agent_step(historical_task.id, 0)
        _, tool_calls = self.persistence.save_tool_calls_action(
            historical_task.id,
            tool_step.id,
            ToolCallsAction(
                tool_calls=(
                    ToolCallRequest(
                        tool_call_id="historical-provider-call",
                        tool_name="read_file",
                        arguments={"path": "secret-history.txt"},
                    ),
                )
            ),
        )
        self.persistence.start_tool_call(tool_calls[0].id)
        self.persistence.save_tool_result(
            tool_calls[0].id,
            ToolResult(
                tool_call_id="historical-provider-call",
                tool_name="read_file",
                status=ToolResultStatus.COMPLETED,
                content="历史文件的完整内容不应进入新 Task",
            ),
        )
        self.persistence.finish_agent_step(
            tool_step.id,
            AgentStepStatus.COMPLETED,
        )
        final_step = self.persistence.create_agent_step(historical_task.id, 1)
        self.persistence.save_assistant_message(
            historical_task.id,
            final_step.id,
            "旧任务已经完成。",
            MessageType.FINAL,
        )
        self.persistence.finish_agent_step(
            final_step.id,
            AgentStepStatus.COMPLETED,
        )
        self.persistence.finish_task(
            historical_task.id,
            AgentResult(
                TaskStatus.COMPLETED,
                final_answer="旧任务已经完成。",
            ),
        )
        current_task = self.persistence.create_task_in_session(
            coding_session.id,
            original_prompt="基于上一轮继续处理",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(current_task.id)
        current_step = self.persistence.create_agent_step(current_task.id, 0)
        _, current_calls = self.persistence.save_tool_calls_action(
            current_task.id,
            current_step.id,
            ToolCallsAction(
                content="检查当前目录。",
                tool_calls=(
                    ToolCallRequest(
                        tool_call_id="current-provider-call",
                        tool_name="list_files",
                        arguments={"path": "."},
                    ),
                ),
            ),
        )
        self.persistence.start_tool_call(current_calls[0].id)
        self.persistence.save_tool_result(
            current_calls[0].id,
            ToolResult(
                tool_call_id="current-provider-call",
                tool_name="list_files",
                status=ToolResultStatus.COMPLETED,
                content="README.md\tfile",
            ),
        )

        with (
            patch.object(
                self.persistence,
                "load_messages",
                wraps=self.persistence.load_messages,
            ) as load_messages,
            patch.object(
                self.persistence,
                "load_tool_calls",
                wraps=self.persistence.load_tool_calls,
            ) as load_tool_calls,
        ):
            context = self.manager.build(current_task.id)

        load_messages.assert_called_once_with(current_task.id)
        load_tool_calls.assert_called_once_with(current_task.id)
        self.assertEqual(
            [message.role for message in context.messages],
            [
                LLMMessageRole.SYSTEM,
                LLMMessageRole.USER,
                LLMMessageRole.ASSISTANT,
                LLMMessageRole.USER,
                LLMMessageRole.ASSISTANT,
                LLMMessageRole.TOOL,
            ],
        )
        self.assertEqual(context.messages[1].content, "读取旧文件并总结")
        self.assertEqual(context.messages[2].content, "旧任务已经完成。")
        self.assertEqual(context.messages[3].content, "基于上一轮继续处理")
        self.assertEqual(context.messages[4].content, "检查当前目录。")
        self.assertEqual(
            context.messages[4].tool_calls[0].tool_call_id,
            "current-provider-call",
        )
        self.assertEqual(context.messages[5].content, "README.md\tfile")
        self.assertEqual(
            context.messages[5].tool_call_id,
            "current-provider-call",
        )
        serialized_context = "\n".join(
            message.content or "" for message in context.messages
        )
        self.assertNotIn("历史文件的完整内容", serialized_context)
        self.assertNotIn("secret-history.txt", serialized_context)

    def test_non_completed_prior_tasks_use_deterministic_summaries(self) -> None:
        coding_session, failed_task = self.persistence.create_session_with_task(
            title="多轮终态摘要",
            original_prompt="失败的上一轮",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(failed_task.id)
        self.persistence.finish_task(
            failed_task.id,
            AgentResult(TaskStatus.FAILED, error="公开失败原因"),
        )
        cancelled_task = self.persistence.create_task_in_session(
            coding_session.id,
            "取消的上一轮",
            "/workspace/demo",
        )
        self.persistence.start_task(cancelled_task.id)
        self.persistence.finish_task(
            cancelled_task.id,
            AgentResult(
                TaskStatus.CANCELLED,
                termination_reason="USER_CANCELLED",
            ),
        )
        terminated_task = self.persistence.create_task_in_session(
            coding_session.id,
            "终止的上一轮",
            "/workspace/demo",
        )
        self.persistence.start_task(terminated_task.id)
        self.persistence.finish_task(
            terminated_task.id,
            AgentResult(
                TaskStatus.TERMINATED,
                termination_reason="MAX_STEPS",
            ),
        )
        current_task = self.persistence.create_task_in_session(
            coding_session.id,
            "当前任务",
            "/workspace/demo",
        )

        context = self.manager.build(current_task.id)

        self.assertEqual(
            [message.role for message in context.messages],
            [
                LLMMessageRole.SYSTEM,
                LLMMessageRole.USER,
                LLMMessageRole.ASSISTANT,
                LLMMessageRole.USER,
                LLMMessageRole.ASSISTANT,
                LLMMessageRole.USER,
                LLMMessageRole.ASSISTANT,
                LLMMessageRole.USER,
            ],
        )
        self.assertEqual(
            [context.messages[index].content for index in (2, 4, 6)],
            [
                "上一轮任务执行失败：公开失败原因",
                CANCELLED_TASK_SUMMARY,
                "上一轮任务已终止：MAX_STEPS",
            ],
        )
        self.assertEqual(context.messages[-1].content, "当前任务")

    def test_restore_plain_assistant_message(self) -> None:
        _, task = self.persistence.create_session_with_task(
            title="说明进度。",
            original_prompt="说明进度。",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(task.id)
        step = self.persistence.create_agent_step(task.id, step_number=0)
        self.persistence.save_assistant_message(
            task.id,
            step.id,
            "正在分析。",
            MessageType.TEXT,
        )

        context = self.manager.build(task.id)

        self.assertEqual(context.messages[2].role, LLMMessageRole.ASSISTANT)
        self.assertEqual(context.messages[2].content, "正在分析。")
        self.assertEqual(context.messages[2].tool_calls, ())

    def test_reject_tool_call_without_tool_result_message(self) -> None:
        _, task = self.persistence.create_session_with_task(
            title="读取项目文件。",
            original_prompt="读取项目文件。",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(task.id)
        step = self.persistence.create_agent_step(task.id, step_number=0)
        self.persistence.save_tool_calls_action(
            task.id,
            step.id,
            ToolCallsAction(
                tool_calls=(
                    ToolCallRequest(
                        tool_call_id="provider-missing-result",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                )
            ),
        )
        self.manager = ContextManager(
            self.persistence,
            ContextLimits(1, 100),
        )

        with self.assertRaises(ContextHistoryError) as caught:
            self.manager.build(task.id)

        self.assertIn("no corresponding TOOL_RESULT", str(caught.exception))

    def test_running_task_rejects_final_message_as_history(self) -> None:
        _, task = self.persistence.create_session_with_task(
            title="完成任务。",
            original_prompt="完成任务。",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(task.id)
        step = self.persistence.create_agent_step(task.id, step_number=0)
        self.persistence.save_assistant_message(
            task.id,
            step.id,
            "已经完成。",
            MessageType.FINAL,
        )

        with self.assertRaisesRegex(
            ContextHistoryError,
            "RUNNING Task cannot contain FINAL",
        ):
            self.manager.build(task.id)

    def test_report_missing_task(self) -> None:
        with self.assertRaises(ContextTaskNotFoundError) as caught:
            self.manager.build("missing-task")

        self.assertIn("missing-task", str(caught.exception))

    def test_reject_non_persistence_dependency(self) -> None:
        with self.assertRaises(TypeError):
            ContextManager(  # type: ignore[arg-type]
                object(),
                ContextLimits(60_000, 12_000),
            )

    def test_reject_non_context_limits_dependency(self) -> None:
        with self.assertRaises(TypeError):
            ContextManager(self.persistence, object())  # type: ignore[arg-type]


class ConversationTurnBlockTests(unittest.TestCase):
    def test_requires_one_user_and_one_plain_assistant_message(self) -> None:
        block = ConversationTurnBlock(
            (
                LLMMessage(LLMMessageRole.USER, "上一轮需求"),
                LLMMessage(LLMMessageRole.ASSISTANT, "上一轮回答"),
            )
        )

        self.assertEqual(
            [message.role for message in block.messages],
            [LLMMessageRole.USER, LLMMessageRole.ASSISTANT],
        )
        with self.assertRaises(ValueError):
            ConversationTurnBlock((block.messages[0],))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ConversationTurnBlock(
                (block.messages[1], block.messages[0])
            )

    def test_rejects_non_terminal_or_incomplete_historical_task(self) -> None:
        pending_history = Task(
            id="pending-history",
            session_id="session-1",
            original_prompt="尚未结束",
            workspace="/workspace/demo",
            status=TaskStatus.PENDING.value,
        )
        completed_without_answer = Task(
            id="broken-completed",
            session_id="session-1",
            original_prompt="声称完成",
            workspace="/workspace/demo",
            status=TaskStatus.COMPLETED.value,
        )
        current = Task(
            id="current",
            session_id="session-1",
            original_prompt="当前任务",
            workspace="/workspace/demo",
            status=TaskStatus.PENDING.value,
        )

        for historical_task, expected_error in (
            (pending_history, "is not terminal"),
            (completed_without_answer, "has no final_answer"),
        ):
            with self.subTest(task=historical_task.id):
                with self.assertRaisesRegex(
                    ContextHistoryError,
                    expected_error,
                ):
                    _restore_conversation_turn_blocks(
                        current,
                        [historical_task, current],
                    )

    def test_rejects_current_task_that_is_not_latest(self) -> None:
        current = Task(
            id="current",
            session_id="session-1",
            original_prompt="当前任务",
            workspace="/workspace/demo",
            status=TaskStatus.RUNNING.value,
        )
        later = Task(
            id="later",
            session_id="session-1",
            original_prompt="异常的后续任务",
            workspace="/workspace/demo",
            status=TaskStatus.COMPLETED.value,
            final_answer="异常记录",
        )

        with self.assertRaisesRegex(ContextHistoryError, "not the latest"):
            _restore_conversation_turn_blocks(current, [current, later])

    def test_uses_fixed_fallbacks_when_public_terminal_details_are_missing(
        self,
    ) -> None:
        failed = Task(
            id="failed",
            session_id="session-1",
            original_prompt="失败任务",
            workspace="/workspace/demo",
            status=TaskStatus.FAILED.value,
        )
        terminated = Task(
            id="terminated",
            session_id="session-1",
            original_prompt="终止任务",
            workspace="/workspace/demo",
            status=TaskStatus.TERMINATED.value,
        )
        current = Task(
            id="current",
            session_id="session-1",
            original_prompt="当前任务",
            workspace="/workspace/demo",
            status=TaskStatus.PENDING.value,
        )

        blocks = _restore_conversation_turn_blocks(
            current,
            [failed, terminated, current],
        )

        self.assertEqual(
            [block.messages[1].content for block in blocks],
            [
                FAILED_TASK_SUMMARY_PREFIX + FAILED_TASK_SUMMARY_FALLBACK,
                TERMINATED_TASK_SUMMARY_PREFIX
                + TERMINATED_TASK_SUMMARY_FALLBACK,
            ],
        )

    def test_rejects_task_from_another_session(self) -> None:
        foreign = Task(
            id="foreign",
            session_id="session-2",
            original_prompt="其他会话",
            workspace="/workspace/demo",
            status=TaskStatus.COMPLETED.value,
            final_answer="其他会话回答",
        )
        current = Task(
            id="current",
            session_id="session-1",
            original_prompt="当前任务",
            workspace="/workspace/demo",
            status=TaskStatus.PENDING.value,
        )

        with self.assertRaisesRegex(ContextHistoryError, "another Session"):
            _restore_conversation_turn_blocks(current, [foreign, current])


class InteractionBlockRestorationTests(unittest.TestCase):
    def test_restores_blocks_by_assistant_sequence_and_calls_by_index(self) -> None:
        first_assistant = stored_assistant("assistant-0", 0)
        first_call = stored_tool_call("call-a", first_assistant.id, 0)
        second_call = stored_tool_call("call-b", first_assistant.id, 1)
        first_result = stored_tool_result("result-a", 1, first_call.id)
        second_result = stored_tool_result("result-b", 2, second_call.id)
        second_assistant = stored_assistant(
            "assistant-1",
            3,
            step_id="step-1",
            content="继续分析。",
        )

        blocks = _restore_interaction_blocks(
            [second_assistant, second_result, first_assistant, first_result],
            [second_call, first_call],
        )

        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(isinstance(block, InteractionBlock) for block in blocks))
        self.assertEqual(
            [call.call_index for call in blocks[0].messages[0].tool_calls],
            [0, 1],
        )
        self.assertEqual(
            [message.tool_call_id for message in blocks[0].messages[1:]],
            ["provider-call-a", "provider-call-b"],
        )
        self.assertEqual(blocks[1].messages[0].content, "继续分析。")

    def test_rejects_tool_call_without_source_assistant(self) -> None:
        call = stored_tool_call("call-a", "missing-assistant", 0)

        with self.assertRaisesRegex(
            ContextHistoryError,
            "outside task history",
        ):
            _restore_interaction_blocks([], [call])

    def test_rejects_tool_call_whose_source_is_a_tool_message(self) -> None:
        result = stored_tool_result("result-a", 0, "call-a")
        call = stored_tool_call("call-a", result.id, 0)

        with self.assertRaisesRegex(
            ContextHistoryError,
            "source Message is not an Assistant",
        ):
            _restore_interaction_blocks([result], [call])

    def test_rejects_missing_duplicate_and_unknown_tool_results(self) -> None:
        assistant = stored_assistant("assistant-0", 0)
        call = stored_tool_call("call-a", assistant.id, 0)
        result = stored_tool_result("result-a", 1, call.id)
        duplicate = stored_tool_result("result-b", 2, call.id)
        unknown = stored_tool_result("result-unknown", 1, "missing-call")

        invalid_histories = (
            ([assistant], [call], "no corresponding TOOL_RESULT"),
            (
                [assistant, result, duplicate],
                [call],
                "multiple TOOL_RESULT Messages",
            ),
            (
                [assistant, unknown],
                [call],
                "unknown internal ToolCall id",
            ),
        )
        for messages, calls, expected_error in invalid_histories:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ContextHistoryError, expected_error):
                    _restore_interaction_blocks(messages, calls)

    def test_rejects_result_before_assistant_tool_call(self) -> None:
        assistant = stored_assistant("assistant-0", 1)
        call = stored_tool_call("call-a", assistant.id, 0)
        result = stored_tool_result("result-a", 0, call.id)

        with self.assertRaisesRegex(ContextHistoryError, "appears before"):
            _restore_interaction_blocks([assistant, result], [call])

    def test_rejects_duplicate_call_index_in_one_block(self) -> None:
        assistant = stored_assistant("assistant-0", 0)
        first_call = stored_tool_call("call-a", assistant.id, 0)
        second_call = stored_tool_call("call-b", assistant.id, 0)
        first_result = stored_tool_result("result-a", 1, first_call.id)
        second_result = stored_tool_result("result-b", 2, second_call.id)

        with self.assertRaisesRegex(ContextHistoryError, "duplicate call_index"):
            _restore_interaction_blocks(
                [assistant, first_result, second_result],
                [first_call, second_call],
            )

    def test_rejects_tool_result_order_that_differs_from_call_index(self) -> None:
        assistant = stored_assistant("assistant-0", 0)
        first_call = stored_tool_call("call-a", assistant.id, 0)
        second_call = stored_tool_call("call-b", assistant.id, 1)
        second_result = stored_tool_result("result-b", 1, second_call.id)
        first_result = stored_tool_result("result-a", 2, first_call.id)

        with self.assertRaisesRegex(ContextHistoryError, "call_index order"):
            _restore_interaction_blocks(
                [assistant, second_result, first_result],
                [first_call, second_call],
            )

    def test_rejects_duplicate_or_unassigned_messages(self) -> None:
        assistant = stored_assistant("assistant-0", 0)
        orphan_result = stored_tool_result("orphan-result", 1, None)

        with self.assertRaisesRegex(ContextHistoryError, "duplicate Message id"):
            _restore_interaction_blocks([assistant, assistant], [])
        with self.assertRaisesRegex(ContextHistoryError, "has no internal ToolCall"):
            _restore_interaction_blocks([assistant, orphan_result], [])

    def test_rejects_duplicate_message_sequence(self) -> None:
        first_assistant = stored_assistant("assistant-0", 0, content="第一条")
        second_assistant = stored_assistant(
            "assistant-1",
            0,
            step_id="step-1",
            content="重复 sequence",
        )

        with self.assertRaisesRegex(ContextHistoryError, "duplicate Message sequence"):
            _restore_interaction_blocks(
                [first_assistant, second_assistant],
                [],
            )

    def test_rejects_interleaved_blocks_that_would_reorder_history(self) -> None:
        first_assistant = stored_assistant("assistant-0", 0)
        first_call = stored_tool_call("call-a", first_assistant.id, 0)
        first_result = stored_tool_result("result-a", 2, first_call.id)
        second_assistant = stored_assistant(
            "assistant-1",
            1,
            step_id="step-1",
            content="不应插入工具交互中间。",
        )

        with self.assertRaisesRegex(ContextHistoryError, "contiguous"):
            _restore_interaction_blocks(
                [first_assistant, second_assistant, first_result],
                [first_call],
            )


if __name__ == "__main__":
    unittest.main()
