import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent import (
    ToolCallRequest,
    ToolCallsAction,
    ToolResult,
    ToolResultStatus,
)
from app.context import ContextLimits, ContextManager
from app.db.base import Base
from app.db.persistence import PersistenceService
from app.llm import LLMContext, LLMMessageRole


class ContextToolResultStatusTests(unittest.TestCase):
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

    def test_every_tool_result_status_uses_same_context_truncation_rule(self) -> None:
        _, task = self.persistence.create_session_with_task(
            title="检查所有工具结果状态的上下文。",
            original_prompt="检查所有工具结果状态的上下文。",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(task.id)
        step = self.persistence.create_agent_step(task.id, 0)
        statuses = (
            ToolResultStatus.COMPLETED,
            ToolResultStatus.ERROR,
            ToolResultStatus.REJECTED,
            ToolResultStatus.TIMEOUT,
        )
        _, records = self.persistence.save_tool_calls_action(
            task.id,
            step.id,
            ToolCallsAction(
                tool_calls=tuple(
                    ToolCallRequest(
                        tool_call_id=f"provider-{status.value.lower()}",
                        tool_name="read_file",
                        arguments={"path": f"{status.value.lower()}.txt"},
                        call_index=index,
                    )
                    for index, status in enumerate(statuses)
                )
            ),
        )

        original_contents: dict[str, str] = {}
        for record, status in zip(records, statuses):
            if status in {
                ToolResultStatus.COMPLETED,
                ToolResultStatus.TIMEOUT,
            }:
                self.persistence.start_tool_call(record.id)
            content = (
                f"{status.value}-BEGIN-"
                + "中文输出" * 300
                + f"-{status.value}-END"
            )
            original_contents[record.id] = content
            self.persistence.save_tool_result(
                record.id,
                ToolResult(
                    tool_call_id=record.provider_call_id,
                    tool_name=record.tool_name,
                    status=status,
                    content=content,
                    error=None if status is ToolResultStatus.COMPLETED else status.value,
                ),
            )

        context_result = ContextManager(
            self.persistence,
            limits=ContextLimits(60_000, 200),
        ).build(task.id)

        self.assertIsInstance(context_result, LLMContext)
        assert isinstance(context_result, LLMContext)
        context_tool_messages = tuple(
            message
            for message in context_result.messages
            if message.role is LLMMessageRole.TOOL
        )
        stored_result_messages = self.persistence.load_messages(task.id)[1:]
        self.assertEqual(len(context_tool_messages), len(statuses))
        self.assertEqual(len(stored_result_messages), len(statuses))

        records_by_provider_id = {
            record.provider_call_id: record for record in records
        }
        for context_message, stored_message in zip(
            context_tool_messages,
            stored_result_messages,
        ):
            assert context_message.tool_call_id is not None
            record = records_by_provider_id[context_message.tool_call_id]
            original_content = original_contents[record.id]
            assert context_message.content is not None
            self.assertLessEqual(len(context_message.content), 200)
            self.assertIn("[Tool Result truncated]", context_message.content)
            self.assertIn(
                f"original_characters: {len(original_content)}",
                context_message.content,
            )
            self.assertIn(original_content[:8], context_message.content)
            self.assertIn(original_content[-8:], context_message.content)
            self.assertEqual(stored_message.content, original_content)
            self.assertGreater(len(stored_message.content), 200)


if __name__ == "__main__":
    unittest.main()
