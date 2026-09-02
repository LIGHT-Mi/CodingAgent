import re
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent import ToolCallRequest, ToolCallsAction, ToolResult, ToolResultStatus
from app.context import (
    ContextLimits,
    ContextManager,
    InteractionBlock,
    ToolResultTruncator,
)
from app.db.base import Base
from app.db.persistence import PersistenceService
from app.llm import LLMMessage, LLMMessageRole, LLMToolCall


def tool_call(arguments_json: str = '{"path":"main.py"}') -> LLMToolCall:
    return LLMToolCall(
        tool_call_id="provider-call",
        tool_name="read_file",
        arguments_json=arguments_json,
        call_index=0,
    )


def tool_message(content: str) -> LLMMessage:
    return LLMMessage(
        role=LLMMessageRole.TOOL,
        content=content,
        tool_call_id="provider-call",
    )


class ToolResultTruncatorTests(unittest.TestCase):
    def test_truncates_only_tool_message_content(self) -> None:
        truncator = ToolResultTruncator(ContextLimits(500, 100))
        long_text = "x" * 1_000
        messages = (
            LLMMessage(LLMMessageRole.SYSTEM, long_text),
            LLMMessage(LLMMessageRole.USER, long_text),
            LLMMessage(LLMMessageRole.ASSISTANT, long_text),
            LLMMessage(
                role=LLMMessageRole.ASSISTANT,
                content=None,
                tool_calls=(tool_call(long_text),),
            ),
        )

        for message in messages:
            with self.subTest(role=message.role, has_calls=bool(message.tool_calls)):
                self.assertIs(truncator.truncate_message(message), message)

    def test_detailed_result_fits_limit_and_preserves_beginning_and_end(self) -> None:
        limit = 200
        content = "BEGIN-" + "x" * 1_000 + "-END"
        original = tool_message(content)

        truncated = ToolResultTruncator(
            ContextLimits(1_000, limit)
        ).truncate_message(original)

        self.assertIsNot(truncated, original)
        assert truncated.content is not None
        self.assertLessEqual(len(truncated.content), limit)
        self.assertIn("[Tool Result truncated]", truncated.content)
        self.assertIn(f"original_characters: {len(content)}", truncated.content)
        self.assertIn("showing: beginning and end", truncated.content)
        beginning_section, end = truncated.content.split(
            "\n\n--- omitted ---\n\n--- end ---\n",
            maxsplit=1,
        )
        beginning = beginning_section.split("--- beginning ---\n", maxsplit=1)[1]
        retained = int(
            re.search(r"retained_characters: (\d+)", truncated.content).group(1)  # type: ignore[union-attr]
        )
        self.assertEqual(len(beginning) + len(end), retained)
        self.assertEqual(len(beginning), (retained + 1) // 2)
        self.assertEqual(len(end), retained // 2)
        self.assertTrue(content.startswith(beginning))
        self.assertTrue(content.endswith(end))
        self.assertEqual(truncated.tool_call_id, original.tool_call_id)

    def test_compact_result_supports_small_injected_limit(self) -> None:
        limit = 100
        content = "BEGIN" + "x" * 1_000 + "END"

        truncated = ToolResultTruncator(
            ContextLimits(500, limit)
        ).truncate_message(tool_message(content))

        assert truncated.content is not None
        self.assertLessEqual(len(truncated.content), limit)
        self.assertIn(f"original_characters:{len(content)}", truncated.content)
        prefix, end = truncated.content.split("\n[omitted]\n", maxsplit=1)
        header, beginning = prefix.rsplit("\n", maxsplit=1)
        retained = int(
            re.search(r"retained_characters:(\d+)", header).group(1)  # type: ignore[union-attr]
        )
        self.assertEqual(len(beginning) + len(end), retained)
        self.assertEqual(len(beginning), (retained + 1) // 2)
        self.assertEqual(len(end), retained // 2)
        self.assertTrue(content.startswith(beginning))
        self.assertTrue(content.endswith(end))

    def test_message_at_limit_is_not_copied_or_truncated(self) -> None:
        message = tool_message("x" * 100)
        truncator = ToolResultTruncator(ContextLimits(500, 100))

        self.assertIs(truncator.truncate_message(message), message)

    def test_truncate_blocks_preserves_complete_interaction_and_call_fields(self) -> None:
        call = tool_call('{"path":"' + "x" * 500 + '"}')
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="读取文件。",
            tool_calls=(call,),
        )
        block = InteractionBlock((assistant, tool_message("r" * 1_000)))

        truncated_block = ToolResultTruncator(
            ContextLimits(500, 100)
        ).truncate_blocks((block,))[0]

        self.assertEqual(len(truncated_block.messages), 2)
        self.assertIs(truncated_block.messages[0], assistant)
        self.assertEqual(truncated_block.messages[0].tool_calls[0], call)
        self.assertEqual(
            truncated_block.messages[1].tool_call_id,
            call.tool_call_id,
        )
        self.assertLessEqual(len(truncated_block.messages[1].content or ""), 100)

    def test_rejects_invalid_dependencies_and_impossible_marker_limit(self) -> None:
        with self.assertRaises(TypeError):
            ToolResultTruncator(object())  # type: ignore[arg-type]

        truncator = ToolResultTruncator(ContextLimits(100, 10))
        with self.assertRaisesRegex(ValueError, "too small"):
            truncator.truncate_message(tool_message("long result" * 20))
        with self.assertRaises(TypeError):
            truncator.truncate_message(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            truncator.truncate_blocks((object(),))  # type: ignore[arg-type]


class ContextManagerToolResultTruncationTests(unittest.TestCase):
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

    def test_context_uses_truncated_copy_while_database_keeps_full_result(self) -> None:
        _, task = self.persistence.create_session_with_task(
            title="读取长文件。",
            original_prompt="读取长文件。",
            workspace="/workspace/demo",
        )
        self.persistence.start_task(task.id)
        step = self.persistence.create_agent_step(task.id, 0)
        _, tool_calls = self.persistence.save_tool_calls_action(
            task.id,
            step.id,
            ToolCallsAction(
                tool_calls=(
                    ToolCallRequest(
                        tool_call_id="provider-call",
                        tool_name="read_file",
                        arguments={"path": "large.txt"},
                    ),
                )
            ),
        )
        full_result = "BEGIN" + "x" * 1_000 + "END"
        self.persistence.start_tool_call(tool_calls[0].id)
        self.persistence.save_tool_result(
            tool_calls[0].id,
            ToolResult(
                tool_call_id="provider-call",
                tool_name="read_file",
                status=ToolResultStatus.COMPLETED,
                content=full_result,
            ),
        )

        context = ContextManager(
            self.persistence,
            ContextLimits(10_000, 200),
        ).build(task.id)
        stored_messages = self.persistence.load_messages(task.id)

        self.assertEqual(stored_messages[-1].content, full_result)
        self.assertEqual(len(stored_messages[-1].content), len(full_result))
        context_result = context.messages[-1]
        self.assertEqual(context_result.role, LLMMessageRole.TOOL)
        self.assertLessEqual(len(context_result.content or ""), 200)
        self.assertIn("[Tool Result truncated]", context_result.content or "")
        self.assertEqual(context_result.tool_call_id, "provider-call")


if __name__ == "__main__":
    unittest.main()
