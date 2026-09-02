import unittest

from app.agent import RuntimeEvent, RuntimeEventType
from app.context import (
    ContextCharacterCounter,
    ConversationTurnBlock,
    InteractionBlock,
)
from app.context.manager import _apply_sliding_window
from app.llm import LLMContext, LLMMessage, LLMMessageRole, LLMToolCall


def base_messages() -> tuple[LLMMessage, LLMMessage]:
    return (
        LLMMessage(LLMMessageRole.SYSTEM, "固定系统提示"),
        LLMMessage(LLMMessageRole.USER, "固定原始任务"),
    )


def plain_block(content: str) -> InteractionBlock:
    return InteractionBlock(
        (LLMMessage(LLMMessageRole.ASSISTANT, content),)
    )


def conversation_block(label: str, size: int = 0) -> ConversationTurnBlock:
    return ConversationTurnBlock(
        (
            LLMMessage(LLMMessageRole.USER, f"{label}-user-" + "u" * size),
            LLMMessage(
                LLMMessageRole.ASSISTANT,
                f"{label}-assistant-" + "a" * size,
            ),
        )
    )


def tool_block(label: str, result_size: int = 80) -> InteractionBlock:
    calls = tuple(
        LLMToolCall(
            tool_call_id=f"{label}-call-{index}",
            tool_name="read_file",
            arguments_json=f'{{"path":"{label}-{index}.py"}}',
            call_index=index,
        )
        for index in range(2)
    )
    assistant = LLMMessage(
        role=LLMMessageRole.ASSISTANT,
        content=f"读取 {label}",
        tool_calls=calls,
    )
    results = tuple(
        LLMMessage(
            role=LLMMessageRole.TOOL,
            content=f"{label}-{index}-" + "x" * result_size,
            tool_call_id=call.tool_call_id,
        )
        for index, call in enumerate(calls)
    )
    return InteractionBlock((assistant, *results))


def assembled(
    base: tuple[LLMMessage, LLMMessage],
    interaction_blocks: tuple[InteractionBlock, ...],
    conversation_blocks: tuple[ConversationTurnBlock, ...] = (),
) -> LLMContext:
    return LLMContext(
        (
            base[0],
            *(
                message
                for block in conversation_blocks
                for message in block.messages
            ),
            base[1],
            *(
                message
                for block in interaction_blocks
                for message in block.messages
            ),
        )
    )


class ContextSlidingWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = ContextCharacterCounter()
        self.base = base_messages()

    def test_returns_all_blocks_when_candidate_is_within_budget(self) -> None:
        blocks = (plain_block("第一轮"), plain_block("第二轮"))
        full_context = assembled(self.base, blocks)
        exact_budget = self.counter.count(full_context)

        result = _apply_sliding_window(
            self.base,
            (),
            blocks,
            self.counter,
            exact_budget,
        )

        self.assertIsInstance(result, LLMContext)
        assert isinstance(result, LLMContext)
        self.assertEqual(result.messages, full_context.messages)
        self.assertEqual(self.counter.count(result), exact_budget)

    def test_removes_oldest_blocks_until_newest_suffix_fits(self) -> None:
        blocks = (
            plain_block("Block 0 看起来非常重要-" + "a" * 100),
            plain_block("Block 1 包含严重错误-" + "b" * 100),
            plain_block("Block 2 普通内容-" + "c" * 100),
            plain_block("Block 3 最新内容-" + "d" * 100),
        )
        newest_two = assembled(self.base, blocks[2:])
        budget = self.counter.count(newest_two)

        result = _apply_sliding_window(
            self.base,
            (),
            blocks,
            self.counter,
            budget,
        )

        self.assertIsInstance(result, LLMContext)
        assert isinstance(result, LLMContext)
        self.assertEqual(result.messages, newest_two.messages)
        self.assertNotIn(blocks[0].messages[0], result.messages)
        self.assertNotIn(blocks[1].messages[0], result.messages)
        self.assertEqual(
            result.messages[2:],
            (*blocks[2].messages, *blocks[3].messages),
        )

    def test_removes_a_multi_tool_interaction_as_one_indivisible_block(self) -> None:
        old_tool_block = tool_block("old", result_size=200)
        newest_block = plain_block("保留最新回复")
        budget = self.counter.count(assembled(self.base, (newest_block,)))

        result = _apply_sliding_window(
            self.base,
            (),
            (old_tool_block, newest_block),
            self.counter,
            budget,
        )

        self.assertIsInstance(result, LLMContext)
        assert isinstance(result, LLMContext)
        self.assertEqual(
            result.messages,
            (*self.base, *newest_block.messages),
        )
        removed_call_ids = {
            call.tool_call_id
            for call in old_tool_block.messages[0].tool_calls
        }
        self.assertTrue(
            all(
                message.tool_call_id not in removed_call_ids
                for message in result.messages
            )
        )

    def test_can_remove_all_history_while_always_retaining_base_messages(self) -> None:
        blocks = (tool_block("old"), plain_block("new"))
        base_context = assembled(self.base, ())
        base_budget = self.counter.count(base_context)

        result = _apply_sliding_window(
            self.base,
            (),
            blocks,
            self.counter,
            base_budget,
        )

        self.assertIsInstance(result, LLMContext)
        assert isinstance(result, LLMContext)
        self.assertEqual(result.messages, self.base)
        self.assertEqual(self.counter.count(result), base_budget)

    def test_returns_context_overflow_when_base_messages_exceed_budget(self) -> None:
        blocks = (plain_block("会先被删除"),)
        base_characters = self.counter.count(assembled(self.base, ()))

        result = _apply_sliding_window(
            self.base,
            (),
            blocks,
            self.counter,
            base_characters - 1,
        )

        self.assertIsInstance(result, RuntimeEvent)
        assert isinstance(result, RuntimeEvent)
        self.assertEqual(result.event_type, RuntimeEventType.CONTEXT_OVERFLOW)
        self.assertEqual(result.source, "context_manager")
        self.assertEqual(
            result.message,
            "System Prompt and Current Task exceed the context character budget",
        )
        self.assertEqual(result.details["required_characters"], base_characters)
        self.assertEqual(
            result.details["max_context_characters"],
            base_characters - 1,
        )
        self.assertEqual(
            result.details["conversation_turn_block_count"],
            0,
        )
        self.assertEqual(result.details["interaction_block_count"], 0)
        self.assertEqual(
            set(result.details),
            {
                "max_context_characters",
                "required_characters",
                "conversation_turn_block_count",
                "interaction_block_count",
            },
        )

    def test_removes_conversation_turns_before_current_interactions(self) -> None:
        conversation_blocks = (
            conversation_block("old-turn", size=120),
            conversation_block("new-turn", size=120),
        )
        interaction_blocks = (
            plain_block("current interaction remains"),
        )
        interactions_only = assembled(self.base, interaction_blocks)
        budget = self.counter.count(interactions_only)

        result = _apply_sliding_window(
            self.base,
            conversation_blocks,
            interaction_blocks,
            self.counter,
            budget,
        )

        self.assertIsInstance(result, LLMContext)
        assert isinstance(result, LLMContext)
        self.assertEqual(result.messages, interactions_only.messages)

    def test_removes_interactions_only_after_all_conversation_turns(self) -> None:
        conversation_blocks = (conversation_block("old", size=100),)
        interaction_blocks = (
            plain_block("old interaction-" + "x" * 100),
            plain_block("latest interaction"),
        )
        latest_interaction_only = assembled(
            self.base,
            (interaction_blocks[1],),
        )
        budget = self.counter.count(latest_interaction_only)

        result = _apply_sliding_window(
            self.base,
            conversation_blocks,
            interaction_blocks,
            self.counter,
            budget,
        )

        self.assertIsInstance(result, LLMContext)
        assert isinstance(result, LLMContext)
        self.assertEqual(result.messages, latest_interaction_only.messages)


if __name__ == "__main__":
    unittest.main()
