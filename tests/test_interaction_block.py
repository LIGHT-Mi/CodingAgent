import unittest
from dataclasses import FrozenInstanceError

from app.context import InteractionBlock
from app.llm import LLMMessage, LLMMessageRole, LLMToolCall


def tool_call(call_id: str, call_index: int) -> LLMToolCall:
    return LLMToolCall(
        tool_call_id=call_id,
        tool_name="read_file",
        arguments_json='{"path":"main.py"}',
        call_index=call_index,
    )


def tool_result(call_id: str, content: str = "result") -> LLMMessage:
    return LLMMessage(
        role=LLMMessageRole.TOOL,
        content=content,
        tool_call_id=call_id,
    )


class InteractionBlockTests(unittest.TestCase):
    def test_plain_assistant_message_is_a_standalone_block(self) -> None:
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="继续分析。",
        )

        block = InteractionBlock(messages=[assistant])

        self.assertEqual(block.messages, (assistant,))

    def test_single_tool_call_and_result_form_one_block(self) -> None:
        call = tool_call("call-a", 0)
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content=None,
            tool_calls=(call,),
        )
        result = tool_result("call-a")

        block = InteractionBlock((assistant, result))

        self.assertEqual(block.messages, (assistant, result))

    def test_multiple_tool_calls_require_all_results_in_order(self) -> None:
        calls = tuple(tool_call(f"call-{index}", index) for index in range(3))
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="读取三个文件。",
            tool_calls=calls,
        )
        results = tuple(
            tool_result(call.tool_call_id, f"result-{call.call_index}")
            for call in calls
        )

        block = InteractionBlock((assistant, *results))

        self.assertEqual(block.messages[0], assistant)
        self.assertEqual(block.messages[1:], results)

    def test_cannot_keep_only_part_of_a_tool_interaction(self) -> None:
        calls = (tool_call("call-a", 0), tool_call("call-b", 1))
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content=None,
            tool_calls=calls,
        )

        for messages in (
            (assistant,),
            (assistant, tool_result("call-a")),
            (assistant, tool_result("call-a"), tool_result("call-b"), tool_result("call-c")),
        ):
            with self.subTest(messages=messages):
                with self.assertRaisesRegex(ValueError, "exactly one result"):
                    InteractionBlock(messages)

    def test_cannot_keep_tool_results_without_their_assistant(self) -> None:
        with self.assertRaisesRegex(ValueError, "start with an assistant"):
            InteractionBlock((tool_result("call-a"),))

    def test_rejects_wrong_missing_or_reordered_result_identity(self) -> None:
        calls = (tool_call("call-a", 0), tool_call("call-b", 1))
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content=None,
            tool_calls=calls,
        )

        invalid_results = (
            (tool_result("call-a"), tool_result("other")),
            (tool_result("call-b"), tool_result("call-a")),
        )
        for results in invalid_results:
            with self.subTest(results=results):
                with self.assertRaisesRegex(ValueError, "match every"):
                    InteractionBlock((assistant, *results))

    def test_rejects_non_tool_messages_after_tool_call_assistant(self) -> None:
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content=None,
            tool_calls=(tool_call("call-a", 0),),
        )
        another_assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="not a tool result",
        )

        with self.assertRaisesRegex(ValueError, "must all be tool results"):
            InteractionBlock((assistant, another_assistant))

    def test_plain_assistant_block_rejects_extra_messages(self) -> None:
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="普通回复",
        )

        with self.assertRaisesRegex(ValueError, "plain assistant"):
            InteractionBlock((assistant, tool_result("call-a")))

    def test_rejects_empty_and_non_message_values(self) -> None:
        with self.assertRaises(ValueError):
            InteractionBlock(())
        with self.assertRaises(TypeError):
            InteractionBlock((object(),))  # type: ignore[arg-type]

    def test_is_an_immutable_value_object(self) -> None:
        assistant = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content="普通回复",
        )
        block = InteractionBlock((assistant,))

        with self.assertRaises(FrozenInstanceError):
            block.messages = ()  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
