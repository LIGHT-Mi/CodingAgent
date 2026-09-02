import json
import unittest

from app.context import ContextCharacterCounter
from app.llm import LLMContext, LLMMessage, LLMMessageRole, LLMToolCall


def stable_size(value: dict[str, object]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


class ContextCharacterCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = ContextCharacterCounter()

    def test_counts_role_content_and_json_structure(self) -> None:
        message = LLMMessage(
            role=LLMMessageRole.SYSTEM,
            content="中文系统提示",
        )
        expected = stable_size(
            {
                "role": "system",
                "content": "中文系统提示",
            }
        )

        self.assertEqual(self.counter.count_message(message), expected)

    def test_counts_all_assistant_tool_call_fields_in_call_index_order(self) -> None:
        first_call = LLMToolCall(
            tool_call_id="call-list",
            tool_name="list_files",
            arguments_json='{"path":"源码"}',
            call_index=0,
        )
        second_call = LLMToolCall(
            tool_call_id="call-read",
            tool_name="read_file",
            arguments_json='{"path":"源码/main.py"}',
            call_index=1,
        )
        message = LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content=None,
            tool_calls=(second_call, first_call),
        )
        expected = stable_size(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "tool_call_id": "call-list",
                        "tool_name": "list_files",
                        "arguments_json": '{"path":"源码"}',
                    },
                    {
                        "tool_call_id": "call-read",
                        "tool_name": "read_file",
                        "arguments_json": '{"path":"源码/main.py"}',
                    },
                ],
            }
        )

        self.assertEqual(self.counter.count_message(message), expected)

    def test_counts_tool_result_identity_and_content(self) -> None:
        message = LLMMessage(
            role=LLMMessageRole.TOOL,
            content="测试失败：期望 2，实际 3",
            tool_call_id="call-test",
        )
        expected = stable_size(
            {
                "role": "tool",
                "content": "测试失败：期望 2，实际 3",
                "tool_call_id": "call-test",
            }
        )

        self.assertEqual(self.counter.count_message(message), expected)

    def test_context_count_is_sum_of_serialized_messages_only(self) -> None:
        call = LLMToolCall(
            tool_call_id="call-read",
            tool_name="read_file",
            arguments_json='{"path":"main.py"}',
            call_index=0,
        )
        context = LLMContext(
            messages=(
                LLMMessage(LLMMessageRole.SYSTEM, "系统提示"),
                LLMMessage(LLMMessageRole.USER, "读取 main.py"),
                LLMMessage(
                    role=LLMMessageRole.ASSISTANT,
                    content="开始读取。",
                    tool_calls=(call,),
                ),
                LLMMessage(
                    role=LLMMessageRole.TOOL,
                    content="print('ok')",
                    tool_call_id=call.tool_call_id,
                ),
            )
        )

        expected = sum(
            self.counter.count_message(message)
            for message in context.messages
        )
        self.assertEqual(self.counter.count(context), expected)
        self.assertEqual(self.counter.count(context), self.counter.count(context))

    def test_unicode_is_counted_as_python_characters_not_escaped_ascii(self) -> None:
        message = LLMMessage(LLMMessageRole.USER, "中文")
        unicode_size = self.counter.count_message(message)
        ascii_escaped_size = len(
            json.dumps(
                {"role": "user", "content": "中文"},
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )

        self.assertLess(unicode_size, ascii_escaped_size)

    def test_tool_call_identity_name_and_arguments_change_the_count(self) -> None:
        short_call = LLMToolCall("a", "read_file", "{}", 0)
        long_call = LLMToolCall(
            "provider-call-with-long-id",
            "search_files",
            '{"query":"a longer query","path":"src"}',
            0,
        )
        short_message = LLMMessage(
            LLMMessageRole.ASSISTANT,
            None,
            (short_call,),
        )
        long_message = LLMMessage(
            LLMMessageRole.ASSISTANT,
            None,
            (long_call,),
        )

        self.assertGreater(
            self.counter.count_message(long_message),
            self.counter.count_message(short_message),
        )

    def test_rejects_non_contract_inputs(self) -> None:
        with self.assertRaises(TypeError):
            self.counter.count(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.counter.count_message(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
