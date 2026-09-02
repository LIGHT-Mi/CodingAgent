import unittest

from app.agent.contracts import FinalAction, InvalidAction, ToolCallsAction
from app.llm.action_parser import AgentActionParser
from app.llm.contracts import NormalizedLLMResponse, NormalizedToolCall


def make_tool_call(
    *,
    call_index: int = 0,
    tool_call_id: str | None = "call-1",
    tool_type: str | None = "function",
    tool_name: str | None = "read_file",
    arguments_json: str | None = '{"path":"main.py"}',
) -> NormalizedToolCall:
    return NormalizedToolCall(
        call_index=call_index,
        tool_call_id=tool_call_id,
        tool_type=tool_type,
        tool_name=tool_name,
        arguments_json=arguments_json,
    )


def make_response(
    *,
    finish_reason: str | None,
    content: str | None = None,
    tool_calls: tuple[NormalizedToolCall, ...] = (),
    metadata: dict | None = None,
) -> NormalizedLLMResponse:
    return NormalizedLLMResponse(
        provider="deepseek",
        response_id="response-1",
        model="deepseek-chat",
        finish_reason=finish_reason,
        content=content,
        tool_calls=tool_calls,
        metadata=metadata or {},
    )


class AgentActionParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = AgentActionParser()

    def test_parse_final_action(self) -> None:
        response = make_response(
            finish_reason="stop",
            content="任务已经完成。",
        )

        action = self.parser.parse(response)

        self.assertIsInstance(action, FinalAction)
        self.assertEqual(action.content, "任务已经完成。")

    def test_parse_multiple_tool_calls_in_stable_order(self) -> None:
        response = make_response(
            finish_reason="tool_calls",
            content="先读取目录和文件。",
            tool_calls=(
                make_tool_call(),
                make_tool_call(
                    call_index=1,
                    tool_call_id="call-2",
                    tool_name="list_files",
                    arguments_json='{"path":"src"}',
                ),
            ),
        )

        action = self.parser.parse(response)

        self.assertIsInstance(action, ToolCallsAction)
        self.assertEqual(action.content, "先读取目录和文件。")
        self.assertEqual(
            [call.tool_name for call in action.tool_calls],
            ["read_file", "list_files"],
        )
        self.assertEqual(action.tool_calls[0].arguments, {"path": "main.py"})

    def test_unknown_tool_name_is_left_for_tool_router(self) -> None:
        response = make_response(
            finish_reason="tool_calls",
            tool_calls=(make_tool_call(tool_name="future_tool"),),
        )

        action = self.parser.parse(response)

        self.assertIsInstance(action, ToolCallsAction)
        self.assertEqual(action.tool_calls[0].tool_name, "future_tool")

    def test_any_invalid_tool_call_invalidates_whole_action(self) -> None:
        invalid_calls = (
            make_tool_call(tool_call_id=None),
            make_tool_call(tool_name=" "),
            make_tool_call(tool_type="custom"),
            make_tool_call(arguments_json=None),
            make_tool_call(arguments_json="{invalid-json"),
            make_tool_call(arguments_json='{"value":NaN}'),
            make_tool_call(arguments_json="[]"),
        )

        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                response = make_response(
                    finish_reason="tool_calls",
                    tool_calls=(
                        make_tool_call(),
                        NormalizedToolCall(
                            call_index=1,
                            tool_call_id=invalid_call.tool_call_id,
                            tool_type=invalid_call.tool_type,
                            tool_name=invalid_call.tool_name,
                            arguments_json=invalid_call.arguments_json,
                        ),
                    ),
                )

                action = self.parser.parse(response)

                self.assertIsInstance(action, InvalidAction)
                self.assertIs(action.raw_response, response)

    def test_duplicate_tool_call_identity_is_invalid(self) -> None:
        response = make_response(
            finish_reason="tool_calls",
            tool_calls=(
                make_tool_call(),
                make_tool_call(call_index=1),
            ),
        )

        action = self.parser.parse(response)

        self.assertIsInstance(action, InvalidAction)
        self.assertIn("tool_call_id must be unique", action.reason)

    def test_finish_reason_and_payload_must_agree(self) -> None:
        responses = (
            make_response(
                finish_reason="stop",
                tool_calls=(make_tool_call(),),
            ),
            make_response(
                finish_reason="tool_calls",
                content="没有实际工具调用",
            ),
            make_response(
                finish_reason="length",
                content="被截断的回答",
            ),
            make_response(
                finish_reason=None,
                content="没有结束原因",
            ),
        )

        for response in responses:
            with self.subTest(finish_reason=response.finish_reason):
                self.assertIsInstance(self.parser.parse(response), InvalidAction)

    def test_empty_final_text_is_invalid(self) -> None:
        for content in (None, "", "   "):
            with self.subTest(content=content):
                action = self.parser.parse(
                    make_response(finish_reason="stop", content=content)
                )
                self.assertIsInstance(action, InvalidAction)

    def test_normalization_error_is_invalid_before_action_parsing(self) -> None:
        response = make_response(
            finish_reason="stop",
            content="看起来有效的文本",
            metadata={"normalization_errors": ["response.id must be a string"]},
        )

        action = self.parser.parse(response)

        self.assertIsInstance(action, InvalidAction)
        self.assertIn("response normalization failed", action.reason)

    def test_reject_wrong_response_type(self) -> None:
        with self.assertRaises(TypeError):
            self.parser.parse({})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
