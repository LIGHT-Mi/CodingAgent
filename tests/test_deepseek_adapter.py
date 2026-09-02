import unittest
from types import SimpleNamespace

from app.llm.deepseek_adapter import DeepSeekResponseAdapter


class DeepSeekResponseAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = DeepSeekResponseAdapter()

    def test_normalize_official_final_response_shape(self) -> None:
        response = self.adapter.normalize(
            {
                "id": "response-1",
                "object": "chat.completion",
                "created": 123456,
                "model": "deepseek-chat",
                "system_fingerprint": "fingerprint-1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "任务已经完成。",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            }
        )

        self.assertEqual(response.provider, "deepseek")
        self.assertEqual(response.response_id, "response-1")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.content, "任务已经完成。")
        self.assertEqual(response.usage.total_tokens, 120)
        self.assertNotIn("normalization_errors", response.metadata)

    def test_normalize_tool_calls_and_preserve_raw_arguments(self) -> None:
        response = self.adapter.normalize(
            {
                "id": "response-2",
                "object": "chat.completion",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "我先查看文件。",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "index": 8,
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"main.py"}',
                                    },
                                },
                                {
                                    "id": "call-2",
                                    "type": "function",
                                    "index": 9,
                                    "function": {
                                        "name": "list_files",
                                        "arguments": "{invalid-json",
                                    },
                                },
                            ],
                        },
                    }
                ],
            }
        )

        self.assertEqual(
            [call.call_index for call in response.tool_calls],
            [0, 1],
        )
        self.assertEqual(response.tool_calls[0].tool_type, "function")
        self.assertEqual(response.tool_calls[0].tool_name, "read_file")
        self.assertEqual(response.tool_calls[1].arguments_json, "{invalid-json")

    def test_normalize_openai_compatible_attribute_objects(self) -> None:
        raw_response = SimpleNamespace(
            id="response-3",
            object="chat.completion",
            model="deepseek-chat",
            created=123456,
            system_fingerprint=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                type="function",
                                function=SimpleNamespace(
                                    name="run_command",
                                    arguments='{"command":"pytest"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
            ),
        )

        response = self.adapter.normalize(raw_response)

        self.assertEqual(response.tool_calls[0].tool_call_id, "call-1")
        self.assertEqual(response.tool_calls[0].tool_name, "run_command")
        self.assertNotIn("normalization_errors", response.metadata)

    def test_normalize_malformed_response_without_raising(self) -> None:
        response = self.adapter.normalize(
            {
                "id": 123,
                "object": "chat.completion.chunk",
                "model": "deepseek-chat",
                "choices": [],
                "usage": {
                    "prompt_tokens": "many",
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

        self.assertIsNone(response.response_id)
        self.assertIsNone(response.content)
        self.assertEqual(response.tool_calls, ())
        self.assertIsNone(response.usage)
        self.assertGreaterEqual(len(response.metadata["normalization_errors"]), 3)

    def test_preserve_incomplete_tool_call_for_action_parser(self) -> None:
        response = self.adapter.normalize(
            {
                "id": "response-4",
                "object": "chat.completion",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": None,
                                    "type": "custom",
                                    "function": {
                                        "name": None,
                                        "arguments": None,
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )

        tool_call = response.tool_calls[0]
        self.assertIsNone(tool_call.tool_call_id)
        self.assertEqual(tool_call.tool_type, "custom")
        self.assertIsNone(tool_call.tool_name)
        self.assertIsNone(tool_call.arguments_json)


if __name__ == "__main__":
    unittest.main()
