"""使用固定 DeepSeek 响应验证完整的离线动作解析链路。"""

import unittest
from typing import Any

from app.agent.contracts import (
    AgentAction,
    FinalAction,
    InvalidAction,
    ToolCallsAction,
)
from app.llm.action_parser import AgentActionParser
from app.llm.deepseek_adapter import DeepSeekResponseAdapter


FIXED_FINAL_RESPONSE: dict[str, Any] = {
    "id": "response-final",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "deepseek-chat",
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
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    },
}

FIXED_SINGLE_TOOL_CALL_RESPONSE: dict[str, Any] = {
    "id": "response-single-tool",
    "object": "chat.completion",
    "created": 1_700_000_001,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "我先读取入口文件。",
                "tool_calls": [
                    {
                        "id": "call-read-main",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"main.py"}',
                        },
                    }
                ],
            },
        }
    ],
}

FIXED_MULTIPLE_TOOL_CALLS_RESPONSE: dict[str, Any] = {
    "id": "response-multiple-tools",
    "object": "chat.completion",
    "created": 1_700_000_002,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "我会按顺序检查目录和配置文件。",
                "tool_calls": [
                    {
                        "id": "call-list-source",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": '{"path":"src"}',
                        },
                    },
                    {
                        "id": "call-read-config",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"pyproject.toml"}',
                        },
                    },
                ],
            },
        }
    ],
}

FIXED_INVALID_ARGUMENTS_RESPONSE: dict[str, Any] = {
    "id": "response-invalid-arguments",
    "object": "chat.completion",
    "created": 1_700_000_003,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-invalid-json",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":',
                        },
                    }
                ],
            },
        }
    ],
}

FIXED_UNKNOWN_RESPONSE: dict[str, Any] = {
    "id": "response-unknown",
    "object": "unknown.response",
    "model": "deepseek-chat",
    "result": {
        "answer": "这个结构不是 DeepSeek Chat Completion。",
    },
}


class LLMResponsePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = DeepSeekResponseAdapter()
        self.parser = AgentActionParser()

    def parse_fixed_response(
        self,
        raw_response: dict[str, Any],
    ) -> AgentAction:
        normalized_response = self.adapter.normalize(raw_response)
        return self.parser.parse(normalized_response)

    def test_fixed_final_response_becomes_final_action(self) -> None:
        action = self.parse_fixed_response(FIXED_FINAL_RESPONSE)

        self.assertIsInstance(action, FinalAction)
        self.assertEqual(action.content, "任务已经完成。")

    def test_fixed_single_tool_call_becomes_tool_calls_action(self) -> None:
        action = self.parse_fixed_response(FIXED_SINGLE_TOOL_CALL_RESPONSE)

        self.assertIsInstance(action, ToolCallsAction)
        self.assertEqual(action.content, "我先读取入口文件。")
        self.assertEqual(len(action.tool_calls), 1)
        self.assertEqual(action.tool_calls[0].tool_call_id, "call-read-main")
        self.assertEqual(action.tool_calls[0].tool_name, "read_file")
        self.assertEqual(action.tool_calls[0].arguments, {"path": "main.py"})
        self.assertEqual(action.tool_calls[0].call_index, 0)

    def test_fixed_multiple_tool_calls_preserve_response_order(self) -> None:
        action = self.parse_fixed_response(FIXED_MULTIPLE_TOOL_CALLS_RESPONSE)

        self.assertIsInstance(action, ToolCallsAction)
        self.assertEqual(
            [tool_call.tool_call_id for tool_call in action.tool_calls],
            ["call-list-source", "call-read-config"],
        )
        self.assertEqual(
            [tool_call.tool_name for tool_call in action.tool_calls],
            ["list_files", "read_file"],
        )
        self.assertEqual(
            [tool_call.arguments for tool_call in action.tool_calls],
            [{"path": "src"}, {"path": "pyproject.toml"}],
        )
        self.assertEqual(
            [tool_call.call_index for tool_call in action.tool_calls],
            [0, 1],
        )

    def test_fixed_invalid_arguments_become_invalid_action(self) -> None:
        action = self.parse_fixed_response(FIXED_INVALID_ARGUMENTS_RESPONSE)

        self.assertIsInstance(action, InvalidAction)
        self.assertIn("arguments_json is not valid JSON", action.reason)

    def test_fixed_unknown_response_becomes_invalid_action(self) -> None:
        action = self.parse_fixed_response(FIXED_UNKNOWN_RESPONSE)

        self.assertIsInstance(action, InvalidAction)
        self.assertIn("response normalization failed", action.reason)
        self.assertIn("response.object", action.reason)
        self.assertIn("response.choices", action.reason)


if __name__ == "__main__":
    unittest.main()
