import unittest
from dataclasses import FrozenInstanceError

from app.llm.contracts import (
    LLMContext,
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
    LLMToolCall,
    LLMToolChoice,
    LLMToolSchema,
    LLMUsage,
    ModelConfig,
    NormalizedLLMResponse,
    NormalizedToolCall,
)


class LLMContextContractTests(unittest.TestCase):
    def test_construct_minimal_system_and_user_context(self) -> None:
        system_message = LLMMessage(
            LLMMessageRole.SYSTEM,
            "你是编程智能体。",
        )
        user_message = LLMMessage(
            LLMMessageRole.USER,
            "解释这段代码。",
        )

        context = LLMContext(messages=(system_message, user_message))

        self.assertEqual(context.messages, (system_message, user_message))

    def test_allow_complete_assistant_and_tool_history(self) -> None:
        system_message = LLMMessage(LLMMessageRole.SYSTEM, "系统提示词")
        user_message = LLMMessage(LLMMessageRole.USER, "用户任务")
        assistant_message = LLMMessage(
            LLMMessageRole.ASSISTANT,
            "先读取文件",
            tool_calls=(
                LLMToolCall(
                    tool_call_id="provider-call-1",
                    tool_name="read_file",
                    arguments_json='{"path":"main.py"}',
                    call_index=0,
                ),
            ),
        )
        tool_message = LLMMessage(
            LLMMessageRole.TOOL,
            "print('hello')",
            tool_call_id="provider-call-1",
        )

        context = LLMContext(
            messages=(system_message, user_message, assistant_message, tool_message)
        )

        self.assertEqual(len(context.messages), 4)
        self.assertEqual(context.messages[3].tool_call_id, "provider-call-1")

    def test_allow_historical_conversation_turns_before_current_task(self) -> None:
        messages = (
            LLMMessage(LLMMessageRole.SYSTEM, "系统提示词"),
            LLMMessage(LLMMessageRole.USER, "历史任务"),
            LLMMessage(LLMMessageRole.ASSISTANT, "历史最终回答"),
            LLMMessage(LLMMessageRole.USER, "当前任务"),
        )

        context = LLMContext(messages)

        self.assertEqual(context.messages, messages)

    def test_reject_invalid_context_shape_or_tool_history(self) -> None:
        system_message = LLMMessage(LLMMessageRole.SYSTEM, "系统提示词")
        user_message = LLMMessage(LLMMessageRole.USER, "用户任务")
        repeated_user = LLMMessage(LLMMessageRole.USER, "额外用户消息")
        orphan_tool = LLMMessage(
            LLMMessageRole.TOOL,
            "结果",
            tool_call_id="missing-call",
        )

        invalid_messages = (
            (),
            (system_message,),
            (user_message, system_message),
            (system_message, user_message, repeated_user),
            (system_message, user_message, orphan_tool),
        )
        for messages in invalid_messages:
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    LLMContext(messages=messages)

        with self.assertRaises(TypeError):
            LLMContext(messages=(system_message, object()))

    def test_reject_assistant_tool_call_without_tool_result(self) -> None:
        system_message = LLMMessage(LLMMessageRole.SYSTEM, "系统提示词")
        user_message = LLMMessage(LLMMessageRole.USER, "用户任务")
        assistant_message = LLMMessage(
            LLMMessageRole.ASSISTANT,
            None,
            tool_calls=(
                LLMToolCall(
                    tool_call_id="provider-call-1",
                    tool_name="read_file",
                    arguments_json='{"path":"main.py"}',
                    call_index=0,
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "exactly one TOOL result"):
            LLMContext(
                messages=(system_message, user_message, assistant_message)
            )


class LLMRequestContractTests(unittest.TestCase):
    def test_construct_request_with_complete_tool_call_history(self) -> None:
        tool_call = LLMToolCall(
            tool_call_id="call-1",
            tool_name="read_file",
            arguments_json='{"path":"src/main.py"}',
            call_index=0,
        )
        messages = (
            LLMMessage(
                role=LLMMessageRole.SYSTEM,
                content="你是编程智能体。",
            ),
            LLMMessage(
                role=LLMMessageRole.USER,
                content="读取并解释 main.py。",
            ),
            LLMMessage(
                role=LLMMessageRole.ASSISTANT,
                content=None,
                tool_calls=(tool_call,),
            ),
            LLMMessage(
                role=LLMMessageRole.TOOL,
                content="print('hello')",
                tool_call_id="call-1",
            ),
        )
        schema = LLMToolSchema(
            name="read_file",
            description="读取 Workspace 中的文本文件。",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        )
        request = LLMRequest(
            model="deepseek-chat",
            messages=messages,
            tool_schemas=(schema,),
            tool_choice=LLMToolChoice.AUTO,
            temperature=0,
            max_output_tokens=2048,
            metadata={"task_id": "task-1", "step_number": 0},
        )

        self.assertEqual(request.messages[2].tool_calls[0].tool_name, "read_file")
        self.assertEqual(request.messages[3].tool_call_id, "call-1")
        self.assertEqual(request.tool_schemas[0].parameters["type"], "object")
        self.assertEqual(request.metadata["task_id"], "task-1")

    def test_construct_text_only_request_without_tools(self) -> None:
        request = LLMRequest(
            model="deepseek-chat",
            messages=(
                LLMMessage(LLMMessageRole.SYSTEM, "你是编程智能体。"),
                LLMMessage(LLMMessageRole.USER, "说明这个任务。"),
            ),
            tool_choice=LLMToolChoice.NONE,
        )

        self.assertEqual(request.tool_schemas, ())
        self.assertEqual(request.tool_choice, LLMToolChoice.NONE)

    def test_message_role_rules_reject_inconsistent_fields(self) -> None:
        with self.assertRaises(ValueError):
            LLMMessage(LLMMessageRole.USER, None)
        with self.assertRaises(ValueError):
            LLMMessage(LLMMessageRole.ASSISTANT, None)
        with self.assertRaises(ValueError):
            LLMMessage(LLMMessageRole.TOOL, "result")
        with self.assertRaises(ValueError):
            LLMMessage(
                LLMMessageRole.SYSTEM,
                "system",
                tool_call_id="call-1",
            )

    def test_request_rejects_invalid_tool_and_generation_settings(self) -> None:
        message = LLMMessage(LLMMessageRole.USER, "任务")
        schema = LLMToolSchema(
            name="read_file",
            description="读取文件",
            parameters={"type": "object"},
        )

        with self.assertRaises(ValueError):
            LLMRequest(model="model", messages=())
        with self.assertRaises(ValueError):
            LLMRequest(
                model="model",
                messages=(message,),
                tool_choice=LLMToolChoice.REQUIRED,
            )
        with self.assertRaises(ValueError):
            LLMRequest(
                model="model",
                messages=(message,),
                tool_schemas=(schema, schema),
            )
        with self.assertRaises(ValueError):
            LLMRequest(model="model", messages=(message,), temperature=-0.1)
        with self.assertRaises(ValueError):
            LLMRequest(model="model", messages=(message,), max_output_tokens=0)
        with self.assertRaises(TypeError):
            LLMRequest(model="model", messages=(message,), stream="false")

    def test_request_contract_is_frozen(self) -> None:
        request = LLMRequest(
            model="deepseek-chat",
            messages=(LLMMessage(LLMMessageRole.USER, "任务"),),
        )

        with self.assertRaises(FrozenInstanceError):
            request.model = "other-model"  # type: ignore[misc]


class NormalizedLLMResponseContractTests(unittest.TestCase):
    def test_construct_normalized_final_response(self) -> None:
        response = NormalizedLLMResponse(
            provider="deepseek",
            response_id="response-1",
            model="deepseek-chat",
            finish_reason="stop",
            content="任务已经完成。",
            usage=LLMUsage(
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
            ),
            metadata={"created": 123456},
        )

        self.assertEqual(response.content, "任务已经完成。")
        self.assertEqual(response.tool_calls, ())
        self.assertEqual(response.usage.total_tokens, 120)

    def test_preserve_tool_calls_and_raw_arguments_for_later_parser(self) -> None:
        response = NormalizedLLMResponse(
            provider="deepseek",
            response_id="response-2",
            model="deepseek-chat",
            finish_reason="tool_calls",
            content="我先查看文件。",
            tool_calls=(
                NormalizedToolCall(
                    call_index=0,
                    tool_call_id="call-1",
                    tool_type="function",
                    tool_name="read_file",
                    arguments_json='{"path":"main.py"}',
                ),
                NormalizedToolCall(
                    call_index=1,
                    tool_call_id="call-2",
                    tool_type="function",
                    tool_name="list_files",
                    arguments_json="{invalid-json",
                ),
            ),
        )

        self.assertEqual(response.tool_calls[0].tool_name, "read_file")
        self.assertEqual(response.tool_calls[1].arguments_json, "{invalid-json")

    def test_allow_incomplete_or_empty_response_for_invalid_action_parser(self) -> None:
        response = NormalizedLLMResponse(
            provider="deepseek",
            response_id=None,
            model=None,
            finish_reason=None,
            content=None,
            tool_calls=(
                NormalizedToolCall(
                    call_index=0,
                    tool_call_id=None,
                    tool_type=None,
                    tool_name="",
                    arguments_json=None,
                ),
            ),
        )
        empty_response = NormalizedLLMResponse(
            provider="deepseek",
            response_id=None,
            model=None,
            finish_reason=None,
            content=None,
        )

        self.assertIsNone(response.tool_calls[0].tool_call_id)
        self.assertEqual(response.tool_calls[0].tool_name, "")
        self.assertIsNone(empty_response.content)
        self.assertEqual(empty_response.tool_calls, ())

    def test_usage_rejects_inconsistent_counts(self) -> None:
        with self.assertRaises(ValueError):
            LLMUsage(input_tokens=10, output_tokens=5, total_tokens=14)


if __name__ == "__main__":
    unittest.main()
