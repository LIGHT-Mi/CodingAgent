import unittest
from dataclasses import FrozenInstanceError

from app.llm.contracts import (
    LLMContext,
    LLMMessage,
    LLMMessageRole,
    LLMToolChoice,
    ModelConfig,
)
from app.llm.request_builder import LLMRequestBuilder


def make_context() -> LLMContext:
    return LLMContext(
        messages=(
            LLMMessage(LLMMessageRole.SYSTEM, "你是编程助手。"),
            LLMMessage(LLMMessageRole.USER, "解释依赖注入。"),
        )
    )


class LLMRequestBuilderTests(unittest.TestCase):
    def test_build_minimal_non_streaming_request_without_tools(self) -> None:
        context = make_context()
        model_config = ModelConfig(
            model="deepseek-v4-flash",
            temperature=0,
            max_output_tokens=1024,
        )

        request = LLMRequestBuilder().build(context, model_config, ())

        self.assertEqual(request.model, "deepseek-v4-flash")
        self.assertIs(request.messages, context.messages)
        self.assertEqual(request.tool_schemas, ())
        self.assertIs(request.tool_choice, LLMToolChoice.NONE)
        self.assertEqual(request.temperature, 0)
        self.assertEqual(request.max_output_tokens, 1024)
        self.assertFalse(request.stream)

    def test_model_config_rejects_invalid_generation_values(self) -> None:
        invalid_configs = (
            {"model": " "},
            {"model": "model", "temperature": -0.1},
            {"model": "model", "temperature": True},
            {"model": "model", "max_output_tokens": 0},
            {"model": "model", "max_output_tokens": True},
        )
        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    ModelConfig(**values)

    def test_model_config_is_frozen(self) -> None:
        config = ModelConfig(model="deepseek-v4-flash")

        with self.assertRaises(FrozenInstanceError):
            config.model = "other-model"  # type: ignore[misc]

    def test_reject_wrong_context_or_model_config(self) -> None:
        builder = LLMRequestBuilder()

        with self.assertRaises(TypeError):
            builder.build(object(), ModelConfig(model="model"), ())
        with self.assertRaises(TypeError):
            builder.build(make_context(), object(), ())


if __name__ == "__main__":
    unittest.main()
