"""将模型上下文和配置组装为供应商无关请求。"""

from __future__ import annotations

from collections.abc import Iterable

from app.llm.contracts import (
    LLMContext,
    LLMRequest,
    LLMToolChoice,
    LLMToolSchema,
    ModelConfig,
)


class LLMRequestBuilder:
    """集中构造模型请求，避免其他模块直接拼接请求字段。"""

    def build(
        self,
        context: LLMContext,
        model_config: ModelConfig,
        tool_schemas: Iterable[LLMToolSchema],
    ) -> LLMRequest:
        """使用 Context、模型配置和可用工具定义构造一次非流式请求。"""

        if not isinstance(context, LLMContext):
            raise TypeError("context must be an LLMContext")
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be a ModelConfig")

        schemas = tuple(tool_schemas)
        return LLMRequest(
            model=model_config.model,
            messages=context.messages,
            tool_schemas=schemas,
            tool_choice=(
                LLMToolChoice.AUTO if schemas else LLMToolChoice.NONE
            ),
            temperature=model_config.temperature,
            max_output_tokens=model_config.max_output_tokens,
            stream=False,
        )
