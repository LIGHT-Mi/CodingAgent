"""DeepSeek Chat Completions 非流式响应标准化。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.contracts import (
    LLMUsage,
    NormalizedLLMResponse,
    NormalizedToolCall,
)


class DeepSeekResponseAdapter:
    """把字典或属性对象形式的 DeepSeek 响应转换为统一响应契约。"""

    provider_name = "deepseek"

    def normalize(self, raw_response: Any) -> NormalizedLLMResponse:
        """标准化非流式 Chat Completion，不在此处解析 arguments JSON。"""

        errors: list[str] = []
        response_id = _optional_string(
            _get_field(raw_response, "id"),
            "response.id",
            errors,
        )
        model = _optional_string(
            _get_field(raw_response, "model"),
            "response.model",
            errors,
        )
        object_type = _optional_string(
            _get_field(raw_response, "object"),
            "response.object",
            errors,
        )
        if object_type is not None and object_type != "chat.completion":
            errors.append(
                f"response.object must be 'chat.completion', got {object_type!r}"
            )

        choices = _as_list(
            _get_field(raw_response, "choices"),
            "response.choices",
            errors,
        )
        choice = choices[0] if choices else None
        if len(choices) > 1:
            errors.append("response contains multiple choices; only the first is used")

        finish_reason = _optional_string(
            _get_field(choice, "finish_reason"),
            "choice.finish_reason",
            errors,
        )
        choice_index = _optional_integer(
            _get_field(choice, "index"),
            "choice.index",
            errors,
        )

        message = _get_field(choice, "message")
        if choice is not None and message is None:
            errors.append("choice.message is missing")
        content = _optional_string(
            _get_field(message, "content"),
            "choice.message.content",
            errors,
        )

        raw_tool_calls = _get_field(message, "tool_calls")
        if raw_tool_calls is None:
            tool_calls: tuple[NormalizedToolCall, ...] = ()
        else:
            calls = _as_list(
                raw_tool_calls,
                "choice.message.tool_calls",
                errors,
            )
            tool_calls = tuple(
                self._normalize_tool_call(raw_call, index, errors)
                for index, raw_call in enumerate(calls)
            )

        usage = self._normalize_usage(_get_field(raw_response, "usage"), errors)
        metadata: dict[str, Any] = {
            "choice_count": len(choices),
        }
        created = _get_field(raw_response, "created")
        if created is not None:
            metadata["created"] = created
        system_fingerprint = _get_field(raw_response, "system_fingerprint")
        if system_fingerprint is not None:
            metadata["system_fingerprint"] = system_fingerprint
        if object_type is not None:
            metadata["object"] = object_type
        if choice_index is not None:
            metadata["choice_index"] = choice_index
        if errors:
            metadata["normalization_errors"] = list(errors)

        return NormalizedLLMResponse(
            provider=self.provider_name,
            response_id=response_id,
            model=model,
            finish_reason=finish_reason,
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_tool_call(
        raw_call: Any,
        call_index: int,
        errors: list[str],
    ) -> NormalizedToolCall:
        prefix = f"choice.message.tool_calls[{call_index}]"
        function = _get_field(raw_call, "function")
        if function is None:
            errors.append(f"{prefix}.function is missing")
        return NormalizedToolCall(
            call_index=call_index,
            tool_call_id=_optional_string(
                _get_field(raw_call, "id"),
                f"{prefix}.id",
                errors,
            ),
            tool_type=_optional_string(
                _get_field(raw_call, "type"),
                f"{prefix}.type",
                errors,
            ),
            tool_name=_optional_string(
                _get_field(function, "name"),
                f"{prefix}.function.name",
                errors,
            ),
            arguments_json=_optional_string(
                _get_field(function, "arguments"),
                f"{prefix}.function.arguments",
                errors,
            ),
        )

    @staticmethod
    def _normalize_usage(raw_usage: Any, errors: list[str]) -> LLMUsage | None:
        if raw_usage is None:
            return None

        input_tokens = _optional_integer(
            _get_field(raw_usage, "prompt_tokens"),
            "usage.prompt_tokens",
            errors,
        )
        output_tokens = _optional_integer(
            _get_field(raw_usage, "completion_tokens"),
            "usage.completion_tokens",
            errors,
        )
        total_tokens = _optional_integer(
            _get_field(raw_usage, "total_tokens"),
            "usage.total_tokens",
            errors,
        )
        if None in {input_tokens, output_tokens, total_tokens}:
            errors.append("usage is incomplete")
            return None

        try:
            return LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"usage is invalid: {exc}")
            return None


def _get_field(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _as_list(value: Any, field_name: str, errors: list[str]) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        errors.append(f"{field_name} is missing")
    else:
        errors.append(f"{field_name} must be a list")
    return []


def _optional_string(
    value: Any,
    field_name: str,
    errors: list[str],
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(f"{field_name} must be a string or null")
        return None
    return value


def _optional_integer(
    value: Any,
    field_name: str,
    errors: list[str],
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{field_name} must be an integer or null")
        return None
    if value < 0:
        errors.append(f"{field_name} must be greater than or equal to zero")
        return None
    return value
