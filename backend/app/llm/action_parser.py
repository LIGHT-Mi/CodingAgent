"""标准化 LLM 响应到 AgentAction 的解析器。"""

from __future__ import annotations

import json
from typing import Any

from app.agent.contracts import (
    AgentAction,
    FinalAction,
    InvalidAction,
    ToolCallRequest,
    ToolCallsAction,
)
from app.llm.contracts import NormalizedLLMResponse, NormalizedToolCall


class AgentActionParser:
    """把完整的标准化响应解析为最终、工具调用或无效动作。"""

    def parse(self, response: NormalizedLLMResponse) -> AgentAction:
        if not isinstance(response, NormalizedLLMResponse):
            raise TypeError("response must be a NormalizedLLMResponse")

        normalization_errors = response.metadata.get("normalization_errors")
        if normalization_errors:
            return self._invalid(
                response,
                "response normalization failed: "
                f"{self._format_errors(normalization_errors)}",
            )

        if response.tool_calls:
            if response.finish_reason != "tool_calls":
                return self._invalid(
                    response,
                    "response contains tool calls but finish_reason is not "
                    "'tool_calls'",
                )
            return self._parse_tool_calls(response)

        if response.finish_reason == "tool_calls":
            return self._invalid(
                response,
                "finish_reason is 'tool_calls' but no tool calls were returned",
            )
        if response.finish_reason != "stop":
            return self._invalid(
                response,
                f"response did not finish normally: {response.finish_reason!r}",
            )
        if response.content is None or not response.content.strip():
            return self._invalid(
                response,
                "response contains neither tool calls nor final text",
            )
        return FinalAction(content=response.content)

    def _parse_tool_calls(self, response: NormalizedLLMResponse) -> AgentAction:
        parsed_calls: list[ToolCallRequest] = []
        for call in response.tool_calls:
            parsed_call_or_error = self._parse_tool_call(call)
            if isinstance(parsed_call_or_error, str):
                return self._invalid(
                    response,
                    f"tool call at index {call.call_index} is invalid: "
                    f"{parsed_call_or_error}",
                )
            parsed_calls.append(parsed_call_or_error)

        try:
            return ToolCallsAction(
                tool_calls=tuple(parsed_calls),
                content=response.content,
            )
        except (TypeError, ValueError) as exc:
            return self._invalid(response, f"tool calls are invalid: {exc}")

    @staticmethod
    def _parse_tool_call(call: NormalizedToolCall) -> ToolCallRequest | str:
        if call.tool_type != "function":
            return f"tool type must be 'function', got {call.tool_type!r}"
        if call.tool_call_id is None or not call.tool_call_id.strip():
            return "tool_call_id is missing or blank"
        if call.tool_name is None or not call.tool_name.strip():
            return "tool_name is missing or blank"
        if call.arguments_json is None or not call.arguments_json.strip():
            return "arguments_json is missing or blank"

        try:
            arguments = json.loads(
                call.arguments_json,
                parse_constant=_reject_non_standard_json_constant,
            )
        except json.JSONDecodeError as exc:
            return f"arguments_json is not valid JSON: {exc.msg}"
        except ValueError as exc:
            return f"arguments_json is not valid JSON: {exc}"
        if not isinstance(arguments, dict):
            return "arguments_json must decode to a JSON object"

        try:
            return ToolCallRequest(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                arguments=arguments,
                call_index=call.call_index,
            )
        except (TypeError, ValueError) as exc:
            return str(exc)

    @staticmethod
    def _invalid(response: NormalizedLLMResponse, reason: str) -> InvalidAction:
        return InvalidAction(reason=reason, raw_response=response)

    @staticmethod
    def _format_errors(errors: Any) -> str:
        if isinstance(errors, (list, tuple)):
            return "; ".join(str(error) for error in errors)
        return str(errors)


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")
