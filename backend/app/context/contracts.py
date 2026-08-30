"""上下文管理使用的供应商无关预算契约。"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.contracts import LLMMessage, LLMMessageRole


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """F2 构造模型上下文时使用的字符上限。

    字符预算基于 Python 字符串长度以及后续对模型消息结构进行的稳定字符估算，
    不表示精确 token 数，也不表示 UTF-8 编码后的字节数。
    """

    max_context_characters: int
    max_tool_result_characters: int

    def __post_init__(self) -> None:
        _require_positive_integer(
            self.max_context_characters,
            "max_context_characters",
        )
        _require_positive_integer(
            self.max_tool_result_characters,
            "max_tool_result_characters",
        )


@dataclass(frozen=True, slots=True)
class InteractionBlock:
    """一个可以整体保留或整体删除的 Assistant/Tool 历史单元。

    带 Tool Call 的 Assistant Message 必须位于首位，后面按 Tool Call 顺序包含
    每个调用恰好一条结果。没有 Tool Call 的普通 Assistant Message 单独构成一个
    Block。该结构不包含 Task 状态和持久化 MessageType；运行中 FINAL Message 的
    合法性由后续历史恢复阶段结合 Task 状态判断。
    """

    messages: tuple[LLMMessage, ...]

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if not messages:
            raise ValueError("an interaction block must contain at least one message")
        if any(not isinstance(message, LLMMessage) for message in messages):
            raise TypeError("messages must contain only LLMMessage values")

        assistant_message = messages[0]
        if assistant_message.role is not LLMMessageRole.ASSISTANT:
            raise ValueError("an interaction block must start with an assistant message")

        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            if len(messages) != 1:
                raise ValueError(
                    "a plain assistant interaction block cannot contain tool results"
                )
            object.__setattr__(self, "messages", messages)
            return

        tool_results = messages[1:]
        if len(tool_results) != len(tool_calls):
            raise ValueError(
                "an assistant tool call block must contain exactly one result "
                "for every tool call"
            )
        if any(
            result.role is not LLMMessageRole.TOOL
            for result in tool_results
        ):
            raise ValueError(
                "messages after an assistant tool call must all be tool results"
            )

        expected_call_ids = tuple(call.tool_call_id for call in tool_calls)
        actual_call_ids = tuple(result.tool_call_id for result in tool_results)
        if actual_call_ids != expected_call_ids:
            raise ValueError(
                "tool results must match every assistant tool call in order"
            )

        object.__setattr__(self, "messages", messages)


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
