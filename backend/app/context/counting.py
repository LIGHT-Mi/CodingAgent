"""供应商无关、可重复的模型上下文字符计数。"""

from __future__ import annotations

import json
from typing import Any

from app.llm.contracts import LLMContext, LLMMessage


class ContextCharacterCounter:
    """按稳定 JSON 结构估算 LLMContext.messages 的字符规模。

    该计数不是 token 数或 UTF-8 字节数，也不包含 Tool Schema、HTTP 字段、
    模型输出预算以及 LLMRequest 的其他配置。
    """

    def count(self, context: LLMContext) -> int:
        """返回 Context 内全部消息序列化后的字符数之和。"""

        if not isinstance(context, LLMContext):
            raise TypeError("context must be an LLMContext")
        return sum(self.count_message(message) for message in context.messages)

    def count_message(self, message: LLMMessage) -> int:
        """返回一条消息的供应商无关稳定 JSON 字符数。"""

        if not isinstance(message, LLMMessage):
            raise TypeError("message must be an LLMMessage")
        serialized = json.dumps(
            _message_to_dict(message),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return len(serialized)


def _message_to_dict(message: LLMMessage) -> dict[str, Any]:
    serialized: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.tool_calls:
        serialized["tool_calls"] = [
            {
                "tool_call_id": tool_call.tool_call_id,
                "tool_name": tool_call.tool_name,
                "arguments_json": tool_call.arguments_json,
            }
            for tool_call in sorted(
                message.tool_calls,
                key=lambda item: item.call_index,
            )
        ]
    if message.tool_call_id is not None:
        serialized["tool_call_id"] = message.tool_call_id
    return serialized
