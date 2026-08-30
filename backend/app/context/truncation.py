"""发送给模型前对 Tool Result 应用固定字符上限。"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from app.context.contracts import ContextLimits, InteractionBlock
from app.llm.contracts import LLMMessage, LLMMessageRole


_DETAILED_PREFIX_TEMPLATE = (
    "[Tool Result truncated]\n"
    "original_characters: {original_characters}\n"
    "retained_characters: {retained_characters}\n"
    "showing: beginning and end\n\n"
    "--- beginning ---\n"
)
_DETAILED_MIDDLE = "\n\n--- omitted ---\n\n--- end ---\n"
_COMPACT_PREFIX_TEMPLATE = (
    "[Tool Result truncated]\n"
    "original_characters:{original_characters}\n"
    "retained_characters:{retained_characters}\n"
)
_COMPACT_MIDDLE = "\n[omitted]\n"


class ToolResultTruncator:
    """只截断模型上下文中的 TOOL Message 内容。"""

    def __init__(self, limits: ContextLimits) -> None:
        if not isinstance(limits, ContextLimits):
            raise TypeError("limits must be a ContextLimits")
        self._max_characters = limits.max_tool_result_characters

    def truncate_blocks(
        self,
        blocks: tuple[InteractionBlock, ...],
    ) -> tuple[InteractionBlock, ...]:
        """返回保持 Block 完整性的上下文副本，不修改输入对象。"""

        normalized_blocks = tuple(blocks)
        if any(
            not isinstance(block, InteractionBlock)
            for block in normalized_blocks
        ):
            raise TypeError("blocks must contain only InteractionBlock values")

        return tuple(self._truncate_block(block) for block in normalized_blocks)

    def truncate_message(self, message: LLMMessage) -> LLMMessage:
        """超过单条上限时，保留 Tool Result 的开头、结尾和长度信息。"""

        if not isinstance(message, LLMMessage):
            raise TypeError("message must be an LLMMessage")
        if message.role is not LLMMessageRole.TOOL:
            return message

        content = message.content
        assert content is not None
        if len(content) <= self._max_characters:
            return message

        truncated_content = _truncate_tool_result(
            content,
            self._max_characters,
        )
        return replace(message, content=truncated_content)

    def _truncate_block(self, block: InteractionBlock) -> InteractionBlock:
        truncated_messages = tuple(
            self.truncate_message(message) for message in block.messages
        )
        if all(
            truncated is original
            for truncated, original in zip(truncated_messages, block.messages)
        ):
            return block
        return InteractionBlock(messages=truncated_messages)


def _truncate_tool_result(content: str, max_characters: int) -> str:
    renderers: tuple[Callable[[str, int], str], ...] = (
        _render_detailed_result,
        _render_compact_result,
    )
    for renderer in renderers:
        rendered = _fit_truncated_result(content, max_characters, renderer)
        if rendered is not None:
            return rendered
    raise ValueError(
        "max_tool_result_characters is too small to retain the required "
        "truncation marker plus the beginning and end"
    )


def _fit_truncated_result(
    content: str,
    max_characters: int,
    renderer: Callable[[str, int], str],
) -> str | None:
    retained_characters = max_characters
    while retained_characters >= 2:
        rendered = renderer(content, retained_characters)
        overflow = len(rendered) - max_characters
        if overflow <= 0:
            return rendered
        retained_characters -= overflow
    return None


def _render_detailed_result(content: str, retained_characters: int) -> str:
    beginning, end = _split_beginning_and_end(content, retained_characters)
    prefix = _DETAILED_PREFIX_TEMPLATE.format(
        original_characters=len(content),
        retained_characters=retained_characters,
    )
    return f"{prefix}{beginning}{_DETAILED_MIDDLE}{end}"


def _render_compact_result(content: str, retained_characters: int) -> str:
    beginning, end = _split_beginning_and_end(content, retained_characters)
    prefix = _COMPACT_PREFIX_TEMPLATE.format(
        original_characters=len(content),
        retained_characters=retained_characters,
    )
    return f"{prefix}{beginning}{_COMPACT_MIDDLE}{end}"


def _split_beginning_and_end(
    content: str,
    retained_characters: int,
) -> tuple[str, str]:
    beginning_characters = (retained_characters + 1) // 2
    end_characters = retained_characters // 2
    return content[:beginning_characters], content[-end_characters:]
