"""从 Task 原始需求和完整持久化历史构造模型上下文。"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TypeAlias

from app.agent.contracts import (
    MessageRole,
    MessageType,
    RuntimeEvent,
    RuntimeEventType,
    TaskStatus,
)
from app.context.contracts import ContextLimits, InteractionBlock
from app.context.counting import ContextCharacterCounter
from app.context.truncation import ToolResultTruncator
from app.db.models.message import Message
from app.db.models.tool_call import ToolCall
from app.db.persistence import PersistenceService
from app.llm.contracts import (
    LLMContext,
    LLMMessage,
    LLMMessageRole,
    LLMToolCall,
)


CODING_AGENT_SYSTEM_PROMPT = (
    "你是一个编程助手。当前可以使用 list_files、read_file、search_files、"
    "create_file、write_file、edit_file 和 run_command。涉及 Workspace 文件信息"
    "时，必须通过工具取得真实目录和文件内容，不得虚构读取结果。修改已有文件前"
    "必须先用 "
    "read_file 读取真实内容；精确修改优先使用 edit_file，新文件使用 "
    "create_file；需要整体覆盖已有文件时，必须显式使用 write_file。文件修改工具"
    "成功后必须再次使用 read_file 验证实际内容，并使用 run_command 运行相关测试、"
    "构建或项目验证。命令的非零 exit code、timeout 和拒绝结果都是需要分析的 "
    "Observation，不等于任务失败；工具返回错误时，根据 Observation 修正后续调用。"
    "当前仍不能删除文件。信息充分且修改已经验证后再返回最终答案，不得声称未验证"
    "的修改已经成功，也不得声称未实际执行的命令已经成功。"
)


class ContextManagerError(RuntimeError):
    """ContextManager 无法为模型构造上下文。"""


class ContextTaskNotFoundError(ContextManagerError):
    """构造上下文所需的 Task 不存在。"""


class ContextHistoryError(ContextManagerError):
    """持久化消息与 ToolCall 无法恢复为合法模型历史。"""


ContextBuildResult: TypeAlias = LLMContext | RuntimeEvent


class ContextManager:
    """校验完整历史并构造受字符限制的模型上下文。"""

    def __init__(
        self,
        persistence: PersistenceService,
        limits: ContextLimits,
    ) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        if not isinstance(limits, ContextLimits):
            raise TypeError("limits must be a ContextLimits")
        self._persistence = persistence
        self._max_context_characters = limits.max_context_characters
        self._tool_result_truncator = ToolResultTruncator(limits)
        self._character_counter = ContextCharacterCounter()

    def build(self, task_id: str) -> ContextBuildResult:
        """恢复完整历史并按最旧 Block 优先应用字符预算。"""

        task = self._persistence.get_task(task_id)
        if task is None:
            raise ContextTaskNotFoundError(f"Task {task_id} was not found")

        stored_messages = self._persistence.load_messages(task_id)
        stored_tool_calls = self._persistence.load_tool_calls(task_id)
        _validate_final_message_history(task.status, stored_messages)
        blocks = _restore_interaction_blocks(stored_messages, stored_tool_calls)
        blocks = self._tool_result_truncator.truncate_blocks(blocks)
        base_messages = (
            LLMMessage(
                role=LLMMessageRole.SYSTEM,
                content=CODING_AGENT_SYSTEM_PROMPT,
            ),
            LLMMessage(
                role=LLMMessageRole.USER,
                content=task.original_prompt,
            ),
        )
        return _apply_sliding_window(
            base_messages,
            blocks,
            self._character_counter,
            self._max_context_characters,
        )


def _apply_sliding_window(
    base_messages: tuple[LLMMessage, LLMMessage],
    blocks: tuple[InteractionBlock, ...],
    character_counter: ContextCharacterCounter,
    max_context_characters: int,
) -> ContextBuildResult:
    remaining_blocks = blocks

    while True:
        candidate = _assemble_context(base_messages, remaining_blocks)
        candidate_characters = character_counter.count(candidate)
        if candidate_characters <= max_context_characters:
            return candidate
        if remaining_blocks:
            remaining_blocks = remaining_blocks[1:]
            continue

        return _build_context_overflow_event(
            max_context_characters=max_context_characters,
            required_characters=candidate_characters,
        )


def _assemble_context(
    base_messages: tuple[LLMMessage, LLMMessage],
    blocks: tuple[InteractionBlock, ...],
) -> LLMContext:
    history = tuple(
        message
        for block in blocks
        for message in block.messages
    )
    return LLMContext(messages=(*base_messages, *history))


def _build_context_overflow_event(
    *,
    max_context_characters: int,
    required_characters: int,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.CONTEXT_OVERFLOW,
        source="context_manager",
        message=(
            "System Prompt and Original Task exceed the context character budget"
        ),
        details={
            "max_context_characters": max_context_characters,
            "required_characters": required_characters,
            "history_block_count": 0,
        },
    )


def _restore_interaction_blocks(
    messages: list[Message],
    tool_calls: list[ToolCall],
) -> tuple[InteractionBlock, ...]:
    ordered_messages, messages_by_id = _order_and_index_messages(messages)
    calls_by_message_id: dict[str, list[ToolCall]] = defaultdict(list)
    calls_by_internal_id: dict[str, ToolCall] = {}

    for tool_call in tool_calls:
        if tool_call.id in calls_by_internal_id:
            raise ContextHistoryError(
                f"duplicate internal ToolCall id in history: {tool_call.id}"
            )
        source_message = messages_by_id.get(tool_call.assistant_message_id)
        if source_message is None:
            raise ContextHistoryError(
                "ToolCall references an Assistant Message outside task history: "
                f"{tool_call.id}"
            )
        if source_message.role != MessageRole.ASSISTANT.value:
            raise ContextHistoryError(
                "ToolCall source Message is not an Assistant Message: "
                f"{tool_call.id}"
            )
        if tool_call.step_id != source_message.step_id:
            raise ContextHistoryError(
                "ToolCall and its source Assistant Message belong to different "
                f"AgentSteps: {tool_call.id}"
            )
        calls_by_internal_id[tool_call.id] = tool_call
        calls_by_message_id[tool_call.assistant_message_id].append(tool_call)

    result_messages_by_call_id = _index_tool_result_messages(
        ordered_messages,
        calls_by_internal_id,
    )
    blocks: list[InteractionBlock] = []
    block_message_ids: list[tuple[str, ...]] = []
    assigned_message_ids: set[str] = set()

    for assistant_message in ordered_messages:
        if assistant_message.role != MessageRole.ASSISTANT.value:
            continue

        stored_calls = calls_by_message_id.get(assistant_message.id, [])
        ordered_calls = _order_block_tool_calls(assistant_message, stored_calls)
        restored_assistant = _restore_assistant_message(
            assistant_message,
            ordered_calls,
        )
        source_ids = [assistant_message.id]
        block_messages = [restored_assistant]

        for tool_call in ordered_calls:
            result_message = result_messages_by_call_id.get(tool_call.id)
            if result_message is None:
                raise ContextHistoryError(
                    "ToolCalls have no corresponding TOOL_RESULT Message: "
                    f"{tool_call.id}"
                )
            if result_message.sequence <= assistant_message.sequence:
                raise ContextHistoryError(
                    "TOOL_RESULT Message appears before its Assistant ToolCall: "
                    f"{result_message.id}"
                )
            if result_message.step_id != tool_call.step_id:
                raise ContextHistoryError(
                    "TOOL_RESULT Message and ToolCall belong to different "
                    f"AgentSteps: {result_message.id}"
                )
            source_ids.append(result_message.id)
            block_messages.append(
                _restore_tool_result_message(
                    result_message,
                    calls_by_internal_id,
                )
            )

        _validate_tool_result_order(
            assistant_message,
            ordered_calls,
            result_messages_by_call_id,
        )
        for message_id in source_ids:
            if message_id in assigned_message_ids:
                raise ContextHistoryError(
                    f"Message was assigned to multiple InteractionBlocks: {message_id}"
                )
            assigned_message_ids.add(message_id)

        try:
            blocks.append(InteractionBlock(messages=tuple(block_messages)))
        except (TypeError, ValueError) as exc:
            raise ContextHistoryError(
                "persisted messages cannot form a complete InteractionBlock at "
                f"Assistant sequence {assistant_message.sequence}: {exc}"
            ) from exc
        block_message_ids.append(tuple(source_ids))

    unassigned_message_ids = set(messages_by_id) - assigned_message_ids
    if unassigned_message_ids:
        unassigned_ids = ", ".join(sorted(unassigned_message_ids))
        raise ContextHistoryError(
            "persisted Messages could not be assigned to an InteractionBlock: "
            f"{unassigned_ids}"
        )

    grouped_message_ids = tuple(
        message_id
        for block_ids in block_message_ids
        for message_id in block_ids
    )
    persisted_message_ids = tuple(message.id for message in ordered_messages)
    if grouped_message_ids != persisted_message_ids:
        raise ContextHistoryError(
            "persisted Message order cannot be represented as complete contiguous "
            "InteractionBlocks"
        )
    return tuple(blocks)


def _validate_final_message_history(
    task_status: str,
    messages: list[Message],
) -> None:
    if task_status != TaskStatus.RUNNING.value:
        return
    final_messages = [
        message
        for message in messages
        if message.role == MessageRole.ASSISTANT.value
        and message.message_type == MessageType.FINAL.value
    ]
    if final_messages:
        sequences = ", ".join(
            str(message.sequence) for message in final_messages
        )
        raise ContextHistoryError(
            "a RUNNING Task cannot contain FINAL Assistant Message history at "
            f"sequence: {sequences}"
        )


def _order_and_index_messages(
    messages: list[Message],
) -> tuple[tuple[Message, ...], dict[str, Message]]:
    messages_by_id: dict[str, Message] = {}
    messages_by_sequence: dict[int, Message] = {}
    for message in messages:
        if message.id in messages_by_id:
            raise ContextHistoryError(
                f"duplicate Message id in history: {message.id}"
            )
        if message.sequence in messages_by_sequence:
            raise ContextHistoryError(
                f"duplicate Message sequence in history: {message.sequence}"
            )
        messages_by_id[message.id] = message
        messages_by_sequence[message.sequence] = message

    ordered_messages = tuple(
        sorted(messages, key=lambda message: message.sequence)
    )
    return ordered_messages, messages_by_id


def _index_tool_result_messages(
    messages: tuple[Message, ...],
    calls_by_internal_id: dict[str, ToolCall],
) -> dict[str, Message]:
    results_by_call_id: dict[str, Message] = {}
    for message in messages:
        if message.role == MessageRole.ASSISTANT.value:
            if message.tool_call_id is not None:
                raise ContextHistoryError(
                    "Assistant Message cannot reference a ToolCall result: "
                    f"{message.id}"
                )
            continue
        if message.role != MessageRole.TOOL.value:
            raise ContextHistoryError(
                f"unsupported persisted Message role: {message.role}"
            )
        if message.message_type != MessageType.TOOL_RESULT.value:
            raise ContextHistoryError(
                f"TOOL Message has unsupported message_type: {message.message_type}"
            )
        if message.tool_call_id is None:
            raise ContextHistoryError(
                f"TOOL_RESULT Message has no internal ToolCall id: {message.id}"
            )
        if message.tool_call_id not in calls_by_internal_id:
            raise ContextHistoryError(
                "TOOL_RESULT Message references an unknown internal ToolCall id: "
                f"{message.tool_call_id}"
            )
        if message.tool_call_id in results_by_call_id:
            raise ContextHistoryError(
                "multiple TOOL_RESULT Messages reference the same ToolCall: "
                f"{message.tool_call_id}"
            )
        results_by_call_id[message.tool_call_id] = message
    return results_by_call_id


def _order_block_tool_calls(
    assistant_message: Message,
    tool_calls: list[ToolCall],
) -> tuple[ToolCall, ...]:
    calls_by_index: dict[int, ToolCall] = {}
    for tool_call in tool_calls:
        if tool_call.call_index in calls_by_index:
            raise ContextHistoryError(
                "duplicate call_index in one Assistant ToolCall block: "
                f"{tool_call.call_index}"
            )
        calls_by_index[tool_call.call_index] = tool_call
    return tuple(sorted(tool_calls, key=lambda call: call.call_index))


def _validate_tool_result_order(
    assistant_message: Message,
    ordered_calls: tuple[ToolCall, ...],
    result_messages_by_call_id: dict[str, Message],
) -> None:
    if len(ordered_calls) < 2:
        return
    result_sequences = tuple(
        result_messages_by_call_id[tool_call.id].sequence
        for tool_call in ordered_calls
    )
    if result_sequences != tuple(sorted(result_sequences)):
        raise ContextHistoryError(
            "TOOL_RESULT Message order does not match ToolCall call_index order "
            f"for Assistant Message {assistant_message.id}"
        )


def _restore_assistant_message(
    message: Message,
    tool_calls: tuple[ToolCall, ...],
) -> LLMMessage:
    if message.message_type not in {
        MessageType.TEXT.value,
        MessageType.FINAL.value,
    }:
        raise ContextHistoryError(
            "Assistant Message has unsupported message_type: "
            f"{message.message_type}"
        )
    if message.message_type == MessageType.FINAL.value and tool_calls:
        raise ContextHistoryError(
            "a FINAL Assistant Message cannot contain ToolCalls"
        )

    restored_calls = tuple(
        _restore_tool_call(tool_call)
        for tool_call in tool_calls
    )
    try:
        return LLMMessage(
            role=LLMMessageRole.ASSISTANT,
            content=message.content,
            tool_calls=restored_calls,
        )
    except (TypeError, ValueError) as exc:
        raise ContextHistoryError(
            f"invalid Assistant Message history at sequence {message.sequence}: {exc}"
        ) from exc


def _restore_tool_call(tool_call: ToolCall) -> LLMToolCall:
    if not isinstance(tool_call.arguments, dict):
        raise ContextHistoryError(
            f"ToolCall arguments are not an object: {tool_call.id}"
        )
    try:
        arguments_json = json.dumps(
            tool_call.arguments,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return LLMToolCall(
            tool_call_id=tool_call.provider_call_id,
            tool_name=tool_call.tool_name,
            arguments_json=arguments_json,
            call_index=tool_call.call_index,
        )
    except (TypeError, ValueError) as exc:
        raise ContextHistoryError(
            f"invalid ToolCall history {tool_call.id}: {exc}"
        ) from exc


def _restore_tool_result_message(
    message: Message,
    calls_by_internal_id: dict[str, ToolCall],
) -> LLMMessage:
    if message.message_type != MessageType.TOOL_RESULT.value:
        raise ContextHistoryError(
            f"TOOL Message has unsupported message_type: {message.message_type}"
        )
    if message.tool_call_id is None:
        raise ContextHistoryError(
            f"TOOL_RESULT Message has no internal ToolCall id: {message.id}"
        )
    tool_call = calls_by_internal_id.get(message.tool_call_id)
    if tool_call is None:
        raise ContextHistoryError(
            "TOOL_RESULT Message references an unknown internal ToolCall id: "
            f"{message.tool_call_id}"
        )
    try:
        return LLMMessage(
            role=LLMMessageRole.TOOL,
            content=message.content,
            tool_call_id=tool_call.provider_call_id,
        )
    except (TypeError, ValueError) as exc:
        raise ContextHistoryError(
            f"invalid TOOL_RESULT history at sequence {message.sequence}: {exc}"
        ) from exc
