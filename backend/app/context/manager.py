"""从 Task 原始需求和完整持久化历史构造模型上下文。"""

from __future__ import annotations

import json
from collections import defaultdict

from app.agent.contracts import MessageRole, MessageType
from app.db.models.message import Message
from app.db.models.tool_call import ToolCall
from app.db.persistence import PersistenceService
from app.llm.contracts import (
    LLMContext,
    LLMMessage,
    LLMMessageRole,
    LLMToolCall,
)


READ_ONLY_SYSTEM_PROMPT = (
    "你是一个编程助手。当前可以使用 list_files、read_file 和 search_files "
    "三个只读工具。涉及 Workspace 文件信息时，必须通过工具取得真实目录和文件"
    "内容，不得虚构读取结果。信息充分后返回最终答案。当前不能修改、创建或删除"
    "文件，也不能执行命令。"
)


class ContextManagerError(RuntimeError):
    """ContextManager 无法为模型构造上下文。"""


class ContextTaskNotFoundError(ContextManagerError):
    """构造上下文所需的 Task 不存在。"""


class ContextHistoryError(ContextManagerError):
    """持久化消息与 ToolCall 无法恢复为合法模型历史。"""


class ContextManager:
    """构造 System、Original User Prompt 和完整 Assistant/Tool 历史。"""

    def __init__(self, persistence: PersistenceService) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        self._persistence = persistence

    def build(self, task_id: str) -> LLMContext:
        """按持久化顺序恢复当前 Task 的完整上下文。"""

        task = self._persistence.get_task(task_id)
        if task is None:
            raise ContextTaskNotFoundError(f"Task {task_id} was not found")

        stored_messages = self._persistence.load_messages(task_id)
        stored_tool_calls = self._persistence.load_tool_calls(task_id)
        history = _restore_history(stored_messages, stored_tool_calls)
        return LLMContext(
            messages=(
                LLMMessage(
                    role=LLMMessageRole.SYSTEM,
                    content=READ_ONLY_SYSTEM_PROMPT,
                ),
                LLMMessage(
                    role=LLMMessageRole.USER,
                    content=task.original_prompt,
                ),
                *history,
            )
        )


def _restore_history(
    messages: list[Message],
    tool_calls: list[ToolCall],
) -> tuple[LLMMessage, ...]:
    calls_by_message_id: dict[str, list[ToolCall]] = defaultdict(list)
    calls_by_internal_id: dict[str, ToolCall] = {}
    answered_call_ids: set[str] = set()
    message_ids = {message.id for message in messages}

    for tool_call in tool_calls:
        if tool_call.id in calls_by_internal_id:
            raise ContextHistoryError(
                f"duplicate internal ToolCall id in history: {tool_call.id}"
            )
        if tool_call.assistant_message_id not in message_ids:
            raise ContextHistoryError(
                "ToolCall references an Assistant Message outside task history: "
                f"{tool_call.id}"
            )
        calls_by_internal_id[tool_call.id] = tool_call
        calls_by_message_id[tool_call.assistant_message_id].append(tool_call)

    restored: list[LLMMessage] = []
    for message in messages:
        if message.role == MessageRole.ASSISTANT.value:
            restored.append(
                _restore_assistant_message(
                    message,
                    calls_by_message_id.pop(message.id, []),
                )
            )
            continue

        if message.role == MessageRole.TOOL.value:
            if message.tool_call_id in answered_call_ids:
                raise ContextHistoryError(
                    "multiple TOOL_RESULT Messages reference the same ToolCall: "
                    f"{message.tool_call_id}"
                )
            restored.append(
                _restore_tool_result_message(message, calls_by_internal_id)
            )
            assert message.tool_call_id is not None
            answered_call_ids.add(message.tool_call_id)
            continue

        raise ContextHistoryError(
            f"unsupported persisted Message role: {message.role}"
        )

    if calls_by_message_id:
        orphan_ids = ", ".join(sorted(calls_by_message_id))
        raise ContextHistoryError(
            f"ToolCalls could not be attached to Assistant Messages: {orphan_ids}"
        )
    unanswered_call_ids = set(calls_by_internal_id) - answered_call_ids
    if unanswered_call_ids:
        missing_ids = ", ".join(sorted(unanswered_call_ids))
        raise ContextHistoryError(
            "ToolCalls have no corresponding TOOL_RESULT Message: "
            f"{missing_ids}"
        )
    return tuple(restored)


def _restore_assistant_message(
    message: Message,
    tool_calls: list[ToolCall],
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
        for tool_call in sorted(tool_calls, key=lambda call: call.call_index)
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
            ensure_ascii=False,
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
