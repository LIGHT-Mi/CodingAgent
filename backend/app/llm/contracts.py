"""LLM 调用边界使用的纯 Python 数据契约。

请求对象描述模型调用需要的数据；标准化响应对象承接供应商适配后的数据，供后续
动作解析器识别最终回答、工具调用或无效响应。本模块不连接模型 API，也不依赖
数据库 ORM 和 AgentAction。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeAlias


Metadata: TypeAlias = Mapping[str, Any]
JSONSchema: TypeAlias = Mapping[str, Any]


class LLMMessageRole(str, Enum):
    """发送给模型的消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMToolChoice(str, Enum):
    """模型是否可以或必须选择工具。"""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    """历史 Assistant Message 中可重新发送给模型的合法 Tool Call。"""

    tool_call_id: str
    tool_name: str
    arguments_json: str
    call_index: int

    def __post_init__(self) -> None:
        _require_non_blank(self.tool_call_id, "tool_call_id")
        _require_non_blank(self.tool_name, "tool_name")
        _require_non_blank(self.arguments_json, "arguments_json")
        _require_non_negative_integer(self.call_index, "call_index")


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """模型上下文中的一条供应商无关消息。"""

    role: LLMMessageRole
    content: str | None
    tool_calls: tuple[LLMToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, LLMMessageRole):
            raise TypeError("role must be an LLMMessageRole")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("content must be a string or None")

        calls = tuple(self.tool_calls)
        if any(not isinstance(call, LLMToolCall) for call in calls):
            raise TypeError("tool_calls must contain only LLMToolCall values")
        object.__setattr__(self, "tool_calls", calls)

        if self.role in {LLMMessageRole.SYSTEM, LLMMessageRole.USER}:
            if self.content is None:
                raise ValueError("system and user messages must contain content")
            _require_non_blank(self.content, "content")
            if calls or self.tool_call_id is not None:
                raise ValueError(
                    "system and user messages cannot contain tool call fields"
                )
            return

        if self.role is LLMMessageRole.ASSISTANT:
            if self.tool_call_id is not None:
                raise ValueError("assistant messages cannot contain tool_call_id")
            if not calls:
                if self.content is None:
                    raise ValueError(
                        "an assistant message must contain content or tool_calls"
                    )
                _require_non_blank(self.content, "content")
            _validate_unique_tool_calls(calls)
            return

        if calls:
            raise ValueError("tool messages cannot contain tool_calls")
        if self.content is None:
            raise ValueError("tool messages must contain content")
        if self.tool_call_id is None:
            raise ValueError("tool messages must contain tool_call_id")
        _require_non_blank(self.tool_call_id, "tool_call_id")


@dataclass(frozen=True, slots=True)
class LLMContext:
    """上下文管理与模型调用之间传递的模型上下文。

    固定以 System Prompt 开头，随后可以包含历史 Session 的 USER/ASSISTANT
    Conversation Turn、当前 Task 的 USER Prompt，以及当前 Task 内完整的
    Assistant/Tool Interaction Blocks。过长 Tool Result 已在构造阶段保留首尾并
    截断；历史按完整 Block 应用滑动窗口。若不可删除的消息本身超出预算，
    ContextManager 返回 RuntimeEvent，而不会构造本对象。
    """

    messages: tuple[LLMMessage, ...]

    def __post_init__(self) -> None:
        messages = tuple(self.messages)
        if len(messages) < 2:
            raise ValueError("LLMContext must contain at least two messages")
        if any(not isinstance(message, LLMMessage) for message in messages):
            raise TypeError("messages must contain only LLMMessage values")
        if messages[0].role is not LLMMessageRole.SYSTEM:
            raise ValueError(
                "LLMContext must start with a SYSTEM message"
            )
        if any(
            message.role is LLMMessageRole.SYSTEM
            for message in messages[1:]
        ):
            raise ValueError(
                "LLMContext can contain only one leading SYSTEM message"
            )
        if messages[1].role is not LLMMessageRole.USER:
            raise ValueError(
                "the first message after SYSTEM must be a USER message"
            )
        for index, message in enumerate(messages[2:], start=2):
            if message.role is not LLMMessageRole.USER:
                continue
            previous = messages[index - 1]
            if (
                previous.role is not LLMMessageRole.ASSISTANT
                or previous.tool_calls
            ):
                raise ValueError(
                    "a later USER message must follow a plain ASSISTANT message"
                )
        _validate_tool_call_history(messages[1:])
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class LLMToolSchema:
    """可随 LLM 请求发送的单个函数工具定义。"""

    name: str
    description: str
    parameters: JSONSchema

    def __post_init__(self) -> None:
        _require_non_blank(self.name, "name")
        _require_non_blank(self.description, "description")
        object.__setattr__(
            self,
            "parameters",
            _copy_mapping(self.parameters, "parameters"),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """一次模型调用所需的供应商无关生成配置。"""

    model: str
    temperature: float | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.model, "model")
        _validate_temperature(self.temperature)
        _validate_max_output_tokens(self.max_output_tokens)


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """模型客户端接收的供应商无关请求。"""

    model: str
    messages: tuple[LLMMessage, ...]
    tool_schemas: tuple[LLMToolSchema, ...] = ()
    tool_choice: LLMToolChoice = LLMToolChoice.AUTO
    temperature: float | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_blank(self.model, "model")

        messages = tuple(self.messages)
        if not messages:
            raise ValueError("messages must contain at least one LLMMessage")
        if any(not isinstance(message, LLMMessage) for message in messages):
            raise TypeError("messages must contain only LLMMessage values")
        object.__setattr__(self, "messages", messages)

        schemas = tuple(self.tool_schemas)
        if any(not isinstance(schema, LLMToolSchema) for schema in schemas):
            raise TypeError("tool_schemas must contain only LLMToolSchema values")
        schema_names = [schema.name for schema in schemas]
        if len(schema_names) != len(set(schema_names)):
            raise ValueError("tool schema names must be unique within one request")
        object.__setattr__(self, "tool_schemas", schemas)

        if not isinstance(self.tool_choice, LLMToolChoice):
            raise TypeError("tool_choice must be an LLMToolChoice")
        if self.tool_choice is LLMToolChoice.REQUIRED and not schemas:
            raise ValueError("REQUIRED tool_choice needs at least one tool schema")

        _validate_temperature(self.temperature)
        _validate_max_output_tokens(self.max_output_tokens)
        if not isinstance(self.stream, bool):
            raise TypeError("stream must be a boolean")

        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedToolCall:
    """供应商 Tool Call 标准化后的解析候选。

    ID、名称和参数允许缺失或为空，因为标准化层必须保留供应商的异常输出，后续
    动作解析器再统一生成 InvalidAction。arguments_json 保留原始字符串，不在此处
    反序列化。
    """

    call_index: int
    tool_call_id: str | None
    tool_type: str | None
    tool_name: str | None
    arguments_json: str | None

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.call_index, "call_index")
        _require_optional_string(self.tool_call_id, "tool_call_id")
        _require_optional_string(self.tool_type, "tool_type")
        _require_optional_string(self.tool_name, "tool_name")
        _require_optional_string(self.arguments_json, "arguments_json")


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """供应商返回的模型用量统计。"""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.input_tokens, "input_tokens")
        _require_non_negative_integer(self.output_tokens, "output_tokens")
        _require_non_negative_integer(self.total_tokens, "total_tokens")
        if self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError(
                "total_tokens must be greater than or equal to input_tokens + "
                "output_tokens"
            )


@dataclass(frozen=True, slots=True)
class NormalizedLLMResponse:
    """供应商响应转换后的统一结构，尚未解析为 AgentAction。"""

    provider: str
    response_id: str | None
    model: str | None
    finish_reason: str | None
    content: str | None
    tool_calls: tuple[NormalizedToolCall, ...] = ()
    usage: LLMUsage | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_blank(self.provider, "provider")
        _require_optional_string(self.response_id, "response_id")
        _require_optional_string(self.model, "model")
        _require_optional_string(self.finish_reason, "finish_reason")
        _require_optional_string(self.content, "content")

        calls = tuple(self.tool_calls)
        if any(not isinstance(call, NormalizedToolCall) for call in calls):
            raise TypeError(
                "tool_calls must contain only NormalizedToolCall values"
            )
        object.__setattr__(self, "tool_calls", calls)

        if self.usage is not None and not isinstance(self.usage, LLMUsage):
            raise TypeError("usage must be an LLMUsage or None")
        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(self.metadata, "metadata"),
        )


def _validate_unique_tool_calls(calls: tuple[LLMToolCall, ...]) -> None:
    call_ids = [call.tool_call_id for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("tool_call_id must be unique within one assistant message")
    call_indexes = [call.call_index for call in calls]
    if len(call_indexes) != len(set(call_indexes)):
        raise ValueError("call_index must be unique within one assistant message")


def _validate_tool_call_history(messages: tuple[LLMMessage, ...]) -> None:
    requested_call_ids: set[str] = set()
    answered_call_ids: set[str] = set()

    for message in messages:
        if message.role is LLMMessageRole.USER:
            continue
        if message.role is LLMMessageRole.ASSISTANT:
            for call in message.tool_calls:
                if call.tool_call_id in requested_call_ids:
                    raise ValueError(
                        "tool_call_id must be unique across LLMContext history"
                    )
                requested_call_ids.add(call.tool_call_id)
            continue

        assert message.role is LLMMessageRole.TOOL
        assert message.tool_call_id is not None
        if message.tool_call_id not in requested_call_ids:
            raise ValueError(
                "a TOOL message must reference an earlier assistant tool call"
            )
        if message.tool_call_id in answered_call_ids:
            raise ValueError(
                "an assistant tool call can have only one TOOL result message"
            )
        answered_call_ids.add(message.tool_call_id)

    if requested_call_ids != answered_call_ids:
        raise ValueError(
            "every assistant tool call must have exactly one TOOL result message"
        )


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_optional_string(value: str | None, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")


def _require_non_negative_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to zero")


def _validate_temperature(value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("temperature must be a number or None")
    if value < 0:
        raise ValueError("temperature must be greater than or equal to zero")


def _validate_max_output_tokens(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_output_tokens must be an integer or None")
    if value <= 0:
        raise ValueError("max_output_tokens must be greater than zero")


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)
