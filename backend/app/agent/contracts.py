"""Agent 各模块之间共享的纯 Python 数据契约。

本模块只描述任务服务、上下文管理、LLM 网关、文件工具、命令工具、Agent
Runtime、运行策略和持久化服务之间交换的数据，不依赖数据库 ORM、模型客户端
或具体工具实现。数据库模型可以使用这些枚举值，但本模块不能反向导入数据库层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeAlias


Metadata: TypeAlias = Mapping[str, Any]


class TaskStatus(str, Enum):
    """Task 的完整生命周期状态。"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TERMINATED = "TERMINATED"


class AgentStepStatus(str, Enum):
    """Agent 单轮执行记录的生命周期状态。"""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ToolCallStatus(str, Enum):
    """ToolCall 从等待执行到终态的持久化状态。"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class ToolResultStatus(str, Enum):
    """工具执行后可以返回给 Agent Runtime 的终态。"""

    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class MessageRole(str, Enum):
    """进入任务对话历史的消息来源。"""

    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"


class MessageType(str, Enum):
    """任务对话历史中的业务消息类型。"""

    TEXT = "TEXT"
    TOOL_RESULT = "TOOL_RESULT"
    FINAL = "FINAL"


class RuntimeEventType(str, Enum):
    """需要交给 Runtime Policy 判断的运行事件类型。"""

    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_NETWORK_ERROR = "LLM_NETWORK_ERROR"
    INVALID_ACTION = "INVALID_ACTION"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    FATAL_TOOL_ERROR = "FATAL_TOOL_ERROR"
    FATAL_SYSTEM_ERROR = "FATAL_SYSTEM_ERROR"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    AGENT_STATE_CORRUPTED = "AGENT_STATE_CORRUPTED"
    USER_CANCELLED = "USER_CANCELLED"


class RuntimeDecisionType(str, Enum):
    """Runtime Policy 对当前状态或事件作出的决定。"""

    CONTINUE = "CONTINUE"
    RETRY = "RETRY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TERMINATED = "TERMINATED"


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """LLM 网关从模型响应中解析出的单次工具调用请求。"""

    tool_call_id: str
    tool_name: str
    arguments: Metadata = field(default_factory=dict)
    call_index: int = 0

    def __post_init__(self) -> None:
        _require_non_blank(self.tool_call_id, "tool_call_id")
        _require_non_blank(self.tool_name, "tool_name")
        if not isinstance(self.call_index, int):
            raise TypeError("call_index must be an integer")
        if self.call_index < 0:
            raise ValueError("call_index must be greater than or equal to zero")
        object.__setattr__(
            self,
            "arguments",
            _copy_mapping(self.arguments, "arguments"),
        )


@dataclass(frozen=True, slots=True)
class FinalAction:
    """模型不再请求工具、直接结束当前 Task 的动作。"""

    content: str

    def __post_init__(self) -> None:
        _require_non_blank(self.content, "content")


@dataclass(frozen=True, slots=True)
class ToolCallsAction:
    """模型在当前 Step 中请求按顺序执行的一组工具调用。"""

    tool_calls: tuple[ToolCallRequest, ...]
    content: str | None = None

    def __post_init__(self) -> None:
        calls = tuple(self.tool_calls)
        if not calls:
            raise ValueError("tool_calls must contain at least one request")
        if any(not isinstance(call, ToolCallRequest) for call in calls):
            raise TypeError("tool_calls must contain only ToolCallRequest values")

        call_ids = [call.tool_call_id for call in calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("tool_call_id must be unique within one action")

        call_indexes = [call.call_index for call in calls]
        if len(call_indexes) != len(set(call_indexes)):
            raise ValueError("call_index must be unique within one action")

        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("content must be a string or None")
        object.__setattr__(self, "tool_calls", calls)


@dataclass(frozen=True, slots=True)
class InvalidAction:
    """模型响应无法被解析为 FinalAction 或 ToolCallsAction。"""

    reason: str
    raw_response: Any | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.reason, "reason")


AgentAction: TypeAlias = FinalAction | ToolCallsAction | InvalidAction


@dataclass(frozen=True, slots=True)
class ToolResult:
    """文件工具或命令工具返回给 Agent Runtime 的统一工具执行结果。

    命令工具把 ``exit_code``、``stdout``、``stderr`` 放入 metadata。非零
    exit_code 仍可使用 COMPLETED，表示命令已经正常运行并产生可供模型分析的观察。
    """

    tool_call_id: str
    tool_name: str
    status: ToolResultStatus
    content: str | None = None
    error: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_blank(self.tool_call_id, "tool_call_id")
        _require_non_blank(self.tool_name, "tool_name")
        if not isinstance(self.status, ToolResultStatus):
            raise TypeError("status must be a ToolResultStatus")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("content must be a string or None")

        if self.status is ToolResultStatus.COMPLETED:
            if self.error is not None:
                raise ValueError("a completed tool result must not contain an error")
        else:
            if self.error is None:
                raise ValueError("an unsuccessful tool result must contain an error")
            _require_non_blank(self.error, "error")

        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(self.metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Agent 核心模块产生并交给运行策略判断的异常或控制事件。"""

    event_type: RuntimeEventType
    message: str
    source: str
    details: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, RuntimeEventType):
            raise TypeError("event_type must be a RuntimeEventType")
        _require_non_blank(self.message, "message")
        _require_non_blank(self.source, "source")
        object.__setattr__(
            self,
            "details",
            _copy_mapping(self.details, "details"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeDecision:
    """运行策略返回给 Agent Runtime 的继续、重试或终止决定。"""

    decision: RuntimeDecisionType
    reason: str | None = None
    retry_after_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RuntimeDecisionType):
            raise TypeError("decision must be a RuntimeDecisionType")
        if self.reason is not None:
            _require_non_blank(self.reason, "reason")
        if self.decision is not RuntimeDecisionType.CONTINUE and self.reason is None:
            raise ValueError("a non-CONTINUE decision must contain a reason")
        if not isinstance(self.retry_after_seconds, (int, float)):
            raise TypeError("retry_after_seconds must be a number")
        if self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be greater than or equal to zero")
        if (
            self.decision is not RuntimeDecisionType.RETRY
            and self.retry_after_seconds != 0
        ):
            raise ValueError("only a RETRY decision can specify retry_after_seconds")


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Agent Runtime 返回给任务服务的 Task 终态结果。"""

    status: TaskStatus
    final_answer: str | None = None
    error: str | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TaskStatus):
            raise TypeError("status must be a TaskStatus")
        if self.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            raise ValueError("AgentResult status must be terminal")

        if self.status is TaskStatus.COMPLETED:
            if self.final_answer is None:
                raise ValueError("a completed AgentResult must contain final_answer")
            _require_non_blank(self.final_answer, "final_answer")
            if self.error is not None or self.termination_reason is not None:
                raise ValueError(
                    "a completed AgentResult cannot contain error or termination_reason"
                )
            return

        if self.status is TaskStatus.FAILED:
            if self.error is None:
                raise ValueError("a failed AgentResult must contain error")
            _require_non_blank(self.error, "error")
            if self.final_answer is not None or self.termination_reason is not None:
                raise ValueError(
                    "a failed AgentResult cannot contain final_answer or "
                    "termination_reason"
                )
            return

        if self.termination_reason is None:
            raise ValueError(
                "a cancelled or terminated AgentResult must contain "
                "termination_reason"
            )
        _require_non_blank(self.termination_reason, "termination_reason")
        if self.final_answer is not None or self.error is not None:
            raise ValueError(
                "a cancelled or terminated AgentResult cannot contain "
                "final_answer or error"
            )
