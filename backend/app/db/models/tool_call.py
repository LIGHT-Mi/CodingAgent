from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.agent.contracts import ToolCallStatus
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.agent_step import AgentStep
    from app.db.models.message import Message


def new_id() -> str:
    return str(uuid4())


class ToolCall(Base):
    """LLM 在某个 Agent Step 中发起的一次具体工具调用及其执行结果。"""

    __tablename__ = "tool_calls"
    __table_args__ = (
        UniqueConstraint("step_id", "call_index", name="uq_tool_calls_step_call_index"),
        UniqueConstraint(
            "assistant_message_id",
            "provider_call_id",
            name="uq_tool_calls_message_provider_call_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_id,
        comment="Tool Call 唯一 ID。",
    )
    step_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_steps.id", ondelete="CASCADE"),
        index=True,
        comment="包含该 Tool Call 的 Agent Step。",
    )
    assistant_message_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("messages.id", ondelete="CASCADE"),
        index=True,
        comment="发起该 Tool Call 的 Assistant Message；一条 Assistant Message 可发起多个 Tool Call。",
    )
    provider_call_id: Mapped[str] = mapped_column(
        String(255),
        index=True,
        comment="模型供应商返回的 Tool Call ID，用于构造对应的 Tool Result 消息。",
    )
    call_index: Mapped[int] = mapped_column(
        Integer,
        comment="该 Tool Call 在同一 Assistant ToolCalls 列表中的位置，用于稳定恢复原始顺序。",
    )
    tool_name: Mapped[str] = mapped_column(
        String(64),
        comment="工具名称：READ_FILE、WRITE_FILE、EDIT_FILE、LIST_FILES、SEARCH_FILES、CREATE_FILE、DELETE_FILE 或 RUN_COMMAND。",
    )
    arguments: Mapped[dict] = mapped_column(
        JSON,
        comment="模型为此次工具调用生成的原始结构化参数。",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=ToolCallStatus.PENDING.value,
        comment="工具运行状态：PENDING、RUNNING、COMPLETED、ERROR、REJECTED 或 TIMEOUT。",
    )
    exit_code: Mapped[int | None] = mapped_column(
        Integer,
        comment="RUN_COMMAND 的进程退出码；非零退出码不自动代表 ToolCall status=ERROR。",
    )
    stdout: Mapped[str | None] = mapped_column(
        Text,
        comment="RUN_COMMAND 捕获到的标准输出；非命令工具通常为空。",
    )
    stderr: Mapped[str | None] = mapped_column(
        Text,
        comment="RUN_COMMAND 捕获到的标准错误输出；非命令工具通常为空。",
    )
    result: Mapped[str | None] = mapped_column(
        Text,
        comment="Tool 执行层面的原始或标准化结果；发送给 LLM 的观察内容记录在 messages.content。",
    )
    result_metadata: Mapped[dict | None] = mapped_column(
        JSON,
        comment="统一 ToolResult 携带的结构化 metadata，包括命令退出信息或文件结果信息。",
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        comment="工具级错误信息，包括参数错误、安全拒绝或运行异常；不自动代表整个 Task 失败。",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="工具运行时开始处理此次调用的时间。",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="工具运行时完成此次调用的时间。",
    )

    step: Mapped[AgentStep] = relationship(back_populates="tool_calls")
    assistant_message: Mapped[Message] = relationship(
        back_populates="requested_tool_calls",
        foreign_keys=[assistant_message_id],
    )
    result_message: Mapped[Message | None] = relationship(
        back_populates="tool_call",
        uselist=False,
        foreign_keys="Message.tool_call_id",
    )
