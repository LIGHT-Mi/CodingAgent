from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.agent_step import AgentStep
    from app.db.models.task import Task
    from app.db.models.tool_call import ToolCall


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Message(Base):
    """构成 Task 对话历史的消息，用于为 LLM 构造上下文。"""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("task_id", "sequence", name="uq_messages_task_sequence"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_id,
        comment="Message 唯一 ID。",
    )
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        index=True,
        comment="该 Message 所属的 Task。",
    )
    step_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("agent_steps.id", ondelete="SET NULL"),
        index=True,
        comment="产生该 Message 的 Agent Step；初始用户输入可以为空。",
    )
    tool_call_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("tool_calls.id", ondelete="SET NULL"),
        unique=True,
        index=True,
        comment="该 Message 对应的 Tool Call，主要用于 Tool Result Message。",
    )
    sequence: Mapped[int] = mapped_column(
        Integer,
        comment="该 Message 在当前 Task 对话历史中的稳定顺序。",
    )
    role: Mapped[str] = mapped_column(
        String(32),
        comment="Message 产生者：USER、ASSISTANT 或 TOOL。",
    )
    message_type: Mapped[str] = mapped_column(
        String(32),
        comment="业务消息类型：USER_INPUT、TEXT、TOOL_CALL、TOOL_RESULT 或 FINAL。",
    )
    content: Mapped[str | None] = mapped_column(
        Text,
        comment="进入 Task 对话历史的文本内容。",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        comment="Message 记录时间。",
    )

    task: Mapped[Task] = relationship(back_populates="messages")
    step: Mapped[AgentStep | None] = relationship(back_populates="messages")
    tool_call: Mapped[ToolCall | None] = relationship(
        back_populates="result_message",
        foreign_keys=[tool_call_id],
    )
