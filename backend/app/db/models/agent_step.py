from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.message import Message
    from app.db.models.task import Task
    from app.db.models.tool_call import ToolCall


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentStep(Base):
    """Task 中 Agent Loop 的一轮自主执行记录。"""

    __tablename__ = "agent_steps"
    __table_args__ = (
        UniqueConstraint("task_id", "step_number", name="uq_agent_steps_task_step_number"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_id,
        comment="Agent Step 唯一 ID。",
    )
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        index=True,
        comment="该 Agent Step 所属的 Task。",
    )
    step_number: Mapped[int] = mapped_column(
        Integer,
        comment="该 Step 在所属 Task 中的轮次编号，用于排序和最大步数检查。",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="RUNNING",
        comment="Step 生命周期：RUNNING、COMPLETED、FAILED 或 INTERRUPTED。",
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        comment="Step 本身无法完成时的运行时错误；普通工具失败应记录为 Tool Result 或 ToolCall error。",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        comment="该轮 Agent Loop 开始时间。",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="该轮 Agent Loop 结束时间。",
    )

    task: Mapped[Task] = relationship(back_populates="agent_steps")
    messages: Mapped[list[Message]] = relationship(
        back_populates="step",
        passive_deletes=True,
    )
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="step",
        cascade="all, delete-orphan",
    )
