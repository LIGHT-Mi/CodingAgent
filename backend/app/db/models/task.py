from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.agent.contracts import TaskStatus
from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.agent_step import AgentStep
    from app.db.models.message import Message
    from app.db.models.session_record import CodingSession


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Task(Base):
    """一次具体的编程任务执行实例，属于某个会话并绑定一个工作区。"""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_id,
        comment="Task 唯一 ID。",
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
        comment="所属 Session 的 ID。",
    )
    original_prompt: Mapped[str] = mapped_column(
        Text,
        comment="用户创建该任务时提交的原始需求文本。",
    )
    workspace: Mapped[str] = mapped_column(
        String(512),
        comment="工具被允许操作的工作区根路径。",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=TaskStatus.PENDING.value,
        comment="Task 生命周期：PENDING、RUNNING、COMPLETED、FAILED、CANCELLED、TERMINATED。",
    )
    final_answer: Mapped[str | None] = mapped_column(
        Text,
        comment="Task 对外最终结果；通常与 FINAL Assistant Message 内容相同。",
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        comment="Task 因不可恢复错误失败时的错误信息；可恢复工具观察结果记录在 tool_calls 中。",
    )
    termination_reason: Mapped[str | None] = mapped_column(
        Text,
        comment="Task 被取消或被控制规则终止时的非错误原因，例如用户取消、达到最大步数或检测到循环。",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        comment="Task 创建时间。",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Agent Loop 开始执行该 Task 的时间。",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Task 进入终态的时间。",
    )

    session: Mapped[CodingSession] = relationship(back_populates="tasks")
    messages: Mapped[list[Message]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
    )
    agent_steps: Mapped[list[AgentStep]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
    )
