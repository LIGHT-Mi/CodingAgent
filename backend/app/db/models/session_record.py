from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.task import Task


SESSION_TITLE_MAX_LENGTH = 128


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CodingSession(Base):
    """一次持续的用户工作会话，用于组织多个相关编程任务。"""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=new_id,
        comment="Session 唯一 ID。",
    )
    title: Mapped[str] = mapped_column(
        String(SESSION_TITLE_MAX_LENGTH),
        nullable=False,
        comment="根据第一条 Task Prompt 确定性生成的持久化会话标题。",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        comment="用户工作会话创建时间。",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        comment="该会话最近一次发生变化的时间。",
    )

    tasks: Mapped[list[Task]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
