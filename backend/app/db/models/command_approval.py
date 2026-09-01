from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.approval.contracts import CommandApprovalStatus
from app.db.base import Base


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CommandApprovalRequest(Base):
    """与一个 Task/Step/ToolCall 唯一绑定的一次性命令批准请求。"""

    __tablename__ = "command_approval_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        index=True,
    )
    step_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_steps.id", ondelete="CASCADE"),
        index=True,
    )
    tool_call_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tool_calls.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=CommandApprovalStatus.PENDING.value,
        index=True,
    )
    command: Mapped[list[str]] = mapped_column(JSON)
    cwd: Mapped[str] = mapped_column(String(512))
    command_fingerprint: Mapped[str] = mapped_column(String(80))
    rule_id: Mapped[str] = mapped_column(String(128))
    risk_level: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text)
    resolution_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
