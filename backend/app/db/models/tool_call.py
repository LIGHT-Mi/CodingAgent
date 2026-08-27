from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.agent_step import AgentStep
    from app.db.models.message import Message


def new_id() -> str:
    return str(uuid4())


class ToolCall(Base):
    """模型发起的一次本地工具调用及其运行结果。"""

    __tablename__ = "tool_calls"

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
    tool_name: Mapped[str] = mapped_column(
        String(64),
        comment="模型请求调用的工具注册名称。",
    )
    arguments: Mapped[dict | None] = mapped_column(
        JSON,
        comment="模型为此次工具调用生成的结构化参数。",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default="PENDING",
        comment="工具运行状态：PENDING、RUNNING、COMPLETED、ERROR、REJECTED 或 TIMEOUT。",
    )
    exit_code: Mapped[int | None] = mapped_column(
        Integer,
        comment="命令类工具的进程退出码；不同于工具运行状态。",
    )
    stdout: Mapped[str | None] = mapped_column(
        Text,
        comment="命令类工具捕获到的标准输出。",
    )
    stderr: Mapped[str | None] = mapped_column(
        Text,
        comment="命令类工具捕获到的标准错误输出。",
    )
    result: Mapped[str | None] = mapped_column(
        Text,
        comment="通用工具结果，用于文件、搜索或格式化后的命令观察结果。",
    )
    error: Mapped[str | None] = mapped_column(
        Text,
        comment="工具级执行错误；不自动代表整个 Task 失败。",
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
    result_message: Mapped[Message | None] = relationship(
        back_populates="tool_call",
        uselist=False,
        foreign_keys="Message.tool_call_id",
    )
