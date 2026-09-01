"""危险命令用户批准工作流使用的普通 Python 状态契约。"""

from __future__ import annotations

from enum import Enum


class CommandApprovalStatus(str, Enum):
    """一次性命令批准请求的完整生命周期。"""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    CONSUMED = "CONSUMED"
    CANCELLED = "CANCELLED"


class CommandApprovalDecision(str, Enum):
    """用户可以对仍有效的请求作出的明确决定。"""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
