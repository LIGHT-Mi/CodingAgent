"""把批准终态转换成模型可见的普通命令 Observation。"""

from __future__ import annotations

from app.agent.contracts import ToolResult
from app.approval.contracts import CommandApprovalStatus
from app.db.models.command_approval import CommandApprovalRequest
from app.tools.command_results import CommandResultBuilder


def build_command_approval_observation(
    provider_tool_call_id: str,
    request: CommandApprovalRequest,
) -> ToolResult:
    status = CommandApprovalStatus(request.status)
    reason_by_status = {
        CommandApprovalStatus.REJECTED: "user rejected the command",
        CommandApprovalStatus.EXPIRED: "command approval request expired",
        CommandApprovalStatus.INVALIDATED: "command approval request became invalid",
        CommandApprovalStatus.CANCELLED: "command approval was cancelled",
    }
    reason = reason_by_status.get(status)
    if reason is None:
        raise ValueError(
            f"approval status {status.value} cannot become an observation"
        )
    return CommandResultBuilder().build_rejected(
        provider_tool_call_id,
        reason,
        argv=tuple(request.command),
        cwd=request.cwd,
        details=(
            f"approval_request_id: {request.id}\n"
            f"approval_status: {status.value}\n"
            f"risk: {request.risk_level}\n"
            f"rule: {request.rule_id}"
        ),
        metadata={
            "approval_request_id": request.id,
            "approval_status": status.value,
            "command_fingerprint": request.command_fingerprint,
            "rule_id": request.rule_id,
            "risk_level": request.risk_level,
            "resolution_reason": request.resolution_reason,
        },
    )
