"""危险命令批准的独立应用服务边界。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.agent.cancellation import CancellationToken
from app.approval.contracts import (
    CommandApprovalDecision,
    CommandApprovalStatus,
)
from app.approval.coordinator import (
    ApprovalWaitOutcome,
    CommandApprovalCoordinator,
)
from app.db.models.command_approval import CommandApprovalRequest
from app.db.persistence import (
    InvalidStateTransitionError,
    PersistenceService,
    PersistenceValidationError,
    RecordNotFoundError,
)
from app.tools.router import CommandApprovalRequirement


class CommandApprovalServiceError(RuntimeError):
    """批准请求无法完成时的应用层基础错误。"""


class CommandApprovalNotFoundError(CommandApprovalServiceError):
    pass


class CommandApprovalNotActiveError(CommandApprovalServiceError):
    pass


class CommandApprovalFingerprintMismatchError(CommandApprovalServiceError):
    pass


class CommandApprovalService:
    """创建、等待和决定一次性命令批准请求。"""

    def __init__(
        self,
        persistence: PersistenceService,
        coordinator: CommandApprovalCoordinator,
        timeout_seconds: float,
    ) -> None:
        if not isinstance(persistence, PersistenceService):
            raise TypeError("persistence must be a PersistenceService")
        if not isinstance(coordinator, CommandApprovalCoordinator):
            raise TypeError("coordinator must be a CommandApprovalCoordinator")
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds,
            (int, float),
        ):
            raise TypeError("timeout_seconds must be a number")
        normalized_timeout = float(timeout_seconds)
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
            raise ValueError(
                "timeout_seconds must be finite and greater than zero"
            )
        self._persistence = persistence
        self._coordinator = coordinator
        self._timeout_seconds = normalized_timeout

    def create_request(
        self,
        *,
        task_id: str,
        step_id: str,
        tool_call_id: str,
        requirement: CommandApprovalRequirement,
    ) -> CommandApprovalRequest:
        request_id = str(uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self._timeout_seconds
        )
        self._coordinator.register(request_id)
        try:
            return self._persistence.create_command_approval_request(
                request_id=request_id,
                task_id=task_id,
                step_id=step_id,
                tool_call_id=tool_call_id,
                command=requirement.arguments.command,
                cwd=str(requirement.resolved_cwd),
                command_fingerprint=requirement.command_fingerprint,
                rule_id=requirement.safety_decision.rule_id,
                risk_level=requirement.safety_decision.risk_level.value,
                reason=requirement.safety_decision.reason,
                expires_at=expires_at,
            )
        except BaseException:
            self._coordinator.unregister(request_id)
            raise

    def wait_for_decision(
        self,
        request: CommandApprovalRequest,
        cancellation_token: CancellationToken,
    ) -> CommandApprovalRequest:
        """等待通知、失效或取消，并返回数据库中的最终事实状态。"""

        try:
            outcome = self._coordinator.wait(
                request.id,
                _as_aware_utc(request.expires_at),
                cancellation_token,
            )
            if outcome is ApprovalWaitOutcome.CANCELLED:
                return self._persistence.cancel_command_approval(request.id)
            if outcome is ApprovalWaitOutcome.EXPIRED:
                current = self._require_request(request.id)
                if current.status == CommandApprovalStatus.PENDING.value:
                    return self._persistence.expire_command_approval(request.id)
                return current
            return self._require_request(request.id)
        finally:
            self._coordinator.unregister(request.id)

    def decide(
        self,
        *,
        task_id: str,
        request_id: str,
        decision: CommandApprovalDecision,
        command_fingerprint: str,
    ) -> CommandApprovalRequest:
        """接受来自 CLI/Web 的明确决定，不接收或重写 argv/cwd。"""

        current = self._require_request(request_id)
        if current.task_id != task_id:
            raise CommandApprovalNotFoundError(
                "approval request does not belong to this Task"
            )
        if current.command_fingerprint != command_fingerprint:
            raise CommandApprovalFingerprintMismatchError(
                "approval fingerprint does not match the displayed command"
            )
        if current.status != CommandApprovalStatus.PENDING.value:
            raise CommandApprovalNotActiveError(
                f"approval request is {current.status}"
            )
        if not self._coordinator.is_registered(request_id):
            self._persistence.invalidate_command_approval(
                request_id,
                "RUNTIME_NOT_WAITING",
            )
            raise CommandApprovalNotActiveError(
                "approval request is no longer attached to a running Task"
            )

        try:
            resolved = self._persistence.decide_command_approval(
                task_id=task_id,
                request_id=request_id,
                decision=decision,
                command_fingerprint=command_fingerprint,
            )
        except RecordNotFoundError as exc:
            raise CommandApprovalNotFoundError(str(exc)) from exc
        except PersistenceValidationError as exc:
            if "fingerprint" in str(exc):
                raise CommandApprovalFingerprintMismatchError(str(exc)) from exc
            raise CommandApprovalNotActiveError(str(exc)) from exc
        except InvalidStateTransitionError as exc:
            raise CommandApprovalNotActiveError(str(exc)) from exc

        self._coordinator.notify(request_id)
        if resolved.status == CommandApprovalStatus.EXPIRED.value:
            raise CommandApprovalNotActiveError("approval request has expired")
        return resolved

    def list_for_task(self, task_id: str) -> list[CommandApprovalRequest]:
        return self._persistence.load_command_approval_requests(task_id)

    def consume(
        self,
        request_id: str,
        command_fingerprint: str,
    ) -> CommandApprovalRequest:
        return self._persistence.consume_command_approval(
            request_id,
            command_fingerprint,
        )

    def invalidate(
        self,
        request_id: str,
        reason: str,
    ) -> CommandApprovalRequest:
        return self._persistence.invalidate_command_approval(request_id, reason)

    def _require_request(self, request_id: str) -> CommandApprovalRequest:
        request = self._persistence.get_command_approval_request(request_id)
        if request is None:
            raise CommandApprovalNotFoundError(
                f"CommandApprovalRequest {request_id} was not found"
            )
        return request


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
