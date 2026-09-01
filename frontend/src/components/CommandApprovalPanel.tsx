import { useEffect, useRef, useState } from "react";

import {
  decideCommandApproval,
  getApiErrorMessage,
  isApiRequestAborted,
} from "../api/client";
import {
  CommandApprovalDecision,
  CommandApprovalStatus,
  type CommandApproval,
} from "../api/contracts";
import { formatInspectorTime } from "./inspectorFormatters";

interface CommandApprovalPanelProps {
  readonly approval: CommandApproval;
  readonly onDecisionRecorded: () => void;
}

export function CommandApprovalPanel({
  approval,
  onDecisionRecorded,
}: CommandApprovalPanelProps) {
  const [submittingDecision, setSubmittingDecision] =
    useState<CommandApprovalDecision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestController = useRef<AbortController | null>(null);
  const isPending = approval.status === CommandApprovalStatus.PENDING;

  useEffect(
    () => () => requestController.current?.abort(),
    [],
  );

  async function submitDecision(decision: CommandApprovalDecision) {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setSubmittingDecision(decision);
    setError(null);
    try {
      await decideCommandApproval(
        approval.task_id,
        approval.id,
        {
          decision,
          command_fingerprint: approval.command_fingerprint,
        },
        { signal: controller.signal },
      );
      onDecisionRecorded();
    } catch (requestError: unknown) {
      if (!isApiRequestAborted(requestError)) {
        setError(getApiErrorMessage(requestError));
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null;
        setSubmittingDecision(null);
      }
    }
  }

  return (
    <section className="command-approval" aria-label="命令执行批准">
      <header>
        <div>
          <strong>需要你的批准</strong>
          <span
            className={`approval-risk risk-${approval.risk_level.toLowerCase()}`}
          >
            {approval.risk_level}
          </span>
        </div>
        <span
          className={`approval-status status-${approval.status.toLowerCase()}`}
        >
          {approval.status}
        </span>
      </header>

      <p>{approval.reason}</p>
      <dl>
        <div>
          <dt>完整 argv</dt>
          <dd>
            <pre>{JSON.stringify(approval.command, null, 2)}</pre>
          </dd>
        </div>
        <div>
          <dt>工作目录</dt>
          <dd>
            <code>{approval.cwd}</code>
          </dd>
        </div>
        <div>
          <dt>安全规则</dt>
          <dd>
            <code>{approval.rule_id}</code>
          </dd>
        </div>
        <div>
          <dt>命令指纹</dt>
          <dd>
            <code>{approval.command_fingerprint}</code>
          </dd>
        </div>
        <div>
          <dt>决定有效期</dt>
          <dd>
            <time dateTime={approval.expires_at}>
              {formatInspectorTime(approval.expires_at)}
            </time>
          </dd>
        </div>
      </dl>

      {isPending ? (
        <div className="approval-actions">
          <button
            type="button"
            className="approval-reject"
            disabled={submittingDecision !== null}
            onClick={() => void submitDecision(CommandApprovalDecision.REJECT)}
          >
            {submittingDecision === CommandApprovalDecision.REJECT
              ? "正在拒绝…"
              : "拒绝执行"}
          </button>
          <button
            type="button"
            className="approval-approve"
            disabled={submittingDecision !== null}
            onClick={() => void submitDecision(CommandApprovalDecision.APPROVE)}
          >
            {submittingDecision === CommandApprovalDecision.APPROVE
              ? "正在批准…"
              : "允许本次执行"}
          </button>
        </div>
      ) : (
        <p className="approval-resolution">
          处理结果：{approval.resolution_reason ?? approval.status}
        </p>
      )}
      {error ? (
        <p className="approval-error" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
