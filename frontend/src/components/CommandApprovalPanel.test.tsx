import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  CommandApprovalDecision,
  CommandApprovalStatus,
  type CommandApproval,
} from "../api/contracts";
import { CommandApprovalPanel } from "./CommandApprovalPanel";

const apiMocks = vi.hoisted(() => ({
  decideCommandApproval: vi.fn(),
}));

vi.mock("../api/client", () => ({
  ...apiMocks,
  getApiErrorMessage: (error: unknown) =>
    error instanceof Error ? error.message : "请求失败",
  isApiRequestAborted: () => false,
}));

const APPROVAL: CommandApproval = {
  id: "approval-0",
  task_id: "task-0",
  step_id: "step-0",
  tool_call_id: "tool-call-0",
  status: CommandApprovalStatus.PENDING,
  command: ["rm", "file with spaces.txt"],
  cwd: "/workspace/project",
  command_fingerprint: "sha256:exact-command",
  rule_id: "DESTRUCTIVE_COMMAND_REQUIRES_APPROVAL",
  risk_level: "HIGH",
  reason: "destructive command requires approval",
  resolution_reason: null,
  created_at: "2026-09-01T08:00:00Z",
  expires_at: "2026-09-01T08:05:00Z",
  decided_at: null,
  consumed_at: null,
};

describe("CommandApprovalPanel", () => {
  beforeEach(() => apiMocks.decideCommandApproval.mockReset());

  it("shows the exact argv, canonical cwd, risk rule and fingerprint", () => {
    render(
      <CommandApprovalPanel
        approval={APPROVAL}
        onDecisionRecorded={() => undefined}
      />,
    );

    expect(screen.getByText(/"file with spaces\.txt"/)).toBeVisible();
    expect(screen.getByText(APPROVAL.cwd)).toBeVisible();
    expect(screen.getByText(APPROVAL.rule_id)).toBeVisible();
    expect(screen.getByText(APPROVAL.command_fingerprint)).toBeVisible();
    expect(screen.getByText("HIGH")).toBeVisible();
  });

  it("sends only the explicit decision and displayed fingerprint", async () => {
    apiMocks.decideCommandApproval.mockResolvedValue({
      ...APPROVAL,
      status: CommandApprovalStatus.APPROVED,
    });
    const recorded = vi.fn();
    const user = userEvent.setup();
    render(
      <CommandApprovalPanel
        approval={APPROVAL}
        onDecisionRecorded={recorded}
      />,
    );

    await user.click(screen.getByRole("button", { name: "允许本次执行" }));

    await waitFor(() => {
      expect(apiMocks.decideCommandApproval).toHaveBeenCalledWith(
        "task-0",
        "approval-0",
        {
          decision: CommandApprovalDecision.APPROVE,
          command_fingerprint: "sha256:exact-command",
        },
        { signal: expect.any(AbortSignal) },
      );
    });
    expect(recorded).toHaveBeenCalledTimes(1);
    const payload = apiMocks.decideCommandApproval.mock.calls[0][2];
    expect(payload).not.toHaveProperty("command");
    expect(payload).not.toHaveProperty("cwd");
  });

  it("does not offer a second decision for a terminal request", () => {
    render(
      <CommandApprovalPanel
        approval={{
          ...APPROVAL,
          status: CommandApprovalStatus.REJECTED,
          resolution_reason: "USER_REJECTED",
        }}
        onDecisionRecorded={() => undefined}
      />,
    );

    expect(screen.queryByRole("button", { name: "允许本次执行" })).toBeNull();
    expect(screen.getByText(/USER_REJECTED/)).toBeVisible();
  });
});
