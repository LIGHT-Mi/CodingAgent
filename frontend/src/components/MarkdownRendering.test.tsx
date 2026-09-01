import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  AgentStepStatus,
  MessageRole,
  MessageType,
} from "../api/contracts";
import { makeTask } from "../test/fixtures";
import { AgentTurn } from "./AgentTurn";
import { StepCard } from "./StepCard";
import { UserTurn } from "./UserTurn";

const MARKDOWN = [
  "欢迎使用 **Coding Agent**。",
  "",
  "- 读取文件",
  "- 运行 `python -m unittest`",
].join("\n");

describe("Markdown rendering", () => {
  it("renders the final Agent answer as Markdown", () => {
    const { container } = render(
      <AgentTurn task={makeTask({ final_answer: MARKDOWN })} />,
    );

    expect(screen.getByText("Coding Agent", { selector: "strong" })).toBeVisible();
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelector("code")).toHaveTextContent(
      "python -m unittest",
    );
  });

  it("renders Assistant Step messages through the same Markdown component", () => {
    const { container } = render(
      <StepCard
        step={{
          id: "step-0",
          task_id: "task-0",
          step_number: 0,
          status: AgentStepStatus.COMPLETED,
          error: null,
          started_at: "2026-09-01T08:00:00Z",
          finished_at: "2026-09-01T08:01:00Z",
        }}
        messages={[
          {
            id: "message-0",
            task_id: "task-0",
            step_id: "step-0",
            tool_call_id: null,
            sequence: 0,
            role: MessageRole.ASSISTANT,
            message_type: MessageType.FINAL,
            content: MARKDOWN,
            created_at: "2026-09-01T08:01:00Z",
          },
        ]}
        toolCalls={[]}
        commandApprovals={[]}
        onApprovalDecisionRecorded={() => undefined}
      />,
    );

    expect(screen.getByText("Coding Agent", { selector: "strong" })).toBeVisible();
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });

  it("does not execute raw HTML from model output", () => {
    const { container } = render(
      <AgentTurn
        task={makeTask({
          final_answer: '<script>window.alert("unsafe")</script>\n\n安全文本',
        })}
      />,
    );

    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByText("安全文本")).toBeVisible();
  });

  it("keeps the user author label in a separate heading above the bubble", () => {
    const { container } = render(<UserTurn content="你好" />);
    const body = container.querySelector(".user-message .message-body");

    expect(body?.children[0]).toHaveClass("user-message-heading");
    expect(body?.children[1]?.tagName).toBe("P");
  });
});
