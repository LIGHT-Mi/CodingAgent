import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TaskStatus } from "../api/contracts";
import { makeSession, makeSnapshot, makeTask } from "../test/fixtures";
import { AppShell } from "./AppShell";

const apiMocks = vi.hoisted(() => ({
  cancelTask: vi.fn(),
  createSession: vi.fn(),
  createSessionTask: vi.fn(),
  getSession: vi.fn(),
  getSessionTasks: vi.fn(),
  getTaskSnapshot: vi.fn(),
  listSessions: vi.fn(),
}));

vi.mock("../api/client", () => ({
  ApiClientError: class ApiClientError extends Error {
    readonly status: number | null;

    constructor(message: string, status: number | null = null) {
      super(message);
      this.status = status;
    }
  },
  ...apiMocks,
  getApiErrorMessage: (error: unknown) =>
    error instanceof Error ? error.message : "请求失败，请稍后重试。",
  isApiRequestAborted: () => false,
}));

describe("AppShell multi-turn workflow", () => {
  beforeEach(() => {
    Object.values(apiMocks).forEach((mock) => mock.mockReset());
    apiMocks.listSessions.mockResolvedValue([]);
    apiMocks.getSessionTasks.mockResolvedValue([]);
  });

  it("keeps a new draft local until the first message is submitted", async () => {
    const user = userEvent.setup();
    render(<AppShell />);

    await screen.findByText("开始一个新的编程任务");
    await user.type(screen.getByLabelText("任务描述"), "只保存在本地的草稿");
    await user.type(
      screen.getByLabelText("Workspace"),
      "/workspace/draft",
    );

    expect(apiMocks.createSession).not.toHaveBeenCalled();
    expect(apiMocks.createSessionTask).not.toHaveBeenCalled();
  });

  it("shows the server-provided latest update time in the Session list", async () => {
    const updatedAt = "2026-09-01T08:05:00Z";
    apiMocks.listSessions.mockResolvedValue([
      makeSession({ updated_at: updatedAt }),
    ]);

    render(<AppShell />);

    const updatedTime = await screen.findByLabelText(/^最近更新：/);
    expect(updatedTime).toHaveAttribute("datetime", updatedAt);
  });

  it("creates the first Session, uses its server title, then creates a new Task in the same Session", async () => {
    const firstTask = makeTask({
      id: "task-0",
      session_id: "session-0",
      original_prompt: "完成第一轮",
      workspace: "/workspace/one",
    });
    const secondTask = makeTask({
      id: "task-1",
      session_id: "session-0",
      original_prompt: "解释修改原因",
      workspace: "/workspace/two",
      final_answer: "第二轮已完成。",
    });
    apiMocks.createSession.mockResolvedValue({
      session_id: "session-0",
      task_id: "task-0",
      title: "服务端生成的标题",
      status: TaskStatus.PENDING,
    });
    apiMocks.createSessionTask.mockResolvedValue({
      session_id: "session-0",
      task_id: "task-1",
      status: TaskStatus.PENDING,
    });
    apiMocks.getSessionTasks
      .mockResolvedValueOnce([firstTask])
      .mockResolvedValue([firstTask, secondTask]);
    apiMocks.getTaskSnapshot.mockImplementation((taskId: string) =>
      Promise.resolve(
        makeSnapshot(taskId === "task-0" ? firstTask : secondTask),
      ),
    );
    const user = userEvent.setup();
    render(<AppShell />);

    await user.type(screen.getByLabelText("任务描述"), "完成第一轮");
    await user.type(screen.getByLabelText("Workspace"), "/workspace/one");
    await user.click(screen.getByRole("button", { name: "发送任务" }));

    await waitFor(() => {
      expect(apiMocks.createSession).toHaveBeenCalledWith(
        { prompt: "完成第一轮", workspace: "/workspace/one" },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });
    expect(await screen.findByText("服务端生成的标题")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByLabelText("任务描述")).toBeEnabled();
    });

    const workspaceInput = screen.getByLabelText("Workspace");
    await user.clear(workspaceInput);
    await user.type(workspaceInput, "/workspace/two");
    await user.type(screen.getByLabelText("任务描述"), "解释修改原因");
    await user.click(screen.getByRole("button", { name: "发送任务" }));

    await waitFor(() => {
      expect(apiMocks.createSessionTask).toHaveBeenCalledWith(
        "session-0",
        {
          prompt: "解释修改原因",
          workspace: "/workspace/two",
        },
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });
    expect(apiMocks.createSession).toHaveBeenCalledTimes(1);
  });

  it("folds the left and right panels independently", async () => {
    const user = userEvent.setup();
    render(<AppShell />);

    const leftPanel = document.getElementById("conversation-sidebar");
    const rightPanel = document.getElementById("agent-inspector");
    expect(leftPanel).toHaveAttribute("aria-hidden", "false");
    expect(rightPanel).toHaveAttribute("aria-hidden", "false");

    await user.click(
      screen.getByRole("button", { name: "折叠会话侧栏" }),
    );
    expect(leftPanel).toHaveAttribute("aria-hidden", "true");
    expect(rightPanel).toHaveAttribute("aria-hidden", "false");

    await user.click(
      screen.getByRole("button", { name: "折叠执行检查器" }),
    );
    expect(leftPanel).toHaveAttribute("aria-hidden", "true");
    expect(rightPanel).toHaveAttribute("aria-hidden", "true");
    expect(window.localStorage.getItem("leftSidebarCollapsed")).toBe("true");
    expect(window.localStorage.getItem("rightInspectorCollapsed")).toBe(
      "true",
    );
  });

  it("restores the URL and switches the inspector with the selected Task", async () => {
    const firstTask = makeTask({
      id: "task-0",
      session_id: "session-0",
      original_prompt: "第一轮",
      final_answer: "第一轮完成。",
    });
    const secondTask = makeTask({
      id: "task-1",
      session_id: "session-0",
      original_prompt: "第二轮",
      final_answer: "第二轮完成。",
    });
    apiMocks.getSession.mockResolvedValue(
      makeSession({ latest_task_id: "task-1" }),
    );
    apiMocks.getSessionTasks.mockResolvedValue([firstTask, secondTask]);
    apiMocks.getTaskSnapshot.mockImplementation((taskId: string) =>
      Promise.resolve(
        makeSnapshot(taskId === "task-0" ? firstTask : secondTask),
      ),
    );
    window.history.replaceState(
      null,
      "",
      "/?session=session-0&task=task-1",
    );
    const user = userEvent.setup();
    render(<AppShell />);

    await waitFor(() => {
      expect(apiMocks.getSession).toHaveBeenCalledWith(
        "session-0",
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(apiMocks.getSessionTasks).toHaveBeenCalledWith(
        "session-0",
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(apiMocks.getTaskSnapshot).toHaveBeenCalledWith(
        "task-1",
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
    });
    expect(window.location.search).toBe("?session=session-0&task=task-1");

    await user.click(
      screen.getByRole("button", { name: "选择任务：第一轮" }),
    );
    await waitFor(() => {
      expect(apiMocks.getTaskSnapshot).toHaveBeenCalledWith(
        "task-0",
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(window.location.search).toBe("?session=session-0&task=task-0");
    });
    expect(screen.getByText("task-0")).toBeInTheDocument();
  });

  it("keeps the composer disabled while running and waits for a CANCELLED snapshot", async () => {
    const runningTask = makeTask({
      status: TaskStatus.RUNNING,
      final_answer: null,
      finished_at: null,
    });
    const cancelledTask = makeTask({
      status: TaskStatus.CANCELLED,
      final_answer: null,
      termination_reason: "USER_CANCELLED",
    });
    apiMocks.getSession.mockResolvedValue(
      makeSession({ latest_task_status: TaskStatus.RUNNING }),
    );
    apiMocks.getSessionTasks.mockResolvedValue([runningTask]);
    apiMocks.getTaskSnapshot
      .mockResolvedValueOnce(makeSnapshot(runningTask))
      .mockResolvedValue(makeSnapshot(cancelledTask));
    apiMocks.cancelTask.mockResolvedValue({
      task_id: runningTask.id,
      status: TaskStatus.RUNNING,
      cancellation_requested: true,
      outcome: "REQUESTED",
    });
    window.history.replaceState(
      null,
      "",
      "/?session=session-0&task=task-0",
    );
    const user = userEvent.setup();
    render(<AppShell />);

    const promptInput = await screen.findByLabelText("任务描述");
    await waitFor(() => expect(promptInput).toBeDisabled());
    await user.click(
      screen.getByRole("button", { name: "取消当前任务" }),
    );
    expect(await screen.findByText("已请求取消")).toBeInTheDocument();
    expect(promptInput).toBeDisabled();

    await waitFor(
      () => {
        expect(promptInput).toBeEnabled();
        expect(screen.getAllByText(TaskStatus.CANCELLED).length).toBeGreaterThan(
          0,
        );
      },
      { timeout: 3_000 },
    );
    expect(apiMocks.cancelTask).toHaveBeenCalledTimes(1);
  });

  it("does not render server secrets from the public DTO workflow", async () => {
    const task = makeTask();
    apiMocks.getSession.mockResolvedValue(makeSession());
    apiMocks.getSessionTasks.mockResolvedValue([task]);
    apiMocks.getTaskSnapshot.mockResolvedValue(makeSnapshot(task));
    window.history.replaceState(
      null,
      "",
      "/?session=session-0&task=task-0",
    );
    render(<AppShell />);

    await screen.findByText("任务已经完成。");
    expect(document.body).not.toHaveTextContent("DEEPSEEK_API_KEY");
    expect(document.body).not.toHaveTextContent("DATABASE_URL");
    expect(document.body).not.toHaveTextContent("api-secret");
  });
});

describe("useTaskSnapshot polling", () => {
  beforeEach(() => {
    Object.values(apiMocks).forEach((mock) => mock.mockReset());
  });

  it("stops requesting snapshots after the Task reaches a terminal status", async () => {
    const { renderHook } = await import("@testing-library/react");
    const { useTaskSnapshot } = await import("../hooks/useTaskSnapshot");
    const runningTask = makeTask({
      status: TaskStatus.RUNNING,
      final_answer: null,
      finished_at: null,
    });
    const completedTask = makeTask();
    apiMocks.getTaskSnapshot
      .mockResolvedValueOnce(makeSnapshot(runningTask))
      .mockResolvedValueOnce(makeSnapshot(completedTask));
    vi.useFakeTimers();

    renderHook(() => useTaskSnapshot("task-0"));
    await act(async () => Promise.resolve());
    expect(apiMocks.getTaskSnapshot).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_500);
    });
    expect(apiMocks.getTaskSnapshot).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(6_000);
    });
    expect(apiMocks.getTaskSnapshot).toHaveBeenCalledTimes(2);
  });
});
