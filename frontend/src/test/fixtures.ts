import {
  TaskStatus,
  type SessionSummary,
  type Task,
  type TaskSnapshot,
} from "../api/contracts";

const CREATED_AT = "2026-09-01T08:00:00Z";

export function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: "task-0",
    session_id: "session-0",
    original_prompt: "检查项目",
    workspace: "/workspace/project",
    status: TaskStatus.COMPLETED,
    final_answer: "任务已经完成。",
    error: null,
    termination_reason: null,
    created_at: CREATED_AT,
    started_at: CREATED_AT,
    finished_at: CREATED_AT,
    ...overrides,
  };
}

export function makeSession(
  overrides: Partial<SessionSummary> = {},
): SessionSummary {
  return {
    id: "session-0",
    title: "服务端会话标题",
    created_at: CREATED_AT,
    updated_at: CREATED_AT,
    latest_task_id: "task-0",
    latest_task_status: TaskStatus.COMPLETED,
    latest_workspace: "/workspace/project",
    ...overrides,
  };
}

export function makeSnapshot(task: Task): TaskSnapshot {
  return {
    task,
    steps: [],
    messages: [],
    tool_calls: [],
  };
}
