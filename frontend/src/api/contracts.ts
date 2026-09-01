/** 与 backend/app/web/contracts.py 一一对应的 HTTP 数据契约。 */

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | JsonObject;
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export type IsoDateTimeString = string;

function enumValues<const T extends Record<string, string>>(values: T): T {
  return Object.freeze(values);
}

export const TaskStatus = enumValues({
  PENDING: "PENDING",
  RUNNING: "RUNNING",
  COMPLETED: "COMPLETED",
  FAILED: "FAILED",
  CANCELLED: "CANCELLED",
  TERMINATED: "TERMINATED",
});
export type TaskStatus = (typeof TaskStatus)[keyof typeof TaskStatus];

export const AgentStepStatus = enumValues({
  RUNNING: "RUNNING",
  COMPLETED: "COMPLETED",
  FAILED: "FAILED",
  INTERRUPTED: "INTERRUPTED",
});
export type AgentStepStatus =
  (typeof AgentStepStatus)[keyof typeof AgentStepStatus];

export const ToolCallStatus = enumValues({
  PENDING: "PENDING",
  RUNNING: "RUNNING",
  COMPLETED: "COMPLETED",
  ERROR: "ERROR",
  REJECTED: "REJECTED",
  TIMEOUT: "TIMEOUT",
});
export type ToolCallStatus =
  (typeof ToolCallStatus)[keyof typeof ToolCallStatus];

export const MessageRole = enumValues({
  ASSISTANT: "ASSISTANT",
  TOOL: "TOOL",
});
export type MessageRole = (typeof MessageRole)[keyof typeof MessageRole];

export const MessageType = enumValues({
  TEXT: "TEXT",
  TOOL_RESULT: "TOOL_RESULT",
  FINAL: "FINAL",
});
export type MessageType = (typeof MessageType)[keyof typeof MessageType];

export const TaskCancellationOutcome = enumValues({
  REQUESTED: "REQUESTED",
  ALREADY_REQUESTED: "ALREADY_REQUESTED",
  TASK_NOT_FOUND: "TASK_NOT_FOUND",
  TASK_FINISHED: "TASK_FINISHED",
  TASK_NOT_ACTIVE: "TASK_NOT_ACTIVE",
});
export type TaskCancellationOutcome =
  (typeof TaskCancellationOutcome)[keyof typeof TaskCancellationOutcome];

export const CommandApprovalStatus = enumValues({
  PENDING: "PENDING",
  APPROVED: "APPROVED",
  REJECTED: "REJECTED",
  EXPIRED: "EXPIRED",
  INVALIDATED: "INVALIDATED",
  CONSUMED: "CONSUMED",
  CANCELLED: "CANCELLED",
});
export type CommandApprovalStatus =
  (typeof CommandApprovalStatus)[keyof typeof CommandApprovalStatus];

export const CommandApprovalDecision = enumValues({
  APPROVE: "APPROVE",
  REJECT: "REJECT",
});
export type CommandApprovalDecision =
  (typeof CommandApprovalDecision)[keyof typeof CommandApprovalDecision];

export interface Task {
  readonly id: string;
  readonly session_id: string;
  readonly original_prompt: string;
  readonly workspace: string;
  readonly status: TaskStatus;
  readonly final_answer: string | null;
  readonly error: string | null;
  readonly termination_reason: string | null;
  readonly created_at: IsoDateTimeString;
  readonly started_at: IsoDateTimeString | null;
  readonly finished_at: IsoDateTimeString | null;
}

export interface AgentStep {
  readonly id: string;
  readonly task_id: string;
  readonly step_number: number;
  readonly status: AgentStepStatus;
  readonly error: string | null;
  readonly started_at: IsoDateTimeString;
  readonly finished_at: IsoDateTimeString | null;
}

export interface Message {
  readonly id: string;
  readonly task_id: string;
  readonly step_id: string;
  readonly tool_call_id: string | null;
  readonly sequence: number;
  readonly role: MessageRole;
  readonly message_type: MessageType;
  readonly content: string;
  readonly created_at: IsoDateTimeString;
}

export interface ToolCall {
  readonly id: string;
  readonly step_id: string;
  readonly assistant_message_id: string;
  readonly call_index: number;
  readonly tool_name: string;
  readonly arguments: JsonObject;
  readonly status: ToolCallStatus;
  readonly exit_code: number | null;
  readonly stdout: string | null;
  readonly stderr: string | null;
  readonly result: string | null;
  readonly result_metadata: JsonObject | null;
  readonly error: string | null;
  readonly started_at: IsoDateTimeString | null;
  readonly finished_at: IsoDateTimeString | null;
}

export interface CommandApproval {
  readonly id: string;
  readonly task_id: string;
  readonly step_id: string;
  readonly tool_call_id: string;
  readonly status: CommandApprovalStatus;
  readonly command: string[];
  readonly cwd: string;
  readonly command_fingerprint: string;
  readonly rule_id: string;
  readonly risk_level: string;
  readonly reason: string;
  readonly resolution_reason: string | null;
  readonly created_at: IsoDateTimeString;
  readonly expires_at: IsoDateTimeString;
  readonly decided_at: IsoDateTimeString | null;
  readonly consumed_at: IsoDateTimeString | null;
}

export interface CommandApprovalDecisionRequest {
  readonly decision: CommandApprovalDecision;
  readonly command_fingerprint: string;
}

export interface CreateSessionRequest {
  readonly prompt: string;
  readonly workspace: string;
}

export type CreateSessionTaskRequest = CreateSessionRequest;

export interface CreateSessionResponse {
  readonly session_id: string;
  readonly task_id: string;
  readonly title: string;
  readonly status: typeof TaskStatus.PENDING;
}

export interface CreateSessionTaskResponse {
  readonly session_id: string;
  readonly task_id: string;
  readonly status: typeof TaskStatus.PENDING;
}

export interface SessionSummary {
  readonly id: string;
  readonly title: string;
  readonly created_at: IsoDateTimeString;
  readonly updated_at: IsoDateTimeString;
  readonly latest_task_id: string;
  readonly latest_task_status: TaskStatus;
  readonly latest_workspace: string;
}

export interface TaskSnapshot {
  readonly task: Task;
  readonly steps: AgentStep[];
  readonly messages: Message[];
  readonly tool_calls: ToolCall[];
  readonly command_approvals: CommandApproval[];
}

export interface CancelTaskResponse {
  readonly task_id: string;
  readonly status: TaskStatus;
  readonly cancellation_requested: boolean;
  readonly outcome: TaskCancellationOutcome;
}

export interface ErrorResponse {
  readonly detail: string;
}
