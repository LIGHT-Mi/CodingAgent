import type {
  AgentStep,
  CancelTaskResponse,
  CommandApproval,
  CommandApprovalDecisionRequest,
  CreateSessionRequest,
  CreateSessionResponse,
  CreateSessionTaskRequest,
  CreateSessionTaskResponse,
  ErrorResponse,
  Message,
  SessionSummary,
  Task,
  TaskSnapshot,
  ToolCall,
} from "./contracts";

const TASKS_PATH = "/api/tasks";
const SESSIONS_PATH = "/api/sessions";
const DEFAULT_REQUEST_ERROR_MESSAGE = "请求失败，请稍后重试。";

export const ApiErrorKind = Object.freeze({
  ABORTED: "ABORTED",
  CONFIGURATION: "CONFIGURATION",
  HTTP: "HTTP",
  INVALID_RESPONSE: "INVALID_RESPONSE",
  NETWORK: "NETWORK",
});
export type ApiErrorKind =
  (typeof ApiErrorKind)[keyof typeof ApiErrorKind];

export interface ApiRequestOptions {
  readonly signal?: AbortSignal;
}

export class ApiClientError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly userMessage: string;

  constructor(
    kind: ApiErrorKind,
    userMessage: string,
    status: number | null = null,
  ) {
    super(userMessage);
    this.name = "ApiClientError";
    this.kind = kind;
    this.status = status;
    this.userMessage = userMessage;
  }
}

export type FetchImplementation = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export interface ApiClientOptions {
  readonly baseUrl: string;
  readonly fetchImplementation?: FetchImplementation;
}

function normalizeBaseUrl(baseUrl: string): string {
  if (typeof baseUrl !== "string" || !baseUrl.trim()) {
    throw new ApiClientError(
      ApiErrorKind.CONFIGURATION,
      "前端 API 地址尚未配置。",
    );
  }

  let parsedUrl: URL;
  try {
    parsedUrl = new URL(baseUrl.trim());
  } catch {
    throw new ApiClientError(
      ApiErrorKind.CONFIGURATION,
      "前端 API 地址配置无效。",
    );
  }

  if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
    throw new ApiClientError(
      ApiErrorKind.CONFIGURATION,
      "前端 API 地址必须使用 HTTP 或 HTTPS。",
    );
  }
  if (
    parsedUrl.username ||
    parsedUrl.password ||
    parsedUrl.search ||
    parsedUrl.hash
  ) {
    throw new ApiClientError(
      ApiErrorKind.CONFIGURATION,
      "前端 API 地址不能包含凭据、查询参数或片段。",
    );
  }

  parsedUrl.pathname = parsedUrl.pathname.replace(/\/+$/, "");
  return parsedUrl.toString().replace(/\/$/, "");
}

function taskPath(taskId: string, suffix = ""): string {
  return `${TASKS_PATH}/${encodeURIComponent(taskId)}${suffix}`;
}

function sessionPath(sessionId: string, suffix = ""): string {
  return `${SESSIONS_PATH}/${encodeURIComponent(sessionId)}${suffix}`;
}

function isErrorResponse(value: unknown): value is ErrorResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "detail" in value &&
    typeof value.detail === "string" &&
    value.detail.trim().length > 0
  );
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

async function parseJsonResponse(response: Response): Promise<unknown> {
  const responseText = await response.text();
  if (!responseText) {
    return null;
  }
  try {
    return JSON.parse(responseText) as unknown;
  } catch {
    if (!response.ok) {
      return null;
    }
    throw new ApiClientError(
      ApiErrorKind.INVALID_RESPONSE,
      "服务端返回了无法识别的数据。",
      response.status,
    );
  }
}

export class ApiClient {
  readonly baseUrl: string;
  private readonly fetchImplementation: FetchImplementation;

  constructor({ baseUrl, fetchImplementation }: ApiClientOptions) {
    this.baseUrl = normalizeBaseUrl(baseUrl);
    this.fetchImplementation =
      fetchImplementation ??
      ((input, init) => globalThis.fetch(input, init));
  }

  createSession(
    payload: CreateSessionRequest,
    options: ApiRequestOptions = {},
  ): Promise<CreateSessionResponse> {
    return this.request<CreateSessionResponse>(SESSIONS_PATH, {
      method: "POST",
      body: JSON.stringify(payload),
      signal: options.signal,
    });
  }

  listSessions(
    options: ApiRequestOptions = {},
  ): Promise<SessionSummary[]> {
    return this.request<SessionSummary[]>(SESSIONS_PATH, {
      signal: options.signal,
    });
  }

  getSession(
    sessionId: string,
    options: ApiRequestOptions = {},
  ): Promise<SessionSummary> {
    return this.request<SessionSummary>(sessionPath(sessionId), {
      signal: options.signal,
    });
  }

  getSessionTasks(
    sessionId: string,
    options: ApiRequestOptions = {},
  ): Promise<Task[]> {
    return this.request<Task[]>(sessionPath(sessionId, "/tasks"), {
      signal: options.signal,
    });
  }

  createSessionTask(
    sessionId: string,
    payload: CreateSessionTaskRequest,
    options: ApiRequestOptions = {},
  ): Promise<CreateSessionTaskResponse> {
    return this.request<CreateSessionTaskResponse>(
      sessionPath(sessionId, "/tasks"),
      {
        method: "POST",
        body: JSON.stringify(payload),
        signal: options.signal,
      },
    );
  }

  getTask(
    taskId: string,
    options: ApiRequestOptions = {},
  ): Promise<Task> {
    return this.request<Task>(taskPath(taskId), {
      signal: options.signal,
    });
  }

  getTaskSteps(
    taskId: string,
    options: ApiRequestOptions = {},
  ): Promise<AgentStep[]> {
    return this.request<AgentStep[]>(taskPath(taskId, "/steps"), {
      signal: options.signal,
    });
  }

  getTaskMessages(
    taskId: string,
    options: ApiRequestOptions = {},
  ): Promise<Message[]> {
    return this.request<Message[]>(taskPath(taskId, "/messages"), {
      signal: options.signal,
    });
  }

  getTaskToolCalls(
    taskId: string,
    options: ApiRequestOptions = {},
  ): Promise<ToolCall[]> {
    return this.request<ToolCall[]>(taskPath(taskId, "/tool-calls"), {
      signal: options.signal,
    });
  }

  getTaskCommandApprovals(
    taskId: string,
    options: ApiRequestOptions = {},
  ): Promise<CommandApproval[]> {
    return this.request<CommandApproval[]>(
      taskPath(taskId, "/command-approvals"),
      { signal: options.signal },
    );
  }

  decideCommandApproval(
    taskId: string,
    approvalId: string,
    payload: CommandApprovalDecisionRequest,
    options: ApiRequestOptions = {},
  ): Promise<CommandApproval> {
    return this.request<CommandApproval>(
      taskPath(
        taskId,
        `/command-approvals/${encodeURIComponent(approvalId)}/decision`,
      ),
      {
        method: "POST",
        body: JSON.stringify(payload),
        signal: options.signal,
      },
    );
  }

  getTaskSnapshot(
    taskId: string,
    options: ApiRequestOptions = {},
  ): Promise<TaskSnapshot> {
    return this.request<TaskSnapshot>(taskPath(taskId, "/snapshot"), {
      signal: options.signal,
    });
  }

  cancelTask(
    taskId: string,
    options: ApiRequestOptions = {},
  ): Promise<CancelTaskResponse> {
    return this.request<CancelTaskResponse>(taskPath(taskId, "/cancel"), {
      method: "POST",
      signal: options.signal,
    });
  }

  private async request<T>(
    path: string,
    init: RequestInit,
  ): Promise<T> {
    let response: Response;
    let payload: unknown;
    try {
      response = await this.fetchImplementation(`${this.baseUrl}${path}`, {
        ...init,
        headers: {
          Accept: "application/json",
          ...(init.body === undefined
            ? {}
            : { "Content-Type": "application/json" }),
        },
      });
      payload = await parseJsonResponse(response);
    } catch (error: unknown) {
      if (error instanceof ApiClientError) {
        throw error;
      }
      if (init.signal?.aborted || isAbortError(error)) {
        throw new ApiClientError(
          ApiErrorKind.ABORTED,
          "请求已取消。",
        );
      }
      throw new ApiClientError(
        ApiErrorKind.NETWORK,
        "无法连接到服务端，请检查网络和服务状态。",
      );
    }

    if (!response.ok) {
      throw new ApiClientError(
        ApiErrorKind.HTTP,
        isErrorResponse(payload)
          ? payload.detail
          : DEFAULT_REQUEST_ERROR_MESSAGE,
        response.status,
      );
    }
    if (payload === null) {
      throw new ApiClientError(
        ApiErrorKind.INVALID_RESPONSE,
        "服务端没有返回预期数据。",
        response.status,
      );
    }
    return payload as T;
  }
}

let defaultApiClient: ApiClient | null = null;

function getDefaultApiClient(): ApiClient {
  if (defaultApiClient === null) {
    defaultApiClient = new ApiClient({
      baseUrl: import.meta.env.VITE_API_BASE_URL,
    });
  }
  return defaultApiClient;
}

export function createSession(
  payload: CreateSessionRequest,
  options?: ApiRequestOptions,
): Promise<CreateSessionResponse> {
  return getDefaultApiClient().createSession(payload, options);
}

export function listSessions(
  options?: ApiRequestOptions,
): Promise<SessionSummary[]> {
  return getDefaultApiClient().listSessions(options);
}

export function getSession(
  sessionId: string,
  options?: ApiRequestOptions,
): Promise<SessionSummary> {
  return getDefaultApiClient().getSession(sessionId, options);
}

export function getSessionTasks(
  sessionId: string,
  options?: ApiRequestOptions,
): Promise<Task[]> {
  return getDefaultApiClient().getSessionTasks(sessionId, options);
}

export function createSessionTask(
  sessionId: string,
  payload: CreateSessionTaskRequest,
  options?: ApiRequestOptions,
): Promise<CreateSessionTaskResponse> {
  return getDefaultApiClient().createSessionTask(
    sessionId,
    payload,
    options,
  );
}

export function getTask(
  taskId: string,
  options?: ApiRequestOptions,
): Promise<Task> {
  return getDefaultApiClient().getTask(taskId, options);
}

export function getTaskSteps(
  taskId: string,
  options?: ApiRequestOptions,
): Promise<AgentStep[]> {
  return getDefaultApiClient().getTaskSteps(taskId, options);
}

export function getTaskMessages(
  taskId: string,
  options?: ApiRequestOptions,
): Promise<Message[]> {
  return getDefaultApiClient().getTaskMessages(taskId, options);
}

export function getTaskToolCalls(
  taskId: string,
  options?: ApiRequestOptions,
): Promise<ToolCall[]> {
  return getDefaultApiClient().getTaskToolCalls(taskId, options);
}

export function getTaskCommandApprovals(
  taskId: string,
  options?: ApiRequestOptions,
): Promise<CommandApproval[]> {
  return getDefaultApiClient().getTaskCommandApprovals(taskId, options);
}

export function decideCommandApproval(
  taskId: string,
  approvalId: string,
  payload: CommandApprovalDecisionRequest,
  options?: ApiRequestOptions,
): Promise<CommandApproval> {
  return getDefaultApiClient().decideCommandApproval(
    taskId,
    approvalId,
    payload,
    options,
  );
}

export function getTaskSnapshot(
  taskId: string,
  options?: ApiRequestOptions,
): Promise<TaskSnapshot> {
  return getDefaultApiClient().getTaskSnapshot(taskId, options);
}

export function cancelTask(
  taskId: string,
  options?: ApiRequestOptions,
): Promise<CancelTaskResponse> {
  return getDefaultApiClient().cancelTask(taskId, options);
}

export function getApiErrorMessage(error: unknown): string {
  return error instanceof ApiClientError
    ? error.userMessage
    : DEFAULT_REQUEST_ERROR_MESSAGE;
}

export function isApiRequestAborted(error: unknown): boolean {
  return (
    error instanceof ApiClientError && error.kind === ApiErrorKind.ABORTED
  );
}
