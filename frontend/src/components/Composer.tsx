import { useEffect, useRef, useState, type FormEvent } from "react";

import {
  cancelTask,
  createSession,
  createSessionTask,
  getApiErrorMessage,
  isApiRequestAborted,
} from "../api/client";
import type {
  CreateSessionResponse,
  CreateSessionTaskResponse,
  Task,
  TaskStatus,
} from "../api/contracts";
import { isActiveTaskStatus } from "../api/taskStatus";
import { PromptInput } from "./PromptInput";
import { SubmitOrCancelButton } from "./SubmitOrCancelButton";
import { WorkspaceControl } from "./WorkspaceControl";

export interface TaskSubmission {
  readonly prompt: string;
  readonly workspace: string;
}

interface ComposerProps {
  readonly sessionId: string | null;
  readonly latestTask: Task | null;
  readonly latestTaskId: string | null;
  readonly latestTaskStatus: TaskStatus | null;
  readonly defaultWorkspace: string;
  readonly disabled: boolean;
  readonly onSessionCreated: (
    session: CreateSessionResponse,
    submission: TaskSubmission,
  ) => void;
  readonly onTaskCreated: (
    task: CreateSessionTaskResponse,
    submission: TaskSubmission,
  ) => void;
}

interface FormErrors {
  prompt?: string;
  workspace?: string;
  submit?: string;
}

export function Composer({
  sessionId,
  latestTask,
  latestTaskId,
  latestTaskStatus,
  defaultWorkspace,
  disabled,
  onSessionCreated,
  onTaskCreated,
}: ComposerProps) {
  const [prompt, setPrompt] = useState("");
  const [workspace, setWorkspace] = useState(defaultWorkspace);
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [cancellationRequested, setCancellationRequested] = useState(false);
  const activeRequest = useRef<AbortController | null>(null);
  const isTaskActive = isActiveTaskStatus(latestTaskStatus);

  useEffect(() => {
    setPrompt("");
    setWorkspace(defaultWorkspace);
    setErrors({});
  }, [latestTaskId, sessionId]);

  useEffect(() => {
    if (!isTaskActive) {
      setWorkspace(latestTask?.workspace ?? defaultWorkspace);
    }
  }, [defaultWorkspace, isTaskActive, latestTask?.workspace]);

  useEffect(() => {
    setCancellationRequested(false);
    setIsCancelling(false);
  }, [latestTaskId]);

  useEffect(() => {
    return () => activeRequest.current?.abort();
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled || isTaskActive) {
      return;
    }

    const validationErrors: FormErrors = {};
    if (!prompt.trim()) {
      validationErrors.prompt = "请输入需要 Agent 完成的任务。";
    }
    if (!workspace.trim()) {
      validationErrors.workspace = "请输入后端能够访问的 Workspace 路径。";
    }
    if (validationErrors.prompt || validationErrors.workspace) {
      setErrors(validationErrors);
      return;
    }

    activeRequest.current?.abort();
    const requestController = new AbortController();
    activeRequest.current = requestController;
    const submission = {
      prompt: prompt.trim(),
      workspace: workspace.trim(),
    };
    setErrors({});
    setIsSubmitting(true);

    try {
      if (sessionId === null) {
        const response = await createSession(submission, {
          signal: requestController.signal,
        });
        onSessionCreated(response, submission);
      } else {
        const response = await createSessionTask(sessionId, submission, {
          signal: requestController.signal,
        });
        onTaskCreated(response, submission);
      }
      setPrompt("");
    } catch (requestError: unknown) {
      if (!isApiRequestAborted(requestError)) {
        setErrors({ submit: getApiErrorMessage(requestError) });
      }
    } finally {
      if (activeRequest.current === requestController) {
        activeRequest.current = null;
        setIsSubmitting(false);
      }
    }
  }

  async function handleCancel() {
    if (disabled || !isTaskActive || latestTaskId === null) {
      return;
    }
    activeRequest.current?.abort();
    const requestController = new AbortController();
    activeRequest.current = requestController;
    setErrors({});
    setIsCancelling(true);
    try {
      const response = await cancelTask(latestTaskId, {
        signal: requestController.signal,
      });
      setCancellationRequested(response.cancellation_requested);
      if (!response.cancellation_requested) {
        setErrors({ submit: "当前任务已经无法取消，请等待状态刷新。" });
      }
    } catch (requestError: unknown) {
      if (!isApiRequestAborted(requestError)) {
        setErrors({ submit: getApiErrorMessage(requestError) });
      }
    } finally {
      if (activeRequest.current === requestController) {
        activeRequest.current = null;
        setIsCancelling(false);
      }
    }
  }

  const controlsDisabled =
    disabled || isTaskActive || isSubmitting || isCancelling;

  return (
    <form className="task-form" onSubmit={handleSubmit} noValidate>
      <WorkspaceControl
        value={workspace}
        disabled={controlsDisabled}
        error={errors.workspace ?? null}
        onChange={(value) => {
          setWorkspace(value);
          if (errors.workspace) {
            setErrors((current) => ({ ...current, workspace: undefined }));
          }
        }}
      />

      <div className="prompt-control">
        <div className="prompt-input-wrapper">
          <PromptInput
            value={prompt}
            disabled={controlsDisabled}
            error={errors.prompt ?? null}
            onChange={(value) => {
              setPrompt(value);
              if (errors.prompt) {
                setErrors((current) => ({ ...current, prompt: undefined }));
              }
            }}
          />
        </div>
        <SubmitOrCancelButton
          isTaskActive={isTaskActive}
          isSubmitting={isSubmitting}
          isCancelling={isCancelling}
          cancellationRequested={cancellationRequested}
          disabled={disabled}
          onCancel={() => void handleCancel()}
        />
      </div>

      {errors.submit ? (
        <div className="submit-error" role="alert">
          {errors.submit}
        </div>
      ) : null}

      <p className="sr-only" role="status" aria-live="polite">
        {cancellationRequested
          ? "已向服务端请求取消任务，正在等待任务进入已取消状态。"
          : isCancelling
            ? "正在请求取消任务。"
            : ""}
      </p>

      <p className="composer-note" id="task-workspace-note">
        Workspace 是下一轮 Task 使用的服务端本地目录，不会修改历史 Task。
      </p>
    </form>
  );
}
