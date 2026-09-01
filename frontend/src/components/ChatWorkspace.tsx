import type {
  CreateSessionResponse,
  CreateSessionTaskResponse,
  Task,
} from "../api/contracts";
import type { ActiveConversation } from "./AppShell";
import { ChatHeader } from "./ChatHeader";
import { Composer, type TaskSubmission } from "./Composer";
import { ConversationFeed } from "./ConversationFeed";

interface ChatWorkspaceProps {
  readonly activeConversation: ActiveConversation | null;
  readonly tasks: Task[];
  readonly latestTask: Task | null;
  readonly selectedTaskId: string | null;
  readonly tasksLoading: boolean;
  readonly tasksError: string | null;
  readonly navigationLoading: boolean;
  readonly leftSidebarCollapsed: boolean;
  readonly rightInspectorCollapsed: boolean;
  readonly pendingApprovalCount: number;
  readonly onToggleLeftSidebar: () => void;
  readonly onToggleRightInspector: () => void;
  readonly onTaskSelected: (taskId: string) => void;
  readonly onRetryTasks: () => void;
  readonly onSessionCreated: (
    session: CreateSessionResponse,
    submission: TaskSubmission,
  ) => void;
  readonly onTaskCreated: (
    task: CreateSessionTaskResponse,
    submission: TaskSubmission,
  ) => void;
}

export function ChatWorkspace({
  activeConversation,
  tasks,
  latestTask,
  selectedTaskId,
  tasksLoading,
  tasksError,
  navigationLoading,
  leftSidebarCollapsed,
  rightInspectorCollapsed,
  pendingApprovalCount,
  onToggleLeftSidebar,
  onToggleRightInspector,
  onTaskSelected,
  onRetryTasks,
  onSessionCreated,
  onTaskCreated,
}: ChatWorkspaceProps) {
  const latestTaskId = activeConversation?.taskId ?? null;
  const latestTaskStatus =
    latestTask?.status ?? activeConversation?.initialStatus ?? null;
  const defaultWorkspace =
    latestTask?.workspace ?? activeConversation?.latestWorkspace ?? "";

  return (
    <section className="chat-workspace" aria-label="对话工作区">
      <ChatHeader
        title={activeConversation?.title ?? "新会话"}
        taskId={latestTaskId}
        task={latestTask}
        leftSidebarCollapsed={leftSidebarCollapsed}
        rightInspectorCollapsed={rightInspectorCollapsed}
        pendingApprovalCount={pendingApprovalCount}
        onToggleLeftSidebar={onToggleLeftSidebar}
        onToggleRightInspector={onToggleRightInspector}
      />

      <div className="chat-scroll-region">
        {navigationLoading ? (
          <div className="conversation-loading" role="status">
            <span className="sidebar-spinner" aria-hidden="true" />
            正在从 URL 恢复会话
          </div>
        ) : activeConversation === null ? (
          <div className="new-conversation-empty">
            <span className="empty-orbit" aria-hidden="true">
              <span />
            </span>
            <h1>开始一个新的编程任务</h1>
            <p>
              设置后端可以访问的 Workspace，然后告诉 Agent 需要完成什么。
              会话会在发送第一条消息时创建。
            </p>
            {tasksError ? (
              <p className="navigation-error" role="status">
                {tasksError}
              </p>
            ) : null}
          </div>
        ) : (
          <ConversationFeed
            tasks={tasks}
            selectedTaskId={selectedTaskId}
            isLoading={tasksLoading}
            error={tasksError}
            onTaskSelected={onTaskSelected}
            onRetry={onRetryTasks}
          />
        )}
      </div>

      <div className="composer-region">
        <Composer
          sessionId={activeConversation?.sessionId ?? null}
          latestTask={latestTask}
          latestTaskId={latestTaskId}
          latestTaskStatus={latestTaskStatus}
          defaultWorkspace={defaultWorkspace}
          disabled={navigationLoading}
          onSessionCreated={onSessionCreated}
          onTaskCreated={onTaskCreated}
        />
      </div>
    </section>
  );
}
