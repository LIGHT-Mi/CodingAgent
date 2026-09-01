import type { Task, TaskSnapshot } from "../api/contracts";
import { AgentTimeline } from "./AgentTimeline";
import { InspectorHeader } from "./InspectorHeader";
import { TaskDetails } from "./TaskDetails";

interface AgentInspectorProps {
  readonly isOpen: boolean;
  readonly isDrawer: boolean;
  readonly selectedTaskId: string | null;
  readonly fallbackTask: Task | null;
  readonly snapshot: TaskSnapshot | null;
  readonly isLoading: boolean;
  readonly error: string | null;
  readonly notFound: boolean;
  readonly onClose: () => void;
  readonly onRetry: () => void;
}

export function AgentInspector({
  isOpen,
  isDrawer,
  selectedTaskId,
  fallbackTask,
  snapshot,
  isLoading,
  error,
  notFound,
  onClose,
  onRetry,
}: AgentInspectorProps) {
  const selectedTask = snapshot?.task ?? fallbackTask;

  return (
    <aside
      className="agent-inspector"
      id="agent-inspector"
      aria-label="Agent 执行检查器"
      aria-hidden={!isOpen}
      aria-modal={isDrawer && isOpen ? "true" : undefined}
      role={isDrawer ? "dialog" : undefined}
      tabIndex={isDrawer ? -1 : undefined}
      aria-busy={isLoading}
    >
      <InspectorHeader
        task={selectedTask}
        isDrawer={isDrawer}
        onClose={onClose}
      />

      {selectedTaskId === null ? (
        <div className="inspector-empty">
          <span aria-hidden="true">⌁</span>
          <h3>尚无执行记录</h3>
          <p>发送任务后，这里将展示该轮 Task 的状态与执行时间线。</p>
        </div>
      ) : (
        <div className="inspector-content">
          {error ? (
            <div
              className={`inspector-load-error${notFound ? " is-not-found" : ""}`}
              role="status"
            >
              <p>{error}</p>
              <button type="button" onClick={onRetry}>
                重新加载
              </button>
            </div>
          ) : null}

          {snapshot ? (
            <AgentTimeline
              steps={snapshot.steps}
              messages={snapshot.messages}
              toolCalls={snapshot.tool_calls}
              commandApprovals={snapshot.command_approvals}
              onApprovalDecisionRecorded={onRetry}
            />
          ) : isLoading ? (
            <div className="inspector-loading" role="status">
              <span className="sidebar-spinner" aria-hidden="true" />
              正在载入执行快照
            </div>
          ) : null}

          {selectedTask ? <TaskDetails task={selectedTask} /> : null}
        </div>
      )}
    </aside>
  );
}
