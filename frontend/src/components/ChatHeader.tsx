import type { Task } from "../api/contracts";

interface ChatHeaderProps {
  readonly title: string;
  readonly taskId: string | null;
  readonly task: Task | null;
  readonly leftSidebarCollapsed: boolean;
  readonly rightInspectorCollapsed: boolean;
  readonly onToggleLeftSidebar: () => void;
  readonly onToggleRightInspector: () => void;
}

export function ChatHeader({
  title,
  taskId,
  task,
  leftSidebarCollapsed,
  rightInspectorCollapsed,
  onToggleLeftSidebar,
  onToggleRightInspector,
}: ChatHeaderProps) {
  return (
    <header className="workspace-header">
      <div className="workspace-header-side">
        <button
          className="panel-toggle"
          id="toggle-left-panel"
          type="button"
          onClick={onToggleLeftSidebar}
          aria-label={
            leftSidebarCollapsed ? "展开会话侧栏" : "折叠会话侧栏"
          }
          aria-controls="conversation-sidebar"
          aria-expanded={!leftSidebarCollapsed}
          title={leftSidebarCollapsed ? "展开会话侧栏" : "折叠会话侧栏"}
        >
          <span aria-hidden="true">{leftSidebarCollapsed ? "»" : "«"}</span>
        </button>
      </div>

      <div className="workspace-title">
        <strong>{title}</strong>
        <span aria-live="polite" aria-atomic="true">
          {taskId === null
            ? "尚未创建 Session"
            : `Task ${taskId.slice(0, 8)} · ${task?.status ?? "LOADING"}`}
        </span>
      </div>

      <div className="workspace-header-side is-right">
        <button
          className="panel-toggle"
          id="toggle-right-panel"
          type="button"
          onClick={onToggleRightInspector}
          aria-label={
            rightInspectorCollapsed ? "展开执行检查器" : "折叠执行检查器"
          }
          aria-controls="agent-inspector"
          aria-expanded={!rightInspectorCollapsed}
          title={
            rightInspectorCollapsed ? "展开执行检查器" : "折叠执行检查器"
          }
        >
          <span aria-hidden="true">{rightInspectorCollapsed ? "«" : "»"}</span>
        </button>
      </div>
    </header>
  );
}
