import type { Task } from "../api/contracts";

interface InspectorHeaderProps {
  readonly task: Task | null;
  readonly isDrawer: boolean;
  readonly onClose: () => void;
}

export function InspectorHeader({
  task,
  isDrawer,
  onClose,
}: InspectorHeaderProps) {
  return (
    <header className="inspector-header">
      <div>
        <span>Execution</span>
        <h2>Agent 执行</h2>
      </div>
      {task ? (
        <span className={`inspector-status status-${task.status.toLowerCase()}`}>
          {task.status}
        </span>
      ) : null}
      {isDrawer ? (
        <button
          className="drawer-close-button inspector-drawer-close"
          type="button"
          onClick={onClose}
          aria-label="关闭执行抽屉"
        >
          <span aria-hidden="true">×</span>
        </button>
      ) : null}
    </header>
  );
}
