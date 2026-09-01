import type { Task } from "../api/contracts";
import { formatInspectorTime } from "./inspectorFormatters";

interface TaskDetailsProps {
  readonly task: Task;
}

export function TaskDetails({ task }: TaskDetailsProps) {
  return (
    <section className="inspector-section" aria-labelledby="task-info-title">
      <div className="inspector-section-title">
        <h3 id="task-info-title">任务信息</h3>
      </div>
      <dl className="task-info-list">
        <div>
          <dt>Task ID</dt>
          <dd>{task.id}</dd>
        </div>
        <div>
          <dt>Workspace</dt>
          <dd>{task.workspace}</dd>
        </div>
        <div>
          <dt>Created</dt>
          <dd>{formatInspectorTime(task.created_at)}</dd>
        </div>
        <div>
          <dt>Started</dt>
          <dd>{formatInspectorTime(task.started_at)}</dd>
        </div>
        <div>
          <dt>Finished</dt>
          <dd>{formatInspectorTime(task.finished_at)}</dd>
        </div>
        {task.error ? (
          <div className="task-detail-alert is-error">
            <dt>Error</dt>
            <dd>{task.error}</dd>
          </div>
        ) : null}
        {task.termination_reason ? (
          <div className="task-detail-alert is-termination">
            <dt>Termination</dt>
            <dd>{task.termination_reason}</dd>
          </div>
        ) : null}
      </dl>
    </section>
  );
}
