import type { Task } from "../api/contracts";
import { AgentTurn } from "./AgentTurn";
import { UserTurn } from "./UserTurn";

interface ConversationFeedProps {
  readonly tasks: Task[];
  readonly selectedTaskId: string | null;
  readonly isLoading: boolean;
  readonly error: string | null;
  readonly onTaskSelected: (taskId: string) => void;
  readonly onRetry: () => void;
}

function handleTurnKeyDown(
  event: React.KeyboardEvent<HTMLElement>,
  onSelect: () => void,
) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    onSelect();
  }
}

export function ConversationFeed({
  tasks,
  selectedTaskId,
  isLoading,
  error,
  onTaskSelected,
  onRetry,
}: ConversationFeedProps) {
  if (isLoading && tasks.length === 0) {
    return (
      <div className="conversation-loading" role="status">
        <span className="sidebar-spinner" aria-hidden="true" />
        正在载入会话内容
      </div>
    );
  }

  if (error && tasks.length === 0) {
    return (
      <div className="conversation-load-error" role="status">
        <p>{error}</p>
        <button type="button" onClick={onRetry}>
          重新加载会话
        </button>
      </div>
    );
  }

  return (
    <div
      className="message-thread"
      aria-live="polite"
      aria-busy={isLoading}
    >
      {error ? <p className="feed-refresh-error">{error}</p> : null}
      {tasks.map((task) => {
        const isSelected = task.id === selectedTaskId;
        return (
          <article
            className={`conversation-turn${isSelected ? " is-selected" : ""}`}
            key={task.id}
            role="button"
            tabIndex={0}
            aria-current={isSelected ? "true" : undefined}
            aria-label={`选择任务：${task.original_prompt}`}
            onClick={() => onTaskSelected(task.id)}
            onKeyDown={(event) =>
              handleTurnKeyDown(event, () => onTaskSelected(task.id))
            }
          >
            <UserTurn content={task.original_prompt} />
            <AgentTurn task={task} />
          </article>
        );
      })}
    </div>
  );
}
