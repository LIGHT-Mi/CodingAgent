import { TaskStatus, type Task } from "../api/contracts";
import { MarkdownContent } from "./MarkdownContent";

interface AgentTurnProps {
  readonly task: Task;
}

function getPendingAgentContent(task: Task): string {
  if (task.status === TaskStatus.PENDING) {
    return "任务正在等待执行。";
  }
  return "Agent 正在分析任务并执行必要操作。";
}

export function AgentTurn({ task }: AgentTurnProps) {
  return (
    <div className="chat-message agent-message">
      <div className="message-avatar agent-avatar" aria-hidden="true">
        A
      </div>
      <div className="message-body">
        <div className="agent-message-heading">
          <span className="message-author">Agent</span>
          <span className="task-status-pill">{task.status}</span>
        </div>
        {task.final_answer ? (
          <MarkdownContent content={task.final_answer} />
        ) : null}
        {task.error ? (
          <p className="agent-turn-error">任务执行失败：{task.error}</p>
        ) : null}
        {task.termination_reason ? (
          <p className="agent-turn-termination">
            任务已结束：{task.termination_reason}
          </p>
        ) : null}
        {!task.final_answer && !task.error && !task.termination_reason ? (
          <p>{getPendingAgentContent(task)}</p>
        ) : null}
      </div>
    </div>
  );
}
