import {
  MessageRole,
  type AgentStep,
  type CommandApproval,
  type Message,
  type ToolCall,
} from "../api/contracts";
import { formatInspectorTime } from "./inspectorFormatters";
import { MarkdownContent } from "./MarkdownContent";
import { ToolCallCard } from "./ToolCallCard";

interface StepCardProps {
  readonly step: AgentStep;
  readonly messages: Message[];
  readonly toolCalls: ToolCall[];
  readonly commandApprovals: CommandApproval[];
  readonly onApprovalDecisionRecorded: () => void;
}

export function StepCard({
  step,
  messages,
  toolCalls,
  commandApprovals,
  onApprovalDecisionRecorded,
}: StepCardProps) {
  const assistantMessages = messages.filter(
    (message) => message.role === MessageRole.ASSISTANT,
  );

  return (
    <article className="step-card">
      <header className="step-card-header">
        <div>
          <span className="step-number">Step {step.step_number}</span>
          <time>{formatInspectorTime(step.started_at)}</time>
        </div>
        <span className={`step-status status-${step.status.toLowerCase()}`}>
          {step.status}
        </span>
      </header>

      {step.error ? <p className="step-error">{step.error}</p> : null}

      <div className="step-events">
        {assistantMessages.length === 0 ? (
          <p className="step-waiting">等待 Assistant 动作。</p>
        ) : null}
        {assistantMessages.map((message) => {
          const messageToolCalls = toolCalls.filter(
            (toolCall) => toolCall.assistant_message_id === message.id,
          );
          return (
            <section className="assistant-event" key={message.id}>
              <header>
                <span>
                  {messageToolCalls.length > 0
                    ? "Assistant ToolCalls"
                    : `Assistant · ${message.message_type}`}
                </span>
                <span>sequence {message.sequence}</span>
              </header>
              {message.content ? (
                <MarkdownContent content={message.content} />
              ) : null}
              {messageToolCalls.length > 0 ? (
                <div className="tool-call-list">
                  {messageToolCalls.map((toolCall) => {
                    const resultMessage =
                      messages.find(
                        (candidate) =>
                          candidate.tool_call_id === toolCall.id,
                      ) ?? null;
                    return (
                      <ToolCallCard
                        key={toolCall.id}
                        toolCall={toolCall}
                        resultMessage={resultMessage}
                        approval={
                          commandApprovals.find(
                            (approval) =>
                              approval.tool_call_id === toolCall.id,
                          ) ?? null
                        }
                        onApprovalDecisionRecorded={
                          onApprovalDecisionRecorded
                        }
                      />
                    );
                  })}
                </div>
              ) : null}
            </section>
          );
        })}
      </div>

      <footer className="step-card-footer">
        <span>完成时间</span>
        <time>{formatInspectorTime(step.finished_at)}</time>
      </footer>
    </article>
  );
}
