import type {
  AgentStep,
  CommandApproval,
  Message,
  ToolCall,
} from "../api/contracts";
import { StepCard } from "./StepCard";

interface AgentTimelineProps {
  readonly steps: AgentStep[];
  readonly messages: Message[];
  readonly toolCalls: ToolCall[];
  readonly commandApprovals: CommandApproval[];
  readonly onApprovalDecisionRecorded: () => void;
}

export function AgentTimeline({
  steps,
  messages,
  toolCalls,
  commandApprovals,
  onApprovalDecisionRecorded,
}: AgentTimelineProps) {
  return (
    <section className="agent-timeline" aria-labelledby="timeline-title">
      <div className="inspector-section-title timeline-heading">
        <h3 id="timeline-title">执行时间线</h3>
        <span>{steps.length} Steps</span>
      </div>

      {steps.length === 0 ? (
        <p className="timeline-empty">当前 Task 尚未创建执行步骤。</p>
      ) : (
        <div className="step-card-list">
          {steps.map((step) => (
            <StepCard
              key={step.id}
              step={step}
              messages={messages.filter(
                (message) => message.step_id === step.id,
              )}
              toolCalls={toolCalls.filter(
                (toolCall) => toolCall.step_id === step.id,
              )}
              commandApprovals={commandApprovals.filter(
                (approval) => approval.step_id === step.id,
              )}
              onApprovalDecisionRecorded={onApprovalDecisionRecorded}
            />
          ))}
        </div>
      )}
    </section>
  );
}
