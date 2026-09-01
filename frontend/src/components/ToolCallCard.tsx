import { useState } from "react";

import type { Message, ToolCall } from "../api/contracts";
import {
  formatArguments,
  formatInspectorTime,
} from "./inspectorFormatters";

const COLLAPSED_TEXT_THRESHOLD = 600;

interface ToolCallCardProps {
  readonly toolCall: ToolCall;
  readonly resultMessage: Message | null;
}

interface OutputBlockProps {
  readonly label: string;
  readonly value: string;
  readonly className?: string;
}

function OutputBlock({ label, value, className = "" }: OutputBlockProps) {
  const isLong = value.length > COLLAPSED_TEXT_THRESHOLD;
  const [isOpen, setIsOpen] = useState(!isLong);
  return (
    <details
      className={`tool-output-block${className ? ` ${className}` : ""}`}
      open={isOpen}
      onToggle={(event) => setIsOpen(event.currentTarget.open)}
    >
      <summary>
        {label}
        <span>{isLong ? `${value.length} 字符` : ""}</span>
      </summary>
      <pre>{value || "（空输出）"}</pre>
    </details>
  );
}

export function ToolCallCard({
  toolCall,
  resultMessage,
}: ToolCallCardProps) {
  const resultContent =
    resultMessage?.content ?? toolCall.result ?? toolCall.error;

  return (
    <article className="tool-call-card">
      <header className="tool-call-header">
        <div>
          <span className="tool-call-index">#{toolCall.call_index}</span>
          <strong>{toolCall.tool_name}</strong>
        </div>
        <span
          className={`tool-call-status status-${toolCall.status.toLowerCase()}`}
        >
          {toolCall.status}
        </span>
      </header>

      <details className="tool-arguments" open>
        <summary>参数</summary>
        <pre>{formatArguments(toolCall.arguments)}</pre>
      </details>

      {resultContent !== null ? (
        <OutputBlock
          label={
            resultMessage
              ? `Tool Result · sequence ${resultMessage.sequence}`
              : "Tool Result"
          }
          value={resultContent}
          className={toolCall.error ? "is-error" : ""}
        />
      ) : null}

      {toolCall.exit_code !== null ? (
        <div className="command-exit-code">
          <span>Exit code</span>
          <strong>{toolCall.exit_code}</strong>
        </div>
      ) : null}
      {toolCall.stdout !== null ? (
        <OutputBlock label="stdout" value={toolCall.stdout} />
      ) : null}
      {toolCall.stderr !== null ? (
        <OutputBlock
          label="stderr"
          value={toolCall.stderr}
          className={toolCall.stderr ? "is-error" : ""}
        />
      ) : null}

      {toolCall.error && resultContent !== toolCall.error ? (
        <p className="tool-call-error">{toolCall.error}</p>
      ) : null}

      <dl className="tool-call-times">
        <div>
          <dt>Started</dt>
          <dd>{formatInspectorTime(toolCall.started_at)}</dd>
        </div>
        <div>
          <dt>Finished</dt>
          <dd>{formatInspectorTime(toolCall.finished_at)}</dd>
        </div>
      </dl>
    </article>
  );
}
