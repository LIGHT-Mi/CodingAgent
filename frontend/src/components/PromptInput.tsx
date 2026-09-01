interface PromptInputProps {
  readonly value: string;
  readonly disabled: boolean;
  readonly error: string | null;
  readonly onChange: (value: string) => void;
}

export function PromptInput({
  value,
  disabled,
  error,
  onChange,
}: PromptInputProps) {
  return (
    <>
      <label className="sr-only" htmlFor="task-prompt">
        任务描述
      </label>
      <textarea
        id="task-prompt"
        name="prompt"
        rows={3}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="告诉 Agent 需要完成什么……"
        aria-invalid={Boolean(error)}
        aria-describedby={error ? "task-prompt-error" : undefined}
        aria-errormessage={error ? "task-prompt-error" : undefined}
        disabled={disabled}
      />
      {error ? (
        <p className="field-error" id="task-prompt-error" role="alert">
          {error}
        </p>
      ) : null}
    </>
  );
}
