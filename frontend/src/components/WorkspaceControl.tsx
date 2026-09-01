interface WorkspaceControlProps {
  readonly value: string;
  readonly disabled: boolean;
  readonly error: string | null;
  readonly onChange: (value: string) => void;
}

const WORKSPACE_EXAMPLE =
  "/Users/myx/Documents/GitHub/CodingAgent/examples/demo-project";

export function WorkspaceControl({
  value,
  disabled,
  error,
  onChange,
}: WorkspaceControlProps) {
  return (
    <>
      <div className="workspace-control">
        <label htmlFor="task-workspace">Workspace</label>
        <input
          id="task-workspace"
          name="workspace"
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={WORKSPACE_EXAMPLE}
          aria-invalid={Boolean(error)}
          aria-errormessage={error ? "task-workspace-error" : undefined}
          aria-describedby={
            error
              ? "task-workspace-note task-workspace-error"
              : "task-workspace-note"
          }
          autoComplete="off"
          spellCheck={false}
          disabled={disabled}
        />
        <span className="workspace-local-label">服务端本地目录</span>
      </div>
      {error ? (
        <p
          className="field-error workspace-error"
          id="task-workspace-error"
          role="alert"
        >
          {error}
        </p>
      ) : null}
    </>
  );
}
