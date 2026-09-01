interface SubmitOrCancelButtonProps {
  readonly isTaskActive: boolean;
  readonly isSubmitting: boolean;
  readonly isCancelling: boolean;
  readonly cancellationRequested: boolean;
  readonly disabled: boolean;
  readonly onCancel: () => void;
}

export function SubmitOrCancelButton({
  isTaskActive,
  isSubmitting,
  isCancelling,
  cancellationRequested,
  disabled,
  onCancel,
}: SubmitOrCancelButtonProps) {
  if (isTaskActive) {
    return (
      <button
        className="cancel-task-button"
        type="button"
        onClick={onCancel}
        disabled={disabled || isCancelling || cancellationRequested}
      >
        <span className="stop-icon" aria-hidden="true" />
        {isCancelling
          ? "正在请求取消"
          : cancellationRequested
            ? "已请求取消"
            : "取消当前任务"}
      </button>
    );
  }

  return (
    <button
      className="send-button"
      type="submit"
      disabled={disabled || isSubmitting}
      aria-label={isSubmitting ? "正在提交任务" : "发送任务"}
    >
      {isSubmitting ? (
        <span className="button-spinner" aria-hidden="true" />
      ) : (
        <span aria-hidden="true">↑</span>
      )}
    </button>
  );
}
