import type {
  SessionSummary,
  TaskStatus,
} from "../api/contracts";

interface ConversationSidebarProps {
  readonly isOpen: boolean;
  readonly isDrawer: boolean;
  readonly sessions: SessionSummary[];
  readonly selectedSessionId: string | null;
  readonly selectedTaskStatus: TaskStatus | null;
  readonly isLoading: boolean;
  readonly error: string | null;
  readonly onNewConversation: () => void;
  readonly onSessionSelected: (session: SessionSummary) => void;
  readonly onClose: () => void;
  readonly onRetry: () => void;
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  );
}

function formatConversationUpdatedAt(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "时间未知";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

export function ConversationSidebar({
  isOpen,
  isDrawer,
  sessions,
  selectedSessionId,
  selectedTaskStatus,
  isLoading,
  error,
  onNewConversation,
  onSessionSelected,
  onClose,
  onRetry,
}: ConversationSidebarProps) {
  return (
    <aside
      className="conversation-sidebar"
      id="conversation-sidebar"
      aria-label="会话列表"
      aria-hidden={!isOpen}
      aria-modal={isDrawer && isOpen ? "true" : undefined}
      role={isDrawer ? "dialog" : undefined}
      tabIndex={isDrawer ? -1 : undefined}
    >
      <div className="sidebar-brand">
        <BrandMark />
        <div>
          <strong>Coding Agent</strong>
          <span>Local workspace</span>
        </div>
        <button
          className="drawer-close-button sidebar-drawer-close"
          type="button"
          onClick={onClose}
          aria-label="关闭会话抽屉"
        >
          <span aria-hidden="true">×</span>
        </button>
      </div>

      <button
        className="new-conversation-button"
        type="button"
        onClick={onNewConversation}
      >
        <span aria-hidden="true">＋</span>
        新建会话
      </button>

      <div className="sidebar-section-heading">
        <span>会话</span>
      </div>

      <nav className="conversation-list" aria-label="最近会话">
        {isLoading && sessions.length === 0 ? (
          <div className="sidebar-loading" role="status">
            <span className="sidebar-spinner" aria-hidden="true" />
            正在载入会话
          </div>
        ) : null}

        {error ? (
          <div className="sidebar-list-error" role="status">
            <p>{error}</p>
            <button type="button" onClick={onRetry}>
              重新加载
            </button>
          </div>
        ) : null}

        {!isLoading && error === null && sessions.length === 0 ? (
          <p className="sidebar-empty">发送第一条消息后，会话会显示在这里。</p>
        ) : null}

        {sessions.map((session) => {
          const isSelected = session.id === selectedSessionId;
          const status =
            isSelected && selectedTaskStatus !== null
              ? selectedTaskStatus
              : session.latest_task_status;
          const updatedAtText = formatConversationUpdatedAt(session.updated_at);
          return (
            <button
              className={`conversation-item${isSelected ? " is-active" : ""}`}
              type="button"
              key={session.id}
              onClick={() => onSessionSelected(session)}
              aria-current={isSelected ? "page" : undefined}
            >
              <span className="conversation-title">{session.title}</span>
              <span className="conversation-meta">
                <span
                  className={`conversation-status-dot status-${status.toLowerCase()}`}
                  aria-hidden="true"
                />
                <span>{status}</span>
                <span aria-hidden="true">·</span>
                <time
                  className="conversation-updated-at"
                  dateTime={session.updated_at}
                  aria-label={`最近更新：${updatedAtText}`}
                >
                  {updatedAtText}
                </time>
              </span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footnote">单机原型</div>
    </aside>
  );
}
