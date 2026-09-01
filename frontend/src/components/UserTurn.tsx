interface UserTurnProps {
  readonly content: string;
}

export function UserTurn({ content }: UserTurnProps) {
  return (
    <div className="chat-message user-message">
      <div className="message-avatar" aria-hidden="true">
        你
      </div>
      <div className="message-body">
        <div className="user-message-heading">
          <span className="message-author">你</span>
        </div>
        <p>{content}</p>
      </div>
    </div>
  );
}
