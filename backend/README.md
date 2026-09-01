# Backend

## Environment

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Fill in `DATABASE_URL` and `DEEPSEEK_API_KEY` in the untracked `.env`, then
initialize the prototype database:

```bash
.venv/bin/python -m app.db.init_db
```

## CLI

```bash
.venv/bin/python -m app.cli --workspace .. "Describe this project"
```

## Web API

Start the FastAPI application with:

```bash
.venv/bin/python -m uvicorn app.web.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
The multi-turn polling API provides:

```text
POST /api/sessions
GET  /api/sessions
GET  /api/sessions/{session_id}
GET  /api/sessions/{session_id}/tasks
POST /api/sessions/{session_id}/tasks
GET  /api/tasks/{task_id}
GET  /api/tasks/{task_id}/snapshot
GET  /api/tasks/{task_id}/steps
GET  /api/tasks/{task_id}/messages
GET  /api/tasks/{task_id}/tool-calls
GET  /api/tasks/{task_id}/command-approvals
POST /api/tasks/{task_id}/command-approvals/{approval_id}/decision
POST /api/tasks/{task_id}/cancel
```

需要批准的命令会保持 ToolCall 为 `PENDING`，直到用户通过 Web API 或前端明确允许、拒绝，或者请求超过 `COMMAND_APPROVAL_TIMEOUT_SECONDS`。决定接口只接收决定和页面展示的命令指纹；argv 与规范工作目录以服务端持久化记录为准。

For a separate frontend development server, configure an explicit JSON array
of allowed origins in `.env`; wildcard origins are rejected:

```dotenv
WEB_CORS_ALLOWED_ORIGINS=["http://localhost:5173"]
```
