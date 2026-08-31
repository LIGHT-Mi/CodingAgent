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
The current polling API provides:

```text
POST /api/tasks
GET  /api/tasks/{task_id}
GET  /api/tasks/{task_id}/steps
GET  /api/tasks/{task_id}/messages
GET  /api/tasks/{task_id}/tool-calls
POST /api/tasks/{task_id}/cancel
```

For a separate frontend development server, configure an explicit JSON array
of allowed origins in `.env`; wildcard origins are rejected:

```dotenv
WEB_CORS_ALLOWED_ORIGINS=["http://localhost:5173"]
```
