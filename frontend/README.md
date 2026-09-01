# Frontend

## Environment

The frontend requires Node.js 24.20.x. Install dependencies and copy the public environment example:

```bash
npm install
cp .env.example .env
```

`frontend/.env` must contain only the public API base URL:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Never place `DEEPSEEK_API_KEY`, `DATABASE_URL`, or backend `.env` values in the frontend.

## Commands

```bash
npm run dev
npm test
npm run build
```

The development UI runs at `http://localhost:5173`. The backend must allow that exact origin through `WEB_CORS_ALLOWED_ORIGINS`.
