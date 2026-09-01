# Coding Agent

这是一个自行实现 Agent Loop、上下文管理、本地文件工具、受策略限制的命令执行、运行策略和多轮 Web UI 的编程智能体原型。

## 运行环境

- Python 3.12
- PostgreSQL
- Node.js 24.20.x
- DeepSeek API Key

API Key 和数据库连接串只允许写入未提交的 `backend/.env`，不得写入前端环境文件或源码。

## 1. 配置并重建数据库

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

在 `backend/.env` 中配置：

```dotenv
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/database_name
DEEPSEEK_API_KEY=your-api-key
ALLOWED_WORKSPACE_ROOT=/absolute/path/to/allowed/workspaces
WEB_CORS_ALLOWED_ORIGINS=["http://localhost:5173"]
```

本项目是原型，不使用 Alembic。Schema 变化后通过以下命令重建：

```bash
cd backend
.venv/bin/python -m app.db.init_db
```

警告：该命令会删除当前数据库中的全部 Session、Task、Step、Message 和 ToolCall，然后重新建表。只应对原型或演示数据库执行。

## 2. 启动后端

```bash
cd backend
.venv/bin/python -m uvicorn app.web.main:app --reload --host 127.0.0.1 --port 8000
```

API 文档位于 `http://127.0.0.1:8000/docs`。

`ALLOWED_WORKSPACE_ROOT` 是用户可选择 Workspace 的允许根目录。前端输入的 Workspace 必须是后端所在机器可访问的目录，并位于该根目录中；它不是浏览器上传目录。

## 3. 配置并启动前端

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

`frontend/.env` 只包含公开的后端地址：

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

浏览器打开 `http://localhost:5173`。启动顺序建议为：PostgreSQL → 后端 → 前端。

如果前端地址发生变化，需要同步修改后端 `WEB_CORS_ALLOWED_ORIGINS`。禁止配置通配符 `*`。

## 4. 离线测试

后端测试使用内存数据库和模拟模型响应：

```bash
cd backend
.venv/bin/python -m unittest discover -s ../tests -q
```

前端测试使用模拟 HTTP API：

```bash
cd frontend
npm test
npm run build
```

这些测试不会调用 DeepSeek，不消耗 API 额度。

## 本地命令执行风险

`run_command` 使用结构化 argv、`shell=False`、允许策略、固定超时、有限输出和独立进程组，但它不是严格沙箱。

- `cwd` 被限制在 Task Workspace 内，不代表进程只能访问 Workspace。
- 被执行的 Python 代码、测试和构建脚本仍拥有当前操作系统用户的本地权限。
- 命令仍可能访问网络、启动其他进程或通过绝对路径访问外部文件。
- 不应把不可信仓库直接交给当前原型执行。

真正隔离需要容器或操作系统级文件系统、网络、进程和资源限制。
