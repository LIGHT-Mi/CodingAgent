# Coding Agent

一个自行实现核心 Agent 机制、能够在本地 Workspace 中完成真实编程任务的编程智能体。

- Git 仓库：<https://github.com/LIGHT-Mi/CodingAgent>
- 模型：DeepSeek 原生 Tool Calling
- 后端：Python、FastAPI、SQLAlchemy、PostgreSQL
- 前端：React、TypeScript、Vite

项目没有使用 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI
等 Agent 框架或 SDK。Agent Loop、模型动作解析、上下文管理、本地工具执行、
运行策略、循环终止和错误处理均由项目自行实现。

## 核心能力

Agent 可以完成以下闭环：

```text
理解用户任务
→ 查看和搜索项目
→ 读取真实源码
→ 运行测试或构建命令
→ 根据失败 Observation 定位问题
→ 精确修改文件
→ 回读修改结果
→ 再次运行验证
→ 返回最终答案
```

### 七个本地工具

| 工具 | 功能 |
| --- | --- |
| `list_files` | 稳定排序并列出目录直接子项 |
| `read_file` | 读取 Workspace 内的 UTF-8 文本文件 |
| `search_files` | 按字面文本搜索文件并返回路径、行号和匹配内容 |
| `create_file` | 独占创建新文件，不覆盖已有目标 |
| `write_file` | 安全地整体覆盖已有文本文件 |
| `edit_file` | 只在 `old_text` 唯一出现时进行精确替换 |
| `run_command` | 运行受策略限制的测试、构建和项目命令 |

文件工具和命令工具均在本地执行，不依赖 Code Interpreter、Files API 或
其他服务端托管的代码执行能力。

### 错误是 Observation

测试失败、编译失败、非零退出码、文件不存在、编辑文本不匹配、命令超时和
策略拒绝都是模型可以继续分析的 Observation，不会被简单等同为 Agent Runtime
失败。例如：

```text
run_command exit_code=1
→ ToolResult COMPLETED
→ 模型读取失败输出
→ edit_file 修复源码
→ read_file 回读
→ run_command exit_code=0
→ Task COMPLETED
```

### 上下文管理

- System Prompt 和当前 Task Prompt 始终保留。
- 历史会话以不可拆分的 Conversation Turn Block 管理。
- Assistant Tool Calls 及其全部 Tool Result 构成不可拆分的 Interaction Block。
- 过长 Tool Result 在发送模型前保留开头和结尾，并明确标记截断。
- 超出字符预算时先删除最旧历史 Block，基础消息仍超预算才返回 Context Overflow。
- 历史 Task 只传递 Prompt 和最终结论，不无限传递旧文件内容及命令输出。

### Runtime Policy

项目自行处理：

- LLM timeout、rate limit、临时网络错误和 InvalidAction 重试；
- 确定性指数退避；
- 最大 Agent Step；
- 连续工具循环指纹检测；
- Context Overflow；
- Fatal Tool/System Error；
- 协作式用户取消；
- Step、Message 和 ToolCall 生命周期闭合。

LLM 重试发生在同一个 Step、相同 Context 内，不重复保存 Assistant Message 或
ToolCall。

### Workspace 与命令安全

文件访问会拒绝：

- `../` 越界；
- Workspace 外绝对路径；
- Workspace 内 symlink 指向外部；
- 文件和目录类型不匹配。

`run_command` 使用结构化 argv、`shell=False`、固定超时、有界 stdout/stderr、
独立进程组和最小子进程环境。安全策略区分：

```text
ALLOW             自动允许的测试和项目命令
REQUIRE_APPROVAL  必须等待用户对当前命令作出一次性决定
REJECT            永久拒绝，用户批准也不能覆盖
```

批准请求绑定当前 Task、ToolCall、完整 argv、规范工作目录、有效期和 SHA-256
命令指纹；批准后仍会重新进行路径与安全校验。

### 多轮 Web UI 与执行时间线

三栏前端将运行数据映射为：

```text
左侧：Session 会话列表
中间：同一 Session 下的多轮 Task 对话
右侧：AgentStep、Message、ToolCall 和命令批准时间线
```

支持状态轮询、历史 Task 切换、URL 恢复、Markdown/GFM 渲染、左右栏折叠和
运行中取消。Session、Task、Step、Message、ToolCall 和批准请求均持久化到
PostgreSQL，便于解释 Agent 每一步做了什么。

## 运行环境

- Python 3.12
- PostgreSQL
- Node.js 24.20.x
- npm 11.12.x
- DeepSeek API Key

API Key 和数据库连接串只能写入未提交的 `backend/.env`，不得写入前端环境、
源码或演示材料。

## 快速启动

### 1. 克隆仓库

```bash
git clone https://github.com/LIGHT-Mi/CodingAgent.git
cd CodingAgent
```

### 2. 配置后端

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

在未提交的 `backend/.env` 中填写：

```dotenv
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/database_name
DEEPSEEK_API_KEY=your-api-key
ALLOWED_WORKSPACE_ROOT=/absolute/path/to/allowed/workspaces
WEB_CORS_ALLOWED_ORIGINS=["http://localhost:5173"]
COMMAND_APPROVAL_TIMEOUT_SECONDS=300
```

`ALLOWED_WORKSPACE_ROOT` 是后端允许用户选择的 Workspace 根目录。前端输入的是
后端所在机器能够访问的目录路径，不是浏览器上传目录。

### 3. 初始化原型数据库

```bash
cd backend
.venv/bin/python -m app.db.init_db
```

本项目是原型，不使用 Alembic。`init_db` 会删除演示数据库中的全部 Session、
Task、Step、Message、ToolCall 和命令批准记录后重新建表，不要对需要保留数据的
数据库执行。

### 4. 启动后端

```bash
cd backend
.venv/bin/python -m uvicorn app.web.main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000
```

Web API 文档：<http://127.0.0.1:8000/docs>

停止服务时使用 `Ctrl+C`；`Ctrl+Z` 只会挂起进程并继续占用端口。

### 5. 启动前端

另开终端：

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

`frontend/.env` 只保存公开的后端地址：

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

浏览器访问：<http://localhost:5173>

建议启动顺序：PostgreSQL → 后端 → 前端。如果前端地址发生变化，需要同步修改
`WEB_CORS_ALLOWED_ORIGINS`；后端拒绝 CORS 通配符 `*`。

## 演示 Workspace

仓库包含一个可复现的故障项目：

```text
examples/coding-agent-demo-workspace
```

初始测试命令：

```bash
cd examples/coding-agent-demo-workspace
python3 -m unittest discover -s tests -p 'test_*.py'
```

初始版本的 `multiply()` 错误地执行加法，预期有两个测试失败。可向 Agent 输入：

```text
请检查这个项目并运行现有测试，根据失败信息定位并修复问题。修改已有文件前先读取真实源码，优先使用精确编辑；修改后回读文件，并重新运行同一测试命令。最后总结失败原因、修改内容和实际验证结果。
```

完整录制分镜见 [项目两分钟演示脚本](docs/项目两分钟演示脚本.md)，可复制的提示词见
[演示用 Agent 指令合集](docs/演示用Agent指令合集.md)。

## CLI

项目保留复用同一应用装配的薄 CLI：

```bash
cd backend
.venv/bin/python -m app.cli --workspace .. "请说明这个项目做了什么"
```

正式的危险命令批准交互入口位于 Web API 和前端。

## 离线测试

后端测试使用内存数据库和模拟模型响应，不调用 DeepSeek：

```bash
cd backend
.venv/bin/python -m unittest discover -s ../tests -q
```

前端测试及生产构建：

```bash
cd frontend
npm test
npm run build
```

## 设计文档

- [项目需求](docs/PROJECT_REQUIREMENTS.md)
- [总体设计](docs/构建编程智能体V2-4.md)
- [分步实现计划](docs/构建编程智能体V2-4-实现计划.md)
- [前端多轮会话设计](docs/构建编程智能体V2-4-前端多轮会话设计.md)
- [在线纵向验收记录](docs/第10步-在线纵向验收记录.md)

## 安全边界

当前命令能力是“受策略限制的本地命令执行”，不是容器或操作系统级严格沙箱：

- `cwd` 位于 Workspace 内，不代表子进程只能访问 Workspace；
- 被执行的代码仍拥有当前操作系统用户权限；
- 项目代码和测试可能访问网络、启动其他进程或访问外部绝对路径；
- 不应将不可信仓库直接交给当前原型执行。

真正隔离需要容器或操作系统级文件系统、网络、进程和资源限制。
