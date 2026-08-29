"""编程智能体的最小命令行入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from sqlalchemy.orm import Session

from app.agent.contracts import TaskStatus
from app.agent.runtime import AgentRuntime
from app.api.task_service import TaskService
from app.api.workspace import WorkspaceValidator
from app.context.manager import ContextManager
from app.core.config import settings
from app.db.persistence import PersistenceService
from app.db.session import SessionLocal
from app.llm.factory import create_configured_llm_gateway
from app.llm.gateway import LLMGateway
from app.tools import (
    ToolRouter,
    WorkspacePathGuard,
    create_read_only_file_tool_registry,
)


SessionFactory = Callable[[], Session]
LLMGatewayFactory = Callable[[], LLMGateway]


def build_parser() -> argparse.ArgumentParser:
    """创建 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="在指定 Workspace 中执行一个编程任务。",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="任务使用的 Workspace 目录。",
    )
    parser.add_argument("prompt", help="交给编程智能体的任务描述。")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: SessionFactory = SessionLocal,
    llm_gateway_factory: LLMGatewayFactory = create_configured_llm_gateway,
    allowed_workspace_root: str | Path | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """解析参数、装配服务、执行任务，并返回进程退出码。"""

    args = build_parser().parse_args(argv)
    configured_root = (
        settings.ALLOWED_WORKSPACE_ROOT
        if allowed_workspace_root is None
        else allowed_workspace_root
    )

    try:
        with session_factory() as db:
            persistence = PersistenceService(db)
            context_manager = ContextManager(persistence)
            llm_gateway = llm_gateway_factory()
            agent_runtime = AgentRuntime(
                persistence,
                context_manager,
                llm_gateway,
                ToolRouter(
                    create_read_only_file_tool_registry(),
                    WorkspacePathGuard(),
                ),
                max_agent_steps=settings.MAX_AGENT_STEPS,
            )
            task_service = TaskService(
                persistence,
                WorkspaceValidator(configured_root),
                agent_runtime,
            )
            result = task_service.run(args.prompt, args.workspace)
    except Exception as exc:
        print(f"任务执行失败：{exc}", file=stderr)
        return 1

    if result.status is TaskStatus.COMPLETED:
        print(result.final_answer, file=stdout)
        return 0

    error = result.error or result.termination_reason or result.status.value
    print(f"任务执行失败：{error}", file=stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
