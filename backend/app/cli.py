"""编程智能体的最小命令行入口。"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import TextIO

from app.agent import CancellationToken
from app.agent.contracts import TaskStatus
from app.application import (
    ApplicationFactory,
    LLMGatewayFactory,
    SessionFactory,
)
from app.core.config import settings
from app.db.session import SessionLocal
from app.llm.factory import create_configured_llm_gateway


CLI_CANCELLATION_REASON = "USER_CANCELLED"


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
    application_factory = ApplicationFactory(
        settings,
        session_factory=session_factory,
        llm_gateway_factory=llm_gateway_factory,
        allowed_workspace_root=allowed_workspace_root,
    )

    cancellation_token = CancellationToken()
    try:
        with _translate_sigint_to_cancellation(cancellation_token):
            with application_factory.create_db_session() as db:
                persistence = application_factory.create_persistence_service(db)
                task_service = application_factory.create_task_service(
                    persistence,
                )
                result = task_service.run_task_and_wait(
                    args.prompt,
                    args.workspace,
                    cancellation_token,
                )
    except Exception as exc:
        print(f"任务执行失败：{exc}", file=stderr)
        return 1

    if result.status is TaskStatus.COMPLETED:
        print(result.final_answer, file=stdout)
        return 0

    if result.status is TaskStatus.CANCELLED:
        reason = result.termination_reason or CLI_CANCELLATION_REASON
        print(f"任务已取消：{reason}", file=stderr)
        return 1

    error = result.error or result.termination_reason or result.status.value
    print(f"任务执行失败：{error}", file=stderr)
    return 1


@contextmanager
def _translate_sigint_to_cancellation(
    cancellation_token: CancellationToken,
) -> Iterator[None]:
    """运行期间把 Ctrl+C 转换为协作式取消，并在结束后恢复处理器。"""

    if not isinstance(cancellation_token, CancellationToken):
        raise TypeError("cancellation_token must be a CancellationToken")

    previous_handler = signal.getsignal(signal.SIGINT)

    def request_cancellation(
        signum: int,
        frame: FrameType | None,
    ) -> None:
        del signum, frame
        cancellation_token.cancel(CLI_CANCELLATION_REASON)

    signal.signal(signal.SIGINT, request_cancellation)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
