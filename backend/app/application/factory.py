"""CLI 和 Web API 共用的应用组件装配入口。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from app.agent import RetryWaiter, RuntimePolicy, RuntimePolicyConfig
from app.agent.runtime import AgentRuntime
from app.api.task_service import TaskService
from app.api.workspace import WorkspaceValidator
from app.context import ContextLimits, ContextManager
from app.core.config import Settings, settings
from app.db.persistence import PersistenceService
from app.db.session import SessionLocal
from app.llm.factory import create_configured_llm_gateway
from app.llm.gateway import LLMGateway
from app.tools import (
    CommandExecutor,
    CommandResultBuilder,
    CommandSafetyPolicy,
    RunCommandTool,
    ToolRouter,
    WorkingDirectoryGuard,
    WorkspacePathGuard,
    create_local_tool_registry,
)


SessionFactory = Callable[[], Session]
LLMGatewayFactory = Callable[[], LLMGateway]


class ApplicationFactory:
    """根据同一份配置为一个数据库 Session 装配应用服务。"""

    def __init__(
        self,
        config: Settings = settings,
        *,
        session_factory: SessionFactory = SessionLocal,
        llm_gateway_factory: LLMGatewayFactory = create_configured_llm_gateway,
        allowed_workspace_root: str | Path | None = None,
    ) -> None:
        if not isinstance(config, Settings):
            raise TypeError("config must be a Settings")
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not callable(llm_gateway_factory):
            raise TypeError("llm_gateway_factory must be callable")

        self._config = config
        self._session_factory = session_factory
        self._llm_gateway_factory = llm_gateway_factory
        self._allowed_workspace_root = (
            config.ALLOWED_WORKSPACE_ROOT
            if allowed_workspace_root is None
            else allowed_workspace_root
        )

    @property
    def config(self) -> Settings:
        return self._config

    def create_db_session(self) -> Session:
        """创建由调用方负责关闭的数据库 Session。"""

        db = self._session_factory()
        if not isinstance(db, Session):
            close = getattr(db, "close", None)
            if callable(close):
                close()
            raise TypeError("session_factory must return a SQLAlchemy Session")
        return db

    @staticmethod
    def create_persistence_service(db: Session) -> PersistenceService:
        return PersistenceService(db)

    def create_context_manager(
        self,
        persistence: PersistenceService,
    ) -> ContextManager:
        return ContextManager(
            persistence,
            limits=ContextLimits(
                max_context_characters=(
                    self._config.MAX_LLM_CONTEXT_CHARACTERS
                ),
                max_tool_result_characters=(
                    self._config.MAX_CONTEXT_TOOL_RESULT_CHARACTERS
                ),
            ),
        )

    def create_llm_gateway(self) -> LLMGateway:
        gateway = self._llm_gateway_factory()
        if not isinstance(gateway, LLMGateway):
            raise TypeError("llm_gateway_factory must return an LLMGateway")
        return gateway

    def create_tool_router(self) -> ToolRouter:
        command_result_builder = CommandResultBuilder()
        command_tool = RunCommandTool(
            CommandExecutor(
                timeout_seconds=self._config.COMMAND_TIMEOUT_SECONDS,
                termination_grace_seconds=(
                    self._config.COMMAND_TERMINATION_GRACE_SECONDS
                ),
                max_output_bytes_per_stream=(
                    self._config.MAX_COMMAND_OUTPUT_BYTES_PER_STREAM
                ),
            ),
            command_result_builder,
        )
        return ToolRouter(
            create_local_tool_registry(command_tool),
            WorkspacePathGuard(),
            WorkingDirectoryGuard(),
            CommandSafetyPolicy(),
            command_result_builder,
        )

    def create_runtime(
        self,
        persistence: PersistenceService,
    ) -> AgentRuntime:
        return AgentRuntime(
            persistence,
            self.create_context_manager(persistence),
            self.create_llm_gateway(),
            self.create_tool_router(),
            RuntimePolicy(
                RuntimePolicyConfig(
                    max_agent_steps=self._config.MAX_AGENT_STEPS,
                    max_llm_retries=self._config.MAX_LLM_RETRIES,
                    retry_base_seconds=self._config.LLM_RETRY_BASE_SECONDS,
                    retry_max_seconds=self._config.LLM_RETRY_MAX_SECONDS,
                    loop_repeat_threshold=(
                        self._config.AGENT_LOOP_REPEAT_THRESHOLD
                    ),
                )
            ),
            RetryWaiter(),
        )

    def create_task_service(
        self,
        persistence: PersistenceService,
    ) -> TaskService:
        return TaskService(
            persistence,
            WorkspaceValidator(self._allowed_workspace_root),
            self.create_runtime(persistence),
        )
