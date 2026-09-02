import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.runtime import AgentRuntime
from app.api.conversation_service import ConversationService
from app.api.task_service import TaskService
from app.application import ApplicationFactory
from app.context.manager import ContextManager
from app.core.config import Settings
from app.db.persistence import PersistenceService
from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry
from app.tools import CODING_TOOL_SCHEMAS, ToolRouter


class _TaskSubmitter:
    def submit(self, task_id: str) -> object:
        del task_id
        return object()


class ApplicationFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.allowed_root = Path(self.temporary_directory.name)
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.addCleanup(self.engine.dispose)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.config = Settings(
            DATABASE_URL="sqlite+pysqlite:///:memory:",
            DEEPSEEK_API_KEY="test-secret",
            ALLOWED_WORKSPACE_ROOT=self.allowed_root,
            MAX_AGENT_STEPS=4,
            MAX_LLM_RETRIES=1,
            LLM_RETRY_BASE_SECONDS=0.5,
            LLM_RETRY_MAX_SECONDS=3,
            AGENT_LOOP_REPEAT_THRESHOLD=2,
            MAX_LLM_CONTEXT_CHARACTERS=500,
            MAX_CONTEXT_TOOL_RESULT_CHARACTERS=100,
            COMMAND_TIMEOUT_SECONDS=2,
            MAX_COMMAND_OUTPUT_BYTES_PER_STREAM=1024,
            COMMAND_TERMINATION_GRACE_SECONDS=0.1,
        )

    def test_factory_builds_cli_and_web_reusable_service_graph(self) -> None:
        gateway = LLMGateway(
            DeepSeekClient(
                api_key="test-secret",
                open_url=lambda request, timeout: None,
            ),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
        )
        factory = ApplicationFactory(
            self.config,
            session_factory=self.session_factory,
            llm_gateway_factory=lambda: gateway,
        )

        with factory.create_db_session() as db:
            persistence = factory.create_persistence_service(db)
            context_manager = factory.create_context_manager(persistence)
            tool_router = factory.create_tool_router()
            runtime = factory.create_runtime(persistence)
            task_service = factory.create_task_service(persistence)
            conversation_service = factory.create_conversation_service(
                persistence,
                _TaskSubmitter(),
            )

            self.assertIsInstance(persistence, PersistenceService)
            self.assertIsInstance(context_manager, ContextManager)
            self.assertIsInstance(tool_router, ToolRouter)
            self.assertIsInstance(runtime, AgentRuntime)
            self.assertIsInstance(task_service, TaskService)
            self.assertIsInstance(conversation_service, ConversationService)
            self.assertIs(runtime._persistence, persistence)
            self.assertEqual(
                runtime._runtime_policy.config.max_agent_steps,
                4,
            )
            self.assertEqual(
                runtime._runtime_policy.config.max_llm_retries,
                1,
            )
            self.assertEqual(
                task_service._workspace_validator.allowed_root,
                self.allowed_root.resolve(),
            )

    def test_factory_rejects_invalid_dependency_factories(self) -> None:
        with self.assertRaises(TypeError):
            ApplicationFactory(
                self.config,
                session_factory=None,  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            ApplicationFactory(
                self.config,
                llm_gateway_factory=None,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
