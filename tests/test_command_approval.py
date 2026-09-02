import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import (
    CancellationToken,
    RetryWaiter,
    RuntimePolicy,
    RuntimePolicyConfig,
    TaskStatus,
    ToolCallStatus,
    ToolCallsAction,
    ToolCallRequest,
)
from app.agent.runtime import AgentRuntime
from app.approval import (
    CommandApprovalCoordinator,
    CommandApprovalDecision,
    CommandApprovalStatus,
)
from app.approval.service import CommandApprovalFingerprintMismatchError
from app.approval.service import (
    CommandApprovalService,
    CommandApprovalNotActiveError,
    CommandApprovalNotFoundError,
)
from app.context import ContextLimits, ContextManager
from app.application import ApplicationFactory, TaskRunner
from app.core.config import Settings
from app.db.base import Base
from app.db.persistence import PersistenceService
from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry
from app.tools import (
    CODING_TOOL_SCHEMAS,
    CommandApprovalRequirement,
    CommandExecutor,
    RunCommandTool,
    ToolRouter,
    WorkspacePathGuard,
    create_local_tool_registry,
)
from app.web.main import create_web_app
from test_web_api import call_asgi


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class CommandApprovalWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        database_path = self.root / "approval.sqlite"
        self.engine = create_engine(
            f"sqlite+pysqlite:///{database_path}",
            connect_args={"check_same_thread": False},
        )
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.coordinator = CommandApprovalCoordinator()

    def test_approved_exact_command_waits_then_executes_once(self) -> None:
        runtime, task_id, executor, result_box = self._start_runtime(
            ["echo", "approved once"]
        )

        approval = self._wait_for_pending_approval(task_id)
        self.assertEqual(executor.call_count, 0)
        self.assertEqual(approval.command, ["echo", "approved once"])
        self.assertEqual(approval.cwd, str(self.workspace))
        self._assert_tool_call_is_pending(task_id)

        with self.session_factory() as api_db:
            decided = CommandApprovalService(
                PersistenceService(api_db),
                self.coordinator,
                2,
            ).decide(
                task_id=task_id,
                request_id=approval.id,
                decision=CommandApprovalDecision.APPROVE,
                command_fingerprint=approval.command_fingerprint,
            )
        self.assertEqual(decided.status, CommandApprovalStatus.APPROVED.value)

        runtime.join(3)
        self.assertFalse(runtime.is_alive())
        self.assertEqual(
            result_box["result"].status,
            TaskStatus.COMPLETED,
            result_box["result"],
        )
        self.assertEqual(executor.call_count, 1)
        with self.session_factory() as verification_db:
            persistence = PersistenceService(verification_db)
            stored = persistence.load_command_approval_requests(task_id)[0]
            tool_call = persistence.load_tool_calls(task_id)[0]
            self.assertEqual(stored.status, CommandApprovalStatus.CONSUMED.value)
            self.assertIsNotNone(stored.consumed_at)
            self.assertEqual(tool_call.status, ToolCallStatus.COMPLETED.value)
            self.assertEqual(
                tool_call.result_metadata["approval_request_id"],
                approval.id,
            )
        with self.session_factory() as api_db:
            with self.assertRaises(CommandApprovalNotActiveError):
                CommandApprovalService(
                    PersistenceService(api_db),
                    self.coordinator,
                    2,
                ).decide(
                    task_id=task_id,
                    request_id=approval.id,
                    decision=CommandApprovalDecision.APPROVE,
                    command_fingerprint=approval.command_fingerprint,
                )

    def test_rejection_and_fingerprint_tampering_never_execute(self) -> None:
        runtime, task_id, executor, result_box = self._start_runtime(
            ["rm", "missing.txt"]
        )
        approval = self._wait_for_pending_approval(task_id)

        with self.session_factory() as api_db:
            service = CommandApprovalService(
                PersistenceService(api_db),
                self.coordinator,
                2,
            )
            with self.assertRaises(CommandApprovalNotFoundError):
                service.decide(
                    task_id="another-task",
                    request_id=approval.id,
                    decision=CommandApprovalDecision.APPROVE,
                    command_fingerprint=approval.command_fingerprint,
                )
            with self.assertRaises(CommandApprovalFingerprintMismatchError):
                service.decide(
                    task_id=task_id,
                    request_id=approval.id,
                    decision=CommandApprovalDecision.APPROVE,
                    command_fingerprint="sha256:tampered",
                )
            still_pending = service.list_for_task(task_id)[0]
            self.assertEqual(
                still_pending.status,
                CommandApprovalStatus.PENDING.value,
            )
            service.decide(
                task_id=task_id,
                request_id=approval.id,
                decision=CommandApprovalDecision.REJECT,
                command_fingerprint=approval.command_fingerprint,
            )

        runtime.join(3)
        self.assertFalse(runtime.is_alive())
        self.assertEqual(
            result_box["result"].status,
            TaskStatus.COMPLETED,
            result_box["result"],
        )
        self.assertEqual(executor.call_count, 0)
        with self.session_factory() as verification_db:
            persistence = PersistenceService(verification_db)
            stored = persistence.load_command_approval_requests(task_id)[0]
            tool_call = persistence.load_tool_calls(task_id)[0]
            self.assertEqual(stored.status, CommandApprovalStatus.REJECTED.value)
            self.assertEqual(tool_call.status, ToolCallStatus.REJECTED.value)
            self.assertIsNone(tool_call.started_at)
            self.assertEqual(
                tool_call.result_metadata["approval_status"],
                CommandApprovalStatus.REJECTED.value,
            )

    def test_cancellation_and_changed_cwd_never_execute(self) -> None:
        runtime, task_id, executor, result_box = self._start_runtime(
            ["rm", "cancelled.txt"]
        )
        approval = self._wait_for_pending_approval(task_id)
        token = result_box["cancellation_token"]
        assert isinstance(token, CancellationToken)
        token.cancel("USER_CANCELLED")
        runtime.join(3)
        self.assertFalse(runtime.is_alive())
        self.assertEqual(result_box["result"].status, TaskStatus.CANCELLED)
        self.assertEqual(executor.call_count, 0)
        with self.session_factory() as verification_db:
            persistence = PersistenceService(verification_db)
            stored = persistence.get_command_approval_request(approval.id)
            self.assertEqual(stored.status, CommandApprovalStatus.CANCELLED.value)
            tool_call = persistence.load_tool_calls(task_id)[0]
            self.assertEqual(tool_call.status, ToolCallStatus.ERROR.value)
            self.assertTrue(tool_call.result_metadata["interrupted"])

        approved_cwd = self.workspace / "temporary-cwd"
        approved_cwd.mkdir()
        changed_runtime, changed_task_id, changed_executor, changed_box = (
            self._start_runtime(
                ["echo", "revalidate"],
                cwd="temporary-cwd",
            )
        )
        changed_approval = self._wait_for_pending_approval(changed_task_id)
        approved_cwd.rmdir()
        with self.session_factory() as api_db:
            CommandApprovalService(
                PersistenceService(api_db),
                self.coordinator,
                2,
            ).decide(
                task_id=changed_task_id,
                request_id=changed_approval.id,
                decision=CommandApprovalDecision.APPROVE,
                command_fingerprint=changed_approval.command_fingerprint,
            )
        changed_runtime.join(3)
        self.assertFalse(changed_runtime.is_alive())
        self.assertEqual(changed_box["result"].status, TaskStatus.COMPLETED)
        self.assertEqual(changed_executor.call_count, 0)
        with self.session_factory() as verification_db:
            persistence = PersistenceService(verification_db)
            stored = persistence.get_command_approval_request(
                changed_approval.id
            )
            self.assertEqual(
                stored.status,
                CommandApprovalStatus.INVALIDATED.value,
            )
            tool_call = persistence.load_tool_calls(changed_task_id)[0]
            self.assertEqual(tool_call.status, ToolCallStatus.ERROR.value)
            self.assertTrue(
                tool_call.result_metadata["approval_revalidation_failed"]
            )
            self.assertIsNone(tool_call.started_at)
            self.assertEqual(
                tool_call.result_metadata["approval_status"],
                CommandApprovalStatus.INVALIDATED.value,
            )

    def test_expired_request_and_permanent_rejection_create_no_process(self) -> None:
        runtime, task_id, executor, result_box = self._start_runtime(
            ["echo", "too late"],
            approval_timeout=0.05,
        )
        self._wait_for_pending_approval(task_id)
        runtime.join(3)
        self.assertFalse(runtime.is_alive())
        self.assertEqual(result_box["result"].status, TaskStatus.COMPLETED)
        self.assertEqual(executor.call_count, 0)
        with self.session_factory() as verification_db:
            persistence = PersistenceService(verification_db)
            approval = persistence.load_command_approval_requests(task_id)[0]
            tool_call = persistence.load_tool_calls(task_id)[0]
            self.assertEqual(approval.status, CommandApprovalStatus.EXPIRED.value)
            self.assertEqual(tool_call.status, ToolCallStatus.REJECTED.value)
            self.assertIsNone(tool_call.started_at)

        permanent_runtime, permanent_task_id, permanent_executor, permanent_box = (
            self._start_runtime(["sudo", "pytest"])
        )
        permanent_runtime.join(3)
        self.assertFalse(permanent_runtime.is_alive())
        self.assertEqual(permanent_box["result"].status, TaskStatus.COMPLETED)
        self.assertEqual(permanent_executor.call_count, 0)
        with self.session_factory() as verification_db:
            persistence = PersistenceService(verification_db)
            self.assertEqual(
                persistence.load_command_approval_requests(permanent_task_id),
                [],
            )
            tool_call = persistence.load_tool_calls(permanent_task_id)[0]
            self.assertEqual(tool_call.status, ToolCallStatus.REJECTED.value)

    def test_web_api_lists_and_decides_the_persisted_exact_request(self) -> None:
        database_url = str(self.engine.url)
        gateway = LLMGateway(
            DeepSeekClient(
                api_key="test-secret",
                open_url=lambda request, timeout: None,
            ),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
        )
        factory = ApplicationFactory(
            Settings(
                DATABASE_URL=database_url,
                DEEPSEEK_API_KEY="test-secret",
                ALLOWED_WORKSPACE_ROOT=self.root,
                COMMAND_APPROVAL_TIMEOUT_SECONDS=2,
            ),
            session_factory=self.session_factory,
            llm_gateway_factory=lambda: gateway,
            command_approval_coordinator=self.coordinator,
        )
        runner = TaskRunner(factory, max_workers=1)
        self.addCleanup(runner.shutdown)
        app = create_web_app(factory, runner)

        with self.session_factory() as db:
            persistence = PersistenceService(db)
            _, task = persistence.create_session_with_task(
                title="API approval",
                original_prompt="approve command",
                workspace=str(self.workspace),
            )
            persistence.start_task(task.id)
            step = persistence.create_agent_step(task.id, 0)
            action = ToolCallsAction(
                tool_calls=(
                    ToolCallRequest(
                        tool_call_id="provider-api-command",
                        tool_name="run_command",
                        arguments={"command": ["echo", "from api"], "cwd": "."},
                    ),
                )
            )
            _, records = persistence.save_tool_calls_action(
                task.id,
                step.id,
                action,
            )
            requirement = factory.create_tool_router().prepare(
                action.tool_calls[0],
                self.workspace,
            )
            self.assertIsInstance(requirement, CommandApprovalRequirement)
            assert isinstance(requirement, CommandApprovalRequirement)
            approval = factory.create_command_approval_service(
                persistence
            ).create_request(
                task_id=task.id,
                step_id=step.id,
                tool_call_id=records[0].id,
                requirement=requirement,
            )

        listed = asyncio.run(
            call_asgi(
                app,
                "GET",
                f"/api/tasks/{task.id}/command-approvals",
            )
        )
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json[0]["command"], ["echo", "from api"])
        self.assertEqual(listed.json[0]["cwd"], str(self.workspace))

        decided = asyncio.run(
            call_asgi(
                app,
                "POST",
                f"/api/tasks/{task.id}/command-approvals/{approval.id}/decision",
                json_body={
                    "decision": "APPROVE",
                    "command_fingerprint": approval.command_fingerprint,
                },
            )
        )
        self.assertEqual(decided.status_code, 200)
        self.assertEqual(decided.json["status"], "APPROVED")
        self.assertNotIn("DEEPSEEK_API_KEY", decided.text)

    def _start_runtime(
        self,
        command: list[str],
        *,
        approval_timeout: float = 2,
        cwd: str = ".",
    ):
        runtime_db = self.session_factory()
        self.addCleanup(runtime_db.close)
        persistence = PersistenceService(runtime_db)
        _, task = persistence.create_session_with_task(
            title="命令批准测试",
            original_prompt="执行需要批准的命令后回答",
            workspace=str(self.workspace),
        )
        persistence.start_task(task.id)
        response_number = 0

        def open_url(request, *, timeout):
            nonlocal response_number
            del request, timeout
            response_number += 1
            if response_number == 1:
                return FakeHTTPResponse(self._tool_call_response(command, cwd))
            return FakeHTTPResponse(self._final_response())

        executor = CommandExecutor(
            timeout_seconds=2,
            termination_grace_seconds=0.1,
            max_output_bytes_per_stream=1024,
        )
        real_execute = executor.execute
        execute_patch = patch.object(executor, "execute", wraps=real_execute)
        execute_mock = execute_patch.start()
        self.addCleanup(execute_patch.stop)
        gateway = LLMGateway(
            DeepSeekClient(api_key="test-secret", open_url=open_url),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
        )
        runtime = AgentRuntime(
            persistence,
            ContextManager(persistence, ContextLimits(60_000, 12_000)),
            gateway,
            ToolRouter(
                create_local_tool_registry(RunCommandTool(executor)),
                WorkspacePathGuard(),
            ),
            RuntimePolicy(RuntimePolicyConfig()),
            RetryWaiter(lambda seconds: None),
            command_approval_service=CommandApprovalService(
                persistence,
                self.coordinator,
                approval_timeout,
            ),
        )
        cancellation_token = CancellationToken()
        result_box: dict[str, object] = {
            "cancellation_token": cancellation_token,
        }

        def run_runtime() -> None:
            result_box["result"] = runtime.run(task.id, cancellation_token)

        thread = threading.Thread(target=run_runtime, daemon=True)
        thread.start()
        return thread, task.id, execute_mock, result_box

    def _wait_for_pending_approval(self, task_id: str):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.session_factory() as db:
                requests = PersistenceService(
                    db
                ).load_command_approval_requests(task_id)
                if requests:
                    self.assertEqual(
                        requests[0].status,
                        CommandApprovalStatus.PENDING.value,
                    )
                    return requests[0]
            time.sleep(0.01)
        self.fail("runtime did not create a command approval request")

    def _assert_tool_call_is_pending(self, task_id: str) -> None:
        with self.session_factory() as db:
            tool_call = PersistenceService(db).load_tool_calls(task_id)[0]
            self.assertEqual(tool_call.status, ToolCallStatus.PENDING.value)
            self.assertIsNone(tool_call.started_at)

    @staticmethod
    def _tool_call_response(
        command: list[str],
        cwd: str = ".",
    ) -> dict[str, object]:
        return {
            "id": "response-tool",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "provider-command",
                                "type": "function",
                                "function": {
                                    "name": "run_command",
                                    "arguments": json.dumps(
                                        {"command": command, "cwd": cwd}
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
        }

    @staticmethod
    def _final_response() -> dict[str, object]:
        return {
            "id": "response-final",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "已处理命令结果。",
                    },
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
