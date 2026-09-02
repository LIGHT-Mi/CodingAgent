import json
import tempfile
import threading
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.agent import AgentStepStatus, TaskStatus
from app.application import (
    ApplicationFactory,
    TaskAlreadySubmittedError,
    TaskCancellationOutcome,
    TaskNotPendingError,
    TaskRunner,
    TaskRunnerShutdownError,
)
from app.core.config import Settings
from app.db.base import Base
from app.db.models.task import Task
from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry
from app.tools import CODING_TOOL_SCHEMAS


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class TaskRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        database_path = root / "task-runner.sqlite"
        self.engine = create_engine(
            f"sqlite+pysqlite:///{database_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        session_type = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.session_creation_threads: list[int] = []

        def tracked_session_factory():
            self.session_creation_threads.append(threading.get_ident())
            return session_type()

        self.session_factory = tracked_session_factory
        self.llm_started = threading.Event()
        self.release_llm = threading.Event()
        self.llm_call_count = 0

        def open_url(request, *, timeout):
            self.llm_call_count += 1
            self.llm_started.set()
            if not self.release_llm.wait(5):
                raise TimeoutError("test did not release fake LLM")
            return self._final_response("后台任务完成。")

        def gateway_factory() -> LLMGateway:
            return LLMGateway(
                DeepSeekClient(api_key="test-secret", open_url=open_url),
                ModelConfig(model="deepseek-v4-flash"),
                ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
            )

        self.application_factory = ApplicationFactory(
            Settings(
                DATABASE_URL=f"sqlite+pysqlite:///{database_path}",
                DEEPSEEK_API_KEY="test-secret",
                ALLOWED_WORKSPACE_ROOT=root,
                LLM_RETRY_BASE_SECONDS=0,
                LLM_RETRY_MAX_SECONDS=0,
            ),
            session_factory=self.session_factory,
            llm_gateway_factory=gateway_factory,
        )

    def test_submit_returns_before_worker_finishes_and_uses_worker_session(
        self,
    ) -> None:
        task_id = self._create_pending_task("后台执行")
        main_thread_id = threading.get_ident()
        runner = TaskRunner(self.application_factory, max_workers=1)
        self.addCleanup(runner.shutdown)

        future = runner.submit(task_id)

        self.assertFalse(future.done())
        self.assertTrue(self.llm_started.wait(2))
        self.assertEqual(self._load_task(task_id).status, TaskStatus.RUNNING.value)
        self.assertIn(task_id, runner.active_task_ids)
        self.assertTrue(
            any(
                thread_id != main_thread_id
                for thread_id in self.session_creation_threads
            )
        )

        self.release_llm.set()
        result = future.result(timeout=5)

        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(
            self._load_task(task_id).status,
            TaskStatus.COMPLETED.value,
        )
        self._wait_for_registry_cleanup(future)
        self.assertNotIn(task_id, runner.active_task_ids)

    def test_cancel_reaches_runtime_and_closes_task_and_step(self) -> None:
        task_id = self._create_pending_task("取消后台执行")
        runner = TaskRunner(self.application_factory, max_workers=1)
        self.addCleanup(runner.shutdown)
        future = runner.submit(task_id)
        self.assertTrue(self.llm_started.wait(2))

        first_cancel = runner.cancel(task_id)
        repeated_cancel = runner.cancel(task_id)
        unknown_cancel = runner.cancel("unknown-task")
        self.assertEqual(
            first_cancel.outcome,
            TaskCancellationOutcome.REQUESTED,
        )
        self.assertTrue(first_cancel.cancellation_requested)
        self.assertEqual(
            repeated_cancel.outcome,
            TaskCancellationOutcome.ALREADY_REQUESTED,
        )
        self.assertTrue(repeated_cancel.cancellation_requested)
        self.assertEqual(
            unknown_cancel.outcome,
            TaskCancellationOutcome.TASK_NOT_FOUND,
        )
        self.assertFalse(unknown_cancel.cancellation_requested)
        self.release_llm.set()
        result = future.result(timeout=5)

        self.assertEqual(result.status, TaskStatus.CANCELLED)
        with self.application_factory.create_db_session() as db:
            task = db.get(Task, task_id)
            self.assertEqual(task.status, TaskStatus.CANCELLED.value)
            self.assertEqual(task.termination_reason, "USER_CANCELLED")
            self.assertEqual(len(task.agent_steps), 1)
            self.assertEqual(
                task.agent_steps[0].status,
                AgentStepStatus.INTERRUPTED.value,
            )

    def test_queued_cancel_is_consumed_before_creating_a_step(self) -> None:
        first_task_id = self._create_pending_task("占用工作线程")
        second_task_id = self._create_pending_task("队列中取消")
        runner = TaskRunner(self.application_factory, max_workers=1)
        self.addCleanup(runner.shutdown)
        first_future = runner.submit(first_task_id)
        self.assertTrue(self.llm_started.wait(2))
        second_future = runner.submit(second_task_id)

        self.assertEqual(
            runner.cancel(second_task_id).outcome,
            TaskCancellationOutcome.REQUESTED,
        )
        self.release_llm.set()

        self.assertEqual(
            first_future.result(timeout=5).status,
            TaskStatus.COMPLETED,
        )
        second_result = second_future.result(timeout=5)
        self.assertEqual(second_result.status, TaskStatus.CANCELLED)
        with self.application_factory.create_db_session() as db:
            task = db.get(Task, second_task_id)
            self.assertEqual(task.status, TaskStatus.CANCELLED.value)
            self.assertEqual(task.agent_steps, [])
        self.assertEqual(self.llm_call_count, 1)

    def test_rejects_duplicate_non_pending_and_unknown_submissions(self) -> None:
        task_id = self._create_pending_task("重复提交")
        runner = TaskRunner(self.application_factory, max_workers=1)
        self.addCleanup(runner.shutdown)
        future = runner.submit(task_id)

        with self.assertRaises(TaskAlreadySubmittedError):
            runner.submit(task_id)
        with self.assertRaises(TaskNotPendingError):
            runner.submit("missing-task")

        self.assertTrue(self.llm_started.wait(2))
        self.release_llm.set()
        future.result(timeout=5)
        with self.assertRaises(TaskNotPendingError):
            runner.submit(task_id)

    def test_cancel_distinguishes_not_submitted_and_finished_tasks(self) -> None:
        pending_task_id = self._create_pending_task("尚未提交")
        runner = TaskRunner(self.application_factory, max_workers=1)
        self.addCleanup(runner.shutdown)

        pending_result = runner.cancel(pending_task_id)

        self.assertEqual(
            pending_result.outcome,
            TaskCancellationOutcome.TASK_NOT_ACTIVE,
        )
        self.assertEqual(pending_result.task_status, TaskStatus.PENDING)

        completed_task_id = self._create_pending_task("已经结束")
        future = runner.submit(completed_task_id)
        self.assertTrue(self.llm_started.wait(2))
        self.release_llm.set()
        self.assertEqual(
            future.result(timeout=5).status,
            TaskStatus.COMPLETED,
        )
        self._wait_for_registry_cleanup(future)

        finished_result = runner.cancel(completed_task_id)
        self.assertEqual(
            finished_result.outcome,
            TaskCancellationOutcome.TASK_FINISHED,
        )
        self.assertEqual(finished_result.task_status, TaskStatus.COMPLETED)
        self.assertFalse(finished_result.cancellation_requested)
        self.assertNotIn(
            completed_task_id,
            runner.cancellation_registry.registered_task_ids,
        )

    def test_completion_and_cancel_race_leave_no_registered_token(self) -> None:
        task_id = self._create_pending_task("完成与取消并发")
        runner = TaskRunner(self.application_factory, max_workers=1)
        self.addCleanup(runner.shutdown)
        future = runner.submit(task_id)
        self.assertTrue(self.llm_started.wait(2))
        race_start = threading.Barrier(3)
        cancellation_results = []

        def complete_llm() -> None:
            race_start.wait()
            self.release_llm.set()

        def request_cancel() -> None:
            race_start.wait()
            cancellation_results.append(runner.cancel(task_id))

        completion_thread = threading.Thread(target=complete_llm)
        cancellation_thread = threading.Thread(target=request_cancel)
        completion_thread.start()
        cancellation_thread.start()
        race_start.wait()
        completion_thread.join(timeout=2)
        cancellation_thread.join(timeout=2)
        result = future.result(timeout=5)
        self._wait_for_registry_cleanup(future)

        self.assertIn(
            result.status,
            {TaskStatus.COMPLETED, TaskStatus.CANCELLED},
        )
        self.assertEqual(len(cancellation_results), 1)
        self.assertIn(
            cancellation_results[0].outcome,
            {
                TaskCancellationOutcome.REQUESTED,
                TaskCancellationOutcome.TASK_FINISHED,
            },
        )
        self.assertNotIn(task_id, runner.active_task_ids)
        self.assertNotIn(
            task_id,
            runner.cancellation_registry.registered_task_ids,
        )
        self.assertEqual(
            runner.cancel(task_id).outcome,
            TaskCancellationOutcome.TASK_FINISHED,
        )

    def test_shutdown_requests_cancellation_and_rejects_new_tasks(self) -> None:
        task_id = self._create_pending_task("关闭时取消")
        runner = TaskRunner(self.application_factory, max_workers=1)
        future = runner.submit(task_id)
        self.assertTrue(self.llm_started.wait(2))

        shutdown_finished = threading.Event()

        def shutdown_runner() -> None:
            runner.shutdown(wait=True, cancel_running=True)
            shutdown_finished.set()

        shutdown_thread = threading.Thread(target=shutdown_runner)
        shutdown_thread.start()
        self.release_llm.set()
        self.assertTrue(shutdown_finished.wait(5))
        shutdown_thread.join(timeout=1)

        self.assertEqual(future.result(timeout=1).status, TaskStatus.CANCELLED)
        another_task_id = self._create_pending_task("关闭后提交")
        with self.assertRaises(TaskRunnerShutdownError):
            runner.submit(another_task_id)

    def _create_pending_task(self, prompt: str) -> str:
        with self.application_factory.create_db_session() as db:
            persistence = self.application_factory.create_persistence_service(db)
            service = self.application_factory.create_task_service(persistence)
            return service.create_task(prompt, self.workspace)

    def _load_task(self, task_id: str) -> Task:
        with self.application_factory.create_db_session() as db:
            return db.scalars(select(Task).where(Task.id == task_id)).one()

    @staticmethod
    def _wait_for_registry_cleanup(future) -> None:
        callback_finished = threading.Event()
        future.add_done_callback(lambda completed: callback_finished.set())
        if not callback_finished.wait(2):
            raise AssertionError("TaskRunner completion callback did not finish")

    @staticmethod
    def _final_response(content: str) -> FakeHTTPResponse:
        return FakeHTTPResponse(
            json.dumps(
                {
                    "id": "response-final",
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": content,
                            },
                        }
                    ],
                }
            ).encode("utf-8")
        )


if __name__ == "__main__":
    unittest.main()
