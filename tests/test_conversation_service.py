import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent import TaskStatus
from app.api.conversation_service import (
    TASK_SUBMISSION_FAILURE_ERROR,
    ConversationService,
    ConversationTaskSubmissionError,
)
from app.api.workspace import WorkspaceValidationError, WorkspaceValidator
from app.db.base import Base
from app.db.persistence import (
    InvalidStateTransitionError,
    PersistenceService,
    RecordNotFoundError,
)


class RecordingTaskSubmitter:
    def __init__(self) -> None:
        self.submitted_task_ids: list[str] = []
        self.failure: Exception | None = None

    def submit(self, task_id: str) -> object:
        if self.failure is not None:
            raise self.failure
        self.submitted_task_ids.append(task_id)
        return object()


class ConversationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.allowed_root = Path(self.temporary_directory.name)
        self.workspace = self.allowed_root / "project"
        self.workspace.mkdir()
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        testing_session = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.db: Session = testing_session()
        self.addCleanup(self.db.close)
        self.persistence = PersistenceService(self.db)
        self.submitter = RecordingTaskSubmitter()
        self.service = ConversationService(
            self.persistence,
            WorkspaceValidator(self.allowed_root),
            self.submitter,
        )

    def test_create_first_round_and_query_conversation(self) -> None:
        created = self.service.create_conversation(
            "  检查\n项目并修复测试  ",
            self.workspace,
        )

        self.assertEqual(self.submitter.submitted_task_ids, [created.task_id])
        self.assertEqual(created.title, "检查 项目并修复测试")
        conversation = self.service.get_conversation(created.session_id)
        self.assertEqual(conversation.title, "检查 项目并修复测试")
        self.assertEqual(conversation.latest_task_id, created.task_id)
        self.assertEqual(conversation.latest_task_status, TaskStatus.PENDING)
        self.assertEqual(
            conversation.latest_workspace,
            str(self.workspace.resolve()),
        )
        tasks = self.service.list_tasks(created.session_id)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, created.task_id)
        self.assertEqual(
            tasks[0].original_prompt,
            "  检查\n项目并修复测试  ",
        )
        self.assertEqual(tasks[0].workspace, str(self.workspace.resolve()))
        self.assertEqual(
            [item.id for item in self.service.list_conversations()],
            [created.session_id],
        )

    def test_create_follow_up_rejects_active_task_then_reuses_session(
        self,
    ) -> None:
        created = self.service.create_conversation("第一轮", self.workspace)

        with self.assertRaises(InvalidStateTransitionError):
            self.service.create_task(
                created.session_id,
                "不能并发的第二轮",
                self.workspace,
            )
        self.assertEqual(self.submitter.submitted_task_ids, [created.task_id])

        self.persistence.fail_pending_task(created.task_id, "结束第一轮")
        follow_up_id = self.service.create_task(
            created.session_id,
            "第二轮",
            self.workspace,
        )

        self.assertEqual(
            self.submitter.submitted_task_ids,
            [created.task_id, follow_up_id],
        )
        tasks = self.service.list_tasks(created.session_id)
        self.assertEqual([task.id for task in tasks], [created.task_id, follow_up_id])
        self.assertEqual(tasks[0].status, TaskStatus.FAILED)
        self.assertEqual(tasks[1].status, TaskStatus.PENDING)
        conversation = self.service.get_conversation(created.session_id)
        self.assertEqual(conversation.title, "第一轮")
        self.assertEqual(conversation.latest_task_id, follow_up_id)

    def test_submission_failure_closes_task_and_allows_follow_up(self) -> None:
        self.submitter.failure = RuntimeError("executor unavailable")

        with self.assertRaises(ConversationTaskSubmissionError) as caught:
            self.service.create_conversation("第一轮", self.workspace)

        self.assertTrue(caught.exception.task_closed)
        failed_task = self.persistence.get_task(caught.exception.task_id)
        self.assertEqual(failed_task.status, TaskStatus.FAILED.value)
        self.assertEqual(failed_task.error, TASK_SUBMISSION_FAILURE_ERROR)
        self.assertIsNone(failed_task.started_at)
        self.assertIsNotNone(failed_task.finished_at)
        self.assertFalse(
            self.persistence.has_active_session_task(failed_task.session_id)
        )

        self.submitter.failure = None
        follow_up_id = self.service.create_task(
            failed_task.session_id,
            "重新提交",
            self.workspace,
        )
        self.assertEqual(self.submitter.submitted_task_ids, [follow_up_id])

    def test_validation_and_missing_session_do_not_submit_task(self) -> None:
        outside = self.allowed_root / "missing"
        invalid_calls = (
            lambda: self.service.create_conversation("   ", self.workspace),
            lambda: self.service.create_conversation("任务", outside),
            lambda: self.service.create_task(
                "missing-session",
                "后续任务",
                self.workspace,
            ),
        )
        expected_errors = (
            ValueError,
            WorkspaceValidationError,
            RecordNotFoundError,
        )

        for call, expected_error in zip(invalid_calls, expected_errors):
            with self.subTest(expected_error=expected_error):
                with self.assertRaises(expected_error):
                    call()

        self.assertEqual(self.submitter.submitted_task_ids, [])
