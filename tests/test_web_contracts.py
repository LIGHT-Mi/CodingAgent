import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from app.agent import TaskStatus, ToolCallStatus
from app.web.contracts import (
    API_COMMAND_APPROVAL_DECISION_PATH,
    API_SESSIONS_PATH,
    API_SESSION_PATH,
    API_SESSION_TASKS_PATH,
    API_TASK_CANCEL_PATH,
    API_TASK_COMMAND_APPROVALS_PATH,
    API_TASK_MESSAGES_PATH,
    API_TASK_STEPS_PATH,
    API_TASK_TOOL_CALLS_PATH,
    API_TASK_PATH,
    API_TASK_SNAPSHOT_PATH,
    CancelTaskResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    CreateSessionTaskRequest,
    CreateSessionTaskResponse,
    ToolCallResponse,
)
from app.application import TaskCancellationOutcome
from app.web.main import create_web_app


class WebContractTests(unittest.TestCase):
    def test_fixed_session_and_task_paths(self) -> None:
        self.assertEqual(API_SESSIONS_PATH, "/api/sessions")
        self.assertEqual(API_SESSION_PATH, "/api/sessions/{session_id}")
        self.assertEqual(
            API_SESSION_TASKS_PATH,
            "/api/sessions/{session_id}/tasks",
        )
        self.assertEqual(API_TASK_PATH, "/api/tasks/{task_id}")
        self.assertEqual(
            API_TASK_STEPS_PATH,
            "/api/tasks/{task_id}/steps",
        )
        self.assertEqual(
            API_TASK_MESSAGES_PATH,
            "/api/tasks/{task_id}/messages",
        )
        self.assertEqual(
            API_TASK_TOOL_CALLS_PATH,
            "/api/tasks/{task_id}/tool-calls",
        )
        self.assertEqual(
            API_TASK_CANCEL_PATH,
            "/api/tasks/{task_id}/cancel",
        )
        self.assertEqual(
            API_TASK_SNAPSHOT_PATH,
            "/api/tasks/{task_id}/snapshot",
        )
        self.assertEqual(
            API_TASK_COMMAND_APPROVALS_PATH,
            "/api/tasks/{task_id}/command-approvals",
        )
        self.assertEqual(
            API_COMMAND_APPROVAL_DECISION_PATH,
            "/api/tasks/{task_id}/command-approvals/{approval_id}/decision",
        )

    def test_session_task_requests_are_strict_and_preserve_text(self) -> None:
        request = CreateSessionRequest(
            prompt="  inspect project  ",
            workspace="  .  ",
        )
        self.assertEqual(request.prompt, "  inspect project  ")
        self.assertEqual(request.workspace, "  .  ")

        invalid_payloads = (
            {"prompt": "", "workspace": "."},
            {"prompt": "   ", "workspace": "."},
            {"prompt": "inspect", "workspace": "   "},
            {"prompt": "inspect", "workspace": ".", "unknown": True},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    CreateSessionRequest.model_validate(payload)
                with self.assertRaises(ValidationError):
                    CreateSessionTaskRequest.model_validate(payload)

    def test_session_creation_responses_are_always_pending(self) -> None:
        response = CreateSessionResponse(
            session_id="session-1",
            task_id="task-1",
            title="Inspect project",
        )
        self.assertEqual(
            response.model_dump(mode="json"),
            {
                "session_id": "session-1",
                "task_id": "task-1",
                "title": "Inspect project",
                "status": "PENDING",
            },
        )
        with self.assertRaises(ValidationError):
            CreateSessionResponse(
                session_id="session-1",
                task_id="task-1",
                title="Inspect project",
                status=TaskStatus.RUNNING,
            )
        follow_up = CreateSessionTaskResponse(
            session_id="session-1",
            task_id="task-2",
        )
        self.assertEqual(follow_up.status, TaskStatus.PENDING)

    def test_tool_call_response_serializes_domain_status(self) -> None:
        now = datetime.now(timezone.utc)
        response = ToolCallResponse(
            id="tool-1",
            step_id="step-1",
            assistant_message_id="message-1",
            call_index=0,
            tool_name="run_command",
            arguments={"command": ["python", "-m", "unittest"]},
            status=ToolCallStatus.COMPLETED,
            exit_code=0,
            stdout="OK",
            started_at=now,
            finished_at=now,
        )
        serialized = response.model_dump(mode="json")
        self.assertEqual(serialized["status"], "COMPLETED")
        self.assertEqual(serialized["arguments"]["command"][0], "python")
        self.assertNotIn("provider_call_id", serialized)

    def test_fastapi_app_exposes_session_and_task_routes(self) -> None:
        web_app = create_web_app()
        self.addCleanup(web_app.state.task_runner.shutdown, wait=False)
        paths = set(web_app.openapi()["paths"])
        self.assertTrue(
            {
                API_SESSIONS_PATH,
                API_SESSION_PATH,
                API_SESSION_TASKS_PATH,
                API_TASK_PATH,
                API_TASK_SNAPSHOT_PATH,
                API_TASK_STEPS_PATH,
                API_TASK_MESSAGES_PATH,
                API_TASK_TOOL_CALLS_PATH,
                API_TASK_CANCEL_PATH,
                API_TASK_COMMAND_APPROVALS_PATH,
                API_COMMAND_APPROVAL_DECISION_PATH,
            }.issubset(paths)
        )
        self.assertNotIn("/api/tasks", paths)

    def test_cancel_response_exposes_structured_idempotent_outcome(self) -> None:
        response = CancelTaskResponse(
            task_id="task-1",
            status=TaskStatus.RUNNING,
            cancellation_requested=True,
            outcome=TaskCancellationOutcome.ALREADY_REQUESTED,
        )

        self.assertEqual(
            response.model_dump(mode="json"),
            {
                "task_id": "task-1",
                "status": "RUNNING",
                "cancellation_requested": True,
                "outcome": "ALREADY_REQUESTED",
            },
        )


if __name__ == "__main__":
    unittest.main()
