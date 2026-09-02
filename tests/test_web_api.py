import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.agent import TaskStatus
from app.application import ApplicationFactory, TaskRunner
from app.core.config import Settings
from app.db.base import Base
from app.db.models.task import Task
from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry
from app.tools import CODING_TOOL_SCHEMAS
from app.web.main import create_web_app


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class WebAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "a.txt").write_text("alpha", encoding="utf-8")
        (self.workspace / "b.txt").write_text("beta", encoding="utf-8")
        database_path = root / "web.sqlite"
        self.database_url = f"sqlite+pysqlite:///{database_path}"
        self.engine = create_engine(
            self.database_url,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.addCleanup(self.engine.dispose)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )
        self.llm_started = threading.Event()
        self.release_llm = threading.Event()
        self.response_number = 0

        def open_url(request, *, timeout):
            del request, timeout
            self.response_number += 1
            self.llm_started.set()
            if not self.release_llm.wait(5):
                raise TimeoutError("test did not release fake LLM")
            if self.response_number == 1:
                return FakeHTTPResponse(self._tool_calls_response())
            return FakeHTTPResponse(self._final_response())

        def gateway_factory() -> LLMGateway:
            return LLMGateway(
                DeepSeekClient(api_key="api-secret", open_url=open_url),
                ModelConfig(model="deepseek-v4-flash"),
                ToolSchemaRegistry(CODING_TOOL_SCHEMAS),
            )

        self.config = Settings(
            DATABASE_URL=self.database_url,
            DEEPSEEK_API_KEY="api-secret",
            ALLOWED_WORKSPACE_ROOT=root,
            WEB_CORS_ALLOWED_ORIGINS=("http://localhost:5173",),
            LLM_RETRY_BASE_SECONDS=0,
            LLM_RETRY_MAX_SECONDS=0,
        )
        self.factory = ApplicationFactory(
            self.config,
            session_factory=self.session_factory,
            llm_gateway_factory=gateway_factory,
        )
        self.runner = TaskRunner(self.factory, max_workers=1)
        self.addCleanup(self.runner.shutdown)
        self.app = create_web_app(self.factory, self.runner)

    def test_create_query_history_and_dto_redaction(self) -> None:
        response = self._request(
            "POST",
            "/api/sessions",
            json_body={
                "prompt": "读取两个文件后回答",
                "workspace": str(self.workspace),
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json["status"], "PENDING")
        self.assertTrue(response.json["session_id"])
        self.assertEqual(response.json["title"], "读取两个文件后回答")
        session_id = response.json["session_id"]
        task_id = response.json["task_id"]
        self.assertTrue(self.llm_started.wait(2))

        running = self._request("GET", f"/api/tasks/{task_id}")
        self.assertEqual(running.status_code, 200)
        self.assertEqual(running.json["status"], "RUNNING")

        self.release_llm.set()
        completed = self._wait_for_terminal_task(task_id)
        self.assertEqual(completed["status"], "COMPLETED")

        steps = self._request("GET", f"/api/tasks/{task_id}/steps")
        messages = self._request("GET", f"/api/tasks/{task_id}/messages")
        tool_calls = self._request(
            "GET",
            f"/api/tasks/{task_id}/tool-calls",
        )
        snapshot = self._request(
            "GET",
            f"/api/tasks/{task_id}/snapshot",
        )

        self.assertEqual(steps.status_code, 200)
        self.assertEqual(
            [step["step_number"] for step in steps.json],
            [0, 1],
        )
        self.assertEqual(
            [message["sequence"] for message in messages.json],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [tool_call["call_index"] for tool_call in tool_calls.json],
            [0, 1],
        )
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.json["task"], completed)
        self.assertEqual(snapshot.json["steps"], steps.json)
        self.assertEqual(snapshot.json["messages"], messages.json)
        self.assertEqual(snapshot.json["tool_calls"], tool_calls.json)
        self.assertEqual(snapshot.json["command_approvals"], [])
        session = self._request("GET", f"/api/sessions/{session_id}")
        sessions = self._request("GET", "/api/sessions")
        session_tasks = self._request(
            "GET",
            f"/api/sessions/{session_id}/tasks",
        )
        self.assertEqual(session.json["latest_task_id"], task_id)
        self.assertEqual(session.json["latest_task_status"], "COMPLETED")
        self.assertEqual(
            session.json["latest_workspace"],
            str(self.workspace.resolve()),
        )
        self.assertEqual(sessions.json, [session.json])
        self.assertEqual([task["id"] for task in session_tasks.json], [task_id])
        self.assertTrue(
            all("provider_call_id" not in item for item in tool_calls.json)
        )
        serialized_responses = json.dumps(
            {
                "task": completed,
                "steps": steps.json,
                "messages": messages.json,
                "tool_calls": tool_calls.json,
            }
        )
        self.assertNotIn("api-secret", serialized_responses)
        self.assertNotIn(self.database_url, serialized_responses)

    def test_session_api_rejects_concurrent_then_creates_follow_up(self) -> None:
        created = self._request(
            "POST",
            "/api/sessions",
            json_body={
                "prompt": "第一轮任务",
                "workspace": str(self.workspace),
            },
        )
        self.assertEqual(created.status_code, 202)
        session_id = created.json["session_id"]
        first_task_id = created.json["task_id"]
        self.assertTrue(self.llm_started.wait(2))

        conflict = self._request(
            "POST",
            f"/api/sessions/{session_id}/tasks",
            json_body={
                "prompt": "不能并发的任务",
                "workspace": str(self.workspace),
            },
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            conflict.json,
            {"detail": "Session already has an active task"},
        )

        self.release_llm.set()
        self.assertEqual(
            self._wait_for_terminal_task(first_task_id)["status"],
            "COMPLETED",
        )
        follow_up = self._request(
            "POST",
            f"/api/sessions/{session_id}/tasks",
            json_body={
                "prompt": "请继续检查",
                "workspace": str(self.workspace),
            },
        )
        self.assertEqual(follow_up.status_code, 202)
        self.assertEqual(follow_up.json["session_id"], session_id)
        self.assertEqual(follow_up.json["status"], "PENDING")
        second_task_id = follow_up.json["task_id"]
        self.assertNotEqual(second_task_id, first_task_id)
        self.assertEqual(
            self._wait_for_terminal_task(second_task_id)["status"],
            "COMPLETED",
        )

        session = self._request("GET", f"/api/sessions/{session_id}")
        tasks = self._request(
            "GET",
            f"/api/sessions/{session_id}/tasks",
        )
        self.assertEqual(session.json["title"], "第一轮任务")
        self.assertEqual(session.json["latest_task_id"], second_task_id)
        self.assertEqual(session.json["latest_task_status"], "COMPLETED")
        self.assertEqual(
            [task["id"] for task in tasks.json],
            [first_task_id, second_task_id],
        )

    def test_cancel_is_idempotent_and_eventually_closes_task(self) -> None:
        created = self._request(
            "POST",
            "/api/sessions",
            json_body={
                "prompt": "等待取消",
                "workspace": str(self.workspace),
            },
        )
        task_id = created.json["task_id"]
        self.assertTrue(self.llm_started.wait(2))

        first = self._request("POST", f"/api/tasks/{task_id}/cancel")
        second = self._request("POST", f"/api/tasks/{task_id}/cancel")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json["outcome"], "REQUESTED")
        self.assertTrue(first.json["cancellation_requested"])
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json["outcome"], "ALREADY_REQUESTED")
        self.release_llm.set()
        terminal = self._wait_for_terminal_task(task_id)
        self.assertEqual(terminal["status"], "CANCELLED")

        finished = self._request("POST", f"/api/tasks/{task_id}/cancel")
        self.assertEqual(finished.status_code, 200)
        self.assertEqual(finished.json["outcome"], "TASK_FINISHED")
        self.assertFalse(finished.json["cancellation_requested"])

    def test_unknown_task_and_invalid_workspace_return_safe_errors(self) -> None:
        unknown = self._request("GET", "/api/tasks/missing")
        invalid = self._request(
            "POST",
            "/api/sessions",
            json_body={
                "prompt": "invalid workspace",
                "workspace": str(self.workspace / "missing"),
            },
        )
        unknown_cancel = self._request(
            "POST",
            "/api/tasks/missing/cancel",
        )
        unknown_session = self._request("GET", "/api/sessions/missing")
        unknown_session_tasks = self._request(
            "GET",
            "/api/sessions/missing/tasks",
        )

        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(unknown.json, {"detail": "Task was not found"})
        self.assertEqual(invalid.status_code, 400)
        self.assertNotIn("Traceback", invalid.text)
        self.assertNotIn("api-secret", invalid.text)
        self.assertEqual(unknown_cancel.status_code, 404)
        self.assertEqual(
            unknown_cancel.json,
            {"detail": "Task was not found"},
        )
        self.assertEqual(unknown_session.status_code, 404)
        self.assertEqual(
            unknown_session.json,
            {"detail": "Session was not found"},
        )
        self.assertEqual(unknown_session_tasks.status_code, 404)
        self.assertEqual(
            unknown_session_tasks.json,
            {"detail": "Session was not found"},
        )

    def test_scheduling_failure_closes_new_task_as_failed(self) -> None:
        self.runner.shutdown(wait=True, cancel_running=False)

        response = self._request(
            "POST",
            "/api/sessions",
            json_body={
                "prompt": "无法提交的任务",
                "workspace": str(self.workspace),
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json, {"detail": "Task could not be scheduled"})
        with self.factory.create_db_session() as db:
            tasks = list(db.scalars(select(Task)).all())
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].status, TaskStatus.FAILED.value)
            self.assertEqual(
                tasks[0].error,
                "Background task scheduling failed",
            )
            self.assertIsNone(tasks[0].started_at)
            self.assertIsNotNone(tasks[0].finished_at)

    def test_cors_allows_only_configured_origin(self) -> None:
        allowed = self._request(
            "GET",
            "/api/tasks/missing",
            headers={"origin": "http://localhost:5173"},
        )
        rejected = self._request(
            "GET",
            "/api/tasks/missing",
            headers={"origin": "http://untrusted.example"},
        )

        self.assertEqual(
            allowed.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )
        self.assertNotIn("access-control-allow-origin", rejected.headers)

        preflight = self._request(
            "OPTIONS",
            "/api/sessions",
            headers={
                "origin": "http://localhost:5173",
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type",
            },
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers.get("access-control-allow-origin"),
            "http://localhost:5173",
        )
        self.assertIn(
            "POST",
            preflight.headers.get("access-control-allow-methods", ""),
        )

    def _wait_for_terminal_task(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = self._request("GET", f"/api/tasks/{task_id}")
            if response.json["status"] in {
                TaskStatus.COMPLETED.value,
                TaskStatus.FAILED.value,
                TaskStatus.CANCELLED.value,
                TaskStatus.TERMINATED.value,
            }:
                return response.json
            threading.Event().wait(0.01)
        raise AssertionError("Task did not reach a terminal state")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> "ASGIResponse":
        return asyncio.run(
            call_asgi(
                self.app,
                method,
                path,
                json_body=json_body,
                headers=headers,
            )
        )

    @staticmethod
    def _tool_calls_response() -> dict[str, Any]:
        return {
            "id": "response-tools",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "读取文件",
                        "tool_calls": [
                            {
                                "id": "provider-call-a",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"a.txt"}',
                                },
                            },
                            {
                                "id": "provider-call-b",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"b.txt"}',
                                },
                            },
                        ],
                    },
                }
            ],
        }

    @staticmethod
    def _final_response() -> dict[str, Any]:
        return {
            "id": "response-final",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "读取完成。",
                    },
                }
            ],
        }


class ASGIResponse:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        start = next(
            message
            for message in messages
            if message["type"] == "http.response.start"
        )
        self.status_code = start["status"]
        self.headers = {
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in start.get("headers", [])
        }
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        self.text = body.decode("utf-8")
        try:
            self.json = json.loads(self.text) if self.text else None
        except json.JSONDecodeError:
            self.json = None


async def call_asgi(
    app,
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> ASGIResponse:
    body = b"" if json_body is None else json.dumps(json_body).encode("utf-8")
    request_headers = {
        "accept": "application/json",
        **({"content-type": "application/json"} if json_body is not None else {}),
        **(headers or {}),
    }
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in request_headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    request_consumed = False
    response_messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_consumed
        if not request_consumed:
            request_consumed = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        response_messages.append(message)

    await app(scope, receive, send)
    return ASGIResponse(response_messages)


if __name__ == "__main__":
    unittest.main()
