import json
import unittest
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

from app.agent.contracts import FinalAction, RuntimeEvent, RuntimeEventType
from app.llm.contracts import (
    LLMContext,
    LLMMessage,
    LLMMessageRole,
    ModelConfig,
)
from app.llm.deepseek_client import DeepSeekClient
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry

from test_deepseek_client import FakeHTTPResponse


def make_context() -> LLMContext:
    return LLMContext(
        messages=(
            LLMMessage(LLMMessageRole.SYSTEM, "你是编程助手。"),
            LLMMessage(LLMMessageRole.USER, "完成任务。"),
        )
    )


class LLMGatewayTests(unittest.TestCase):
    def test_success_response_becomes_agent_action(self) -> None:
        captured = {}
        response_body = json.dumps(
            {
                "id": "response-1",
                "object": "chat.completion",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "已完成。"},
                    }
                ],
            }
        ).encode("utf-8")

        def open_url(request, *, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse(response_body)

        gateway = LLMGateway(
            DeepSeekClient(api_key="secret", open_url=open_url),
            ModelConfig(model="deepseek-v4-flash"),
            ToolSchemaRegistry(),
        )

        result = gateway.invoke(make_context())

        self.assertIsInstance(result, FinalAction)
        self.assertEqual(result.content, "已完成。")
        self.assertEqual(gateway.tool_schema_registry.get_all(), ())
        payload = captured["payload"]
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "user"],
        )
        self.assertFalse(payload["stream"])
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_transport_failures_become_specific_runtime_events(self) -> None:
        def timeout_open_url(request, *, timeout):
            raise TimeoutError("timed out")

        def network_open_url(request, *, timeout):
            raise URLError("offline")

        headers = Message()
        headers["Retry-After"] = "3"

        def rate_limit_open_url(request, *, timeout):
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                headers,
                BytesIO(b'{"error":{"message":"slow down"}}'),
            )

        cases = (
            (timeout_open_url, RuntimeEventType.LLM_TIMEOUT),
            (rate_limit_open_url, RuntimeEventType.LLM_RATE_LIMIT),
            (network_open_url, RuntimeEventType.LLM_NETWORK_ERROR),
        )
        for open_url, event_type in cases:
            with self.subTest(event_type=event_type):
                gateway = LLMGateway(
                    DeepSeekClient(api_key="secret", open_url=open_url),
                    ModelConfig(model="deepseek-v4-flash"),
                    ToolSchemaRegistry(),
                )

                result = gateway.invoke(make_context())

                self.assertIsInstance(result, RuntimeEvent)
                self.assertEqual(result.event_type, event_type)
                self.assertEqual(result.source, "deepseek_client")

    def test_invalid_http_or_response_body_becomes_infrastructure_event(self) -> None:
        def unauthorized_open_url(request, *, timeout):
            raise HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                Message(),
                BytesIO(b'{"error":{"message":"invalid credentials"}}'),
            )

        def invalid_json_open_url(request, *, timeout):
            return FakeHTTPResponse(b"not-json")

        for open_url in (unauthorized_open_url, invalid_json_open_url):
            with self.subTest(open_url=open_url):
                gateway = LLMGateway(
                    DeepSeekClient(api_key="secret", open_url=open_url),
                    ModelConfig(model="deepseek-v4-flash"),
                    ToolSchemaRegistry(),
                )

                result = gateway.invoke(make_context())

                self.assertIsInstance(result, RuntimeEvent)
                self.assertEqual(
                    result.event_type,
                    RuntimeEventType.INFRASTRUCTURE_ERROR,
                )
                self.assertNotIn("secret", result.message)


if __name__ == "__main__":
    unittest.main()
