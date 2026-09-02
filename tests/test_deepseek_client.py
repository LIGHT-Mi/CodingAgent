import json
import unittest
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError, URLError

from app.llm.contracts import (
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
    LLMToolCall,
    LLMToolSchema,
)
from app.llm.deepseek_client import (
    DeepSeekClient,
    DeepSeekConfigurationError,
    DeepSeekNetworkError,
    DeepSeekRateLimitError,
    DeepSeekResponseError,
    DeepSeekRequestError,
    DeepSeekTimeoutError,
)


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def make_request() -> LLMRequest:
    return LLMRequest(
        model="deepseek-v4-flash",
        messages=(
            LLMMessage(LLMMessageRole.SYSTEM, "你是编程智能体。"),
            LLMMessage(LLMMessageRole.USER, "读取配置文件。"),
            LLMMessage(
                LLMMessageRole.ASSISTANT,
                None,
                tool_calls=(
                    LLMToolCall(
                        tool_call_id="call-read",
                        tool_name="read_file",
                        arguments_json='{"path":"pyproject.toml"}',
                        call_index=0,
                    ),
                ),
            ),
            LLMMessage(
                LLMMessageRole.TOOL,
                "配置文件内容",
                tool_call_id="call-read",
            ),
        ),
        tool_schemas=(
            LLMToolSchema(
                name="read_file",
                description="读取文本文件",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ),
        temperature=0,
        max_output_tokens=1024,
    )


class DeepSeekClientTests(unittest.TestCase):
    def test_send_real_chat_completion_shape_and_decode_response(self) -> None:
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
                        "message": {"role": "assistant", "content": "完成。"},
                    }
                ],
            }
        ).encode("utf-8")

        def open_url(request, *, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeHTTPResponse(response_body)

        client = DeepSeekClient(
            api_key="test-secret",
            timeout_seconds=15,
            open_url=open_url,
        )

        response = client.create_chat_completion(make_request())

        http_request = captured["request"]
        payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(http_request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(http_request.method, "POST")
        self.assertEqual(http_request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(captured["timeout"], 15)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][2]["tool_calls"][0]["id"], "call-read")
        self.assertEqual(payload["messages"][3]["tool_call_id"], "call-read")
        self.assertEqual(payload["tools"][0]["function"]["name"], "read_file")
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(response["id"], "response-1")

    def test_reject_missing_credentials_and_invalid_configuration(self) -> None:
        with self.assertRaises(DeepSeekConfigurationError):
            DeepSeekClient(api_key=" ")
        with self.assertRaises(DeepSeekConfigurationError):
            DeepSeekClient(api_key="secret", base_url="not-a-url")
        with self.assertRaises(DeepSeekConfigurationError):
            DeepSeekClient(api_key="secret", timeout_seconds=0)

    def test_classify_timeout_network_and_rate_limit_failures(self) -> None:
        def timeout_open_url(request, *, timeout):
            raise TimeoutError("timed out")

        def network_open_url(request, *, timeout):
            raise URLError("offline")

        headers = Message()
        headers["Retry-After"] = "2.5"

        def rate_limit_open_url(request, *, timeout):
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                headers,
                BytesIO(b'{"error":{"message":"slow down"}}'),
            )

        cases = (
            (timeout_open_url, DeepSeekTimeoutError),
            (network_open_url, DeepSeekNetworkError),
            (rate_limit_open_url, DeepSeekRateLimitError),
        )
        for open_url, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                client = DeepSeekClient(api_key="secret", open_url=open_url)
                with self.assertRaises(expected_error) as caught:
                    client.create_chat_completion(make_request())
                self.assertNotIn("secret", str(caught.exception))

        rate_error = caught.exception
        self.assertEqual(rate_error.retry_after_seconds, 2.5)

    def test_reject_non_json_success_response(self) -> None:
        def open_url(request, *, timeout):
            return FakeHTTPResponse(b"not-json")

        client = DeepSeekClient(api_key="secret", open_url=open_url)

        with self.assertRaises(DeepSeekResponseError):
            client.create_chat_completion(make_request())

    def test_reject_streaming_request(self) -> None:
        request = make_request()
        streaming_request = LLMRequest(
            model=request.model,
            messages=request.messages,
            stream=True,
        )

        with self.assertRaises(DeepSeekRequestError):
            DeepSeekClient.build_payload(streaming_request)


if __name__ == "__main__":
    unittest.main()
