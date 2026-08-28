"""使用 DeepSeek Chat Completions API 的同步 HTTP 客户端。"""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Mapping
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.llm.contracts import LLMMessage, LLMRequest


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClientError(RuntimeError):
    """DeepSeek 客户端无法返回一个可供适配器处理的响应。"""


class DeepSeekConfigurationError(DeepSeekClientError):
    """DeepSeek 客户端配置不完整或无效。"""


class DeepSeekTimeoutError(DeepSeekClientError):
    """DeepSeek 请求在配置的时间内没有完成。"""


class DeepSeekNetworkError(DeepSeekClientError):
    """与 DeepSeek 建立连接或传输数据时发生网络错误。"""


class DeepSeekRateLimitError(DeepSeekClientError):
    """DeepSeek 拒绝了超过速率限制的请求。"""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class DeepSeekAPIError(DeepSeekClientError):
    """DeepSeek 返回了速率限制以外的非成功 HTTP 状态。"""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class DeepSeekResponseError(DeepSeekClientError):
    """DeepSeek 的成功 HTTP 响应不是可用的 JSON 对象。"""


class DeepSeekRequestError(DeepSeekClientError):
    """供应商无关请求包含 DeepSeek 不接受的参数。"""


OpenURL = Callable[..., Any]


class DeepSeekClient:
    """向 DeepSeek OpenAI 兼容端点发送非流式 Chat Completion 请求。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 60.0,
        open_url: OpenURL = urlopen,
    ) -> None:
        self._api_key = _validate_api_key(api_key)
        self.base_url = _validate_base_url(base_url)
        self.timeout_seconds = _validate_timeout(timeout_seconds)
        if not callable(open_url):
            raise TypeError("open_url must be callable")
        self._open_url = open_url

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def create_chat_completion(self, request: LLMRequest) -> Mapping[str, Any]:
        """发送一个非流式请求并返回解码后的供应商原始响应。"""

        payload = self.build_payload(request)
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        http_request = Request(
            self.chat_completions_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with self._open_url(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read()
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (TimeoutError, socket.timeout) as exc:
            raise DeepSeekTimeoutError(
                f"DeepSeek request timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise DeepSeekTimeoutError(
                    "DeepSeek request timed out while establishing the connection"
                ) from exc
            raise DeepSeekNetworkError(
                f"DeepSeek network request failed: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise DeepSeekNetworkError(
                f"DeepSeek network request failed: {exc}"
            ) from exc

        return _decode_response(response_body)

    @staticmethod
    def build_payload(request: LLMRequest) -> dict[str, Any]:
        """把供应商无关请求转换为 DeepSeek Chat Completions 请求体。"""

        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        if request.temperature is not None and request.temperature > 2:
            raise DeepSeekRequestError(
                "DeepSeek temperature must be less than or equal to 2"
            )
        if len(request.tool_schemas) > 128:
            raise DeepSeekRequestError(
                "DeepSeek accepts at most 128 tools in one request"
            )

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                _serialize_message(message) for message in request.messages
            ],
            "stream": False,
        }
        if request.tool_schemas:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": dict(schema.parameters),
                    },
                }
                for schema in request.tool_schemas
            ]
            payload["tool_choice"] = request.tool_choice.value
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        return payload

    @staticmethod
    def _raise_http_error(error: HTTPError) -> NoReturn:
        status_code = error.code
        response_body = error.read()
        message = _extract_api_error_message(response_body, status_code)
        if status_code == 429:
            raise DeepSeekRateLimitError(
                message,
                retry_after_seconds=_parse_retry_after(error.headers),
            ) from error
        if status_code in {408, 504}:
            raise DeepSeekTimeoutError(message) from error
        raise DeepSeekAPIError(message, status_code=status_code) from error


def _serialize_message(message: LLMMessage) -> dict[str, Any]:
    serialized: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.tool_calls:
        serialized["tool_calls"] = [
            {
                "id": tool_call.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": tool_call.arguments_json,
                },
            }
            for tool_call in sorted(
                message.tool_calls,
                key=lambda item: item.call_index,
            )
        ]
    if message.tool_call_id is not None:
        serialized["tool_call_id"] = message.tool_call_id
    return serialized


def _decode_response(response_body: bytes) -> Mapping[str, Any]:
    try:
        decoded_text = response_body.decode("utf-8")
        decoded_json = json.loads(
            decoded_text,
            parse_constant=_reject_non_standard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeepSeekResponseError(
            "DeepSeek returned a response that is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(decoded_json, Mapping):
        raise DeepSeekResponseError("DeepSeek response JSON must be an object")
    return decoded_json


def _extract_api_error_message(response_body: bytes, status_code: int) -> str:
    fallback = f"DeepSeek API request failed with HTTP {status_code}"
    try:
        decoded_json = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback
    if not isinstance(decoded_json, Mapping):
        return fallback
    error = decoded_json.get("error")
    if not isinstance(error, Mapping):
        return fallback
    message = error.get("message")
    if not isinstance(message, str) or not message.strip():
        return fallback
    return f"{fallback}: {message.strip()}"


def _parse_retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def _validate_api_key(api_key: str) -> str:
    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if not api_key.strip():
        raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is missing or blank")
    return api_key.strip()


def _validate_base_url(base_url: str) -> str:
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string")
    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DeepSeekConfigurationError(
            "DEEPSEEK_BASE_URL must be an absolute HTTP or HTTPS URL"
        )
    if parsed.query or parsed.fragment:
        raise DeepSeekConfigurationError(
            "DEEPSEEK_BASE_URL must not contain a query or fragment"
        )
    return normalized


def _validate_timeout(timeout_seconds: float) -> float:
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds,
        (int, float),
    ):
        raise TypeError("timeout_seconds must be a number")
    if timeout_seconds <= 0:
        raise DeepSeekConfigurationError(
            "DEEPSEEK_TIMEOUT_SECONDS must be greater than zero"
        )
    return float(timeout_seconds)


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")
