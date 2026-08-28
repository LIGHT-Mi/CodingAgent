"""模型调用、供应商响应标准化和 AgentAction 解析的统一入口。"""

from __future__ import annotations

from typing import TypeAlias

from app.agent.contracts import (
    AgentAction,
    RuntimeEvent,
    RuntimeEventType,
)
from app.llm.action_parser import AgentActionParser
from app.llm.contracts import LLMContext, ModelConfig
from app.llm.deepseek_adapter import DeepSeekResponseAdapter
from app.llm.deepseek_client import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekNetworkError,
    DeepSeekRateLimitError,
    DeepSeekResponseError,
    DeepSeekTimeoutError,
)
from app.llm.request_builder import LLMRequestBuilder
from app.llm.tool_schema_registry import ToolSchemaRegistry


LLMGatewayResult: TypeAlias = AgentAction | RuntimeEvent


class LLMGateway:
    """调用 DeepSeek，并把成功响应或调用异常转换为统一数据契约。"""

    def __init__(
        self,
        client: DeepSeekClient,
        model_config: ModelConfig,
        tool_schema_registry: ToolSchemaRegistry,
        *,
        request_builder: LLMRequestBuilder | None = None,
        response_adapter: DeepSeekResponseAdapter | None = None,
        action_parser: AgentActionParser | None = None,
    ) -> None:
        if not isinstance(client, DeepSeekClient):
            raise TypeError("client must be a DeepSeekClient")
        if not isinstance(model_config, ModelConfig):
            raise TypeError("model_config must be a ModelConfig")
        if not isinstance(
            tool_schema_registry,
            ToolSchemaRegistry,
        ):
            raise TypeError("tool_schema_registry must be a ToolSchemaRegistry")
        if request_builder is not None and not isinstance(
            request_builder,
            LLMRequestBuilder,
        ):
            raise TypeError(
                "request_builder must be an LLMRequestBuilder or None"
            )
        self.client = client
        self.model_config = model_config
        self.tool_schema_registry = tool_schema_registry
        self.request_builder = (
            LLMRequestBuilder()
            if request_builder is None
            else request_builder
        )
        self.response_adapter = response_adapter or DeepSeekResponseAdapter()
        self.action_parser = action_parser or AgentActionParser()

    def invoke(self, context: LLMContext) -> LLMGatewayResult:
        """根据 Context 构造请求并完成模型调用，不读写数据库。"""

        request = self.request_builder.build(
            context,
            self.model_config,
            self.tool_schema_registry.get_all(),
        )

        try:
            raw_response = self.client.create_chat_completion(request)
        except DeepSeekTimeoutError as exc:
            return RuntimeEvent(
                event_type=RuntimeEventType.LLM_TIMEOUT,
                message=str(exc),
                source="deepseek_client",
                details={"provider": "deepseek"},
            )
        except DeepSeekRateLimitError as exc:
            details = {"provider": "deepseek", "status_code": 429}
            if exc.retry_after_seconds is not None:
                details["retry_after_seconds"] = exc.retry_after_seconds
            return RuntimeEvent(
                event_type=RuntimeEventType.LLM_RATE_LIMIT,
                message=str(exc),
                source="deepseek_client",
                details=details,
            )
        except DeepSeekNetworkError as exc:
            return RuntimeEvent(
                event_type=RuntimeEventType.LLM_NETWORK_ERROR,
                message=str(exc),
                source="deepseek_client",
                details={"provider": "deepseek"},
            )
        except (DeepSeekAPIError, DeepSeekResponseError) as exc:
            details: dict[str, object] = {
                "provider": "deepseek",
                "error_type": type(exc).__name__,
            }
            if isinstance(exc, DeepSeekAPIError):
                details["status_code"] = exc.status_code
            return RuntimeEvent(
                event_type=RuntimeEventType.INFRASTRUCTURE_ERROR,
                message=str(exc),
                source="deepseek_client",
                details=details,
            )
        except DeepSeekClientError as exc:
            return RuntimeEvent(
                event_type=RuntimeEventType.INFRASTRUCTURE_ERROR,
                message=str(exc),
                source="deepseek_client",
                details={
                    "provider": "deepseek",
                    "error_type": type(exc).__name__,
                },
            )

        normalized_response = self.response_adapter.normalize(raw_response)
        return self.action_parser.parse(normalized_response)
