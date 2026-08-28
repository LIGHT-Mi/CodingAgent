"""LLM 请求与响应的供应商无关契约。"""

from app.llm.action_parser import AgentActionParser
from app.llm.contracts import (
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
    LLMToolCall,
    LLMToolChoice,
    LLMToolSchema,
    LLMUsage,
    NormalizedLLMResponse,
    NormalizedToolCall,
)
from app.llm.deepseek_adapter import DeepSeekResponseAdapter
from app.llm.deepseek_client import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfigurationError,
    DeepSeekNetworkError,
    DeepSeekRateLimitError,
    DeepSeekRequestError,
    DeepSeekResponseError,
    DeepSeekTimeoutError,
)
from app.llm.factory import create_configured_llm_gateway
from app.llm.gateway import LLMGateway, LLMGatewayResult
from app.llm.tool_schema_registry import (
    DuplicateToolSchemaError,
    InvalidToolSchemaError,
    ToolSchemaNotFoundError,
    ToolSchemaRegistry,
    ToolSchemaRegistryError,
)

__all__ = [
    "AgentActionParser",
    "LLMMessage",
    "LLMMessageRole",
    "LLMRequest",
    "LLMToolCall",
    "LLMToolChoice",
    "LLMToolSchema",
    "LLMUsage",
    "NormalizedLLMResponse",
    "NormalizedToolCall",
    "DeepSeekResponseAdapter",
    "DEFAULT_DEEPSEEK_BASE_URL",
    "DeepSeekAPIError",
    "DeepSeekClient",
    "DeepSeekClientError",
    "DeepSeekConfigurationError",
    "DeepSeekNetworkError",
    "DeepSeekRateLimitError",
    "DeepSeekRequestError",
    "DeepSeekResponseError",
    "DeepSeekTimeoutError",
    "LLMGateway",
    "LLMGatewayResult",
    "create_configured_llm_gateway",
    "DuplicateToolSchemaError",
    "InvalidToolSchemaError",
    "ToolSchemaNotFoundError",
    "ToolSchemaRegistry",
    "ToolSchemaRegistryError",
]
