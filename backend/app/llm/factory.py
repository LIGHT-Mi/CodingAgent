"""根据应用配置装配 DeepSeek 模型网关。"""

from __future__ import annotations

from app.llm.contracts import ModelConfig
from app.llm.deepseek_client import DeepSeekClient, DeepSeekConfigurationError
from app.llm.gateway import LLMGateway
from app.llm.tool_schema_registry import ToolSchemaRegistry


def create_configured_llm_gateway() -> LLMGateway:
    """从 ``backend/.env`` 或进程环境变量创建可调用的模型网关。"""

    from app.core.config import settings
    from app.tools.schemas import READ_ONLY_FILE_TOOL_SCHEMAS

    secret = settings.DEEPSEEK_API_KEY
    if secret is None:
        raise DeepSeekConfigurationError(
            "DEEPSEEK_API_KEY is missing; set it in backend/.env or the process "
            "environment"
        )
    return LLMGateway(
        DeepSeekClient(
            api_key=secret.get_secret_value(),
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout_seconds=settings.DEEPSEEK_TIMEOUT_SECONDS,
        ),
        ModelConfig(model=settings.DEEPSEEK_MODEL),
        ToolSchemaRegistry(READ_ONLY_FILE_TOOL_SCHEMAS),
    )
