"""LLM 可用函数工具 Schema 的注册表。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from app.llm.contracts import LLMToolSchema


TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ToolSchemaRegistryError(ValueError):
    """Tool Schema Registry 无法接受或返回请求的 Schema。"""


class DuplicateToolSchemaError(ToolSchemaRegistryError):
    """注册表中已经存在同名 Tool Schema。"""


class ToolSchemaNotFoundError(ToolSchemaRegistryError):
    """注册表中不存在指定名称的 Tool Schema。"""


class InvalidToolSchemaError(ToolSchemaRegistryError):
    """Tool Schema 不满足本项目的函数工具约束。"""


class ToolSchemaRegistry:
    """按注册顺序保存并查询可发送给模型的函数工具 Schema。"""

    def __init__(self, schemas: Iterable[LLMToolSchema] = ()) -> None:
        self._schemas: dict[str, LLMToolSchema] = {}
        for schema in schemas:
            self.register(schema)

    def register(self, schema: LLMToolSchema) -> None:
        """注册一个 Schema；同名工具不会被静默覆盖。"""

        if not isinstance(schema, LLMToolSchema):
            raise TypeError("schema must be an LLMToolSchema")
        self._validate_schema(schema)
        if schema.name in self._schemas:
            raise DuplicateToolSchemaError(
                f"tool schema {schema.name!r} is already registered"
            )
        self._schemas[schema.name] = schema

    def get(self, tool_name: str) -> LLMToolSchema | None:
        """查询 Schema，不存在时返回 None。"""

        _require_non_blank(tool_name, "tool_name")
        return self._schemas.get(tool_name)

    def require(self, tool_name: str) -> LLMToolSchema:
        """查询 Schema，不存在时抛出明确异常。"""

        schema = self.get(tool_name)
        if schema is None:
            raise ToolSchemaNotFoundError(
                f"tool schema {tool_name!r} is not registered"
            )
        return schema

    def get_all(self) -> tuple[LLMToolSchema, ...]:
        """按稳定注册顺序返回全部 Schema。"""

        return tuple(self._schemas.values())

    def names(self) -> tuple[str, ...]:
        """按稳定注册顺序返回全部工具名称。"""

        return tuple(self._schemas)

    def __contains__(self, tool_name: object) -> bool:
        return tool_name in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)

    @staticmethod
    def _validate_schema(schema: LLMToolSchema) -> None:
        if TOOL_NAME_PATTERN.fullmatch(schema.name) is None:
            raise InvalidToolSchemaError(
                "tool name must contain only letters, digits, underscores or "
                "hyphens and be at most 64 characters"
            )

        parameters = schema.parameters
        if parameters.get("type") != "object":
            raise InvalidToolSchemaError(
                "tool parameters must be a JSON Schema object"
            )

        properties = parameters.get("properties", {})
        if not isinstance(properties, Mapping):
            raise InvalidToolSchemaError(
                "tool parameter properties must be a mapping"
            )

        required = parameters.get("required", ())
        if not isinstance(required, (list, tuple)) or any(
            not isinstance(name, str) or not name for name in required
        ):
            raise InvalidToolSchemaError(
                "tool required parameters must be a list of non-empty names"
            )
        unknown_required = set(required) - set(properties)
        if unknown_required:
            names = ", ".join(sorted(unknown_required))
            raise InvalidToolSchemaError(
                f"required tool parameters are not defined in properties: {names}"
            )


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
