import unittest

from app.llm.contracts import LLMToolSchema
from app.llm.tool_schema_registry import (
    DuplicateToolSchemaError,
    InvalidToolSchemaError,
    ToolSchemaNotFoundError,
    ToolSchemaRegistry,
)


def make_schema(name: str, required: tuple[str, ...] = ()) -> LLMToolSchema:
    properties = {
        parameter: {"type": "string"}
        for parameter in required
    }
    return LLMToolSchema(
        name=name,
        description=f"{name} tool",
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    )


class ToolSchemaRegistryTests(unittest.TestCase):
    def test_register_query_and_list_schemas_in_stable_order(self) -> None:
        read_file = make_schema("read_file", ("path",))
        run_command = make_schema("run-command", ("command",))
        registry = ToolSchemaRegistry((read_file,))

        registry.register(run_command)

        self.assertEqual(len(registry), 2)
        self.assertIn("read_file", registry)
        self.assertIs(registry.get("read_file"), read_file)
        self.assertIs(registry.require("run-command"), run_command)
        self.assertEqual(registry.names(), ("read_file", "run-command"))
        self.assertEqual(registry.get_all(), (read_file, run_command))

    def test_reject_duplicate_schema_without_overwriting_original(self) -> None:
        original = make_schema("read_file", ("path",))
        duplicate = make_schema("read_file")
        registry = ToolSchemaRegistry((original,))

        with self.assertRaises(DuplicateToolSchemaError):
            registry.register(duplicate)

        self.assertIs(registry.require("read_file"), original)

    def test_report_missing_schema(self) -> None:
        registry = ToolSchemaRegistry()

        self.assertIsNone(registry.get("missing_tool"))
        with self.assertRaises(ToolSchemaNotFoundError):
            registry.require("missing_tool")

    def test_reject_invalid_tool_name_and_parameter_schema(self) -> None:
        invalid_name = make_schema("invalid tool name")
        missing_object_type = LLMToolSchema(
            name="read_file",
            description="read file",
            parameters={"properties": {}},
        )
        unknown_required = LLMToolSchema(
            name="read_file",
            description="read file",
            parameters={
                "type": "object",
                "properties": {},
                "required": ["path"],
            },
        )

        for schema in (invalid_name, missing_object_type, unknown_required):
            with self.subTest(schema=schema):
                with self.assertRaises(InvalidToolSchemaError):
                    ToolSchemaRegistry((schema,))

    def test_reject_non_schema_value(self) -> None:
        registry = ToolSchemaRegistry()

        with self.assertRaises(TypeError):
            registry.register({"name": "read_file"})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
