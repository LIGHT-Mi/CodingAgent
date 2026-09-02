import unittest
from unittest.mock import patch

from pydantic import SecretStr

from app.core.config import settings
from app.llm import LLMContext, LLMMessage, LLMMessageRole, LLMToolChoice
from app.llm.factory import create_configured_llm_gateway


class LLMFactoryTests(unittest.TestCase):
    def test_register_seven_strict_coding_tool_schemas(self) -> None:
        with patch.object(
            settings,
            "DEEPSEEK_API_KEY",
            SecretStr("test-secret"),
        ):
            gateway = create_configured_llm_gateway()

        schemas = gateway.tool_schema_registry.get_all()
        self.assertEqual(
            tuple(schema.name for schema in schemas),
            (
                "list_files",
                "read_file",
                "search_files",
                "create_file",
                "write_file",
                "edit_file",
                "run_command",
            ),
        )
        for schema in schemas:
            self.assertEqual(schema.parameters["type"], "object")
            self.assertIn("properties", schema.parameters)
            self.assertIn("required", schema.parameters)
            self.assertFalse(schema.parameters["additionalProperties"])

        forbidden_names = {"delete_file"}
        self.assertTrue(forbidden_names.isdisjoint(gateway.tool_schema_registry.names()))

        context = LLMContext(
            messages=(
                LLMMessage(LLMMessageRole.SYSTEM, "你是编程助手。"),
                LLMMessage(LLMMessageRole.USER, "查看项目。"),
            )
        )
        request = gateway.request_builder.build(
            context,
            gateway.model_config,
            schemas,
        )
        self.assertEqual(request.tool_schemas, schemas)
        self.assertIs(request.tool_choice, LLMToolChoice.AUTO)
        self.assertFalse(request.stream)
        payload = gateway.client.build_payload(request)
        self.assertEqual(
            tuple(tool["function"]["name"] for tool in payload["tools"]),
            (
                "list_files",
                "read_file",
                "search_files",
                "create_file",
                "write_file",
                "edit_file",
                "run_command",
            ),
        )
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertFalse(payload["stream"])


if __name__ == "__main__":
    unittest.main()
