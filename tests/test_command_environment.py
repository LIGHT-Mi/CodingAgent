import os
import sys
import tempfile
import unittest
from pathlib import Path

from app.tools import CommandEnvironmentBuilder, DEFAULT_COMMAND_LOCALE


class CommandEnvironmentBuilderTests(unittest.TestCase):
    def test_copies_only_explicitly_allowed_environment_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = {
                "PATH": "/safe/bin:/usr/bin",
                "LANG": "zh_CN.UTF-8",
                "LC_ALL": "C.UTF-8",
                "LC_CTYPE": "UTF-8",
                "TMPDIR": temporary_directory,
                "DEEPSEEK_API_KEY": "deepseek-secret",
                "DATABASE_URL": "postgresql://secret",
                "OPENAI_API_KEY": "openai-secret",
                "AWS_SECRET_ACCESS_KEY": "aws-secret",
                "HOME": "/private/home",
                "PYTHONPATH": "/private/python",
                "VIRTUAL_ENV": "/private/venv",
            }

            environment = CommandEnvironmentBuilder(source).build()

            self.assertEqual(
                dict(environment),
                {
                    "PATH": "/safe/bin:/usr/bin",
                    "LANG": "zh_CN.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "LC_CTYPE": "UTF-8",
                    "TMPDIR": str(Path(temporary_directory).resolve()),
                },
            )

    def test_uses_fixed_defaults_when_allowed_values_are_missing(self) -> None:
        environment = CommandEnvironmentBuilder({}).build()

        self.assertEqual(environment["PATH"], os.defpath)
        self.assertEqual(environment["LANG"], DEFAULT_COMMAND_LOCALE)
        self.assertEqual(
            environment["TMPDIR"],
            str(Path(tempfile.gettempdir()).resolve()),
        )

    def test_default_source_prioritizes_the_current_python_directory(self) -> None:
        environment = CommandEnvironmentBuilder().build()

        self.assertEqual(
            environment["PATH"].split(os.pathsep)[0],
            str(Path(sys.executable).parent),
        )

    def test_invalid_or_missing_configured_temp_directory_uses_fallback(self) -> None:
        for configured in ("relative-missing", "/definitely/missing/path"):
            with self.subTest(configured=configured):
                environment = CommandEnvironmentBuilder(
                    {"TMPDIR": configured}
                ).build()
                self.assertEqual(
                    environment["TMPDIR"],
                    str(Path(tempfile.gettempdir()).resolve()),
                )

    def test_source_and_result_cannot_override_future_builds(self) -> None:
        source = {"PATH": "/first/bin"}
        builder = CommandEnvironmentBuilder(source)
        source["PATH"] = "/changed/bin"

        first = builder.build()
        self.assertEqual(first["PATH"], "/first/bin")
        with self.assertRaises(TypeError):
            first["PATH"] = "/mutated/bin"  # type: ignore[index]
        self.assertEqual(builder.build()["PATH"], "/first/bin")

    def test_rejects_invalid_source_and_allowed_value_types(self) -> None:
        with self.assertRaises(TypeError):
            CommandEnvironmentBuilder([])  # type: ignore[arg-type]

        for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"):
            with self.subTest(key=key):
                with self.assertRaises(TypeError):
                    CommandEnvironmentBuilder({key: 1}).build()  # type: ignore[dict-item]

    def test_build_has_no_model_environment_override_parameter(self) -> None:
        builder = CommandEnvironmentBuilder({})

        with self.assertRaises(TypeError):
            builder.build(env={"DEEPSEEK_API_KEY": "secret"})  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
