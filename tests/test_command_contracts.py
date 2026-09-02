import unittest
from dataclasses import FrozenInstanceError

from app.tools import DEFAULT_COMMAND_CWD, RunCommandArguments


class RunCommandArgumentsTests(unittest.TestCase):
    def test_constructs_immutable_argv_and_default_cwd(self) -> None:
        arguments = RunCommandArguments(command=["python", "-m", "unittest"])

        self.assertEqual(arguments.command, ("python", "-m", "unittest"))
        self.assertIsInstance(arguments.command, tuple)
        self.assertEqual(arguments.cwd, DEFAULT_COMMAND_CWD)
        with self.assertRaises(FrozenInstanceError):
            arguments.cwd = "src"  # type: ignore[misc]

    def test_accepts_explicit_relative_cwd(self) -> None:
        arguments = RunCommandArguments(
            command=("python", "-m", "pytest"),
            cwd="backend",
        )

        self.assertEqual(arguments.cwd, "backend")

    def test_rejects_shell_string_and_invalid_command_arrays(self) -> None:
        invalid_commands = (
            "python -m pytest",
            (),
            ("python", 1),
            ("python", ""),
            ("   ", "-m", "pytest"),
        )

        for command in invalid_commands:
            with self.subTest(command=command):
                with self.assertRaises((TypeError, ValueError)):
                    RunCommandArguments(command=command)  # type: ignore[arg-type]

    def test_rejects_invalid_cwd(self) -> None:
        for cwd in ("", "   ", "/tmp/outside", None):
            with self.subTest(cwd=cwd):
                with self.assertRaises((TypeError, ValueError)):
                    RunCommandArguments(
                        command=("python", "-m", "unittest"),
                        cwd=cwd,  # type: ignore[arg-type]
                    )

    def test_rejects_execution_control_fields(self) -> None:
        forbidden_fields = (
            {"env": {"TOKEN": "secret"}},
            {"timeout": 1},
            {"shell": True},
            {"background": True},
        )

        for fields in forbidden_fields:
            with self.subTest(fields=fields):
                with self.assertRaises(TypeError):
                    RunCommandArguments(
                        command=("python", "-m", "unittest"),
                        **fields,
                    )


if __name__ == "__main__":
    unittest.main()
