import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from app.agent import ToolResultStatus
from app.tools import (
    CommandRiskLevel,
    CommandSafetyDecision,
    CommandSafetyPolicy,
    CommandSafetyVerdict,
    RunCommandArguments,
    build_rejected_command_result,
    command_fingerprint,
)


class CommandSafetyPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temporary_directory.name).resolve()
        self.policy = CommandSafetyPolicy()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_allows_small_python_test_build_and_run_set(self) -> None:
        allowed_commands = (
            ("pytest", "-q"),
            ("python", "-m", "pytest"),
            ("python3", "-m", "unittest", "discover"),
            ("python", "-m", "compileall", "src"),
            ("python", "src/main.py"),
            ("python", "--version"),
        )

        for command in allowed_commands:
            with self.subTest(command=command):
                decision = self.policy.evaluate(
                    RunCommandArguments(command=command),
                    self.cwd,
                )
                self.assertEqual(decision.verdict, CommandSafetyVerdict.ALLOW)
                self.assertFalse(decision.approval_eligible)

    def test_permanently_rejects_privilege_power_and_disk_commands(self) -> None:
        for command in (
            ("sudo", "pytest"),
            ("/usr/bin/sudo", "pytest"),
            ("shutdown", "-h", "now"),
            ("mkfs.ext4", "/dev/example"),
        ):
            with self.subTest(command=command):
                decision = self.policy.evaluate(
                    RunCommandArguments(command=command),
                    self.cwd,
                )
                self.assertEqual(decision.verdict, CommandSafetyVerdict.REJECT)
                self.assertEqual(decision.risk_level, CommandRiskLevel.CRITICAL)
                self.assertFalse(decision.approval_eligible)

    def test_marks_shell_destructive_and_unknown_commands_as_approvable(self) -> None:
        cases = (
            (("bash", "-lc", "pytest"), "SHELL_INTERPRETER_REQUIRES_APPROVAL"),
            (("rm", "result.txt"), "DESTRUCTIVE_COMMAND_REQUIRES_APPROVAL"),
            (("git", "status"), "UNKNOWN_EXECUTABLE_REQUIRES_APPROVAL"),
            (("python", "-c", "print('x')"), "PYTHON_MODE_REQUIRES_APPROVAL"),
            (("python", "-m", "pip", "list"), "PYTHON_MODULE_REQUIRES_APPROVAL"),
        )

        for command, rule_id in cases:
            with self.subTest(command=command):
                decision = self.policy.evaluate(
                    RunCommandArguments(command=command),
                    self.cwd,
                )
                self.assertEqual(
                    decision.verdict,
                    CommandSafetyVerdict.REQUIRE_APPROVAL,
                )
                self.assertEqual(decision.rule_id, rule_id)
                self.assertTrue(decision.approval_eligible)

    def test_rejects_shell_syntax_as_unsupported_and_not_approvable(self) -> None:
        for command in (
            ("pytest", "&&", "python", "main.py"),
            ("pytest", "$(touch outside)"),
            ("pytest", "`touch outside`"),
            ("pytest", "results\nrm file"),
        ):
            with self.subTest(command=command):
                decision = self.policy.evaluate(
                    RunCommandArguments(command=command),
                    self.cwd,
                )
                self.assertEqual(
                    decision.rule_id,
                    "UNSUPPORTED_SHELL_SYNTAX",
                )
                self.assertFalse(decision.approval_eligible)

    def test_rejects_executable_paths_even_when_basename_is_allowed(self) -> None:
        decision = self.policy.evaluate(
            RunCommandArguments(command=("/tmp/python", "-m", "pytest")),
            self.cwd,
        )

        self.assertEqual(
            decision.verdict,
            CommandSafetyVerdict.REQUIRE_APPROVAL,
        )
        self.assertEqual(
            decision.rule_id,
            "EXECUTABLE_PATH_REQUIRES_APPROVAL",
        )
        self.assertTrue(decision.approval_eligible)


class CommandSafetyResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temporary_directory.name).resolve()
        self.arguments = RunCommandArguments(command=("sudo", "pytest"))
        self.decision = CommandSafetyPolicy().evaluate(
            self.arguments,
            self.cwd,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_model_visible_rejected_observation_with_metadata(self) -> None:
        result = build_rejected_command_result(
            "provider-call-1",
            self.arguments,
            self.cwd,
            self.decision,
        )

        self.assertEqual(result.status, ToolResultStatus.REJECTED)
        self.assertEqual(result.tool_name, "run_command")
        self.assertIn("status: REJECTED", result.content or "")
        self.assertIn("exit_code: unavailable", result.content or "")
        self.assertIn("stdout_truncated: false", result.content or "")
        self.assertEqual(result.metadata["argv"], ["sudo", "pytest"])
        self.assertEqual(result.metadata["cwd"], str(self.cwd))
        self.assertEqual(result.metadata["rule_id"], self.decision.rule_id)
        self.assertEqual(
            result.metadata["risk_level"],
            self.decision.risk_level.value,
        )
        self.assertFalse(result.metadata["approval_eligible"])
        self.assertTrue(
            str(result.metadata["command_fingerprint"]).startswith("sha256:")
        )

    def test_command_fingerprint_is_stable_and_binds_argv_and_cwd(self) -> None:
        original = command_fingerprint(self.arguments.command, self.cwd)

        self.assertEqual(
            original,
            command_fingerprint(self.arguments.command, self.cwd),
        )
        self.assertNotEqual(
            original,
            command_fingerprint(("git", "diff"), self.cwd),
        )
        self.assertNotEqual(
            original,
            command_fingerprint(
                self.arguments.command,
                (self.cwd / "subdirectory").resolve(),
            ),
        )

    def test_cannot_build_rejected_result_from_allow_decision(self) -> None:
        allowed = CommandSafetyPolicy().evaluate(
            RunCommandArguments(command=("pytest",)),
            self.cwd,
        )

        with self.assertRaises(ValueError):
            build_rejected_command_result(
                "provider-call-2",
                RunCommandArguments(command=("pytest",)),
                self.cwd,
                allowed,
            )

    def test_decision_is_a_validated_frozen_value_object(self) -> None:
        with self.assertRaises(ValueError):
            CommandSafetyDecision(
                verdict=CommandSafetyVerdict.ALLOW,
                reason="allowed",
                rule_id="INVALID_ALLOW",
                risk_level=CommandRiskLevel.LOW,
                approval_eligible=True,
            )

        with self.assertRaises(FrozenInstanceError):
            self.decision.rule_id = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
