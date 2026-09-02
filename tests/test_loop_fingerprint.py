import unittest

from app.agent import (
    ToolCallRequest,
    ToolResult,
    ToolResultStatus,
    build_loop_fingerprint,
)


class LoopFingerprintTests(unittest.TestCase):
    def test_arguments_are_normalized_and_provider_ids_are_excluded(self) -> None:
        first = build_loop_fingerprint(
            (
                ToolCallRequest(
                    "provider-a",
                    "read_file",
                    {"options": {"b": 2, "a": 1}, "path": "main.py"},
                    0,
                ),
            ),
            (
                ToolResult(
                    "provider-a",
                    "read_file",
                    ToolResultStatus.COMPLETED,
                    content="source",
                ),
            ),
        )
        second = build_loop_fingerprint(
            (
                ToolCallRequest(
                    "provider-b",
                    "read_file",
                    {"path": "main.py", "options": {"a": 1, "b": 2}},
                    0,
                ),
            ),
            (
                ToolResult(
                    "provider-b",
                    "read_file",
                    ToolResultStatus.COMPLETED,
                    content="source",
                ),
            ),
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_multiple_calls_are_ordered_by_call_index(self) -> None:
        calls = (
            ToolCallRequest("call-read", "read_file", {"path": "a.py"}, 1),
            ToolCallRequest("call-list", "list_files", {"path": "."}, 0),
        )
        results = (
            ToolResult(
                "call-read",
                "read_file",
                ToolResultStatus.COMPLETED,
                content="pass\n",
            ),
            ToolResult(
                "call-list",
                "list_files",
                ToolResultStatus.COMPLETED,
                content="a.py\tfile",
            ),
        )

        self.assertEqual(
            build_loop_fingerprint(calls, results),
            build_loop_fingerprint(tuple(reversed(calls)), tuple(reversed(results))),
        )

    def test_command_duration_and_absolute_cwd_do_not_change_fingerprint(
        self,
    ) -> None:
        def fingerprint(provider_id: str, duration: float, cwd: str) -> str:
            return build_loop_fingerprint(
                (
                    ToolCallRequest(
                        provider_id,
                        "run_command",
                        {"command": ["pytest", "-q"], "cwd": "."},
                    ),
                ),
                (
                    ToolResult(
                        provider_id,
                        "run_command",
                        ToolResultStatus.COMPLETED,
                        content=f"duration_seconds: {duration}",
                        metadata={
                            "exit_code": 1,
                            "stdout": "1 failed",
                            "stderr": "",
                            "timeout": False,
                            "duration_seconds": duration,
                            "cwd": cwd,
                        },
                    ),
                ),
            )

        self.assertEqual(
            fingerprint("provider-a", 0.12, "/tmp/run-a"),
            fingerprint("provider-b", 0.98, "/tmp/run-b"),
        )

    def test_stable_result_fields_change_fingerprint(self) -> None:
        call = ToolCallRequest(
            "call-command",
            "run_command",
            {"command": ["pytest"], "cwd": "."},
        )

        def fingerprint(stdout: str) -> str:
            return build_loop_fingerprint(
                (call,),
                (
                    ToolResult(
                        "call-command",
                        "run_command",
                        ToolResultStatus.COMPLETED,
                        content="command observation",
                        metadata={
                            "exit_code": 0,
                            "stdout": stdout,
                            "stderr": "",
                            "timeout": False,
                        },
                    ),
                ),
            )

        self.assertNotEqual(fingerprint("first"), fingerprint("second"))

    def test_invalid_unicode_argument_still_has_a_stable_fingerprint(self) -> None:
        call = ToolCallRequest(
            "call-create",
            "create_file",
            {"path": "new.txt", "content": "\ud800"},
        )
        result = ToolResult(
            "call-create",
            "create_file",
            ToolResultStatus.ERROR,
            error="content cannot be encoded as UTF-8",
        )

        self.assertEqual(
            build_loop_fingerprint((call,), (result,)),
            build_loop_fingerprint((call,), (result,)),
        )

    def test_requires_one_matching_result_per_call(self) -> None:
        call = ToolCallRequest("call-read", "read_file", {"path": "a.py"})

        with self.assertRaises(ValueError):
            build_loop_fingerprint((call,), ())
        with self.assertRaises(ValueError):
            build_loop_fingerprint(
                (call,),
                (
                    ToolResult(
                        "call-read",
                        "list_files",
                        ToolResultStatus.COMPLETED,
                        content="a.py",
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
