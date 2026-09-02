import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent import ToolCallRequest, ToolResult, ToolResultStatus
from app.tools import (
    DEFAULT_FILE_TOOL_LIMITS,
    CommandExecutor,
    CreateFileArguments,
    DuplicateLocalToolError,
    EditFileArguments,
    FileToolLimits,
    ListFilesTool,
    LocalToolRegistry,
    PreparedFileToolCall,
    RunCommandTool,
    ToolRouter,
    WorkspacePathGuard,
    WriteFileArguments,
    create_local_tool_registry,
)


def create_test_local_tool_registry(
    limits: FileToolLimits = DEFAULT_FILE_TOOL_LIMITS,
) -> LocalToolRegistry:
    """使用测试资源上限装配六个文件工具和命令工具。"""

    command_tool = RunCommandTool(
        CommandExecutor(
            timeout_seconds=2,
            termination_grace_seconds=0.1,
            max_output_bytes_per_stream=1024,
        )
    )
    return create_local_tool_registry(command_tool, limits=limits)


class FileToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.outside_file = self.outside / "secret.txt"
        self.outside_file.write_text("secret\n", encoding="utf-8")
        self.router = ToolRouter(
            create_test_local_tool_registry(),
            WorkspacePathGuard(),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def request(self, tool_name: str, **arguments: object) -> ToolCallRequest:
        return ToolCallRequest(
            tool_call_id=f"call-{tool_name}",
            tool_name=tool_name,
            arguments=arguments,
        )

    def execute_request(
        self,
        request: ToolCallRequest,
        *,
        router: ToolRouter | None = None,
    ) -> ToolResult:
        selected_router = self.router if router is None else router
        prepared = selected_router.prepare(request, self.workspace)
        if isinstance(prepared, ToolResult):
            return prepared
        return selected_router.execute(prepared)


class LocalToolRegistryTests(FileToolTestCase):
    def test_default_registry_contains_seven_tools_in_stable_order(self) -> None:
        registry = create_test_local_tool_registry()

        self.assertEqual(
            registry.names(),
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
        self.assertEqual(len(registry), 7)

    def test_registry_rejects_duplicate_tool(self) -> None:
        registry = LocalToolRegistry((ListFilesTool(),))

        with self.assertRaises(DuplicateLocalToolError):
            registry.register(ListFilesTool())

    def test_registry_accepts_only_local_tools(self) -> None:
        registry = LocalToolRegistry()

        with self.assertRaises(TypeError):
            registry.register(object())  # type: ignore[arg-type]


class ListFilesToolTests(FileToolTestCase):
    def test_list_only_direct_children_in_stable_order_with_types(self) -> None:
        (self.workspace / "z.txt").write_text("z", encoding="utf-8")
        source = self.workspace / "src"
        source.mkdir()
        (source / "nested.py").write_text("nested", encoding="utf-8")
        (self.workspace / "a-link").symlink_to(self.workspace / "z.txt")

        result = self.execute_request(self.request("list_files"))

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(
            result.content,
            "a-link\tsymlink\nsrc\tdirectory\nz.txt\tfile",
        )
        self.assertNotIn("nested.py", result.content or "")
        self.assertEqual(result.metadata["requested_path"], ".")
        self.assertEqual(result.metadata["resolved_path"], str(self.workspace))
        self.assertEqual(result.metadata["entry_count"], 3)
        self.assertFalse(result.metadata["limit_reached"])


class ReadFileToolTests(FileToolTestCase):
    def test_read_utf8_file_without_changing_content_or_line_endings(self) -> None:
        file_path = self.workspace / "message.txt"
        original_bytes = "第一行\r\nsecond\n".encode("utf-8")
        file_path.write_bytes(original_bytes)

        result = self.execute_request(
            self.request("read_file", path="message.txt")
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.content, original_bytes.decode("utf-8"))
        self.assertEqual(result.metadata["byte_count"], len(original_bytes))
        self.assertEqual(
            result.metadata["character_count"],
            len(original_bytes.decode("utf-8")),
        )
        self.assertEqual(file_path.read_bytes(), original_bytes)

    def test_binary_and_non_utf8_files_are_ordinary_errors(self) -> None:
        binary_file = self.workspace / "binary.dat"
        invalid_utf8_file = self.workspace / "invalid.txt"
        binary_file.write_bytes(b"text\x00data")
        invalid_utf8_file.write_bytes(b"\xff")

        for path in (binary_file, invalid_utf8_file):
            with self.subTest(path=path):
                result = self.execute_request(
                    self.request("read_file", path=path.name)
                )
                self.assertEqual(result.status, ToolResultStatus.ERROR)
                self.assertIsNotNone(result.error)

    def test_file_size_limit_returns_error_without_partial_content(self) -> None:
        file_path = self.workspace / "large.txt"
        file_path.write_text("12345", encoding="utf-8")
        router = ToolRouter(
            create_test_local_tool_registry(
                limits=FileToolLimits(
                    max_file_bytes=4,
                    max_search_files=10,
                    max_search_matches=10,
                )
            ),
            WorkspacePathGuard(),
        )

        result = self.execute_request(
            self.request("read_file", path=file_path.name),
            router=router,
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIsNone(result.content)
        self.assertTrue(result.metadata["limit_reached"])


class CreateFileToolTests(FileToolTestCase):
    def test_create_utf8_file_with_stable_summary(self) -> None:
        untouched_file = self.workspace / "untouched.txt"
        untouched_file.write_text("unchanged", encoding="utf-8")
        content = "第一行\r\nsecond line\n"

        result = self.execute_request(
            self.request("create_file", path="created.txt", content=content)
        )

        created_file = self.workspace / "created.txt"
        encoded_content = content.encode("utf-8")
        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(created_file.read_bytes(), encoded_content)
        self.assertEqual(untouched_file.read_text(encoding="utf-8"), "unchanged")
        self.assertIn("Created created.txt", result.content or "")
        self.assertIn(f"characters: 0 → {len(content)}", result.content or "")
        self.assertIn(f"bytes: 0 → {len(encoded_content)}", result.content or "")
        self.assertIn("--- /dev/null", result.content or "")
        self.assertIn("+++ b/created.txt", result.content or "")
        self.assertEqual(result.metadata["relative_path"], "created.txt")
        self.assertEqual(result.metadata["operation"], "create_file")
        self.assertTrue(result.metadata["changed"])
        self.assertEqual(result.metadata["before_character_count"], 0)
        self.assertEqual(result.metadata["after_character_count"], len(content))
        self.assertEqual(result.metadata["before_byte_count"], 0)
        self.assertEqual(result.metadata["after_byte_count"], len(encoded_content))
        self.assertEqual(result.metadata["replacement_count"], 0)
        self.assertFalse(result.metadata["diff_truncated"])

    def test_existing_target_is_error_and_is_not_overwritten(self) -> None:
        existing_file = self.workspace / "existing.txt"
        original_bytes = b"original\n"
        existing_file.write_bytes(original_bytes)

        result = self.execute_request(
            self.request(
                "create_file",
                path="existing.txt",
                content="replacement\n",
            )
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("already exists", result.error or "")
        self.assertEqual(existing_file.read_bytes(), original_bytes)

    def test_exclusive_create_prevents_prepare_execute_race_overwrite(self) -> None:
        request = self.request(
            "create_file",
            path="raced.txt",
            content="agent content",
        )
        prepared = self.router.prepare(request, self.workspace)
        self.assertIsInstance(prepared, PreparedFileToolCall)
        assert isinstance(prepared, PreparedFileToolCall)
        raced_file = self.workspace / "raced.txt"
        raced_file.write_text("concurrent content", encoding="utf-8")

        result = self.router.execute(prepared)

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(
            raced_file.read_text(encoding="utf-8"),
            "concurrent content",
        )

    def test_content_limit_is_checked_before_file_creation(self) -> None:
        router = ToolRouter(
            create_test_local_tool_registry(
                limits=FileToolLimits(
                    max_file_bytes=4,
                    max_search_files=10,
                    max_search_matches=10,
                )
            ),
            WorkspacePathGuard(),
        )

        result = self.execute_request(
            self.request("create_file", path="large.txt", content="12345"),
            router=router,
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertTrue(result.metadata["limit_reached"])
        self.assertFalse((self.workspace / "large.txt").exists())


class WriteFileToolTests(FileToolTestCase):
    def test_atomically_overwrite_utf8_file_and_preserve_mode(self) -> None:
        target_file = self.workspace / "target.txt"
        target_file.write_text("old content\n", encoding="utf-8")
        target_file.chmod(0o640)
        untouched_file = self.workspace / "untouched.txt"
        untouched_file.write_bytes(b"unchanged\n")
        new_content = "新内容\r\nsecond\n"

        result = self.execute_request(
            self.request(
                "write_file",
                path="target.txt",
                content=new_content,
            )
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(target_file.read_bytes(), new_content.encode("utf-8"))
        self.assertEqual(untouched_file.read_bytes(), b"unchanged\n")
        self.assertEqual(stat.S_IMODE(target_file.stat().st_mode), 0o640)
        self.assertTrue(result.metadata["changed"])
        self.assertEqual(result.metadata["relative_path"], "target.txt")
        self.assertEqual(result.metadata["operation"], "write_file")
        self.assertEqual(result.metadata["replacement_count"], 0)
        self.assertFalse(result.metadata["diff_truncated"])
        self.assertEqual(
            result.metadata["after_byte_count"],
            len(new_content.encode("utf-8")),
        )
        self.assertIn("Wrote target.txt", result.content or "")
        self.assertIn("--- a/target.txt", result.content or "")
        self.assertIn("+++ b/target.txt", result.content or "")
        self.assertIn("-old content", result.content or "")
        self.assertIn("+新内容", result.content or "")
        self.assertFalse(
            any(path.name.endswith(".tmp") for path in self.workspace.iterdir())
        )

    def test_missing_and_non_utf8_targets_are_errors_without_changes(self) -> None:
        invalid_file = self.workspace / "invalid.txt"
        original_bytes = b"\xff\x00"
        invalid_file.write_bytes(original_bytes)

        missing_result = self.execute_request(
            self.request("write_file", path="missing.txt", content="new")
        )
        invalid_result = self.execute_request(
            self.request("write_file", path="invalid.txt", content="new")
        )

        self.assertEqual(missing_result.status, ToolResultStatus.ERROR)
        self.assertEqual(invalid_result.status, ToolResultStatus.ERROR)
        self.assertFalse((self.workspace / "missing.txt").exists())
        self.assertEqual(invalid_file.read_bytes(), original_bytes)

    def test_atomic_replace_failure_preserves_original_and_cleans_temporary_file(
        self,
    ) -> None:
        target_file = self.workspace / "target.txt"
        original_bytes = b"original\n"
        target_file.write_bytes(original_bytes)

        with patch(
            "app.tools.file_tools.os.replace",
            side_effect=PermissionError("replace denied"),
        ):
            result = self.execute_request(
                self.request(
                    "write_file",
                    path="target.txt",
                    content="replacement\n",
                )
            )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("replace denied", result.error or "")
        self.assertEqual(target_file.read_bytes(), original_bytes)
        self.assertEqual(
            [path.name for path in self.workspace.iterdir()],
            ["target.txt"],
        )

    def test_same_content_completes_without_replacing_file(self) -> None:
        target_file = self.workspace / "target.txt"
        content = "same\n"
        target_file.write_text(content, encoding="utf-8")

        with patch("app.tools.file_tools.os.replace") as replace:
            result = self.execute_request(
                self.request("write_file", path="target.txt", content=content)
            )

        replace.assert_not_called()
        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertFalse(result.metadata["changed"])
        self.assertFalse(result.metadata["diff_truncated"])
        self.assertNotIn("diff:", result.content or "")
        self.assertEqual(target_file.read_text(encoding="utf-8"), content)

    def test_diff_output_is_limited_without_truncating_written_file(self) -> None:
        target_file = self.workspace / "target.txt"
        original_content = "\n".join(f"old-{index}" for index in range(20)) + "\n"
        new_content = "\n".join(f"new-{index}" for index in range(20)) + "\n"
        target_file.write_text(original_content, encoding="utf-8")
        router = ToolRouter(
            create_test_local_tool_registry(
                limits=FileToolLimits(
                    max_file_bytes=1024,
                    max_search_files=10,
                    max_search_matches=10,
                    max_diff_lines=5,
                    max_diff_characters=100,
                )
            ),
            WorkspacePathGuard(),
        )

        result = self.execute_request(
            self.request("write_file", path="target.txt", content=new_content),
            router=router,
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertTrue(result.metadata["diff_truncated"])
        self.assertIn("... diff truncated ...", result.content or "")
        diff_content = (result.content or "").split("diff:\n", maxsplit=1)[1]
        self.assertLessEqual(len(diff_content), 100)
        self.assertEqual(target_file.read_text(encoding="utf-8"), new_content)

    def test_new_content_limit_preserves_existing_file(self) -> None:
        target_file = self.workspace / "target.txt"
        original_bytes = b"old\n"
        target_file.write_bytes(original_bytes)
        router = ToolRouter(
            create_test_local_tool_registry(
                limits=FileToolLimits(
                    max_file_bytes=4,
                    max_search_files=10,
                    max_search_matches=10,
                )
            ),
            WorkspacePathGuard(),
        )

        result = self.execute_request(
            self.request("write_file", path="target.txt", content="12345"),
            router=router,
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertTrue(result.metadata["limit_reached"])
        self.assertEqual(target_file.read_bytes(), original_bytes)


class EditFileToolTests(FileToolTestCase):
    def test_replace_one_exact_multiline_match_and_preserve_other_content(
        self,
    ) -> None:
        target_file = self.workspace / "target.py"
        original_content = (
            "def greet():\r\n"
            "    message = 'old'\r\n"
            "    return message\r\n"
        )
        old_text = "    message = 'old'\r\n    return message"
        new_text = "    message = 'new'\r\n    return message.upper()"
        target_file.write_bytes(original_content.encode("utf-8"))
        target_file.chmod(0o640)
        untouched_file = self.workspace / "untouched.txt"
        untouched_file.write_text("unchanged", encoding="utf-8")

        result = self.execute_request(
            self.request(
                "edit_file",
                path="target.py",
                old_text=old_text,
                new_text=new_text,
            )
        )

        expected_content = original_content.replace(old_text, new_text)
        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(
            target_file.read_bytes(),
            expected_content.encode("utf-8"),
        )
        self.assertEqual(untouched_file.read_text(encoding="utf-8"), "unchanged")
        self.assertEqual(stat.S_IMODE(target_file.stat().st_mode), 0o640)
        self.assertEqual(result.metadata["match_count"], 1)
        self.assertEqual(result.metadata["replacement_count"], 1)
        self.assertEqual(result.metadata["relative_path"], "target.py")
        self.assertEqual(result.metadata["operation"], "edit_file")
        self.assertTrue(result.metadata["changed"])
        self.assertFalse(result.metadata["diff_truncated"])
        self.assertIn("Edited target.py", result.content or "")
        self.assertIn("1 exact replacement", result.content or "")
        self.assertIn("--- a/target.py", result.content or "")
        self.assertIn("+++ b/target.py", result.content or "")

    def test_zero_multiple_and_overlapping_matches_are_errors_without_changes(
        self,
    ) -> None:
        cases = (
            ("alpha\n", "missing", 0),
            ("old and old\n", "old", 2),
            ("aaa", "aa", 2),
        )

        for index, (content, old_text, expected_matches) in enumerate(cases):
            with self.subTest(content=content, old_text=old_text):
                target_file = self.workspace / f"target-{index}.txt"
                original_bytes = content.encode("utf-8")
                target_file.write_bytes(original_bytes)

                result = self.execute_request(
                    self.request(
                        "edit_file",
                        path=target_file.name,
                        old_text=old_text,
                        new_text="new",
                    )
                )

                self.assertEqual(result.status, ToolResultStatus.ERROR)
                self.assertEqual(result.metadata["match_count"], expected_matches)
                self.assertEqual(target_file.read_bytes(), original_bytes)

    def test_empty_new_text_deletes_only_the_exact_match(self) -> None:
        target_file = self.workspace / "target.txt"
        target_file.write_text("before REMOVE after", encoding="utf-8")

        result = self.execute_request(
            self.request(
                "edit_file",
                path="target.txt",
                old_text="REMOVE ",
                new_text="",
            )
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(
            target_file.read_text(encoding="utf-8"),
            "before after",
        )

    def test_non_utf8_target_is_error_without_modification(self) -> None:
        target_file = self.workspace / "target.bin"
        original_bytes = b"old\xfftext"
        target_file.write_bytes(original_bytes)

        result = self.execute_request(
            self.request(
                "edit_file",
                path="target.bin",
                old_text="old",
                new_text="new",
            )
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(target_file.read_bytes(), original_bytes)

    def test_atomic_replace_failure_preserves_original_and_cleans_temporary_file(
        self,
    ) -> None:
        target_file = self.workspace / "target.txt"
        original_bytes = b"old value\n"
        target_file.write_bytes(original_bytes)

        with patch(
            "app.tools.file_tools.os.replace",
            side_effect=PermissionError("replace denied"),
        ):
            result = self.execute_request(
                self.request(
                    "edit_file",
                    path="target.txt",
                    old_text="old",
                    new_text="new",
                )
            )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertEqual(target_file.read_bytes(), original_bytes)
        self.assertEqual(
            [path.name for path in self.workspace.iterdir()],
            ["target.txt"],
        )

    def test_updated_content_limit_preserves_original_file(self) -> None:
        target_file = self.workspace / "target.txt"
        original_bytes = b"old\n"
        target_file.write_bytes(original_bytes)
        router = ToolRouter(
            create_test_local_tool_registry(
                limits=FileToolLimits(
                    max_file_bytes=5,
                    max_search_files=10,
                    max_search_matches=10,
                )
            ),
            WorkspacePathGuard(),
        )

        result = self.execute_request(
            self.request(
                "edit_file",
                path="target.txt",
                old_text="old",
                new_text="12345",
            ),
            router=router,
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertTrue(result.metadata["limit_reached"])
        self.assertEqual(target_file.read_bytes(), original_bytes)


class SearchFilesToolTests(FileToolTestCase):
    def test_literal_search_is_sorted_by_relative_path_and_line_number(self) -> None:
        source = self.workspace / "src"
        source.mkdir()
        (self.workspace / "b.txt").write_text(
            "literal [x]\nnone\nliteral [x] twice [x]\n",
            encoding="utf-8",
        )
        (source / "a.txt").write_text(
            "first\nliteral [x]\n",
            encoding="utf-8",
        )

        result = self.execute_request(self.request("search_files", query="[x]"))

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(
            result.content,
            "b.txt:1:literal [x]\n"
            "b.txt:3:literal [x] twice [x]\n"
            "src/a.txt:2:literal [x]",
        )
        self.assertEqual(result.metadata["searched_file_count"], 2)
        self.assertEqual(result.metadata["match_count"], 3)
        self.assertFalse(result.metadata["limit_reached"])

    def test_no_search_result_is_completed(self) -> None:
        (self.workspace / "file.txt").write_text("content\n", encoding="utf-8")

        result = self.execute_request(
            self.request("search_files", query="missing")
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.content, "")
        self.assertEqual(result.metadata["match_count"], 0)

    def test_search_match_and_file_limits_return_partial_completed_result(self) -> None:
        (self.workspace / "a.txt").write_text("hit\nhit\nhit\n", encoding="utf-8")
        (self.workspace / "b.txt").write_text("hit\n", encoding="utf-8")
        router = ToolRouter(
            create_test_local_tool_registry(
                limits=FileToolLimits(
                    max_file_bytes=100,
                    max_search_files=10,
                    max_search_matches=2,
                )
            ),
            WorkspacePathGuard(),
        )

        result = self.execute_request(
            self.request("search_files", query="hit"),
            router=router,
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.content, "a.txt:1:hit\na.txt:2:hit")
        self.assertEqual(result.metadata["match_count"], 2)
        self.assertTrue(result.metadata["limit_reached"])

    def test_search_file_count_limit_returns_partial_completed_result(self) -> None:
        (self.workspace / "a.txt").write_text("hit\n", encoding="utf-8")
        (self.workspace / "b.txt").write_text("hit\n", encoding="utf-8")
        router = ToolRouter(
            create_test_local_tool_registry(
                limits=FileToolLimits(
                    max_file_bytes=100,
                    max_search_files=1,
                    max_search_matches=10,
                )
            ),
            WorkspacePathGuard(),
        )

        result = self.execute_request(
            self.request("search_files", query="hit"),
            router=router,
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.content, "a.txt:1:hit")
        self.assertEqual(result.metadata["searched_file_count"], 1)
        self.assertTrue(result.metadata["limit_reached"])

    def test_non_utf8_search_target_is_an_error(self) -> None:
        invalid_file = self.workspace / "invalid.txt"
        invalid_file.write_bytes(b"\xff")

        result = self.execute_request(
            self.request(
                "search_files",
                query="value",
                path="invalid.txt",
            ),
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)


class FileToolFailureTests(FileToolTestCase):
    def test_all_read_only_tools_leave_workspace_file_contents_unchanged(self) -> None:
        source_directory = self.workspace / "src"
        source_directory.mkdir()
        (self.workspace / "README.md").write_text(
            "Coding Agent\n",
            encoding="utf-8",
        )
        (source_directory / "main.py").write_text(
            "print('Coding Agent')\n",
            encoding="utf-8",
        )
        before = self._workspace_file_contents()

        requests = (
            self.request("list_files", path="."),
            self.request("read_file", path="README.md"),
            self.request("search_files", query="Coding Agent", path="."),
        )
        for request in requests:
            with self.subTest(tool_name=request.tool_name):
                result = self.execute_request(request)
                self.assertEqual(result.status, ToolResultStatus.COMPLETED)
                self.assertEqual(self._workspace_file_contents(), before)

    def _workspace_file_contents(self) -> dict[str, bytes]:
        return {
            path.relative_to(self.workspace).as_posix(): path.read_bytes()
            for path in sorted(self.workspace.rglob("*"))
            if path.is_file()
        }

    def test_argument_errors_and_unknown_tools_return_error(self) -> None:
        requests = (
            self.request("read_file"),
            self.request("read_file", path="file.txt", extra="value"),
            self.request("search_files", query=1),
            self.request("unknown_tool"),
        )

        for request in requests:
            with self.subTest(request=request):
                result = self.execute_request(request)
                self.assertEqual(result.status, ToolResultStatus.ERROR)
                self.assertIsNotNone(result.error)

    def test_invalid_path_and_unencodable_contents_are_ordinary_errors(
        self,
    ) -> None:
        write_target = self.workspace / "write.txt"
        edit_target = self.workspace / "edit.txt"
        write_target.write_text("original write\n", encoding="utf-8")
        edit_target.write_text("old edit\n", encoding="utf-8")

        requests = (
            self.request(
                "create_file",
                path="bad\x00name",
                content="content",
            ),
            self.request(
                "create_file",
                path="create.txt",
                content="\ud800",
            ),
            self.request(
                "write_file",
                path="write.txt",
                content="\ud800",
            ),
            self.request(
                "edit_file",
                path="edit.txt",
                old_text="old edit",
                new_text="\ud800",
            ),
        )

        for request in requests:
            with self.subTest(tool_name=request.tool_name, arguments=request.arguments):
                result = self.execute_request(request)
                self.assertEqual(result.status, ToolResultStatus.ERROR)
                self.assertIsNotNone(result.error)

        self.assertFalse((self.workspace / "create.txt").exists())
        self.assertEqual(
            write_target.read_text(encoding="utf-8"),
            "original write\n",
        )
        self.assertEqual(
            edit_target.read_text(encoding="utf-8"),
            "old edit\n",
        )

    def test_missing_path_and_target_type_errors_return_error(self) -> None:
        directory = self.workspace / "src"
        directory.mkdir()
        file_path = self.workspace / "file.txt"
        file_path.write_text("content", encoding="utf-8")
        requests = (
            self.request("read_file", path="missing.txt"),
            self.request("read_file", path="src"),
            self.request("list_files", path="file.txt"),
        )

        for request in requests:
            with self.subTest(request=request):
                result = self.execute_request(request)
                self.assertEqual(result.status, ToolResultStatus.ERROR)

    def test_workspace_escape_and_external_symlink_return_rejected(self) -> None:
        external_link = self.workspace / "secret-link"
        external_link.symlink_to(self.outside_file)
        requests = (
            self.request("read_file", path="../outside/secret.txt"),
            self.request("read_file", path=str(self.outside_file)),
            self.request("read_file", path="secret-link"),
            self.request("search_files", query="secret", path="secret-link"),
        )

        for request in requests:
            with self.subTest(request=request):
                result = self.execute_request(request)
                self.assertEqual(result.status, ToolResultStatus.REJECTED)
                self.assertIsNone(result.content)

    def test_recursive_search_does_not_read_external_file_symlink(self) -> None:
        external_link = self.workspace / "secret-link.txt"
        external_link.symlink_to(self.outside_file)

        result = self.execute_request(
            self.request("search_files", query="secret")
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.content, "")

    def test_expected_io_error_is_returned_without_timeout(self) -> None:
        file_path = self.workspace / "file.txt"
        file_path.write_text("content", encoding="utf-8")

        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            result = self.execute_request(
                self.request("read_file", path="file.txt")
            )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("denied", result.error or "")
        self.assertNotEqual(result.status, ToolResultStatus.TIMEOUT)


class TwoPhaseToolRouterTests(FileToolTestCase):
    def test_prepared_call_accepts_writable_argument_contracts(self) -> None:
        writable_arguments = (
            CreateFileArguments("new.py", ""),
            WriteFileArguments("main.py", "new content"),
            EditFileArguments("main.py", "old", "new"),
        )

        for arguments in writable_arguments:
            with self.subTest(arguments=arguments):
                prepared = PreparedFileToolCall(
                    tool_call_id="call-write",
                    tool_name="future-file-tool",
                    workspace=self.workspace,
                    arguments=arguments,
                    resolved_path=self.workspace / arguments.path,
                )
                self.assertIs(prepared.arguments, arguments)

    def test_prepare_returns_typed_call_without_reading_file_content(self) -> None:
        file_path = self.workspace / "file.txt"
        file_path.write_text("content", encoding="utf-8")
        request = self.request("read_file", path="file.txt")
        original_read_bytes = Path.read_bytes

        with patch.object(
            Path,
            "read_bytes",
            autospec=True,
            side_effect=original_read_bytes,
        ) as read_bytes:
            prepared = self.router.prepare(request, self.workspace)
            read_bytes.assert_not_called()
            self.assertIsInstance(prepared, PreparedFileToolCall)
            assert isinstance(prepared, PreparedFileToolCall)
            self.assertEqual(prepared.resolved_path, file_path)
            result = self.router.execute(prepared)

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        read_bytes.assert_called_once()

    def test_prepare_failures_do_not_enter_execute_phase(self) -> None:
        external_request = self.request(
            "read_file",
            path=str(self.outside_file),
        )
        requests_and_statuses = (
            (self.request("unknown"), ToolResultStatus.ERROR),
            (self.request("read_file"), ToolResultStatus.ERROR),
            (external_request, ToolResultStatus.REJECTED),
        )

        for request, expected_status in requests_and_statuses:
            with self.subTest(request=request):
                prepared = self.router.prepare(request, self.workspace)
                self.assertIsInstance(prepared, ToolResult)
                assert isinstance(prepared, ToolResult)
                self.assertEqual(prepared.status, expected_status)

    def test_execute_accepts_only_prepared_tool_call(self) -> None:
        with self.assertRaises(TypeError):
            self.router.execute(self.request("list_files"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
