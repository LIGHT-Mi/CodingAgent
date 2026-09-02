import unittest

from app.agent import ToolResult, ToolResultStatus
from app.tools import (
    CREATE_FILE_SCHEMA,
    CODING_TOOL_SCHEMAS,
    DEFAULT_FILE_TOOL_LIMITS,
    CreateFileArguments,
    EDIT_FILE_SCHEMA,
    EditFileArguments,
    FileEntryType,
    FileToolLimits,
    LIST_FILES_SCHEMA,
    ListedFile,
    ListFilesArguments,
    READ_FILE_SCHEMA,
    RUN_COMMAND_SCHEMA,
    ReadFileArguments,
    SEARCH_FILES_SCHEMA,
    SearchFilesArguments,
    SearchMatch,
    UnsupportedTextFileError,
    WRITE_FILE_SCHEMA,
    WriteFileArguments,
    decode_utf8_text,
    format_list_files_result,
    format_search_files_result,
)


class FileToolArgumentTests(unittest.TestCase):
    def test_optional_paths_default_to_workspace_root(self) -> None:
        self.assertEqual(ListFilesArguments().path, ".")
        self.assertEqual(SearchFilesArguments(query="needle").path, ".")

    def test_required_arguments_and_non_blank_values(self) -> None:
        self.assertEqual(ReadFileArguments(path="src/main.py").path, "src/main.py")
        self.assertEqual(
            SearchFilesArguments(query="Agent", path="backend").query,
            "Agent",
        )

        for constructor in (
            lambda: ListFilesArguments(path=" "),
            lambda: ReadFileArguments(path=""),
            lambda: SearchFilesArguments(query=""),
            lambda: SearchFilesArguments(query="value", path=" "),
        ):
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_construct_writable_file_tool_arguments(self) -> None:
        create_arguments = CreateFileArguments("new.py", "")
        write_arguments = WriteFileArguments("main.py", "print('new')\n")
        edit_arguments = EditFileArguments("main.py", "old", "")

        self.assertEqual(create_arguments.content, "")
        self.assertEqual(write_arguments.path, "main.py")
        self.assertEqual(edit_arguments.new_text, "")

    def test_reject_invalid_writable_file_tool_arguments(self) -> None:
        invalid_calls = (
            lambda: CreateFileArguments("", "content"),
            lambda: CreateFileArguments("new.py", None),
            lambda: WriteFileArguments("main.py", None),
            lambda: EditFileArguments("main.py", "", "new"),
            lambda: EditFileArguments("main.py", "same", "same"),
            lambda: EditFileArguments("main.py", "old", None),
        )
        for invalid_call in invalid_calls:
            with self.subTest(invalid_call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class CurrentToolSchemaTests(unittest.TestCase):
    def test_define_seven_tools_in_stable_order(self) -> None:
        self.assertEqual(
            tuple(schema.name for schema in CODING_TOOL_SCHEMAS),
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

    def test_schema_required_and_default_parameters_match_contracts(self) -> None:
        self.assertEqual(LIST_FILES_SCHEMA.parameters["required"], [])
        self.assertEqual(
            LIST_FILES_SCHEMA.parameters["properties"]["path"]["default"],
            ".",
        )
        self.assertEqual(READ_FILE_SCHEMA.parameters["required"], ["path"])
        self.assertEqual(SEARCH_FILES_SCHEMA.parameters["required"], ["query"])
        self.assertEqual(
            SEARCH_FILES_SCHEMA.parameters["properties"]["path"]["default"],
            ".",
        )
        self.assertEqual(
            CREATE_FILE_SCHEMA.parameters["required"],
            ["path", "content"],
        )
        self.assertEqual(
            WRITE_FILE_SCHEMA.parameters["required"],
            ["path", "content"],
        )
        self.assertEqual(
            EDIT_FILE_SCHEMA.parameters["required"],
            ["path", "old_text", "new_text"],
        )
        self.assertEqual(RUN_COMMAND_SCHEMA.parameters["required"], ["command"])
        command = RUN_COMMAND_SCHEMA.parameters["properties"]["command"]
        self.assertEqual(command["type"], "array")
        self.assertEqual(command["minItems"], 1)
        self.assertEqual(command["items"]["type"], "string")
        self.assertEqual(command["items"]["minLength"], 1)
        self.assertEqual(
            RUN_COMMAND_SCHEMA.parameters["properties"]["cwd"]["default"],
            ".",
        )
        for schema in CODING_TOOL_SCHEMAS:
            self.assertEqual(schema.parameters["type"], "object")
            self.assertIn("properties", schema.parameters)
            self.assertIn("required", schema.parameters)
            self.assertFalse(schema.parameters["additionalProperties"])


class FileToolResultFormatTests(unittest.TestCase):
    def test_list_result_is_sorted_and_reports_entry_types(self) -> None:
        result = format_list_files_result(
            (
                ListedFile("src/main.py", FileEntryType.FILE),
                ListedFile("README.md", FileEntryType.FILE),
                ListedFile("src", FileEntryType.DIRECTORY),
            )
        )

        self.assertEqual(
            result,
            "README.md\tfile\nsrc\tdirectory\nsrc/main.py\tfile",
        )
        self.assertEqual(format_list_files_result(()), "")

    def test_search_result_is_sorted_by_path_then_one_based_line_number(self) -> None:
        result = format_search_files_result(
            (
                SearchMatch("b.py", 3, "Agent Agent"),
                SearchMatch("a.py", 10, "Agent"),
                SearchMatch("a.py", 2, "CodingAgent"),
            )
        )

        self.assertEqual(
            result,
            "a.py:2:CodingAgent\na.py:10:Agent\nb.py:3:Agent Agent",
        )
        self.assertEqual(format_search_files_result(()), "")

    def test_utf8_decoder_preserves_text_and_rejects_unsupported_content(self) -> None:
        content = "第一行\r\nsecond line\n"

        self.assertEqual(decode_utf8_text(content.encode("utf-8")), content)
        with self.assertRaises(UnsupportedTextFileError):
            decode_utf8_text(b"text\x00binary")
        with self.assertRaises(UnsupportedTextFileError):
            decode_utf8_text(b"\xff")

    def test_empty_results_can_be_completed_tool_results(self) -> None:
        result = ToolResult(
            tool_call_id="call-1",
            tool_name="search_files",
            status=ToolResultStatus.COMPLETED,
            content=format_search_files_result(()),
            metadata={"match_count": 0, "truncated": False},
        )

        self.assertEqual(result.status, ToolResultStatus.COMPLETED)
        self.assertEqual(result.content, "")
        self.assertIsNone(result.error)

    def test_resource_limits_are_fixed_positive_values(self) -> None:
        limits = DEFAULT_FILE_TOOL_LIMITS

        self.assertEqual(limits.max_file_bytes, 1024 * 1024)
        self.assertEqual(limits.max_search_files, 1000)
        self.assertEqual(limits.max_search_matches, 200)
        self.assertEqual(limits.max_diff_lines, 200)
        self.assertEqual(limits.max_diff_characters, 12_000)

        for invalid_limits in (
            {"max_file_bytes": 0},
            {"max_diff_lines": 0},
            {"max_diff_characters": 0},
        ):
            with self.subTest(invalid_limits=invalid_limits):
                with self.assertRaises(ValueError):
                    FileToolLimits(**invalid_limits)


if __name__ == "__main__":
    unittest.main()
