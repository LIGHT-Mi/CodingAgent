import tempfile
import unittest
from pathlib import Path

from app.agent import ToolResultStatus
from app.tools import (
    WorkspacePathAlreadyExistsError,
    WorkspacePathConfigurationError,
    WorkspacePathError,
    WorkspacePathGuard,
    WorkspacePathNotFoundError,
    WorkspacePathRejectedError,
    WorkspacePathTypeError,
)


class WorkspacePathGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.source_directory = self.workspace / "src"
        self.source_directory.mkdir()
        self.source_file = self.source_directory / "main.py"
        self.source_file.write_text("print('safe')\n", encoding="utf-8")
        self.outside_directory = self.root / "outside"
        self.outside_directory.mkdir()
        self.outside_file = self.outside_directory / "secret.txt"
        self.outside_file.write_text("secret\n", encoding="utf-8")
        self.guard = WorkspacePathGuard()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_resolve_relative_existing_path_from_task_workspace(self) -> None:
        resolved = self.guard.resolve_existing(
            str(self.workspace),
            "src/../src/main.py",
        )

        self.assertEqual(resolved, self.source_file.resolve())
        self.assertTrue(resolved.is_absolute())

    def test_allow_workspace_root_and_in_workspace_absolute_path(self) -> None:
        self.assertEqual(
            self.guard.resolve_existing(self.workspace, "."),
            self.workspace,
        )
        self.assertEqual(
            self.guard.resolve_existing(self.workspace, self.source_file),
            self.source_file,
        )

    def test_reject_parent_escape_and_outside_absolute_path(self) -> None:
        missing_outside_path = self.outside_directory / "missing.txt"
        cases = (
            "../outside/secret.txt",
            self.outside_file,
            missing_outside_path,
        )

        for requested_path in cases:
            with self.subTest(requested_path=requested_path):
                with self.assertRaises(WorkspacePathRejectedError) as raised:
                    self.guard.resolve_existing(self.workspace, requested_path)
                self.assertEqual(raised.exception.status, ToolResultStatus.REJECTED)

    def test_reject_symlinks_to_outside_file_and_directory(self) -> None:
        outside_file_link = self.workspace / "outside-file"
        outside_directory_link = self.workspace / "outside-directory"
        missing_outside_link = self.workspace / "missing-outside-file"
        outside_file_link.symlink_to(self.outside_file)
        outside_directory_link.symlink_to(
            self.outside_directory,
            target_is_directory=True,
        )
        missing_outside_link.symlink_to(self.outside_directory / "missing.txt")

        for requested_path in (
            outside_file_link,
            outside_directory_link,
            missing_outside_link,
        ):
            with self.subTest(requested_path=requested_path):
                with self.assertRaises(WorkspacePathRejectedError) as raised:
                    self.guard.resolve_existing(self.workspace, requested_path)
                self.assertEqual(raised.exception.status, ToolResultStatus.REJECTED)

    def test_missing_workspace_path_is_an_ordinary_tool_error(self) -> None:
        with self.assertRaises(WorkspacePathNotFoundError) as raised:
            self.guard.resolve_existing(self.workspace, "missing.txt")

        self.assertEqual(raised.exception.status, ToolResultStatus.ERROR)

    def test_specific_entry_points_reject_file_type_mismatches(self) -> None:
        self.assertEqual(
            self.guard.resolve_existing_file(self.workspace, "src/main.py"),
            self.source_file,
        )
        self.assertEqual(
            self.guard.resolve_existing_directory(self.workspace, "src"),
            self.source_directory,
        )

        for resolver, requested_path in (
            (self.guard.resolve_existing_file, "src"),
            (self.guard.resolve_existing_directory, "src/main.py"),
        ):
            with self.subTest(requested_path=requested_path):
                with self.assertRaises(WorkspacePathTypeError) as raised:
                    resolver(self.workspace, requested_path)
                self.assertEqual(raised.exception.status, ToolResultStatus.ERROR)

    def test_reject_invalid_requested_path_without_reading_content(self) -> None:
        for requested_path in ("", "   "):
            with self.subTest(requested_path=requested_path):
                with self.assertRaises(WorkspacePathError) as raised:
                    self.guard.resolve_existing(self.workspace, requested_path)
                self.assertEqual(raised.exception.status, ToolResultStatus.ERROR)

    def test_embedded_nul_is_an_ordinary_path_error_with_safe_message(self) -> None:
        for resolver in (
            self.guard.resolve_existing,
            self.guard.resolve_new_file_target,
        ):
            with self.subTest(resolver=resolver.__name__):
                with self.assertRaises(WorkspacePathError) as raised:
                    resolver(self.workspace, "bad\x00name")

                self.assertEqual(raised.exception.status, ToolResultStatus.ERROR)
                self.assertIn(r"bad\x00name", str(raised.exception))
                self.assertNotIn("\x00", str(raised.exception))

    def test_require_prevalidated_canonical_task_workspace(self) -> None:
        relative_workspace = Path(self.workspace.name)
        unnormalized_workspace = self.workspace / ".." / self.workspace.name

        for workspace in ("", relative_workspace, unnormalized_workspace):
            with self.subTest(workspace=workspace):
                with self.assertRaises(WorkspacePathConfigurationError):
                    self.guard.resolve_existing(workspace, ".")

    def test_resolve_new_file_target_from_existing_parent(self) -> None:
        inside_parent_link = self.workspace / "inside-parent"
        inside_parent_link.symlink_to(
            self.source_directory,
            target_is_directory=True,
        )
        relative_target = self.guard.resolve_new_file_target(
            self.workspace,
            "src/new.py",
        )
        absolute_target = self.guard.resolve_new_file_target(
            self.workspace,
            self.workspace / "root.txt",
        )
        linked_parent_target = self.guard.resolve_new_file_target(
            self.workspace,
            "inside-parent/linked.py",
        )

        self.assertEqual(relative_target, self.source_directory / "new.py")
        self.assertEqual(absolute_target, self.workspace / "root.txt")
        self.assertEqual(
            linked_parent_target,
            self.source_directory / "linked.py",
        )
        self.assertFalse(relative_target.exists())
        self.assertFalse(absolute_target.exists())
        self.assertFalse(linked_parent_target.exists())

    def test_new_file_target_requires_existing_directory_parent(self) -> None:
        with self.assertRaises(WorkspacePathNotFoundError) as missing_parent:
            self.guard.resolve_new_file_target(
                self.workspace,
                "missing/new.py",
            )
        with self.assertRaises(WorkspacePathTypeError) as file_parent:
            self.guard.resolve_new_file_target(
                self.workspace,
                "src/main.py/child.py",
            )

        self.assertEqual(missing_parent.exception.status, ToolResultStatus.ERROR)
        self.assertEqual(file_parent.exception.status, ToolResultStatus.ERROR)

    def test_new_file_target_rejects_existing_target_without_overwriting(self) -> None:
        existing_directory = self.workspace / "existing-directory"
        existing_directory.mkdir()
        inside_link = self.workspace / "inside-link"
        inside_link.symlink_to(self.source_file)
        dangling_inside_link = self.workspace / "dangling-inside-link"
        dangling_inside_link.symlink_to(self.workspace / "missing-target.py")
        original_content = self.source_file.read_text(encoding="utf-8")

        for requested_path in (
            "src/main.py",
            "existing-directory",
            "inside-link",
            "dangling-inside-link",
        ):
            with self.subTest(requested_path=requested_path):
                with self.assertRaises(WorkspacePathAlreadyExistsError) as raised:
                    self.guard.resolve_new_file_target(
                        self.workspace,
                        requested_path,
                    )
                self.assertEqual(raised.exception.status, ToolResultStatus.ERROR)

        self.assertEqual(
            self.source_file.read_text(encoding="utf-8"),
            original_content,
        )

    def test_new_file_target_rejects_workspace_escape_and_external_symlinks(
        self,
    ) -> None:
        outside_parent_link = self.workspace / "outside-parent"
        outside_parent_link.symlink_to(
            self.outside_directory,
            target_is_directory=True,
        )
        outside_target_link = self.workspace / "outside-target"
        outside_target_link.symlink_to(self.outside_directory / "missing.txt")

        for requested_path in (
            "../outside/new.py",
            self.outside_directory / "new.py",
            "outside-parent/new.py",
            "outside-target",
        ):
            with self.subTest(requested_path=requested_path):
                with self.assertRaises(WorkspacePathRejectedError) as raised:
                    self.guard.resolve_new_file_target(
                        self.workspace,
                        requested_path,
                    )
                self.assertEqual(
                    raised.exception.status,
                    ToolResultStatus.REJECTED,
                )

    def test_new_file_target_rejects_invalid_path_and_workspace(self) -> None:
        with self.assertRaises(WorkspacePathError):
            self.guard.resolve_new_file_target(self.workspace, " ")
        with self.assertRaises(WorkspacePathConfigurationError):
            self.guard.resolve_new_file_target(Path("relative"), "new.py")


if __name__ == "__main__":
    unittest.main()
