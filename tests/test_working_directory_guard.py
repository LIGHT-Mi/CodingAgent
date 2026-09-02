import tempfile
import unittest
from pathlib import Path

from app.agent import ToolResultStatus
from app.tools import (
    WorkingDirectoryConfigurationError,
    WorkingDirectoryError,
    WorkingDirectoryGuard,
    WorkingDirectoryNotFoundError,
    WorkingDirectoryRejectedError,
    WorkingDirectoryTypeError,
)


class WorkingDirectoryGuardTests(unittest.TestCase):
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
        self.guard = WorkingDirectoryGuard()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_resolves_workspace_relative_directory_and_dot_segments(self) -> None:
        self.assertEqual(
            self.guard.resolve(self.workspace, "."),
            self.workspace,
        )
        self.assertEqual(
            self.guard.resolve(self.workspace, "src/../src"),
            self.source_directory,
        )

    def test_resolves_internal_directory_symlink_to_real_path(self) -> None:
        internal_link = self.workspace / "source-link"
        internal_link.symlink_to(self.source_directory, target_is_directory=True)

        self.assertEqual(
            self.guard.resolve(self.workspace, "source-link"),
            self.source_directory,
        )

    def test_rejects_parent_absolute_and_symlink_workspace_escape(self) -> None:
        outside_link = self.workspace / "outside-link"
        outside_link.symlink_to(self.outside_directory, target_is_directory=True)
        missing_outside = self.outside_directory / "missing"

        for requested_cwd in (
            "../outside",
            self.outside_directory,
            missing_outside,
            outside_link,
        ):
            with self.subTest(requested_cwd=requested_cwd):
                with self.assertRaises(WorkingDirectoryRejectedError) as raised:
                    self.guard.resolve(self.workspace, requested_cwd)
                self.assertEqual(
                    raised.exception.status,
                    ToolResultStatus.REJECTED,
                )

    def test_missing_path_and_file_path_are_ordinary_errors(self) -> None:
        with self.assertRaises(WorkingDirectoryNotFoundError) as missing:
            self.guard.resolve(self.workspace, "missing")
        with self.assertRaises(WorkingDirectoryTypeError) as wrong_type:
            self.guard.resolve(self.workspace, "src/main.py")

        self.assertEqual(missing.exception.status, ToolResultStatus.ERROR)
        self.assertEqual(wrong_type.exception.status, ToolResultStatus.ERROR)

    def test_invalid_cwd_is_an_ordinary_error_with_safe_message(self) -> None:
        for requested_cwd in ("", "   "):
            with self.subTest(requested_cwd=requested_cwd):
                with self.assertRaises(WorkingDirectoryError) as raised:
                    self.guard.resolve(self.workspace, requested_cwd)
                self.assertEqual(raised.exception.status, ToolResultStatus.ERROR)

        with self.assertRaises(WorkingDirectoryError) as nul_error:
            self.guard.resolve(self.workspace, "bad\x00cwd")
        self.assertIn(r"bad\x00cwd", str(nul_error.exception))
        self.assertNotIn("\x00", str(nul_error.exception))

    def test_requires_prevalidated_canonical_task_workspace(self) -> None:
        workspace_link = self.root / "workspace-link"
        workspace_link.symlink_to(self.workspace, target_is_directory=True)
        unnormalized_workspace = self.workspace / ".." / self.workspace.name

        for workspace in (
            "",
            Path(self.workspace.name),
            unnormalized_workspace,
            workspace_link,
        ):
            with self.subTest(workspace=workspace):
                with self.assertRaises(WorkingDirectoryConfigurationError):
                    self.guard.resolve(workspace, ".")


if __name__ == "__main__":
    unittest.main()
