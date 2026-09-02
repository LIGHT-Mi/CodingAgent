import tempfile
import unittest
from pathlib import Path

from app.api.workspace import (
    WorkspaceConfigurationError,
    WorkspaceValidationError,
    WorkspaceValidator,
)


class WorkspaceValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.allowed_root = Path(self.temporary_directory.name) / "allowed"
        self.allowed_root.mkdir()
        self.validator = WorkspaceValidator(self.allowed_root)

    def test_return_canonical_absolute_workspace_path(self) -> None:
        workspace = self.allowed_root / "project"
        workspace.mkdir()
        unnormalized = workspace / ".." / workspace.name

        validated = self.validator.validate(str(unnormalized))

        self.assertEqual(validated, workspace.resolve())
        self.assertTrue(validated.is_absolute())

    def test_allow_the_configured_root_itself(self) -> None:
        self.assertEqual(
            self.validator.validate(self.allowed_root),
            self.allowed_root.resolve(),
        )

    def test_reject_empty_workspace_path(self) -> None:
        for workspace in ("", "   "):
            with self.subTest(workspace=workspace):
                with self.assertRaises(WorkspaceValidationError):
                    self.validator.validate(workspace)

    def test_reject_missing_path_and_regular_file(self) -> None:
        regular_file = self.allowed_root / "file.txt"
        regular_file.write_text("content", encoding="utf-8")

        for workspace in (self.allowed_root / "missing", regular_file):
            with self.subTest(workspace=workspace):
                with self.assertRaises(WorkspaceValidationError):
                    self.validator.validate(workspace)

    def test_reject_workspace_outside_allowed_root(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()

        with self.assertRaises(WorkspaceValidationError):
            self.validator.validate(outside)

    def test_resolve_symlink_before_checking_workspace_boundary(self) -> None:
        outside = Path(self.temporary_directory.name) / "outside"
        outside.mkdir()
        link = self.allowed_root / "outside-link"
        link.symlink_to(outside, target_is_directory=True)

        with self.assertRaises(WorkspaceValidationError):
            self.validator.validate(link)

    def test_reject_invalid_allowed_root_configuration(self) -> None:
        missing_root = Path(self.temporary_directory.name) / "missing-root"
        regular_file = Path(self.temporary_directory.name) / "root-file"
        regular_file.write_text("content", encoding="utf-8")

        for allowed_root in ("", missing_root, regular_file):
            with self.subTest(allowed_root=allowed_root):
                with self.assertRaises(WorkspaceConfigurationError):
                    WorkspaceValidator(allowed_root)

    def test_reject_non_path_input(self) -> None:
        with self.assertRaises(TypeError):
            self.validator.validate(object())


if __name__ == "__main__":
    unittest.main()
