import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings


class WebConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.allowed_root = Path(self.temporary_directory.name)

    def test_cors_origins_are_explicit_and_normalized(self) -> None:
        config = Settings(
            DATABASE_URL="sqlite+pysqlite:///:memory:",
            ALLOWED_WORKSPACE_ROOT=self.allowed_root,
            WEB_CORS_ALLOWED_ORIGINS=(
                "http://localhost:5173/",
                "https://frontend.example.com",
            ),
        )

        self.assertEqual(
            config.WEB_CORS_ALLOWED_ORIGINS,
            (
                "http://localhost:5173",
                "https://frontend.example.com",
            ),
        )

    def test_cors_rejects_wildcards_duplicates_and_non_origins(self) -> None:
        invalid_values = (
            ("*",),
            ("http://localhost:5173", "http://localhost:5173/"),
            ("file:///tmp/frontend",),
            ("https://frontend.example.com/path",),
            ("https://user:password@frontend.example.com",),
        )
        for origins in invalid_values:
            with self.subTest(origins=origins):
                with self.assertRaises(ValidationError):
                    Settings(
                        DATABASE_URL="sqlite+pysqlite:///:memory:",
                        ALLOWED_WORKSPACE_ROOT=self.allowed_root,
                        WEB_CORS_ALLOWED_ORIGINS=origins,
                    )


if __name__ == "__main__":
    unittest.main()
