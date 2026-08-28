import unittest

from sqlalchemy import create_mock_engine

import app.db.models  # noqa: F401
from app.db.base import Base


class DatabaseSchemaTests(unittest.TestCase):
    def test_postgresql_can_create_and_drop_circular_foreign_key_schema(self) -> None:
        statements: list[str] = []

        def capture_sql(sql, *args, **kwargs) -> None:
            statements.append(str(sql.compile(dialect=engine.dialect)))

        engine = create_mock_engine("postgresql+psycopg://", capture_sql)

        Base.metadata.create_all(engine, checkfirst=False)
        Base.metadata.drop_all(engine, checkfirst=False)

        generated_sql = "\n".join(statements)
        self.assertIn(
            "ALTER TABLE messages ADD CONSTRAINT messages_tool_call_id_fkey",
            generated_sql,
        )
        self.assertIn(
            "ALTER TABLE messages DROP CONSTRAINT messages_tool_call_id_fkey",
            generated_sql,
        )


if __name__ == "__main__":
    unittest.main()
