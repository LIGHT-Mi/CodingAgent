import app.db.models  # noqa: F401
from app.db.base import Base
from app.db.session import engine


def reset_db() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    reset_db()
    print("Database tables have been rebuilt.")
