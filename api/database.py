"""
database.py — SQLAlchemy engine/session setup.

The database URL is configurable via the DATABASE_URL environment variable
so PostgreSQL (or any other SQLAlchemy-supported database) can be introduced
later WITHOUT changing any application code -- the ORM models, session
handling, and query logic in this package are database-agnostic. Default is
a local SQLite file, which needs no running service and no deployment setup,
appropriate for this slice (no Redis/workers/deployment yet, per scope).

Example for a future Postgres switch (no code changes needed elsewhere):
    DATABASE_URL=postgresql://user:password@localhost:5432/hiver_tickets
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./tickets.db")

# check_same_thread=False is only needed/safe for SQLite (FastAPI may handle
# a request in a different thread than the one that created the connection,
# since routes run in a thread pool). This flag is ignored by other DBs, but
# we still gate it to avoid passing a SQLite-only kwarg to e.g. Postgres.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all tables if they don't exist. Called at app startup."""
    import api.db_models  # noqa: F401 -- ensures models are registered on Base before create_all
    Base.metadata.create_all(bind=engine)
