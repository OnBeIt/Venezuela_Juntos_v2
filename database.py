import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, Text,
    create_engine, inspect, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(exist_ok=True)

DB_URL = f"sqlite:///{DATA_DIR / 'db.sqlite'}"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(Text, nullable=False, unique=True)
    password_hash = Column(Text, nullable=False)
    is_admin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class MissingPerson(Base):
    __tablename__ = "missing_persons"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    photo_path = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    quality_flag = Column(Boolean, nullable=False, default=False)
    quality_note = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User")
    matches = relationship("Match", back_populates="missing_person", cascade="all, delete-orphan")


class FoundPerson(Base):
    __tablename__ = "found_persons"

    id = Column(Integer, primary_key=True)
    contact_info = Column(Text, nullable=False)
    notes = Column(Text, nullable=True)
    photo_path = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    quality_flag = Column(Boolean, nullable=False, default=False)
    quality_note = Column(Text, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User")
    matches = relationship("Match", back_populates="found_person", cascade="all, delete-orphan")


class AuditLog(Base):
    """Append-only record of noteworthy platform events, shown in /admin/logs."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    event_type = Column(Text, nullable=False)   # e.g. "user.login", "report.missing"
    actor_email = Column(Text, nullable=True)    # who triggered it (if known)
    actor_id = Column(Integer, nullable=True)
    message = Column(Text, nullable=False)       # human-readable detail
    ip = Column(Text, nullable=True)


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    missing_id = Column(Integer, ForeignKey("missing_persons.id"), nullable=False)
    found_id = Column(Integer, ForeignKey("found_persons.id"), nullable=False)
    similarity = Column(Float, nullable=False)
    matched_at = Column(DateTime, default=datetime.utcnow)

    missing_person = relationship("MissingPerson", back_populates="matches")
    found_person = relationship("FoundPerson", back_populates="matches")


# Columns added after the app's initial release. On an existing Railway volume
# the tables already exist, so create_all() will not add these — we ALTER them
# in manually (SQLite supports ADD COLUMN). Each entry is (table, column, DDL).
_ADDED_COLUMNS = [
    ("missing_persons", "owner_id", "INTEGER REFERENCES users(id)"),
    ("missing_persons", "quality_flag", "BOOLEAN NOT NULL DEFAULT 0"),
    ("missing_persons", "quality_note", "TEXT"),
    ("found_persons", "owner_id", "INTEGER REFERENCES users(id)"),
    ("found_persons", "quality_flag", "BOOLEAN NOT NULL DEFAULT 0"),
    ("found_persons", "quality_note", "TEXT"),
]


def _run_migrations() -> None:
    """Idempotently add columns introduced after the first release.

    Guarded by the live table schema so it is safe to run on every startup and
    against the existing production database on the Railway volume.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, column, ddl in _ADDED_COLUMNS:
            if table not in existing_tables:
                continue  # create_all already built it with every column
            cols = {c["name"] for c in inspector.get_columns(table)}
            if column not in cols:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {column} {ddl}'))


def init_db() -> None:
    Base.metadata.create_all(engine)
    _run_migrations()


def get_session() -> Session:
    return Session(engine)
