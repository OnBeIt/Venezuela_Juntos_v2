import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, LargeBinary, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(exist_ok=True)

DB_URL = f"sqlite:///{DATA_DIR / 'db.sqlite'}"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class MissingPerson(Base):
    __tablename__ = "missing_persons"

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    photo_path = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    matches = relationship("Match", back_populates="missing_person", cascade="all, delete-orphan")


class FoundPerson(Base):
    __tablename__ = "found_persons"

    id = Column(Integer, primary_key=True)
    contact_info = Column(Text, nullable=False)
    notes = Column(Text, nullable=True)
    photo_path = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    matches = relationship("Match", back_populates="found_person", cascade="all, delete-orphan")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    missing_id = Column(Integer, ForeignKey("missing_persons.id"), nullable=False)
    found_id = Column(Integer, ForeignKey("found_persons.id"), nullable=False)
    similarity = Column(Float, nullable=False)
    matched_at = Column(DateTime, default=datetime.utcnow)

    missing_person = relationship("MissingPerson", back_populates="matches")
    found_person = relationship("FoundPerson", back_populates="matches")


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
