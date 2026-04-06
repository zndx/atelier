"""SQLAlchemy ORM models for Atelier state."""

from sqlalchemy import BigInteger, Column, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Agent(Base):
    """Persisted keystone agent definition."""

    __tablename__ = "agents"

    id = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    role = Column(String, nullable=True)
    tool_ids = Column(Text, nullable=True)  # JSON array


class Dataset(Base):
    """Reference to a classification parquet dataset."""

    __tablename__ = "datasets"

    id = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=True)
    parquet_path = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    row_count = Column(BigInteger, nullable=True)
