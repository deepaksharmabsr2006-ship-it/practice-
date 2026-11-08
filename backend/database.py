"""
Database Configuration & Engine Setup.
Uses SQLAlchemy for SQLite (local edge execution) with seamless support
for PostgreSQL (cloud deployment like Supabase, Render, Railway, AWS RDS).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL: Defaults to /tmp on Vercel serverless, or local file, or DATABASE_URL env var
if os.getenv("VERCEL"):
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/telematics_fleet.db")
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./telematics_fleet.db")

# Fix for Heroku / Render postgres:// vs postgresql:// scheme
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SQLite-specific connect args
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI Dependency for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
