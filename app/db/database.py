import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

logger = logging.getLogger(__name__)

# Get database configuration from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "meeran")
DB_PASSWORD = os.getenv("DB_PASSWORD", "toor")
DB_NAME = os.getenv("DB_NAME", "volt")

# Construct database URL from individual components
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Convert the PostgreSQL URL to async format for asyncpg
DATABASE_URL_ASYNC = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create SQLAlchemy engine with connection pooling that doesn't fail on import
engine = create_engine(
    DATABASE_URL, 
    pool_recycle=3600, 
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"connect_timeout": 10}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create async engine and session
if DATABASE_URL_ASYNC:
    async_engine = create_async_engine(
        DATABASE_URL_ASYNC, pool_recycle=3600, pool_pre_ping=True
    )
    AsyncSessionLocal = sessionmaker(
        bind=async_engine, expire_on_commit=False, class_=AsyncSession
    )
    # Create an alias for AsyncSessionLocal
    async_session_maker = AsyncSessionLocal

# Base class for models
Base = declarative_base()


def get_db() -> Session:
    """
    Get database session.
    This is used as a dependency in FastAPI routes to get a database session.

    Returns:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SessionManager:
    def __init__(self):
        self.session = SessionLocal()
        self.session.expire_on_commit = False

    def __enter__(self):
        return self.session

    def __exit__(self, type, value, traceback):
        self.session.close()


def get_db_session():
    return SessionManager()


class AsyncSessionManager:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = AsyncSessionLocal()
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:  # No exception, commit the transaction
                await self.session.commit()
            else:  # Exception occurred, rollback
                await self.session.rollback()
        finally:
            await self.session.close()


async def get_async_db_session():
    return AsyncSessionManager()


async def get_async_db():
    """
    Get async database session.
    This is used as a dependency in FastAPI routes to get an async database session.

    Returns:
        AsyncSession: SQLAlchemy async database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def init_db():
    """
    Initialize the database.
    Creates all tables if they don't exist.
    This should only be used for development or testing.
    In production, use Alembic migrations instead.
    """
    try:
        # Import the models to ensure they're registered with the metadata
        from app.db.models import Assistant, Call, Recording, Transcript

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise Exception(f"Error initializing database: {e}")
