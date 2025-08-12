import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlmodel import create_engine, Session, SQLModel
from sqlalchemy.pool import StaticPool


class DatabaseConfig:
    """Database configuration and connection management."""
    
    # Database connection settings
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5433")
    DB_USER = os.getenv("DB_USER", "cvollama")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "cvollama123")
    DB_NAME = os.getenv("DB_NAME", "cv_ollama")
    
    # Connection URL
    DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SYNC_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Engine configuration
    ENGINE_KWARGS = {
        "echo": os.getenv("DB_ECHO", "false").lower() == "true",
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self.async_engine: AsyncEngine = None
        self.sync_engine = None
    
    def initialize_engines(self):
        """Initialize both async and sync database engines."""
        self.async_engine = create_async_engine(
            DatabaseConfig.DATABASE_URL,
            **DatabaseConfig.ENGINE_KWARGS
        )
        
        self.sync_engine = create_engine(
            DatabaseConfig.SYNC_DATABASE_URL,
            **DatabaseConfig.ENGINE_KWARGS
        )
    
    async def create_tables(self):
        """Create all database tables."""
        if not self.async_engine:
            self.initialize_engines()
        
        async with self.async_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
    
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session."""
        if not self.async_engine:
            self.initialize_engines()
        
        async with AsyncSession(self.async_engine) as session:
            yield session
    
    def get_sync_session(self) -> Session:
        """Get sync database session."""
        if not self.sync_engine:
            self.initialize_engines()
        
        return Session(self.sync_engine)
    
    async def close(self):
        """Close database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.sync_engine:
            self.sync_engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session for FastAPI."""
    async for session in db_manager.get_async_session():
        yield session


def init_database():
    """Initialize database on application startup."""
    db_manager.initialize_engines()


async def create_tables():
    """Create all database tables."""
    await db_manager.create_tables()