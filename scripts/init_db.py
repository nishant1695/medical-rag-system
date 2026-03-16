"""
Database initialization script

Creates all tables and initial data.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
from app.models import Base


async def init_db():
    """Initialize database tables."""
    print(f"Connecting to database: {settings.DATABASE_URL}")

    engine = create_async_engine(settings.DATABASE_URL, echo=True)

    print("Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("✅ Database initialized successfully!")


if __name__ == "__main__":
    asyncio.run(init_db())
