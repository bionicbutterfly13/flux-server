"""Environment configuration management for ThoughtSeed pipeline."""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Server configuration
    APP_NAME: str = "ThoughtSeed Pipeline API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database connections
    REDIS_URL: str = "redis://localhost:6379"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    # QDRANT_URL removed 2025-10-01: Using Neo4j unified search only

    # API Keys
    OPENAI_API_KEY: Optional[str] = None

    # File processing limits (from clarification)
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    MAX_FILES_PER_BATCH: int = 1000
    MAX_TOTAL_BATCH_SIZE: int = 500 * 1024 * 1024 * 1024  # 500GB

    # ThoughtSeed processing
    CONSCIOUSNESS_DETECTION_THRESHOLD: float = 0.3
    VECTOR_DIMENSIONS: int = 384
    THOUGHTSEED_LAYERS: int = 5

    # Redis TTL settings (from clarification)
    NEURONAL_PACKET_TTL: int = 24 * 60 * 60  # 24 hours
    ATTRACTOR_BASIN_TTL: int = 7 * 24 * 60 * 60  # 7 days
    PROCESSING_RESULTS_TTL: int = 30 * 24 * 60 * 60  # 30 days

    # WebSocket configuration
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30
    WEBSOCKET_TIMEOUT: int = 60

    # CORS settings
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Performance
    WORKER_PROCESSES: int = 1
    MAX_CONCURRENT_UPLOADS: int = 10
    PROCESSING_QUEUE_SIZE: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra env vars without error

# Global settings instance
settings = Settings()

def get_database_url(db_type: str) -> str:
    """Get database URL by type."""
    urls = {
        "redis": settings.REDIS_URL,
        "neo4j": settings.NEO4J_URI,
    }
    return urls.get(db_type, "")

def is_development() -> bool:
    """Check if running in development mode."""
    return settings.DEBUG or os.getenv("ENVIRONMENT", "development") == "development"

def get_cors_config() -> dict:
    """Get CORS configuration."""
    return {
        "allow_origins": settings.CORS_ORIGINS,
        "allow_credentials": settings.CORS_CREDENTIALS,
        "allow_methods": settings.CORS_METHODS,
        "allow_headers": settings.CORS_HEADERS,
    }