"""
Flux Server - Minimal Version (Query Only)
Consciousness-Enhanced Knowledge Retrieval
Standalone microservice for querying knowledge graphs

This is the minimal working version with just query functionality.
CLAUSE features will be added incrementally once imports are fixed.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.config.settings import settings, get_cors_config

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure minimal Flux Server application."""

    app = FastAPI(
        title="Flux Server (Minimal)",
        description="Consciousness-enhanced knowledge retrieval - Query only",
        version="1.0.0-minimal",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS
    cors_config = get_cors_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config["allow_origins"],
        allow_credentials=cors_config["allow_credentials"],
        allow_methods=cors_config["allow_methods"],
        allow_headers=cors_config["allow_headers"],
    )

    # Import only query route (CLAUSE disabled for now)
    try:
        from src.api.routes import query
        app.include_router(query.router)
        logger.info("✅ Query route loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load query route: {e}")

    @app.get("/")
    async def root():
        """Root endpoint - service information"""
        return {
            "service": "Flux Server",
            "version": "1.0.0-minimal",
            "description": "Consciousness-enhanced knowledge retrieval (query only)",
            "status": "operational",
            "endpoints": {
                "query": "/api/query",
                "health": "/health",
                "docs": "/docs"
            },
            "note": "CLAUSE features will be added once imports are fixed"
        }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "ok",
            "service": "flux-server-minimal",
            "version": "1.0.0-minimal"
        }

    logger.info(f"✅ Flux Server (Minimal) initialized on port {settings.PORT}")
    return app

# Create app instance
app = create_app()
