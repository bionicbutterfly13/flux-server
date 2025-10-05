"""
Flux Server - Consciousness-Enhanced Knowledge Retrieval
Standalone microservice for querying knowledge graphs

Architecture:
- Query Engine: Natural language query processing
- Neo4j Searcher: Graph + vector + full-text unified search
- Response Synthesizer: LLM-based answer generation
- CLAUSE: Consciousness-enhanced path navigation

Installation:
- Native installation (no Docker required)
- Uses local Neo4j and Redis instances
- Designed for customer deployment
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.config.settings import settings, get_cors_config

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure Flux Server application."""

    app = FastAPI(
        title="Flux Server",
        description="Consciousness-enhanced knowledge retrieval and query system",
        version="1.0.0",
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

    # Import routes (deferred to avoid circular imports)
    from src.api.routes import query, clause

    # Register routes
    app.include_router(query.router)
    app.include_router(clause.router)

    @app.get("/")
    async def root():
        """Root endpoint - service information"""
        return {
            "service": "Flux Server",
            "version": "1.0.0",
            "description": "Consciousness-enhanced knowledge retrieval",
            "status": "operational",
            "endpoints": {
                "query": "/api/query",
                "clause_navigate": "/api/clause/navigate",
                "clause_curate": "/api/clause/curate",
                "health": "/health",
                "docs": "/docs"
            }
        }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        # TODO: Add database connection checks
        return {
            "status": "ok",
            "service": "flux-server",
            "version": "1.0.0"
        }

    logger.info(f"Flux Server initialized on port {settings.PORT}")
    return app

# Create app instance
app = create_app()
