"""
Query API Routes - Natural language query interface
Per Spec 006: POST /api/query endpoint
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from src.models.query import Query
from src.models.response import QueryResponse
from src.services.query_engine import QueryEngine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["query"])

# Lazy initialization of query engine to avoid database connection at import time
_query_engine: Optional[QueryEngine] = None


def get_query_engine() -> QueryEngine:
    """Get or create the query engine instance."""
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine()
    return _query_engine


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session context")
    thoughtseed_id: Optional[str] = Field(None, description="ThoughtSeed ID for tracking")


class QueryResponseModel(BaseModel):
    """Response model for query endpoint (extends QueryResponse for API)."""
    response_id: str
    query_id: str
    answer: str
    sources: list
    confidence: float
    processing_time_ms: int
    thoughtseed_trace: Optional[Dict[str, Any]] = None


@router.post("/query", response_model=QueryResponseModel, status_code=status.HTTP_200_OK)
async def process_query(request: QueryRequest) -> QueryResponseModel:
    """
    Process natural language query.

    Accepts a natural language question and returns synthesized answer
    with sources from both Neo4j graph database and Qdrant vector database.

    **Performance**: Target <2s response time
    **Concurrency**: Supports 10+ concurrent queries

    Args:
        request: QueryRequest with question and optional context

    Returns:
        QueryResponse with synthesized answer, sources, and confidence

    Raises:
        400: Invalid question (empty or too long)
        500: Internal processing error
    """
    try:
        # Validate request
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        # Process query through engine
        engine = get_query_engine()
        response = await engine.process_query(
            question=request.question,
            user_id=request.user_id,
            context=request.context,
            thoughtseed_id=request.thoughtseed_id
        )

        # Convert to response model
        return QueryResponseModel(
            response_id=response.response_id,
            query_id=response.query_id,
            answer=response.answer,
            sources=[s.model_dump() for s in response.sources],
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms,
            thoughtseed_trace=response.thoughtseed_trace
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid query request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error processing query"
        )


@router.get("/query/health", status_code=status.HTTP_200_OK)
async def query_health_check() -> Dict[str, Any]:
    """
    Check health of query engine components.

    Returns status of Neo4j, Qdrant, and overall query system.

    Returns:
        Health status dictionary
    """
    try:
        engine = get_query_engine()
        health = await engine.health_check()
        return {
            "status": "healthy" if health["overall"] else "degraded",
            "components": health,
            "message": "Query engine operational" if health["overall"] else "Some components unavailable"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "components": {"neo4j": False, "qdrant": False, "overall": False},
            "message": f"Health check error: {str(e)}"
        }
