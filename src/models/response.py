"""
Response Models - Query response with synthesized answer and sources
Per Spec 006 data model specification
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid


class SearchSource(str, Enum):
    """Source database for search results"""
    NEO4J = "neo4j"
    QDRANT = "qdrant"


class SearchResult(BaseModel):
    """
    Single search result from Neo4j or Qdrant.

    Represents a piece of information retrieved from databases
    with relevance scoring and source attribution.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this result"
    )

    source: SearchSource = Field(
        ...,
        description="Which database this result came from"
    )

    content: str = Field(
        ...,
        description="Text content of the search result"
    )

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score in range [0, 1]"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from database (node type, properties, etc.)"
    )

    relationships: List[str] = Field(
        default_factory=list,
        description="Graph relationships for Neo4j results (e.g., EVOLVED_FROM, HAS_STATE)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "result_id": "result-123",
                "source": "neo4j",
                "content": "Architecture demonstrates consciousness through self-reflection",
                "relevance_score": 0.92,
                "metadata": {"node_type": "Architecture", "consciousness_level": "high"},
                "relationships": ["EVOLVED_FROM", "HAS_STATE", "EXHIBITS"]
            }
        }


class QueryResponse(BaseModel):
    """
    Complete response to a query with synthesized answer and sources.

    Combines results from multiple database searches into a coherent
    response, tracking confidence and processing performance.
    """

    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this response"
    )

    query_id: str = Field(
        ...,
        description="ID of the query this responds to"
    )

    answer: str = Field(
        ...,
        min_length=1,
        description="Synthesized answer combining multiple sources"
    )

    sources: List[SearchResult] = Field(
        default_factory=list,
        description="Source search results used to generate answer"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in the response accuracy [0, 1]"
    )

    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Time taken to process query in milliseconds"
    )

    thoughtseed_trace: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional consciousness processing trace from ThoughtSeed flow"
    )

    @validator('confidence')
    def validate_confidence_range(cls, v):
        """Ensure confidence is in valid range"""
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    @validator('answer')
    def validate_answer_not_empty(cls, v):
        """Ensure answer is substantive"""
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        if len(v.strip()) < 10:
            raise ValueError("Answer must be substantive (at least 10 characters)")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "resp-456",
                "query_id": "query-123",
                "answer": "Neural architectures demonstrate consciousness emergence through...",
                "sources": [
                    {
                        "result_id": "r1",
                        "source": "neo4j",
                        "content": "Graph-based evidence",
                        "relevance_score": 0.9,
                        "metadata": {},
                        "relationships": ["EVOLVED_FROM"]
                    }
                ],
                "confidence": 0.85,
                "processing_time_ms": 1234,
                "thoughtseed_trace": {
                    "layers_processed": ["L1", "L2", "L3"],
                    "consciousness_level": "high"
                }
            }
        }
