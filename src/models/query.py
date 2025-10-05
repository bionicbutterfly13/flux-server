"""
Query Model - Natural language question submitted to research engine
Per Spec 006 data model specification
"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import uuid


class Query(BaseModel):
    """
    Natural language query submitted to research engine.

    Represents a user's question that will be processed through
    Neo4j graph search and vector semantic search, creating a
    ThoughtSeed that flows through consciousness processing.
    """

    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this query"
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question from user"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When query was submitted"
    )

    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for authentication"
    )

    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session context for follow-up questions"
    )

    thoughtseed_id: Optional[str] = Field(
        default=None,
        description="Associated ThoughtSeed ID for consciousness tracking"
    )

    @validator('question')
    def validate_question_not_empty(cls, v):
        """Ensure question is not empty or only whitespace"""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "What neural architectures show consciousness emergence?",
                "timestamp": "2025-09-30T20:30:00Z",
                "user_id": "researcher-123",
                "context": {"previous_query_id": "previous-uuid"},
                "thoughtseed_id": "seed-789"
            }
        }
