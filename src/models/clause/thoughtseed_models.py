"""
T014: ThoughtSeed Models

Pydantic V2 models for ThoughtSeed generation per Spec 028.
Integrates with PathNavigator to enable cross-document linking and basin context.
"""

from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime


class BasinContext(BaseModel):
    """Attractor Basin context from Phase 1"""

    strength: float = Field(..., ge=1.0, le=2.0, description="Basin strength from Phase 1")
    activation_count: int = Field(..., ge=0, description="Activation count")
    co_occurring: Dict[str, int] = Field(..., description="Co-occurrence counts")


class ThoughtSeed(BaseModel):
    """ThoughtSeed for cross-document linking (Spec 028)"""

    id: str = Field(..., description="Unique ThoughtSeed ID")
    concept: str = Field(..., description="Concept node")
    source_doc: str = Field(..., description="Source document/query")
    basin_context: BasinContext = Field(..., description="Basin context from Phase 1")
    similarity_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Similarity threshold for linking"
    )
    linked_documents: List[str] = Field(
        default_factory=list, description="Cross-document links"
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "ts_climate_change_001",
                    "concept": "climate_change",
                    "source_doc": "query_2025-10-02_12345",
                    "basin_context": {
                        "strength": 1.8,
                        "activation_count": 42,
                        "co_occurring": {"greenhouse_gases": 15, "global_warming": 12},
                    },
                    "similarity_threshold": 0.8,
                    "linked_documents": ["doc_climate_001", "doc_climate_042"],
                    "created_at": "2025-10-02T10:30:15Z",
                }
            ]
        }
    }
