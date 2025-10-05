"""
T013: Provenance Models

Pydantic V2 models for provenance tracking per Spec 032.
Integrates with ContextCurator to track evidence lineage and trust signals.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class TrustSignals(BaseModel):
    """Trust signal scores for evidence evaluation"""

    reputation_score: float = Field(
        ..., ge=0.0, le=1.0, description="Source reputation (0-1)"
    )
    recency_score: float = Field(
        ..., ge=0.0, le=1.0, description="Information recency (0-1)"
    )
    semantic_consistency: float = Field(
        ..., ge=0.0, le=1.0, description="Consistency with query (0-1)"
    )


class ProvenanceMetadata(BaseModel):
    """Full provenance tracking per Spec 032"""

    source_uri: str = Field(..., description="Neo4j URI or document source")
    extraction_timestamp: datetime = Field(..., description="When evidence was extracted")
    extractor_identity: str = Field(..., description="Agent/service that extracted evidence")
    supporting_evidence: str = Field(
        ..., max_length=200, description="Evidence snippet (200 chars max)"
    )
    verification_status: str = Field(
        ...,
        pattern="^(verified|pending_review|unverified)$",
        description="Verification status",
    )
    corroboration_count: int = Field(
        ..., ge=0, description="Number of corroborating sources"
    )
    trust_signals: TrustSignals = Field(..., description="Trust signal scores")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_uri": "neo4j://concept/greenhouse_gases",
                    "extraction_timestamp": "2025-10-02T10:30:15Z",
                    "extractor_identity": "ContextCurator-v2.0",
                    "supporting_evidence": "Greenhouse gases trap heat in the atmosphere by absorbing infrared radiation...",
                    "verification_status": "verified",
                    "corroboration_count": 5,
                    "trust_signals": {
                        "reputation_score": 0.95,
                        "recency_score": 0.88,
                        "semantic_consistency": 0.91,
                    },
                }
            ]
        }
    }
