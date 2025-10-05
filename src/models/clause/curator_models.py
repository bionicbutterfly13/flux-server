"""
T011: Context Curator Models

Pydantic V2 models for ContextCurator agent per Spec 035.
Defines request/response schemas for POST /api/clause/curate
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any
from src.models.clause.provenance_models import ProvenanceMetadata


class ContextCurationRequest(BaseModel):
    """Input for POST /api/clause/curate"""

    evidence_pool: List[str] = Field(
        ..., min_length=1, description="Pool of evidence snippets to curate"
    )
    token_budget: int = Field(
        default=2048,
        ge=100,
        le=8192,
        description="Maximum tokens for selected evidence",
    )
    enable_provenance: bool = Field(
        default=True, description="Add full provenance metadata"
    )
    lambda_tok: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Token cost multiplier for shaped utility",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "evidence_pool": [
                        "Greenhouse gases trap heat in the atmosphere...",
                        "CO2 is the primary greenhouse gas from human activity...",
                    ],
                    "token_budget": 2048,
                    "enable_provenance": True,
                    "lambda_tok": 0.01,
                }
            ]
        }
    }


class SelectedEvidence(BaseModel):
    """Evidence snippet with provenance"""

    text: str = Field(..., description="Evidence text")
    tokens: int = Field(..., ge=1, description="Token count (tiktoken)")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    shaped_utility: float = Field(..., description="Score minus token cost")
    provenance: ProvenanceMetadata = Field(..., description="Full provenance metadata")


class ContextCurationResponse(BaseModel):
    """Output from POST /api/clause/curate"""

    selected_evidence: List[SelectedEvidence] = Field(
        ..., description="Curated evidence with provenance"
    )
    metadata: Dict[str, Any] = Field(..., description="Curation metadata")
    performance: Dict[str, float] = Field(..., description="Latency breakdown")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "selected_evidence": [
                        {
                            "text": "Greenhouse gases trap heat...",
                            "tokens": 156,
                            "score": 0.92,
                            "shaped_utility": 0.904,
                            "provenance": {
                                "source_uri": "neo4j://concept/greenhouse_gases",
                                "extraction_timestamp": "2025-10-02T10:30:15Z",
                                "extractor_identity": "ContextCurator-v2.0",
                                "supporting_evidence": "Greenhouse gases trap heat...",
                                "verification_status": "verified",
                                "corroboration_count": 5,
                                "trust_signals": {
                                    "reputation_score": 0.95,
                                    "recency_score": 0.88,
                                    "semantic_consistency": 0.91,
                                },
                            },
                        }
                    ],
                    "metadata": {
                        "tokens_used": 428,
                        "tokens_total": 2048,
                        "learned_stop_triggered": True,
                    },
                    "performance": {
                        "latency_ms": 78,
                        "provenance_overhead_ms": 12,
                    },
                }
            ]
        }
    }
