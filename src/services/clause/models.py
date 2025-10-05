#!/usr/bin/env python3
"""
CLAUSE Phase 1 Data Models

Pydantic models for CLAUSE Subgraph Architect with basin strengthening.
Following OpenAPI contract from specs/034-clause-phase1-foundation/contracts/architect_api.yaml
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class SubgraphRequest(BaseModel):
    """Request model for subgraph construction"""

    query: str = Field(..., min_length=1, description="Search query for subgraph construction")
    edge_budget: int = Field(default=50, ge=1, le=1000, description="Maximum edges to select")
    lambda_edge: float = Field(default=0.2, ge=0.0, le=1.0, description="Edge cost weight for shaped gain")
    hop_distance: int = Field(default=2, ge=1, le=5, description="Maximum hop distance from query concepts")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not empty after stripping"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class EdgeScore(BaseModel):
    """Edge scoring breakdown with 5 CLAUSE signals"""

    edge_key: str = Field(..., description="Edge identifier (source_relation_target)")
    source: str = Field(..., description="Source concept")
    relation: str = Field(..., description="Relation type")
    target: str = Field(..., description="Target concept")

    # 5 CLAUSE signals
    phi_ent: float = Field(..., ge=0.0, le=1.0, description="Entity relevance signal")
    phi_rel: float = Field(..., ge=0.0, le=1.0, description="Relation relevance signal")
    phi_nbr: float = Field(..., ge=0.0, le=1.0, description="Neighborhood signal")
    phi_deg: float = Field(..., ge=0.0, le=1.0, description="Degree signal")
    phi_basin: float = Field(..., ge=0.0, le=1.0, description="Basin strength signal")

    # Aggregated score
    total_score: float = Field(..., ge=0.0, le=1.0, description="Weighted sum of signals")
    shaped_gain: float = Field(..., description="Shaped gain: score - λ × cost")


class SubgraphResponse(BaseModel):
    """Response model for subgraph construction"""

    selected_edges: List[Dict[str, str]] = Field(..., description="Selected edges in subgraph")
    edge_scores: Dict[str, float] = Field(..., description="Edge scores (edge_key → total_score)")
    shaped_gains: Dict[str, float] = Field(..., description="Shaped gains (edge_key → gain)")
    budget_used: int = Field(..., ge=0, description="Number of edges selected")
    stopped_reason: str = Field(
        ...,
        description="Reason construction stopped (budget_exhausted, shaped_gain_zero, no_more_candidates)",
    )

    # Optional detailed breakdown
    signal_breakdown: Optional[List[EdgeScore]] = Field(None, description="Detailed signal breakdown")
    computation_time_ms: Optional[float] = Field(None, ge=0.0, description="Computation time in milliseconds")


class BasinStrengtheningRequest(BaseModel):
    """Request model for basin strengthening"""

    concepts: List[str] = Field(..., min_length=1, description="List of concepts to strengthen")
    document_id: str = Field(..., min_length=1, description="Document identifier")
    increment: float = Field(default=0.2, ge=0.0, le=1.0, description="Strength increment per activation")

    @field_validator("concepts")
    @classmethod
    def concepts_not_empty(cls, v: List[str]) -> List[str]:
        """Validate concepts list is not empty and all are non-empty strings"""
        if not v:
            raise ValueError("Concepts list cannot be empty")

        for concept in v:
            if not concept or not concept.strip():
                raise ValueError("Each concept must be a non-empty string")

        return [c.strip() for c in v]

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        """Validate document_id is not empty"""
        if not v.strip():
            raise ValueError("Document ID cannot be empty")
        return v.strip()


class BasinInfo(BaseModel):
    """Basin information for responses"""

    basin_id: str = Field(..., description="Basin identifier")
    basin_name: str = Field(..., description="Basin name")
    strength: float = Field(..., ge=1.0, le=2.0, description="Basin strength (1.0-2.0)")
    activation_count: int = Field(..., ge=0, description="Number of activations")
    co_occurring_concepts: Dict[str, int] = Field(default_factory=dict, description="Co-occurrence counts")


class BasinStrengtheningResponse(BaseModel):
    """Response model for basin strengthening"""

    updated_basins: List[BasinInfo] = Field(..., description="Basins that were updated")
    new_basins: List[BasinInfo] = Field(..., description="Basins that were created")
    cooccurrence_updates: Dict[str, List[str]] = Field(
        ..., description="Co-occurrence updates (concept → [co-occurred concepts])"
    )
    total_strengthening_time_ms: float = Field(..., ge=0.0, description="Total processing time in milliseconds")


class EdgeScoringRequest(BaseModel):
    """Request model for edge scoring"""

    edges: List[Dict[str, str]] = Field(..., min_length=1, description="Edges to score")
    query: str = Field(..., min_length=1, description="Query for relevance scoring")
    signal_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "entity": 0.25,
            "relation": 0.25,
            "neighborhood": 0.20,
            "degree": 0.15,
            "basin": 0.15,
        },
        description="Signal weights (must sum to 1.0)",
    )

    @field_validator("edges")
    @classmethod
    def edges_valid(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate edges have required fields"""
        if not v:
            raise ValueError("Edges list cannot be empty")

        for edge in v:
            if "source" not in edge or "relation" not in edge or "target" not in edge:
                raise ValueError("Each edge must have 'source', 'relation', 'target' fields")

            if not edge["source"] or not edge["relation"] or not edge["target"]:
                raise ValueError("Edge fields cannot be empty")

        return v

    @field_validator("signal_weights")
    @classmethod
    def weights_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate signal weights sum to 1.0"""
        required_signals = {"entity", "relation", "neighborhood", "degree", "basin"}

        if set(v.keys()) != required_signals:
            raise ValueError(f"Must have weights for all signals: {required_signals}")

        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Signal weights must sum to 1.0, got {total}")

        return v


class EdgeScoringResponse(BaseModel):
    """Response model for edge scoring"""

    edge_scores: Dict[str, float] = Field(..., description="Edge scores (edge_key → total_score)")
    signal_breakdown: Dict[str, Dict[str, float]] = Field(
        ..., description="Signal breakdown per edge (edge_key → signal → value)"
    )
    computation_time_ms: float = Field(..., ge=0.0, description="Computation time in milliseconds")
