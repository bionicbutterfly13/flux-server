"""
T017: Shared Models

Pydantic V2 shared models used across CLAUSE Phase 2 agents.
Includes StateEncoding (Navigator) and BudgetUsage (all agents).
"""

from pydantic import BaseModel, Field, computed_field
from typing import List
import numpy as np


class StateEncoding(BaseModel):
    """Navigator state representation (1154-dim feature vector)"""

    query_embedding: List[float] = Field(
        ..., min_length=384, max_length=384, description="Query embedding (384-dim)"
    )
    node_embedding: List[float] = Field(
        ..., min_length=384, max_length=384, description="Current node embedding"
    )
    node_degree: int = Field(..., ge=0, description="Node degree in graph")
    basin_strength: float = Field(..., ge=1.0, le=2.0, description="Basin strength")
    neighborhood_mean: List[float] = Field(
        ..., min_length=384, max_length=384, description="Neighborhood mean embedding"
    )
    budget_remaining: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized budget remaining"
    )

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array for ML processing"""
        return np.concatenate(
            [
                self.query_embedding,
                self.node_embedding,
                [self.node_degree, self.basin_strength],
                self.neighborhood_mean,
                [self.budget_remaining],
            ]
        )

    model_config = {"arbitrary_types_allowed": True}


class BudgetUsage(BaseModel):
    """Track budget consumption across agents"""

    edge_used: int = Field(default=0, ge=0, description="Edges used by Architect")
    step_used: int = Field(default=0, ge=0, description="Steps used by Navigator")
    token_used: int = Field(default=0, ge=0, description="Tokens used by Curator")

    edge_total: int = Field(..., ge=1, description="Total edge budget")
    step_total: int = Field(..., ge=1, description="Total step budget")
    token_total: int = Field(..., ge=1, description="Total token budget")

    @computed_field
    @property
    def edge_remaining(self) -> int:
        """Remaining edge budget"""
        return max(0, self.edge_total - self.edge_used)

    @computed_field
    @property
    def step_remaining(self) -> int:
        """Remaining step budget"""
        return max(0, self.step_total - self.step_used)

    @computed_field
    @property
    def token_remaining(self) -> int:
        """Remaining token budget"""
        return max(0, self.token_total - self.token_used)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "edge_used": 35,
                    "step_used": 7,
                    "token_used": 1523,
                    "edge_total": 50,
                    "step_total": 10,
                    "token_total": 2048,
                }
            ]
        }
    }
