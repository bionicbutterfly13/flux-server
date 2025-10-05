"""
T010: Path Navigator Models

Pydantic V2 models for PathNavigator agent per Spec 035.
Defines request/response schemas for POST /api/clause/navigate
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PathNavigationRequest(BaseModel):
    """Input for POST /api/clause/navigate"""

    query: str = Field(..., min_length=3, description="Natural language query")
    start_node: str = Field(..., description="Starting concept node ID")
    step_budget: int = Field(
        default=10, ge=1, le=20, description="Maximum navigation steps"
    )
    enable_thoughtseeds: bool = Field(
        default=True, description="Generate ThoughtSeeds during exploration"
    )
    enable_curiosity: bool = Field(
        default=True, description="Trigger curiosity agents on prediction errors"
    )
    enable_causal: bool = Field(
        default=True, description="Use causal reasoning for path selection"
    )
    curiosity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Prediction error threshold for curiosity",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What causes climate change?",
                    "start_node": "climate_change",
                    "step_budget": 10,
                    "enable_thoughtseeds": True,
                    "enable_curiosity": True,
                    "enable_causal": True,
                    "curiosity_threshold": 0.7,
                }
            ]
        }
    }


class PathStep(BaseModel):
    """Single step in navigation path"""

    step: int = Field(..., ge=1, description="Step number (1-indexed)")
    from_node: str = Field(..., description="Source node")
    to_node: str = Field(..., description="Target node")
    relation: str = Field(..., description="Edge relation type")
    action: str = Field(
        ...,
        pattern="^(CONTINUE|BACKTRACK|STOP)$",
        description="Navigator action taken",
    )
    causal_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Causal intervention score"
    )
    thoughtseed_id: Optional[str] = Field(None, description="Generated ThoughtSeed ID")


class PathNavigationResponse(BaseModel):
    """Output from POST /api/clause/navigate"""

    path: Dict[str, Any] = Field(..., description="Navigation path details")
    metadata: Dict[str, Any] = Field(..., description="Path metadata (budgets, triggers)")
    performance: Dict[str, float] = Field(..., description="Latency breakdown")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "path": {
                        "nodes": ["climate_change", "greenhouse_gases", "CO2_emissions"],
                        "edges": [
                            {
                                "from": "climate_change",
                                "relation": "caused_by",
                                "to": "greenhouse_gases",
                            }
                        ],
                        "steps": [
                            {
                                "step": 1,
                                "from": "climate_change",
                                "to": "greenhouse_gases",
                                "action": "CONTINUE",
                                "causal_score": 0.85,
                            }
                        ],
                    },
                    "metadata": {
                        "budget_used": 3,
                        "budget_total": 10,
                        "final_action": "STOP",
                        "thoughtseeds_generated": 12,
                        "curiosity_triggers_spawned": 2,
                    },
                    "performance": {
                        "latency_ms": 145,
                        "thoughtseed_gen_ms": 23,
                        "causal_pred_ms": 87,
                    },
                }
            ]
        }
    }
