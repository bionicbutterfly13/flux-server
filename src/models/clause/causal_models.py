"""
T016: Causal Models

Pydantic V2 models for causal reasoning per Spec 033.
Integrates with PathNavigator to estimate intervention effects using do-calculus.
"""

from pydantic import BaseModel, Field


class CausalIntervention(BaseModel):
    """Causal intervention for path selection (Spec 033)"""

    intervention_node: str = Field(..., description="Node to intervene on")
    target_node: str = Field(..., description="Target outcome node")
    intervention_score: float = Field(
        ..., ge=0.0, le=1.0, description="P(target | do(intervention))"
    )
    computation_time_ms: float = Field(..., ge=0.0, description="Inference latency")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "intervention_node": "greenhouse_gases",
                    "target_node": "climate_change",
                    "intervention_score": 0.87,
                    "computation_time_ms": 23.5,
                }
            ]
        }
    }
