"""
T015: Curiosity Models

Pydantic V2 models for curiosity triggers per Spec 029.
Integrates with PathNavigator to spawn background investigation agents.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class CuriosityTrigger(BaseModel):
    """Curiosity trigger for background investigation (Spec 029)"""

    trigger_type: str = Field(
        ..., pattern="^prediction_error$", description="Trigger type (prediction_error only)"
    )
    concept: str = Field(..., description="Concept that triggered curiosity")
    error_magnitude: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction error magnitude"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Trigger timestamp"
    )
    investigation_status: str = Field(
        default="queued",
        pattern="^(queued|investigating|completed)$",
        description="Investigation status",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "trigger_type": "prediction_error",
                    "concept": "quantum_computing",
                    "error_magnitude": 0.82,
                    "timestamp": "2025-10-02T10:30:15Z",
                    "investigation_status": "queued",
                }
            ]
        }
    }
