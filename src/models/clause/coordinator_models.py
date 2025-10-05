"""
T012: LC-MAPPO Coordinator Models

Pydantic V2 models for LC-MAPPO Coordinator per Spec 035.
Defines request/response schemas for POST /api/clause/coordinate
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class BudgetAllocation(BaseModel):
    """Budget allocation across three agents"""

    edge_budget: int = Field(
        default=50, ge=10, le=200, description="Subgraph edge budget (Architect)"
    )
    step_budget: int = Field(
        default=10, ge=1, le=20, description="Path navigation step budget (Navigator)"
    )
    token_budget: int = Field(
        default=2048,
        ge=100,
        le=8192,
        description="Evidence curation token budget (Curator)",
    )


class LambdaParameters(BaseModel):
    """Cost multipliers for shaped utility calculations"""

    edge: float = Field(default=0.01, ge=0.0, le=1.0, description="Edge cost multiplier")
    latency: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Latency cost multiplier"
    )
    token: float = Field(default=0.01, ge=0.0, le=1.0, description="Token cost multiplier")


class CoordinationRequest(BaseModel):
    """Input for POST /api/clause/coordinate"""

    query: str = Field(..., min_length=3, description="Natural language query")
    budgets: BudgetAllocation = Field(..., description="Budget allocation across agents")
    lambdas: LambdaParameters = Field(
        default_factory=LambdaParameters, description="Cost multipliers"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What causes climate change?",
                    "budgets": {
                        "edge_budget": 50,
                        "step_budget": 10,
                        "token_budget": 2048,
                    },
                    "lambdas": {
                        "edge": 0.01,
                        "latency": 0.01,
                        "token": 0.01,
                    },
                }
            ]
        }
    }


class AgentHandoff(BaseModel):
    """Track agent execution in coordination"""

    step: int = Field(
        ..., ge=1, description="Execution order (1=Architect, 2=Navigator, 3=Curator)"
    )
    agent: str = Field(
        ...,
        pattern="^(SubgraphArchitect|PathNavigator|ContextCurator)$",
        description="Agent identifier",
    )
    action: str = Field(..., description="Action taken by agent")
    budget_used: Dict[str, int] = Field(..., description="Budgets consumed")
    latency_ms: float = Field(..., ge=0.0, description="Agent execution latency")


class CoordinationResponse(BaseModel):
    """Output from POST /api/clause/coordinate"""

    result: Dict[str, Any] = Field(..., description="Combined results from all agents")
    agent_handoffs: List[AgentHandoff] = Field(..., description="Agent execution timeline")
    conflicts_detected: int = Field(..., ge=0, description="Number of write conflicts")
    conflicts_resolved: int = Field(..., ge=0, description="Number of conflicts resolved")
    performance: Dict[str, float] = Field(..., description="Total latency breakdown")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "result": {
                        "subgraph": {"nodes": [], "edges": []},
                        "path": {"nodes": [], "edges": []},
                        "evidence": [{"text": "...", "provenance": {}}],
                    },
                    "agent_handoffs": [
                        {
                            "step": 1,
                            "agent": "SubgraphArchitect",
                            "action": "built_subgraph",
                            "budget_used": {"edges": 35},
                            "latency_ms": 287,
                        }
                    ],
                    "conflicts_detected": 0,
                    "conflicts_resolved": 0,
                    "performance": {
                        "total_latency_ms": 542,
                        "architect_ms": 287,
                        "navigator_ms": 145,
                        "curator_ms": 78,
                    },
                }
            ]
        }
    }
