"""
Flux Server Models
Data models for query processing, CLAUSE navigation, and consciousness tracking
"""

# Core query models
from .query import Query
from .response import QueryResponse, SearchResult, SearchSource
from .attractor_basin import AttractorBasin

# CLAUSE models (from clause/ subdirectory)
from .clause.path_models import (
    SubgraphRequest,
    SubgraphResponse,
    EdgeScore,
)
from .clause.curator_models import (
    BasinStrengtheningRequest,
    BasinStrengtheningResponse,
    BasinInfo,
)
from .clause.shared_models import (
    EdgeScoringRequest,
    EdgeScoringResponse,
    StateEncoding,
    BudgetUsage,
)

__all__ = [
    # Core query
    "Query",
    "QueryResponse",
    "SearchResult",
    "SearchSource",
    # Consciousness
    "AttractorBasin",
    # CLAUSE path
    "SubgraphRequest",
    "SubgraphResponse",
    "EdgeScore",
    # CLAUSE curator
    "BasinStrengtheningRequest",
    "BasinStrengtheningResponse",
    "BasinInfo",
    # CLAUSE shared
    "EdgeScoringRequest",
    "EdgeScoringResponse",
    "StateEncoding",
    "BudgetUsage",
]
