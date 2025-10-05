"""
CLAUSE Phase 1 - Subgraph Architect with Basin Strengthening

This module implements the CLAUSE algorithm for budget-aware knowledge graph
subgraph construction with self-improving basin frequency strengthening.

Core Components:
- SubgraphArchitect: CLAUSE edge scoring and subgraph construction
- BasinTracker: Basin frequency strengthening (+0.2/reappearance, cap 2.0)
- EdgeScorer: 5-signal edge scoring (entity, relation, neighborhood, degree, basin)

Performance Targets:
- Edge scoring: <10ms for 1000 edges
- Subgraph construction: <500ms total
- Basin update: <5ms
"""

__version__ = "1.0.0"

# Import models (T016)
from src.models import (
    SubgraphRequest,
    SubgraphResponse,
    EdgeScore,
    BasinStrengtheningRequest,
    BasinStrengtheningResponse,
    BasinInfo,
    EdgeScoringRequest,
    EdgeScoringResponse,
)

__all__ = [
    # Services
    "SubgraphArchitect",
    "BasinTracker",
    "EdgeScorer",
    "GraphLoader",
    "BasinCache",
    # Models (T016)
    "SubgraphRequest",
    "SubgraphResponse",
    "EdgeScore",
    "BasinStrengtheningRequest",
    "BasinStrengtheningResponse",
    "BasinInfo",
    "EdgeScoringRequest",
    "EdgeScoringResponse",
]

# Import services (T017-T021)
# Import services individually for better error handling
import logging as _logging

try:
    from src.services.basin_tracker import BasinTracker
except ImportError as e:
    _logging.getLogger(__name__).warning(f"BasinTracker import failed: {e}")
    BasinTracker = None  # type: ignore

try:
    from src.services.edge_scorer import EdgeScorer
except ImportError as e:
    _logging.getLogger(__name__).warning(f"EdgeScorer import failed: {e}")
    EdgeScorer = None  # type: ignore

try:
    from src.services.subgraph_architect import SubgraphArchitect
except ImportError as e:
    _logging.getLogger(__name__).warning(f"SubgraphArchitect import failed: {e}")
    SubgraphArchitect = None  # type: ignore

try:
    from src.services.graph_loader import GraphLoader
except ImportError as e:
    _logging.getLogger(__name__).warning(f"GraphLoader import failed: {e}")
    GraphLoader = None  # type: ignore

try:
    from src.services.basin_cache import BasinCache
except ImportError as e:
    _logging.getLogger(__name__).warning(f"BasinCache import failed: {e}")
    BasinCache = None  # type: ignore
