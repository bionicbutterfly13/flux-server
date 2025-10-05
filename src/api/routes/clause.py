"""
CLAUSE API Routes - Budget-aware subgraph construction and basin strengthening

Spec 034 Phase 1: CLAUSE Subgraph Architect with Basin Strengthening
Endpoints:
- POST /api/clause/subgraph - Construct query-specific subgraph
- POST /api/clause/basins/strengthen - Strengthen basins for concepts
- GET /api/clause/basins/{concept_id} - Get basin information
- POST /api/clause/edges/score - Score individual edge

Spec 035 Phase 2: CLAUSE Multi-Agent System (T050-T052)
Endpoints:
- POST /api/clause/navigate - Path navigation with ThoughtSeeds & Curiosity
- POST /api/clause/curate - Evidence curation with provenance
- POST /api/clause/coordinate - Multi-agent coordination
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional
import logging
import time

from src.services.clause.models import (
    SubgraphRequest,
    SubgraphResponse,
    BasinStrengtheningRequest,
    BasinStrengtheningResponse,
    BasinInfo,
    EdgeScoringRequest,
    EdgeScoringResponse,
)
from src.services.clause.graph_loader import GraphLoader
# Note: BasinTracker, EdgeScorer, SubgraphArchitect have import issues
# Will be fixed when their absolute imports are converted to relative

# Phase 2 imports (T050-T052)
from src.models.clause.path_models import (
    PathNavigationRequest,
    PathNavigationResponse,
)
from src.models.clause.curator_models import (
    ContextCurationRequest,
    ContextCurationResponse,
)
from src.models.clause.coordinator_models import (
    CoordinationRequest,
    CoordinationResponse,
)
from src.services.clause.path_navigator import PathNavigator
from src.services.clause.context_curator import ContextCurator
from src.services.clause.coordinator import LCMAPPOCoordinator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/clause", tags=["clause"])

# Lazy initialization to avoid database connection at import time
_graph_loader: Optional[GraphLoader] = None
# _basin_tracker: Optional[BasinTracker] = None
# _edge_scorer: Optional[EdgeScorer] = None
# _architect: Optional[SubgraphArchitect] = None


def get_graph_loader() -> GraphLoader:
    """Get or create GraphLoader instance."""
    global _graph_loader
    if _graph_loader is None:
        _graph_loader = GraphLoader()
    return _graph_loader


# Temporary dependency placeholder until services are fully importable
async def require_services_available():
    """Verify CLAUSE services are available."""
    # For T022: Just check GraphLoader is available
    try:
        get_graph_loader()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"CLAUSE services unavailable: {str(e)}"
        )


@router.post("/subgraph", response_model=SubgraphResponse, status_code=status.HTTP_200_OK)
async def construct_subgraph(
    request: SubgraphRequest,
    _services: None = Depends(require_services_available)
) -> SubgraphResponse:
    """
    Construct budget-aware subgraph for query using CLAUSE algorithm.

    Implements:
    - FR-001: Budget-aware edge selection with shaped gain
    - FR-005: Edge budget enforcement
    - NFR-005: Neo4j retry logic (3x exponential backoff)

    Returns:
    - 200 OK: Subgraph constructed successfully
    - 400 Bad Request: Invalid request parameters
    - 503 Service Unavailable: Neo4j connection failed after retries

    Performance targets (Success Criteria):
    - Subgraph construction: <500ms total (p95)
    - Precision@50: ≥85% query-relevant edges
    """
    start_time = time.time()

    try:
        # Validate request
        if request.edge_budget < 1 or request.edge_budget > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="edge_budget must be between 1 and 1000"
            )

        # Load k-hop subgraph from Neo4j (with NFR-005 retry logic)
        loader = get_graph_loader()
        try:
            loader.load_subgraph_from_neo4j(
                query=request.query,
                hop_distance=request.hop_distance,
                max_seed_nodes=20,
            )
        except ConnectionError as e:
            # NFR-005: Return 503 after retry exhaustion
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Neo4j connection failed: {str(e)}"
            )

        # TODO T022: Implement CLAUSE edge scoring and selection
        # Will use loaded graph from GraphLoader instance
        # For now, return basic subgraph info
        selected_edges = []
        stopped_reason = "NOT_IMPLEMENTED"
        budget_used = 0
        avg_shaped_gain = 0.0

        elapsed_ms = int((time.time() - start_time) * 1000)

        return SubgraphResponse(
            query=request.query,
            edge_budget=request.edge_budget,
            edges_selected=selected_edges,
            stopped_reason=stopped_reason,
            budget_used=budget_used,
            avg_shaped_gain=avg_shaped_gain,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Subgraph construction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/basins/strengthen", response_model=BasinStrengtheningResponse, status_code=status.HTTP_200_OK)
async def strengthen_basins(
    request: BasinStrengtheningRequest,
    _services: None = Depends(require_services_available)
) -> BasinStrengtheningResponse:
    """
    Strengthen attractor basins for concepts.

    Implements:
    - FR-003: +0.2 strength increment per activation (cap 2.0)
    - FR-004: Symmetric co-occurrence tracking
    - NFR-002: Atomic basin updates

    Returns:
    - 200 OK: Basins strengthened successfully
    - 400 Bad Request: Invalid concept_ids
    - 503 Service Unavailable: Database unavailable

    Performance targets:
    - Basin update: <5ms per basin
    """
    start_time = time.time()

    try:
        # Validate request
        if not request.concept_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="concept_ids cannot be empty"
            )

        # TODO T023: Implement basin strengthening
        basins_updated = []
        basins_created = []

        elapsed_ms = int((time.time() - start_time) * 1000)

        return BasinStrengtheningResponse(
            basins_updated=basins_updated,
            basins_created=basins_created,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Basin strengthening failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.get("/basins/{concept_id}", response_model=BasinInfo, status_code=status.HTTP_200_OK)
async def get_basin_info(
    concept_id: str,
    _services: None = Depends(require_services_available)
) -> BasinInfo:
    """
    Get basin information for a concept.

    Returns:
    - 200 OK: Basin info retrieved
    - 404 Not Found: Concept has no basin
    - 503 Service Unavailable: Database unavailable
    """
    try:
        # TODO T024: Implement basin retrieval
        loader = get_graph_loader()
        basin_data = loader.get_basin_info(concept_id)

        if basin_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No basin found for concept: {concept_id}"
            )

        return BasinInfo(
            concept_id=concept_id,
            basin_id=basin_data["basin_id"],
            strength=basin_data["strength"],
            activation_count=basin_data["activation_count"],
            co_occurring_concepts=basin_data["co_occurring_concepts"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Basin info retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/edges/score", response_model=EdgeScoringResponse, status_code=status.HTTP_200_OK)
async def score_edge(
    request: EdgeScoringRequest,
    _services: None = Depends(require_services_available)
) -> EdgeScoringResponse:
    """
    Score individual edge using CLAUSE 5-signal scoring.

    Implements:
    - FR-002: 5-signal edge scoring (φ_ent, φ_rel, φ_nbr, φ_deg, φ_basin)
    - FR-006: Basin strength normalization

    Returns:
    - 200 OK: Edge scored successfully
    - 400 Bad Request: Invalid edge data
    - 503 Service Unavailable: Database unavailable

    Performance targets:
    - Edge scoring: <10ms per edge
    """
    start_time = time.time()

    try:
        # TODO T025: Implement edge scoring
        total_score = 0.0
        signal_breakdown = {
            "phi_entity": 0.0,
            "phi_relation": 0.0,
            "phi_neighborhood": 0.0,
            "phi_degree": 0.0,
            "phi_basin": 0.0,
        }

        elapsed_ms = int((time.time() - start_time) * 1000)

        return EdgeScoringResponse(
            total_score=total_score,
            signal_breakdown=signal_breakdown,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Edge scoring failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )

# ==============================================================================
# Phase 2: Multi-Agent System (T050-T052)
# ==============================================================================

# Lazy initialization for Phase 2 agents
_path_navigator: Optional[PathNavigator] = None
_context_curator: Optional[ContextCurator] = None
_lc_mappo_coordinator: Optional[LCMAPPOCoordinator] = None


def get_path_navigator() -> PathNavigator:
    """Get or create PathNavigator instance."""
    global _path_navigator
    if _path_navigator is None:
        _path_navigator = PathNavigator()
    return _path_navigator


def get_context_curator() -> ContextCurator:
    """Get or create ContextCurator instance."""
    global _context_curator
    if _context_curator is None:
        _context_curator = ContextCurator()
    return _context_curator


def get_coordinator() -> LCMAPPOCoordinator:
    """Get or create LC-MAPPO Coordinator instance."""
    global _lc_mappo_coordinator
    if _lc_mappo_coordinator is None:
        _lc_mappo_coordinator = LCMAPPOCoordinator(
            path_navigator=get_path_navigator(),
            context_curator=get_context_curator(),
        )
    return _lc_mappo_coordinator


@router.post("/navigate", response_model=PathNavigationResponse, status_code=status.HTTP_200_OK)
async def navigate_path(
    request: PathNavigationRequest,
) -> PathNavigationResponse:
    """
    T050: Path navigation with ThoughtSeeds, Curiosity, and Causal reasoning.

    Implements Spec 035 Phase 2:
    - Budget-aware path exploration (β_step = 1-20 steps)
    - ThoughtSeed generation (Spec 028)
    - Curiosity triggers (Spec 029)
    - Causal reasoning (Spec 033)
    - State encoding (1154-dim feature vector)
    - Termination head (learned stopping)

    Returns:
    - 200 OK: Path navigated successfully
    - 400 Bad Request: Invalid request parameters
    - 500 Internal Server Error: Navigation failed

    Performance target: <200ms (NFR-001)
    """
    start_time = time.time()

    try:
        navigator = get_path_navigator()
        response = await navigator.navigate(request)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Path navigation complete: {len(response.path['steps'])} steps, "
            f"{response.metadata['thoughtseeds_generated']} ThoughtSeeds, "
            f"{elapsed_ms:.0f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Path navigation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Navigation error: {str(e)}"
        )


@router.post("/curate", response_model=ContextCurationResponse, status_code=status.HTTP_200_OK)
async def curate_evidence(
    request: ContextCurationRequest,
) -> ContextCurationResponse:
    """
    T051: Evidence curation with listwise scoring and provenance tracking.

    Implements Spec 035 Phase 2:
    - Listwise evidence scoring (pairwise similarity + diversity penalty)
    - Token budget enforcement (tiktoken, 10% safety buffer)
    - Learned stopping (shaped utility threshold)
    - Provenance tracking (Spec 032: 7 fields + 3 trust signals)

    Returns:
    - 200 OK: Evidence curated successfully
    - 400 Bad Request: Invalid request parameters
    - 500 Internal Server Error: Curation failed

    Performance target: <100ms (NFR-002)
    """
    start_time = time.time()

    try:
        curator = get_context_curator()
        response = await curator.curate(request)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Evidence curation complete: {len(response.selected_evidence)} selected, "
            f"{response.metadata['tokens_used']}/{response.metadata['tokens_total']} tokens, "
            f"{elapsed_ms:.0f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Evidence curation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Curation error: {str(e)}"
        )


@router.post("/coordinate", response_model=CoordinationResponse, status_code=status.HTTP_200_OK)
async def coordinate_agents(
    request: CoordinationRequest,
) -> CoordinationResponse:
    """
    T052: Multi-agent coordination with LC-MAPPO.

    Implements Spec 035 Phase 2:
    - Sequential agent handoff (Architect → Navigator → Curator)
    - Budget distribution (β_edge, β_step, β_tok)
    - Conflict detection and resolution (Spec 031)
    - Performance tracking per agent
    - Combined result aggregation

    Returns:
    - 200 OK: Coordination successful
    - 400 Bad Request: Invalid request parameters
    - 500 Internal Server Error: Coordination failed

    Performance target: <600ms total (NFR-003)
    """
    start_time = time.time()

    try:
        coordinator = get_coordinator()
        response = await coordinator.coordinate(request)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Multi-agent coordination complete: "
            f"{len(response.agent_handoffs)} agents, "
            f"{response.conflicts_resolved} conflicts resolved, "
            f"{elapsed_ms:.0f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Multi-agent coordination failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coordination error: {str(e)}"
        )
