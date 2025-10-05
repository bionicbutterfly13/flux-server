"""
Demo CLAUSE Pipeline - Document Upload to Knowledge Graph

End-to-end demo showing:
1. Upload document via Daedalus
2. Extract concepts and add to graph
3. Run CLAUSE multi-agent system
4. Show results with full provenance
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File
from typing import List, Dict, Any
import logging
import time
import sys

sys.path.insert(0, '/Volumes/Asylum/dev/Dionysus-2.0/backend/src')

from src.services.demo.in_memory_graph import get_demo_graph, get_demo_embedder
from src.services.clause.path_navigator import PathNavigator
from src.services.clause.context_curator import ContextCurator
from src.services.clause.coordinator import LCMAPPOCoordinator
from src.models.clause.coordinator_models import (
    CoordinationRequest,
    BudgetAllocation,
    LambdaParameters,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["demo"])


class DocumentProcessingResult(BaseModel):
    """Result from processing a document through CLAUSE pipeline"""

    document_text: str
    concepts_extracted: List[str]
    clause_response: Dict[str, Any]
    processing_stages: List[Dict[str, Any]]
    total_time_ms: float


@router.post("/process-document", response_model=DocumentProcessingResult)
async def process_document_through_clause(
    file: UploadFile = File(...),
) -> DocumentProcessingResult:
    """
    Process document through complete CLAUSE pipeline

    Steps:
    1. Receive document via Daedalus gateway
    2. Extract concepts and add to knowledge graph
    3. Run CLAUSE multi-agent coordination
    4. Return results with full trace

    This is the END-TO-END demo you can actually see working!
    """
    start_time = time.time()
    processing_stages = []

    # Stage 1: Read document
    stage_start = time.time()
    content = await file.read()
    document_text = content.decode('utf-8')
    processing_stages.append({
        "stage": 1,
        "name": "Document Upload (Daedalus)",
        "duration_ms": (time.time() - stage_start) * 1000,
        "result": f"Received {len(document_text)} characters",
    })

    # Stage 2: Extract concepts and update graph
    stage_start = time.time()
    graph = get_demo_graph()
    concepts = graph.add_document_concepts(document_text)
    processing_stages.append({
        "stage": 2,
        "name": "Concept Extraction",
        "duration_ms": (time.time() - stage_start) * 1000,
        "result": f"Extracted {len(concepts)} concepts: {concepts}",
    })

    if not concepts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No relevant concepts found in document. Try mentioning: climate, greenhouse, carbon, renewable, etc."
        )

    # Stage 3: Run CLAUSE multi-agent coordination
    stage_start = time.time()

    # Initialize agents with demo graph
    navigator = PathNavigator(
        embedding_service=get_demo_embedder(),
    )
    # Patch helper methods to use demo graph (async wrappers)
    async def get_node_text_async(node_id: str) -> str:
        return graph.get_node_text(node_id)

    async def get_node_degree_async(node_id: str) -> int:
        return graph.get_node_degree(node_id)

    async def get_neighbors_async(node_id: str):
        return graph.get_neighbors(node_id)

    async def get_candidate_hops_async(node_id: str):
        return graph.get_candidate_hops(node_id)

    navigator._get_node_text = get_node_text_async
    navigator._get_node_degree = get_node_degree_async
    navigator._get_neighbors = get_neighbors_async
    navigator._get_candidate_hops = get_candidate_hops_async
    navigator._embed_text = get_demo_embedder().embed

    curator = ContextCurator(
        embedding_service=get_demo_embedder(),
    )
    curator._embed_text = get_demo_embedder().embed

    coordinator = LCMAPPOCoordinator(
        path_navigator=navigator,
        context_curator=curator,
    )

    # Create query from first concept
    query = f"What is {concepts[0]} and how does it relate to other concepts?"
    start_node = concepts[0]

    request = CoordinationRequest(
        query=query,
        budgets=BudgetAllocation(
            edge_budget=30,
            step_budget=5,
            token_budget=1000,
        ),
        lambdas=LambdaParameters(
            edge=0.01,
            latency=0.01,
            token=0.01,
        ),
    )

    clause_response = await coordinator.coordinate(request)
    processing_stages.append({
        "stage": 3,
        "name": "CLAUSE Multi-Agent Coordination",
        "duration_ms": (time.time() - stage_start) * 1000,
        "result": {
            "agents_executed": len(clause_response.agent_handoffs),
            "conflicts_resolved": clause_response.conflicts_resolved,
        },
    })

    # Stage 4: Extract path details
    stage_start = time.time()
    path_details = {
        "query": query,
        "start_node": start_node,
        "nodes_visited": clause_response.result.get("path", {}).get("nodes", []),
        "edges_traversed": clause_response.result.get("path", {}).get("edges", []),
        "evidence_collected": len(clause_response.result.get("evidence", [])),
    }
    processing_stages.append({
        "stage": 4,
        "name": "Result Extraction",
        "duration_ms": (time.time() - stage_start) * 1000,
        "result": path_details,
    })

    total_time = (time.time() - start_time) * 1000

    return DocumentProcessingResult(
        document_text=document_text[:500],  # First 500 chars
        concepts_extracted=concepts,
        clause_response=clause_response.model_dump(),
        processing_stages=processing_stages,
        total_time_ms=total_time,
    )


@router.get("/graph-status")
async def get_graph_status() -> Dict[str, Any]:
    """
    Get current status of demo knowledge graph

    Shows what concepts are available
    """
    graph = get_demo_graph()

    return {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "available_concepts": list(graph.nodes.keys()),
        "sample_edges": graph.edges[:5],
    }


@router.post("/simple-query")
async def simple_query(query: str, start_node: str = "climate_change") -> Dict[str, Any]:
    """
    Simple query endpoint to test CLAUSE without document upload

    Example: /api/demo/simple-query?query=What causes climate change?&start_node=climate_change
    """
    graph = get_demo_graph()

    # Check if start node exists
    if start_node not in graph.nodes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Node '{start_node}' not found. Available: {list(graph.nodes.keys())}"
        )

    # Initialize navigator with demo graph (async wrappers)
    navigator = PathNavigator(embedding_service=get_demo_embedder())

    async def get_node_text_async(node_id: str) -> str:
        return graph.get_node_text(node_id)

    async def get_node_degree_async(node_id: str) -> int:
        return graph.get_node_degree(node_id)

    async def get_neighbors_async(node_id: str):
        return graph.get_neighbors(node_id)

    async def get_candidate_hops_async(node_id: str):
        return graph.get_candidate_hops(node_id)

    navigator._get_node_text = get_node_text_async
    navigator._get_node_degree = get_node_degree_async
    navigator._get_neighbors = get_neighbors_async
    navigator._get_candidate_hops = get_candidate_hops_async
    navigator._embed_text = get_demo_embedder().embed

    from models.clause.path_models import PathNavigationRequest

    request = PathNavigationRequest(
        query=query,
        start_node=start_node,
        step_budget=5,
    )

    response = await navigator.navigate(request)

    return {
        "query": query,
        "start_node": start_node,
        "path": response.path,
        "metadata": response.metadata,
        "performance": response.performance,
    }
