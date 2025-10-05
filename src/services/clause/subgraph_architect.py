#!/usr/bin/env python3
"""
T019: SubgraphArchitect - CLAUSE Subgraph Construction

Implements CLAUSE algorithm with:
- Shaped gain rule: score - λ_edge × cost > 0
- Budget enforcement: edge_budget (default 50)
- Greedy selection: highest shaped gain first
- Stop conditions: budget exhausted OR shaped_gain ≤ 0 OR no more candidates

Performance target: <500ms for subgraph construction
"""

import time
from typing import List, Dict, Optional

from src.services.edge_scorer import EdgeScorer
from src.services.basin_tracker import BasinTracker


class SubgraphArchitect:
    """
    CLAUSE Subgraph Architect with budget-aware edge selection.

    Uses shaped gain rule to select edges that maximize relevance
    while staying within budget constraints.
    """

    def __init__(
        self,
        edge_scorer: Optional[EdgeScorer] = None,
        basin_tracker: Optional[BasinTracker] = None,
    ):
        """
        Initialize subgraph architect.

        Args:
            edge_scorer: EdgeScorer instance for scoring edges
            basin_tracker: BasinTracker instance for basin strength
        """
        self.basin_tracker = basin_tracker or BasinTracker()
        self.edge_scorer = edge_scorer or EdgeScorer(basin_tracker=self.basin_tracker)

    def build_subgraph(
        self,
        query: str,
        edge_budget: int = 50,
        lambda_edge: float = 0.2,
        hop_distance: int = 2,
        candidate_edges: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """
        Build query-specific subgraph using CLAUSE algorithm.

        Args:
            query: Search query
            edge_budget: Maximum edges to select (1-1000)
            lambda_edge: Edge cost weight for shaped gain (0.0-1.0)
            hop_distance: Maximum hop distance from query concepts (1-5)
            candidate_edges: Optional list of candidate edges (for testing)

        Returns:
            Dict with selected_edges, edge_scores, shaped_gains, budget_used, stopped_reason
        """
        start_time = time.perf_counter()

        # Use provided edges or get from graph (for now, require candidate_edges for testing)
        if candidate_edges is None:
            # In production, this would query Neo4j for candidate edges
            # For now, return empty result
            return {
                "selected_edges": [],
                "edge_scores": {},
                "shaped_gains": {},
                "budget_used": 0,
                "stopped_reason": "no_candidates_provided",
                "computation_time_ms": 0.0,
            }

        edges = candidate_edges

        # Score all candidate edges
        edge_scores = self.edge_scorer.score_edges(edges, query)

        # Calculate shaped gains
        shaped_gains = {}
        for edge in edges:
            edge_key = f"{edge['source']}_{edge['relation']}_{edge['target']}"
            score = edge_scores.get(edge_key, 0.0)

            # Shaped gain: score - λ_edge × cost
            # Assume uniform edge cost = 1.0
            edge_cost = 1.0
            shaped_gain = score - lambda_edge * edge_cost

            shaped_gains[edge_key] = shaped_gain

        # Sort edges by shaped gain (descending)
        sorted_edges = sorted(
            edges,
            key=lambda e: shaped_gains[f"{e['source']}_{e['relation']}_{e['target']}"],
            reverse=True,
        )

        # Greedy selection with budget enforcement
        selected_edges = []
        stopped_reason = "complete"

        for edge in sorted_edges:
            edge_key = f"{edge['source']}_{edge['relation']}_{edge['target']}"
            gain = shaped_gains[edge_key]

            # Stop if shaped gain ≤ 0
            if gain <= 0:
                stopped_reason = "shaped_gain_zero"
                break

            # Stop if budget exhausted
            if len(selected_edges) >= edge_budget:
                stopped_reason = "budget_exhausted"
                break

            # Accept edge
            selected_edges.append(edge)

        # If we selected all candidates and didn't hit budget
        if len(selected_edges) == len(edges) and len(selected_edges) < edge_budget:
            stopped_reason = "no_more_candidates"

        # Build response
        budget_used = len(selected_edges)

        # Filter scores and gains to selected edges only
        selected_edge_keys = [
            f"{e['source']}_{e['relation']}_{e['target']}" for e in selected_edges
        ]
        selected_scores = {k: v for k, v in edge_scores.items() if k in selected_edge_keys}
        selected_gains = {k: v for k, v in shaped_gains.items() if k in selected_edge_keys}

        end_time = time.perf_counter()
        computation_time_ms = (end_time - start_time) * 1000

        return {
            "selected_edges": selected_edges,
            "edge_scores": selected_scores,
            "shaped_gains": selected_gains,
            "budget_used": budget_used,
            "stopped_reason": stopped_reason,
            "computation_time_ms": computation_time_ms,
        }
