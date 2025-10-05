"""
T032-T037: LC-MAPPO Coordinator Service

LC-MAPPO Coordinator for CLAUSE Phase 2 multi-agent system.
Orchestrates SubgraphArchitect, PathNavigator, and ContextCurator with conflict resolution.

Per Spec 035:
- Agent handoff: Sequential execution (Architect → Navigator → Curator)
- Budget distribution: β_edge, β_step, β_tok across 3 agents
- Conflict resolution: Optimistic locking + MERGE strategy (Spec 031)
- LC-MAPPO: Centralized critic with 4 heads (research.md Decision 9)
- Latency target: <600ms total (300ms Architect + 200ms Navigator + 100ms Curator)
"""

import logging
from typing import Dict, Any, List, Optional
import time

from src.models.clause.coordinator_models import (
    CoordinationRequest,
    AgentHandoff,
    CoordinationResponse,
    BudgetAllocation,
)
from src.models.clause.shared_models import BudgetUsage

logger = logging.getLogger(__name__)


class LCMAPPOCoordinator:
    """
    LC-MAPPO Coordinator - Orchestrates CLAUSE 3-agent system.

    Implements:
    - Sequential agent handoff (Architect → Navigator → Curator)
    - Budget distribution and tracking
    - Conflict detection and resolution (Spec 031)
    - Performance monitoring per agent
    - Combined result aggregation
    """

    def __init__(
        self,
        subgraph_architect=None,
        path_navigator=None,
        context_curator=None,
        conflict_resolver=None,
        neo4j_client=None,
    ):
        """
        Initialize LC-MAPPO Coordinator with 3 agents.

        Args:
            subgraph_architect: SubgraphArchitect agent (Phase 1)
            path_navigator: PathNavigator agent (Phase 2)
            context_curator: ContextCurator agent (Phase 2)
            conflict_resolver: Conflict resolution service (Spec 031)
            neo4j_client: Neo4j client for write conflict detection
        """
        self.architect = subgraph_architect
        self.navigator = path_navigator
        self.curator = context_curator
        self.conflict_resolver = conflict_resolver
        self.neo4j = neo4j_client

        # Conflict tracking
        self.conflicts_detected = 0
        self.conflicts_resolved = 0

        logger.info("LC-MAPPO Coordinator initialized with 3 agents")

    async def coordinate(
        self, request: CoordinationRequest
    ) -> CoordinationResponse:
        """
        Coordinate 3-agent system execution per research.md Decision 12.

        Sequential handoff:
        1. SubgraphArchitect: Build subgraph (β_edge budget)
        2. PathNavigator: Explore path (β_step budget)
        3. ContextCurator: Select evidence (β_tok budget)

        Args:
            request: CoordinationRequest with query and budgets

        Returns:
            CoordinationResponse with combined results, handoffs, conflicts
        """
        coord_start_time = time.time()

        # T034: Initialize budget tracking
        budget_usage = BudgetUsage(
            edge_total=request.budgets.edge_budget,
            step_total=request.budgets.step_budget,
            token_total=request.budgets.token_budget,
        )

        # T035: Track agent handoffs
        agent_handoffs: List[AgentHandoff] = []

        # T037: Combined results
        combined_result = {
            "subgraph": None,
            "path": None,
            "evidence": None,
        }

        # T033: Sequential Agent Handoff

        # Step 1: SubgraphArchitect
        architect_result = await self._execute_architect(
            query=request.query,
            edge_budget=request.budgets.edge_budget,
            lambda_edge=request.lambdas.edge,
        )

        agent_handoffs.append(
            AgentHandoff(
                step=1,
                agent="SubgraphArchitect",
                action="built_subgraph",
                budget_used={"edges": architect_result["edges_used"]},
                latency_ms=architect_result["latency_ms"],
            )
        )

        budget_usage.edge_used = architect_result["edges_used"]
        combined_result["subgraph"] = architect_result["subgraph"]

        # Step 2: PathNavigator
        navigator_result = await self._execute_navigator(
            query=request.query,
            start_node=architect_result.get("start_node", "root"),
            step_budget=request.budgets.step_budget,
        )

        agent_handoffs.append(
            AgentHandoff(
                step=2,
                agent="PathNavigator",
                action="navigated_path",
                budget_used={"steps": navigator_result["steps_used"]},
                latency_ms=navigator_result["latency_ms"],
            )
        )

        budget_usage.step_used = navigator_result["steps_used"]
        combined_result["path"] = navigator_result["path"]

        # Step 3: ContextCurator
        curator_result = await self._execute_curator(
            evidence_pool=navigator_result.get("evidence_pool", []),
            token_budget=request.budgets.token_budget,
            lambda_tok=request.lambdas.token,
        )

        agent_handoffs.append(
            AgentHandoff(
                step=3,
                agent="ContextCurator",
                action="curated_evidence",
                budget_used={"tokens": curator_result["tokens_used"]},
                latency_ms=curator_result["latency_ms"],
            )
        )

        budget_usage.token_used = curator_result["tokens_used"]
        combined_result["evidence"] = curator_result["selected_evidence"]

        # T036: Conflict detection and resolution
        await self._detect_and_resolve_conflicts(agent_handoffs)

        # T035: Performance metrics
        total_latency = (time.time() - coord_start_time) * 1000
        performance = {
            "total_latency_ms": total_latency,
            "architect_ms": agent_handoffs[0].latency_ms,
            "navigator_ms": agent_handoffs[1].latency_ms,
            "curator_ms": agent_handoffs[2].latency_ms,
        }

        return CoordinationResponse(
            result=combined_result,
            agent_handoffs=agent_handoffs,
            conflicts_detected=self.conflicts_detected,
            conflicts_resolved=self.conflicts_resolved,
            performance=performance,
        )

    # T033: Agent Execution Methods

    async def _execute_architect(
        self, query: str, edge_budget: int, lambda_edge: float
    ) -> Dict[str, Any]:
        """
        Execute SubgraphArchitect agent (Step 1).

        Args:
            query: Natural language query
            edge_budget: β_edge budget
            lambda_edge: Edge cost multiplier

        Returns:
            Architect result with subgraph, edges_used, latency_ms
        """
        start_time = time.time()

        if self.architect:
            # Call SubgraphArchitect.build() - Phase 1 implementation
            result = await self.architect.build(
                query=query,
                edge_budget=edge_budget,
            )
        else:
            # Placeholder result
            result = {
                "subgraph": {"nodes": [], "edges": []},
                "edges_used": 0,
                "start_node": "root",
            }

        latency_ms = (time.time() - start_time) * 1000

        return {
            "subgraph": result.get("subgraph", {}),
            "edges_used": result.get("edges_used", 0),
            "start_node": result.get("start_node", "root"),
            "latency_ms": latency_ms,
        }

    async def _execute_navigator(
        self, query: str, start_node: str, step_budget: int
    ) -> Dict[str, Any]:
        """
        Execute PathNavigator agent (Step 2).

        Args:
            query: Natural language query
            start_node: Starting node from Architect
            step_budget: β_step budget

        Returns:
            Navigator result with path, steps_used, latency_ms
        """
        start_time = time.time()

        if self.navigator:
            # Import PathNavigationRequest here to avoid circular import
            from src.models.clause.path_models import PathNavigationRequest

            request = PathNavigationRequest(
                query=query,
                start_node=start_node,
                step_budget=step_budget,
            )

            result = await self.navigator.navigate(request)

            # Extract evidence pool from path nodes (placeholder logic)
            evidence_pool = [f"Evidence from {node}" for node in result.path["nodes"]]

            return {
                "path": result.path,
                "steps_used": result.metadata["budget_used"],
                "evidence_pool": evidence_pool,
                "latency_ms": result.performance["latency_ms"],
            }
        else:
            # Placeholder result
            latency_ms = (time.time() - start_time) * 1000
            return {
                "path": {"nodes": [], "edges": [], "steps": []},
                "steps_used": 0,
                "evidence_pool": [],
                "latency_ms": latency_ms,
            }

    async def _execute_curator(
        self, evidence_pool: List[str], token_budget: int, lambda_tok: float
    ) -> Dict[str, Any]:
        """
        Execute ContextCurator agent (Step 3).

        Args:
            evidence_pool: Evidence snippets from Navigator
            token_budget: β_tok budget
            lambda_tok: Token cost multiplier

        Returns:
            Curator result with selected_evidence, tokens_used, latency_ms
        """
        start_time = time.time()

        if self.curator and evidence_pool:
            # Import ContextCurationRequest here to avoid circular import
            from src.models.clause.curator_models import ContextCurationRequest

            request = ContextCurationRequest(
                evidence_pool=evidence_pool,
                token_budget=token_budget,
                lambda_tok=lambda_tok,
            )

            result = await self.curator.curate(request)

            return {
                "selected_evidence": [ev.model_dump() for ev in result.selected_evidence],
                "tokens_used": result.metadata["tokens_used"],
                "latency_ms": result.performance["latency_ms"],
            }
        else:
            # Placeholder result
            latency_ms = (time.time() - start_time) * 1000
            return {
                "selected_evidence": [],
                "tokens_used": 0,
                "latency_ms": latency_ms,
            }

    # T036: Conflict Detection and Resolution

    async def _detect_and_resolve_conflicts(
        self, agent_handoffs: List[AgentHandoff]
    ) -> None:
        """
        Detect and resolve write conflicts per Spec 031.

        Uses optimistic locking strategy:
        - Detect concurrent writes to same Neo4j nodes/edges
        - Resolve using MERGE strategy (max strength wins)

        Args:
            agent_handoffs: List of agent execution records
        """
        # Placeholder: In real implementation, would check Neo4j write logs
        # and detect concurrent modifications

        # Reset conflict counters
        self.conflicts_detected = 0
        self.conflicts_resolved = 0

        if self.conflict_resolver and self.neo4j:
            # Check for conflicts
            conflicts = await self.conflict_resolver.detect_conflicts(
                agent_handoffs=agent_handoffs
            )

            self.conflicts_detected = len(conflicts)

            # Resolve conflicts using MERGE strategy
            for conflict in conflicts:
                await self.conflict_resolver.resolve(conflict, strategy="MERGE")
                self.conflicts_resolved += 1

            logger.info(
                f"Conflicts: {self.conflicts_detected} detected, {self.conflicts_resolved} resolved"
            )
        else:
            logger.debug("Conflict resolution disabled (no resolver configured)")
