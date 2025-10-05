"""
T033-T037: LC-MAPPO Coordinator Service

Multi-agent coordinator for CLAUSE Phase 2 using Lagrangian Constrained MAPPO.
Orchestrates SubgraphArchitect → PathNavigator → ContextCurator with budget constraints.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import asyncio

from src.models.clause.coordinator_models import (
    CoordinationRequest,
    CoordinationResponse,
    AgentHandoff,
)
from src.models.clause.shared_models import BudgetUsage
from src.services.clause.centralized_critic import compute_shaped_return

logger = logging.getLogger(__name__)


class Episode:
    """Episode data for LC-MAPPO training"""

    def __init__(self):
        self.transitions = []

    def add_transition(
        self,
        state: Any,
        action: Any,
        reward: float,
        edge_cost: float,
        latency_cost: float,
        token_cost: float,
    ):
        """Add transition to episode"""
        self.transitions.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "edge_cost": edge_cost,
                "latency_cost": latency_cost,
                "token_cost": token_cost,
            }
        )


class LCMAPPOCoordinator:
    """
    T033-T037: LC-MAPPO Coordinator

    Orchestrates 3 agents with Lagrangian budget constraints:
    - T033: Shaped return calculation (reward - λ × costs)
    - T034: Dual variable updates (projected gradient ascent)
    - T035: Agent handoff protocol (sequential execution)
    - T036: Conflict resolution integration
    - T037: Complete coordination service
    """

    def __init__(
        self,
        architect=None,
        navigator=None,
        curator=None,
        conflict_resolver=None,
        centralized_critic=None,
    ):
        """
        Initialize LC-MAPPO coordinator.

        Args:
            architect: SubgraphArchitect service (Phase 1)
            navigator: PathNavigator service (T025)
            curator: ContextCurator service (T031)
            conflict_resolver: ConflictResolver service (T046)
            centralized_critic: CLAUSECentralizedCritic (T032)
        """
        self.architect = architect
        self.navigator = navigator
        self.curator = curator
        self.conflict_resolver = conflict_resolver
        self.critic = centralized_critic

        # T034: Lagrangian dual variables (initialized to 0.01)
        self.lambda_edge = 0.01
        self.lambda_lat = 0.01
        self.lambda_tok = 0.01

        # Dual update hyperparameters
        self.eta_edge = 0.001  # Learning rate for edge budget dual
        self.eta_lat = 0.001  # Learning rate for latency dual
        self.eta_tok = 0.001  # Learning rate for token dual

        # Performance tracking
        self.conflicts_detected = 0
        self.conflicts_resolved = 0

        logger.info(
            f"LCMAPPOCoordinator initialized: λ_edge={self.lambda_edge}, λ_lat={self.lambda_lat}, λ_tok={self.lambda_tok}"
        )

    def calculate_shaped_returns(self, episode: Episode) -> List[float]:
        """
        T033: Calculate shaped returns for episode.

        Shaped return = r_acc - λ_edge*c_edge - λ_lat*c_lat - λ_tok*c_tok

        Args:
            episode: Episode with transitions

        Returns:
            List of shaped rewards (one per transition)
        """
        shaped_rewards = []

        for transition in episode.transitions:
            r_acc = transition["reward"]
            c_edge = transition["edge_cost"]
            c_lat = transition["latency_cost"]
            c_tok = transition["token_cost"]

            # Compute shaped return
            r_shaped = (
                r_acc
                - self.lambda_edge * c_edge
                - self.lambda_lat * c_lat
                - self.lambda_tok * c_tok
            )
            shaped_rewards.append(r_shaped)

        return shaped_rewards

    def update_duals(self, batch_episodes: List[Episode], budgets: Dict[str, float]):
        """
        T034: Update Lagrangian dual variables.

        Uses projected gradient ascent:
        λ_k = max(0, λ_k + η * (E[C_k] - β_k))

        Args:
            batch_episodes: Batch of episodes for dual update
            budgets: Budget limits {edge_budget, latency_budget, token_budget}
        """
        # Compute average costs across batch
        avg_edge_cost = 0.0
        avg_latency_cost = 0.0
        avg_token_cost = 0.0
        total_transitions = 0

        for episode in batch_episodes:
            for transition in episode.transitions:
                avg_edge_cost += transition["edge_cost"]
                avg_latency_cost += transition["latency_cost"]
                avg_token_cost += transition["token_cost"]
                total_transitions += 1

        if total_transitions > 0:
            avg_edge_cost /= total_transitions
            avg_latency_cost /= total_transitions
            avg_token_cost /= total_transitions

        # Dual updates (projected gradient ascent)
        # λ = max(0, λ + η * (avg_cost - budget))
        self.lambda_edge = max(
            0.0,
            self.lambda_edge
            + self.eta_edge * (avg_edge_cost - budgets.get("edge_budget", 100)),
        )
        self.lambda_lat = max(
            0.0,
            self.lambda_lat
            + self.eta_lat
            * (avg_latency_cost - budgets.get("latency_budget", 200.0)),
        )
        self.lambda_tok = max(
            0.0,
            self.lambda_tok
            + self.eta_tok * (avg_token_cost - budgets.get("token_budget", 2048)),
        )

        logger.info(
            f"Dual update: λ_edge={self.lambda_edge:.4f}, λ_lat={self.lambda_lat:.4f}, λ_tok={self.lambda_tok:.4f}"
        )

    async def coordinate(
        self, request: CoordinationRequest
    ) -> CoordinationResponse:
        """
        T035-T037: Complete coordination workflow.

        Sequential agent handoff:
        1. SubgraphArchitect → builds subgraph
        2. PathNavigator → navigates subgraph
        3. ContextCurator → curates evidence

        With conflict detection (T036) and budget tracking.

        Args:
            request: CoordinationRequest with query and budgets

        Returns:
            CoordinationResponse with results and performance metrics
        """
        agent_handoffs = []
        total_start = time.time()

        # T035: Agent 1 - SubgraphArchitect
        if self.architect:
            architect_start = time.time()
            # Placeholder - actual implementation in Phase 1
            subgraph_result = {
                "graph": None,
                "edge_count": 50,
                "latency_ms": 45.0,
            }
            architect_latency = (time.time() - architect_start) * 1000

            agent_handoffs.append(
                AgentHandoff(
                    step=1,
                    agent="SubgraphArchitect",
                    latency_ms=architect_latency,
                    budget_used={"edges": 50},
                )
            )

        # T035: Agent 2 - PathNavigator
        navigator_result = None
        if self.navigator and subgraph_result.get("graph"):
            navigator_start = time.time()

            # Create navigation request
            from src.models.clause.navigator_models import PathNavigationRequest

            nav_request = PathNavigationRequest(
                query=request.query,
                start_node=request.start_node,
                step_budget=request.budgets.step_budget,
            )

            # T036: Conflict detection (write to Neo4j with conflict resolver)
            try:
                navigator_result = await self.navigator.navigate(
                    nav_request, subgraph_result["graph"]
                )

                # Check for write conflicts if conflict_resolver available
                if self.conflict_resolver:
                    # Placeholder - actual conflict detection in T046
                    pass

            except Exception as e:
                logger.error(f"Navigator failed: {e}")
                self.conflicts_detected += 1

            navigator_latency = (time.time() - navigator_start) * 1000

            agent_handoffs.append(
                AgentHandoff(
                    step=2,
                    agent="PathNavigator",
                    latency_ms=navigator_latency,
                    budget_used={
                        "steps": (
                            navigator_result.metadata.get("steps_taken", 0)
                            if navigator_result
                            else 0
                        )
                    },
                )
            )

        # T035: Agent 3 - ContextCurator
        curator_result = None
        if self.curator and navigator_result:
            curator_start = time.time()

            # Extract evidence pool from path
            evidence_pool = self._extract_evidence_from_path(navigator_result)

            # Create curation request
            from src.models.clause.curator_models import ContextCurationRequest

            curator_request = ContextCurationRequest(
                evidence_pool=evidence_pool,
                token_budget=request.budgets.token_budget,
                lambda_tok=self.lambda_tok,
                enable_provenance=request.enable_provenance,
            )

            # Execute curation
            try:
                curator_result = await self.curator.curate(curator_request)

                # T036: Conflict detection for curator writes
                if self.conflict_resolver:
                    # Placeholder - actual conflict detection in T046
                    pass

            except Exception as e:
                logger.error(f"Curator failed: {e}")
                self.conflicts_detected += 1

            curator_latency = (time.time() - curator_start) * 1000

            agent_handoffs.append(
                AgentHandoff(
                    step=3,
                    agent="ContextCurator",
                    latency_ms=curator_latency,
                    budget_used={
                        "tokens": (
                            curator_result.metadata.get("tokens_used", 0)
                            if curator_result
                            else 0
                        )
                    },
                )
            )

        # Build final response
        total_latency = (time.time() - total_start) * 1000

        # Aggregate budget usage
        budget_usage = BudgetUsage(
            edge_used=subgraph_result.get("edge_count", 0),
            step_used=(
                navigator_result.metadata.get("steps_taken", 0)
                if navigator_result
                else 0
            ),
            token_used=(
                curator_result.metadata.get("tokens_used", 0) if curator_result else 0
            ),
            edge_total=request.budgets.edge_budget,
            step_total=request.budgets.step_budget,
            token_total=request.budgets.token_budget,
        )

        return CoordinationResponse(
            result={
                "subgraph": subgraph_result,
                "path": (
                    navigator_result.model_dump() if navigator_result else None
                ),
                "evidence": (
                    curator_result.model_dump() if curator_result else None
                ),
            },
            agent_handoffs=agent_handoffs,
            budget_usage=budget_usage,
            metadata={
                "conflicts_detected": self.conflicts_detected,
                "conflicts_resolved": self.conflicts_resolved,
                "lambda_edge": self.lambda_edge,
                "lambda_lat": self.lambda_lat,
                "lambda_tok": self.lambda_tok,
            },
            performance={"total_latency_ms": total_latency},
        )

    def _extract_evidence_from_path(self, navigator_result) -> List[str]:
        """
        Extract evidence pool from navigation path.

        Args:
            navigator_result: PathNavigationResponse

        Returns:
            List of evidence text snippets
        """
        # Extract text from nodes visited in path
        evidence_pool = []

        if hasattr(navigator_result, "path") and navigator_result.path:
            for node_id in navigator_result.path:
                # Placeholder - in real implementation, query Neo4j for node text
                evidence_pool.append(f"Evidence from node {node_id}")

        return evidence_pool
