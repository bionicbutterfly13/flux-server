"""
T018-T025: Path Navigator Service

PathNavigator agent for CLAUSE Phase 2 multi-agent system.
Implements budget-aware path exploration with ThoughtSeeds, Curiosity, and Causal reasoning.

Per Spec 035:
- Budget enforcement: β_step (1-20 steps)
- Termination head: Binary classifier (stop_prob > 0.5)
- ThoughtSeed generation: Bulk generation with basin context (Spec 028)
- Curiosity triggers: Redis queue for prediction_error > threshold (Spec 029)
- Causal reasoning: Pre-computed DAG with LRU cache (Spec 033)
- T024: AsyncIO causal timeout (30ms) with background queue fallback
- Latency target: <200ms per navigation
"""

import asyncio
import hashlib
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

from src.models.clause.path_models import (
    PathNavigationRequest,
    PathStep,
    PathNavigationResponse,
)
from src.models.clause.shared_models import StateEncoding
from src.services.causal.causal_queue import CausalQueue, CausalQueueItem

logger = logging.getLogger(__name__)


class PathNavigator:
    """
    PathNavigator agent - Phase 2 of CLAUSE 3-agent system.

    Implements budget-aware path exploration with:
    - State encoding (1154-dim feature vector)
    - Termination head (learned stopping criterion)
    - Backtrack mechanism (when all neighbors below threshold)
    - Intelligence integrations: ThoughtSeeds, Curiosity, Causal
    """

    def __init__(
        self,
        neo4j_client=None,
        redis_client=None,
        embedding_service=None,
        basin_tracker=None,
        thoughtseed_generator=None,
        causal_network=None,
        causal_queue=None,
    ):
        """
        Initialize PathNavigator with required dependencies.

        Args:
            neo4j_client: Neo4j graph database client
            redis_client: Redis client for curiosity queue
            embedding_service: Sentence transformer embedding service
            basin_tracker: AttractorBasin tracker from Phase 1
            thoughtseed_generator: ThoughtSeed generation service (Spec 028)
            causal_network: Causal Bayesian network (Spec 033)
            causal_queue: CausalQueue for background causal processing (T041a)
        """
        self.neo4j = neo4j_client
        self.redis = redis_client
        self.embedder = embedding_service
        self.basin_tracker = basin_tracker
        self.thoughtseed_gen = thoughtseed_generator
        self.causal_net = causal_network
        self.causal_queue = causal_queue

        # Termination head weights (placeholder - will be learned via LC-MAPPO)
        self.termination_head_weights = np.random.randn(1155)  # 1154 state + 1 budget
        self.termination_head_bias = 0.0

        # T024: Track previous query hash for causal result lookup
        self._prev_query_hash: Optional[str] = None

        logger.info("PathNavigator initialized")

    async def navigate(self, request: PathNavigationRequest) -> PathNavigationResponse:
        """
        Execute budget-aware path navigation.

        Per research.md Decision 1-5:
        - Encode state as (query_emb, node_emb, neighborhood)
        - Termination head decides when to stop
        - Generate ThoughtSeeds during exploration
        - Trigger curiosity on prediction errors
        - Use causal reasoning for path selection

        Args:
            request: PathNavigationRequest with query, start_node, budgets

        Returns:
            PathNavigationResponse with path, metadata, performance
        """
        start_time = time.time()
        perf_tracker = {"thoughtseed_gen_ms": 0.0, "causal_pred_ms": 0.0}

        # Initialize navigation state
        current_node = request.start_node
        budget_remaining = request.step_budget
        path_nodes = [current_node]
        path_edges = []
        path_steps = []
        thoughtseeds_generated = []
        curiosity_triggers_spawned = 0

        # Embed query once (reused across steps)
        query_embedding = await self._embed_text(request.query)

        # T018-T025: Main navigation loop
        for step_num in range(1, request.step_budget + 1):
            budget_remaining = request.step_budget - step_num + 1

            # T019: Encode current state (1154-dim)
            state = await self._encode_state(
                query_embedding=query_embedding,
                current_node=current_node,
                budget_remaining=budget_remaining,
            )

            # T020: Check termination head
            if self._should_terminate(state):
                path_steps.append(
                    PathStep(
                        step=step_num,
                        from_node=current_node,
                        to_node=current_node,
                        relation="STOP",
                        action="STOP",
                    )
                )
                logger.info(f"Termination head triggered at step {step_num}")
                break

            # Get candidate next hops
            candidates = await self._get_candidate_hops(current_node)

            if not candidates:
                # T021: No candidates - trigger STOP
                path_steps.append(
                    PathStep(
                        step=step_num,
                        from_node=current_node,
                        to_node=current_node,
                        relation="STOP",
                        action="STOP",
                    )
                )
                logger.info(f"No candidates at step {step_num} - STOP")
                break

            # Score candidates with optional causal reasoning
            scored_candidates = []

            # T024: Check for previous step's causal results
            causal_scores_dict = None
            causal_fallback_used = False
            if request.enable_causal and self.causal_queue and self._prev_query_hash:
                causal_scores_dict = await self.causal_queue.get_result(self._prev_query_hash)
                if causal_scores_dict:
                    logger.debug(f"Using cached causal results from previous step")

            for candidate in candidates:
                # T024: Causal reasoning with AsyncIO timeout (if enabled)
                causal_score = None
                if request.enable_causal and self.causal_net:
                    causal_start = time.time()

                    # Try to use cached results from previous step first
                    if causal_scores_dict and candidate["node"] in causal_scores_dict:
                        causal_score = causal_scores_dict[candidate["node"]]
                        logger.debug(f"Using cached causal score for {candidate['node']}")
                    else:
                        # T024: AsyncIO timeout with fallback
                        causal_score, causal_fallback_used = await self._causal_predict_with_timeout(
                            candidates=[c["node"] for c in candidates],
                            query_hash=self._make_query_hash(request.query, step_num),
                            step_num=step_num,
                        )
                        # Extract score for this candidate
                        causal_score = causal_score.get(candidate["node"], 0.5)

                    perf_tracker["causal_pred_ms"] += (time.time() - causal_start) * 1000

                # Base score (semantic similarity)
                base_score = await self._score_candidate(
                    candidate, query_embedding, state
                )

                # T023: ThoughtSeed generation (if enabled)
                thoughtseed_id = None
                if request.enable_thoughtseeds and self.thoughtseed_gen:
                    ts_start = time.time()
                    ts_id = await self._generate_thoughtseed(
                        concept=candidate["node"],
                        source_doc=request.query,
                    )
                    thoughtseed_id = ts_id
                    thoughtseeds_generated.append(ts_id)
                    perf_tracker["thoughtseed_gen_ms"] += (time.time() - ts_start) * 1000

                # Combine scores
                final_score = base_score
                if causal_score is not None:
                    final_score = 0.7 * base_score + 0.3 * causal_score

                scored_candidates.append(
                    {
                        "candidate": candidate,
                        "base_score": base_score,
                        "causal_score": causal_score,
                        "final_score": final_score,
                        "thoughtseed_id": thoughtseed_id,
                    }
                )

            # T024: Curiosity triggers (if enabled)
            if request.enable_curiosity and self.redis:
                for sc in scored_candidates:
                    # Expected score (from termination head or heuristic)
                    expected_score = 0.5  # Placeholder
                    actual_score = sc["final_score"]
                    prediction_error = abs(expected_score - actual_score)

                    if prediction_error > request.curiosity_threshold:
                        await self._trigger_curiosity(
                            concept=sc["candidate"]["node"],
                            error_magnitude=prediction_error,
                        )
                        curiosity_triggers_spawned += 1

            # Select best candidate
            best = max(scored_candidates, key=lambda x: x["final_score"])

            # T021: Check if best score below threshold (backtrack condition)
            if best["final_score"] < 0.3:  # Threshold from research.md
                path_steps.append(
                    PathStep(
                        step=step_num,
                        from_node=current_node,
                        to_node=current_node,
                        relation="BACKTRACK",
                        action="BACKTRACK",
                    )
                )
                logger.info(f"All candidates below threshold - BACKTRACK at step {step_num}")
                # In real implementation, would pop from path and retry
                break

            # Move to next node
            next_node = best["candidate"]["node"]
            relation = best["candidate"]["relation"]

            path_nodes.append(next_node)
            path_edges.append(
                {
                    "from": current_node,
                    "relation": relation,
                    "to": next_node,
                }
            )
            path_steps.append(
                PathStep(
                    step=step_num,
                    from_node=current_node,
                    to_node=next_node,
                    relation=relation,
                    action="CONTINUE",
                    causal_score=best["causal_score"],
                    thoughtseed_id=best["thoughtseed_id"],
                )
            )

            current_node = next_node

        # T022: Budget enforcement verification
        assert len(path_steps) <= request.step_budget, "Budget violation detected"

        # Build response
        total_latency = (time.time() - start_time) * 1000
        final_action = path_steps[-1].action if path_steps else "STOP"

        return PathNavigationResponse(
            path={
                "nodes": path_nodes,
                "edges": path_edges,
                "steps": [step.model_dump() for step in path_steps],
            },
            metadata={
                "budget_used": len(path_steps),
                "budget_total": request.step_budget,
                "final_action": final_action,
                "thoughtseeds_generated": len(thoughtseeds_generated),
                "curiosity_triggers_spawned": curiosity_triggers_spawned,
            },
            performance={
                "latency_ms": total_latency,
                "thoughtseed_gen_ms": perf_tracker["thoughtseed_gen_ms"],
                "causal_pred_ms": perf_tracker["causal_pred_ms"],
            },
        )

    # T019: State Encoding (1154-dim)
    async def _encode_state(
        self,
        query_embedding: np.ndarray,
        current_node: str,
        budget_remaining: int,
    ) -> StateEncoding:
        """
        Encode navigation state per research.md Decision 1.

        State = (query_emb[384], node_emb[384], node_degree[1], basin_strength[1],
                 neighborhood_mean[384], budget_norm[1]) = 1155 features

        Returns:
            StateEncoding with 1154-dim feature vector
        """
        # Node embedding
        node_text = await self._get_node_text(current_node)
        node_embedding = await self._embed_text(node_text)

        # Node degree
        node_degree = await self._get_node_degree(current_node)

        # Basin strength (from Phase 1 AttractorBasin)
        basin_strength = 1.0  # Default
        if self.basin_tracker:
            basin = await self.basin_tracker.get(current_node)
            basin_strength = basin.strength if basin else 1.0

        # Neighborhood mean (1-hop neighbors)
        neighbors = await self._get_neighbors(current_node)
        if neighbors:
            neighbor_embeddings = [
                await self._embed_text(await self._get_node_text(n)) for n in neighbors
            ]
            neighborhood_mean = np.mean(neighbor_embeddings, axis=0).tolist()
        else:
            neighborhood_mean = [0.0] * 384  # Zero vector

        # Budget remaining (normalized)
        budget_norm = budget_remaining / 20.0  # Normalized to [0, 1]

        return StateEncoding(
            query_embedding=query_embedding.tolist(),
            node_embedding=node_embedding.tolist(),
            node_degree=node_degree,
            basin_strength=basin_strength,
            neighborhood_mean=neighborhood_mean,
            budget_remaining=budget_norm,
        )

    # T020: Termination Head
    def _should_terminate(self, state: StateEncoding) -> bool:
        """
        Binary classifier for termination decision per research.md Decision 2.

        Sigmoid classifier: stop_prob = σ(w·state + b)
        Stop if stop_prob > 0.5

        Args:
            state: StateEncoding (1154-dim + budget)

        Returns:
            True if should stop, False otherwise
        """
        # Convert state to numpy (already includes budget_remaining)
        features = state.to_numpy()  # 1155 features (384+384+1+1+384+1)

        # Linear layer + sigmoid
        logit = np.dot(self.termination_head_weights, features) + self.termination_head_bias
        stop_prob = 1.0 / (1.0 + np.exp(-logit))

        return stop_prob > 0.5

    # T023: ThoughtSeed Generation Hook
    async def _generate_thoughtseed(
        self, concept: str, source_doc: str
    ) -> Optional[str]:
        """
        Generate ThoughtSeed with basin context per research.md Decision 3.

        Integrates with Spec 028 for cross-document linking.

        Args:
            concept: Concept node ID
            source_doc: Source query/document

        Returns:
            ThoughtSeed ID or None
        """
        if not self.thoughtseed_gen:
            return None

        # Fetch basin context
        basin_context = {"strength": 1.0, "activation_count": 0, "co_occurring": {}}
        if self.basin_tracker:
            basin = await self.basin_tracker.get(concept)
            if basin:
                basin_context = {
                    "strength": basin.strength,
                    "activation_count": basin.activation_count,
                    "co_occurring": basin.co_occurring_concepts,
                }

        # Generate ThoughtSeed
        ts_id = await self.thoughtseed_gen.create(
            concept=concept,
            source_doc=source_doc,
            basin_context=basin_context,
            similarity_threshold=0.8,
        )

        return ts_id

    # T024: Curiosity Trigger Hook
    async def _trigger_curiosity(self, concept: str, error_magnitude: float) -> None:
        """
        Trigger curiosity agent via Redis queue per research.md Decision 4.

        Integrates with Spec 029 for background investigation.

        Args:
            concept: Concept that triggered curiosity
            error_magnitude: Prediction error magnitude [0, 1]
        """
        if not self.redis:
            return

        trigger_data = {
            "trigger_type": "prediction_error",
            "concept": concept,
            "error_magnitude": error_magnitude,
            "timestamp": datetime.now().isoformat(),
            "investigation_status": "queued",
        }

        # Push to Redis queue (non-blocking)
        await self.redis.lpush("curiosity_queue", str(trigger_data))
        logger.debug(f"Curiosity trigger queued for concept: {concept}")

    # T024: AsyncIO Causal Timeout with Background Queue
    async def _causal_predict_with_timeout(
        self,
        candidates: List[str],
        query_hash: str,
        step_num: int,
    ) -> Tuple[Dict[str, float], bool]:
        """
        T024: Causal prediction with 30ms AsyncIO timeout and background queue fallback.

        Per research.md decision 14:
        - Try causal prediction with 30ms timeout
        - On timeout: queue for background processing, use semantic similarity fallback
        - On success: return causal scores
        - Next step checks queue for previous step's results

        Args:
            candidates: List of candidate node names
            query_hash: Hash identifying this query (for result lookup)
            step_num: Current navigation step

        Returns:
            Tuple of (scores_dict, fallback_used)
            - scores_dict: Dict[candidate -> causal_score]
            - fallback_used: True if semantic fallback was used
        """
        if not self.causal_net:
            return {}, True

        try:
            # T024: AsyncIO timeout (30ms)
            scores = await asyncio.wait_for(
                self.causal_net.predict(candidates),
                timeout=0.03  # 30ms timeout
            )

            # Success - store query hash for next step
            self._prev_query_hash = query_hash
            logger.debug(f"Causal prediction succeeded within 30ms")
            return scores, False

        except asyncio.TimeoutError:
            # T024: Timeout - queue for background processing
            logger.debug(f"Causal prediction timeout - queueing for background processing")

            if self.causal_queue:
                # Queue this prediction
                await self.causal_queue.put(
                    CausalQueueItem(
                        query_hash=query_hash,
                        candidates=candidates,
                        step_num=step_num,
                    )
                )

                # Store query hash for next step result lookup
                self._prev_query_hash = query_hash

            # T024: Fallback to semantic similarity (uniform scores)
            fallback_scores = {candidate: 0.5 for candidate in candidates}
            return fallback_scores, True

    def _make_query_hash(self, query: str, step_num: int) -> str:
        """
        Create hash for query + step to identify causal results.

        Args:
            query: Query text
            step_num: Navigation step number

        Returns:
            SHA256 hash (first 16 chars)
        """
        content = f"{query}:{step_num}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # T025: Causal Reasoning Hook (DEPRECATED - use _causal_predict_with_timeout instead)
    async def _estimate_causal_intervention(
        self, intervention_node: str, target_node: str
    ) -> Optional[float]:
        """
        Estimate causal intervention effect per research.md Decision 5.

        Integrates with Spec 033 for causal reasoning.
        Uses pre-computed DAG + LRU cache for <30ms latency.

        NOTE: This method is deprecated in favor of _causal_predict_with_timeout (T024).
        Kept for backward compatibility.

        Args:
            intervention_node: Node to intervene on
            target_node: Target outcome node

        Returns:
            P(target | do(intervention)) in [0, 1] or None
        """
        if not self.causal_net:
            return None

        # Cache lookup + do-calculus
        score = await self.causal_net.estimate_intervention(
            intervention=intervention_node,
            target=target_node,
        )

        return score

    # Helper methods (stubs - will be implemented in integration phase)
    async def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using sentence transformer (384-dim)"""
        if self.embedder:
            return await self.embedder.embed(text)
        return np.random.randn(384)  # Placeholder

    async def _get_node_text(self, node_id: str) -> str:
        """Get node text content from Neo4j"""
        return node_id  # Placeholder

    async def _get_node_degree(self, node_id: str) -> int:
        """Get node degree from Neo4j"""
        return 5  # Placeholder

    async def _get_neighbors(self, node_id: str) -> List[str]:
        """Get 1-hop neighbors from Neo4j"""
        return []  # Placeholder

    async def _get_candidate_hops(self, node_id: str) -> List[Dict[str, Any]]:
        """Get candidate next hops from Neo4j"""
        return []  # Placeholder

    async def _score_candidate(
        self, candidate: Dict[str, Any], query_embedding: np.ndarray, state: StateEncoding
    ) -> float:
        """Score candidate hop (semantic similarity)"""
        return 0.5  # Placeholder
