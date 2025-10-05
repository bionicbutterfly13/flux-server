"""
T042-T043: CausalBayesianNetwork service

Implements causal reasoning using do-calculus on knowledge graph structure.
Per research.md decision 5: Pre-computed DAG + LRU cache for intervention predictions.

Uses Neo4j graph structure to build causal DAG and estimate intervention effects.
"""

import logging
from functools import lru_cache
from typing import Dict, Optional, Set, Tuple
import hashlib

logger = logging.getLogger(__name__)


class CausalBayesianNetwork:
    """
    Causal Bayesian Network for intervention prediction.

    Per research.md decision 5:
    - Pre-compute causal DAG from Neo4j graph structure
    - Use do-calculus to estimate P(target | do(intervention))
    - LRU cache (size=1000) for prediction results
    """

    def __init__(self, neo4j_client=None):
        """
        Initialize CausalBayesianNetwork.

        Args:
            neo4j_client: Neo4jClient instance (injected)
        """
        self.neo4j = neo4j_client
        self.dag_cache: Dict[str, Set[str]] = {}  # node -> children
        self.dag_built = False

    async def build_causal_dag(self) -> None:
        """
        Pre-compute causal DAG structure from Neo4j.

        Builds directed graph from Neo4j relationships:
        - CAUSES → causal edge
        - RELATED_TO → bidirectional edge
        - HAS_CONCEPT → concept hierarchy

        Stores in self.dag_cache for fast lookup.
        """
        if not self.neo4j:
            logger.warning("Neo4j not available, using empty DAG")
            self.dag_built = True
            return

        logger.info("Building causal DAG from Neo4j graph...")

        try:
            # Query Neo4j for causal relationships
            query = """
            MATCH (source)-[r:CAUSES|RELATED_TO|HAS_CONCEPT]->(target)
            RETURN source.name AS source_name, target.name AS target_name, type(r) AS rel_type
            LIMIT 10000
            """

            # Execute query (mock implementation for now)
            # In production, would use self.neo4j.execute_query(query)
            results = []

            # Build DAG from results
            for record in results:
                source = record.get("source_name", "")
                target = record.get("target_name", "")
                rel_type = record.get("rel_type", "")

                if source and target:
                    # Add edge to DAG
                    if source not in self.dag_cache:
                        self.dag_cache[source] = set()

                    if rel_type == "CAUSES":
                        # Directed causal edge
                        self.dag_cache[source].add(target)
                    elif rel_type in ["RELATED_TO", "HAS_CONCEPT"]:
                        # Bidirectional edge (add both directions)
                        self.dag_cache[source].add(target)
                        if target not in self.dag_cache:
                            self.dag_cache[target] = set()
                        self.dag_cache[target].add(source)

            self.dag_built = True
            logger.info(f"Causal DAG built: {len(self.dag_cache)} nodes")

        except Exception as e:
            logger.error(f"Failed to build causal DAG: {e}", exc_info=True)
            self.dag_built = True  # Mark as built to avoid retry loops

    async def predict(self, candidates: list[str]) -> Dict[str, float]:
        """
        Predict causal scores for candidate nodes.

        Args:
            candidates: List of candidate node names

        Returns:
            Dict[candidate -> causal_score] (0.0-1.0)
        """
        if not self.dag_built:
            await self.build_causal_dag()

        scores = {}

        for candidate in candidates:
            # Use cached do-calculus prediction
            score = self._do_calculus_cached(candidate)
            scores[candidate] = score

        return scores

    @lru_cache(maxsize=1000)
    def _do_calculus_cached(self, target_node: str) -> float:
        """
        Cached do-calculus intervention prediction.

        Args:
            target_node: Target node for intervention

        Returns:
            Intervention score (0.0-1.0)
        """
        # Simplified do-calculus: use graph connectivity as proxy
        # In production, would implement full do-calculus algorithm

        if target_node not in self.dag_cache:
            return 0.5  # Default score for unknown nodes

        # Score based on out-degree (more connections = higher causal influence)
        out_degree = len(self.dag_cache[target_node])
        score = min(1.0, out_degree / 10.0)  # Normalize to [0, 1]

        return score

    async def estimate_intervention(
        self, intervention_node: str, target_node: str
    ) -> float:
        """
        Estimate P(target | do(intervention)) using do-calculus.

        Args:
            intervention_node: Node to intervene on
            target_node: Target outcome node

        Returns:
            Intervention score (0.0-1.0)
        """
        if not self.dag_built:
            await self.build_causal_dag()

        # Use cached prediction
        cache_key = self._make_cache_key(intervention_node, target_node)
        return self._estimate_intervention_cached(cache_key)

    @lru_cache(maxsize=1000)
    def _estimate_intervention_cached(self, cache_key: str) -> float:
        """
        Cached intervention estimation.

        Args:
            cache_key: Hash of (intervention_node, target_node)

        Returns:
            Intervention score (0.0-1.0)
        """
        # Simplified implementation: check if path exists from intervention to target
        # In production, would implement full backdoor criterion + do-calculus

        intervention_node, target_node = cache_key.split("->")

        # BFS to find path
        if intervention_node not in self.dag_cache:
            return 0.3  # Default low score

        visited = set()
        queue = [intervention_node]
        depth = 0
        max_depth = 5

        while queue and depth < max_depth:
            current = queue.pop(0)

            if current == target_node:
                # Path found - score based on depth (shorter = stronger causal link)
                return 1.0 - (depth / max_depth)

            if current in visited:
                continue

            visited.add(current)

            # Add neighbors to queue
            if current in self.dag_cache:
                for neighbor in self.dag_cache[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            depth += 1

        # No path found - weak causal link
        return 0.2

    def _make_cache_key(self, intervention_node: str, target_node: str) -> str:
        """Create cache key for intervention pair"""
        return f"{intervention_node}->{target_node}"

    def do_calculus(self, intervention: str, target: str) -> float:
        """
        T043: Do-calculus intervention prediction (synchronous wrapper).

        Args:
            intervention: Intervention node
            target: Target node

        Returns:
            Intervention score (0.0-1.0)
        """
        # Synchronous wrapper for async estimate_intervention
        # Use cached version directly
        cache_key = self._make_cache_key(intervention, target)
        return self._estimate_intervention_cached(cache_key)

    def get_descendants(self, node: str, max_depth: int = 3) -> Set[str]:
        """
        Get all descendant nodes (children, grandchildren, etc.)

        Args:
            node: Source node
            max_depth: Maximum traversal depth

        Returns:
            Set of descendant node names
        """
        if node not in self.dag_cache:
            return set()

        descendants = set()
        visited = set()
        queue = [(node, 0)]

        while queue:
            current, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            if current in visited:
                continue

            visited.add(current)

            # Add children
            if current in self.dag_cache:
                for child in self.dag_cache[current]:
                    descendants.add(child)
                    queue.append((child, depth + 1))

        return descendants

    def get_dag_stats(self) -> Dict:
        """Get DAG statistics for debugging"""
        return {
            "nodes": len(self.dag_cache),
            "edges": sum(len(children) for children in self.dag_cache.values()),
            "avg_degree": (
                sum(len(children) for children in self.dag_cache.values())
                / len(self.dag_cache)
                if self.dag_cache
                else 0
            ),
            "cache_size": self._do_calculus_cached.cache_info().currsize,
            "cache_hits": self._do_calculus_cached.cache_info().hits,
            "cache_misses": self._do_calculus_cached.cache_info().misses,
        }
