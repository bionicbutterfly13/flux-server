#!/usr/bin/env python3
"""
T018: EdgeScorer - 5-Signal CLAUSE Edge Scoring

Implements CLAUSE edge scoring with 5 signals:
- φ_ent (0.25): Entity-query relevance
- φ_rel (0.25): Relation-query relevance
- φ_nbr (0.20): Neighborhood co-occurrence
- φ_deg (0.15): Degree prior (prefer moderate degree)
- φ_basin (0.15): Basin strength normalized

Performance target: <10ms for 1000 edges (NumPy vectorization)
"""

from typing import Dict, List, Optional
import numpy as np

from src.services.basin_tracker import BasinTracker


class EdgeScorer:
    """
    CLAUSE 5-signal edge scorer.

    Scores edges based on query relevance, neighborhood structure,
    and basin strength for budget-aware subgraph construction.
    """

    def __init__(self, basin_tracker: Optional[BasinTracker] = None):
        """
        Initialize edge scorer.

        Args:
            basin_tracker: BasinTracker instance for basin strength signal
        """
        self.basin_tracker = basin_tracker or BasinTracker()

        # Default signal weights (CLAUSE spec)
        self.default_weights = {
            "entity": 0.25,
            "relation": 0.25,
            "neighborhood": 0.20,
            "degree": 0.15,
            "basin": 0.15,
        }

    def _hash_based_similarity(self, text1: str, text2: str) -> float:
        """
        Hash-based semantic similarity (NumPy 2.0 compatible).

        Uses character n-gram hashing for semantic similarity without
        sentence-transformers (NumPy 2.0 incompatibility workaround).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score [0.0, 1.0]
        """
        # Normalize texts (replace underscores with spaces)
        t1 = text1.lower().strip().replace("_", " ")
        t2 = text2.lower().strip().replace("_", " ")

        # Exact match
        if t1 == t2:
            return 1.0

        # Substring match
        if t1 in t2 or t2 in t1:
            return 0.9

        # Token overlap (simple Jaccard)
        tokens1 = set(t1.split())
        tokens2 = set(t2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        jaccard = len(intersection) / len(union)

        # Boost for partial word matches (stemming-like)
        partial_matches = 0
        for w1 in tokens1:
            for w2 in tokens2:
                # Check if one word is substring of another (min 3 chars)
                if len(w1) >= 3 and len(w2) >= 3:
                    if w1 in w2 or w2 in w1:
                        partial_matches += 1

        partial_score = min(partial_matches / len(union), 0.3)

        return min(jaccard + partial_score, 1.0)

    def _calculate_phi_ent(self, edge: Dict[str, str], query: str) -> float:
        """
        Calculate entity relevance signal.

        Measures how relevant source and target entities are to the query.

        Args:
            edge: Edge dict with 'source', 'target'
            query: Query string

        Returns:
            φ_ent score [0.0, 1.0]
        """
        source = edge.get("source", "")
        target = edge.get("target", "")

        # Calculate similarity of both entities to query
        source_sim = self._hash_based_similarity(source, query)
        target_sim = self._hash_based_similarity(target, query)

        # Take maximum (at least one entity relevant)
        return max(source_sim, target_sim)

    def _calculate_phi_rel(self, edge: Dict[str, str], query: str) -> float:
        """
        Calculate relation relevance signal.

        Measures how relevant the relation type is to the query.

        Args:
            edge: Edge dict with 'relation'
            query: Query string

        Returns:
            φ_rel score [0.0, 1.0]
        """
        relation = edge.get("relation", "")

        # Calculate relation-query similarity
        return self._hash_based_similarity(relation, query)

    def _calculate_phi_nbr(self, edge: Dict[str, str]) -> float:
        """
        Calculate neighborhood signal.

        Measures co-occurrence strength of source and target concepts.

        Args:
            edge: Edge dict with 'source', 'target'

        Returns:
            φ_nbr score [0.0, 1.0]
        """
        source = edge.get("source", "")
        target = edge.get("target", "")

        try:
            # Get source basin
            source_basin = self.basin_tracker.get_basin(source)
            co_occurring = source_basin.get("co_occurring_concepts", {})

            # Get co-occurrence count with target
            count = co_occurring.get(target, 0)

            # Normalize (assume max co-occurrence is 5 for strong signal)
            return min(count / 5.0, 1.0)

        except KeyError:
            # Basin not found, no co-occurrence
            return 0.0

    def _calculate_phi_deg(self, edge: Dict[str, str], all_edges: Optional[List[Dict[str, str]]] = None) -> float:
        """
        Calculate degree signal.

        Prefers moderate degree nodes (not too sparse, not too dense).

        Args:
            edge: Edge dict with 'source', 'target'
            all_edges: All edges for degree calculation (optional)

        Returns:
            φ_deg score [0.0, 1.0]
        """
        if all_edges is None:
            # No context, return neutral score
            return 0.5

        source = edge.get("source", "")
        target = edge.get("target", "")

        # Count edges involving source or target
        source_degree = sum(1 for e in all_edges if e.get("source") == source or e.get("target") == source)
        target_degree = sum(1 for e in all_edges if e.get("source") == target or e.get("target") == target)

        # Prefer moderate degree (5-20 edges)
        def degree_score(deg: int) -> float:
            if deg < 5:
                return deg / 5.0 * 0.7  # Low degree penalty
            elif deg <= 20:
                return 1.0  # Optimal range
            else:
                return max(0.3, 1.0 - (deg - 20) / 50.0)  # High degree penalty

        # Average of source and target degree scores
        return (degree_score(source_degree) + degree_score(target_degree)) / 2.0

    def _calculate_phi_basin(self, edge: Dict[str, str]) -> float:
        """
        Calculate basin strength signal.

        Measures basin frequency strength normalized to [0.0, 1.0].

        Args:
            edge: Edge dict with 'source', 'target'

        Returns:
            φ_basin score [0.0, 1.0]
        """
        source = edge.get("source", "")
        target = edge.get("target", "")

        source_strength = 1.0
        target_strength = 1.0

        try:
            source_basin = self.basin_tracker.get_basin(source)
            source_strength = source_basin.get("strength", 1.0)
        except KeyError:
            pass

        try:
            target_basin = self.basin_tracker.get_basin(target)
            target_strength = target_basin.get("strength", 1.0)
        except KeyError:
            pass

        # Normalize: (strength - 1.0) / 1.0 → [0.0, 1.0]
        source_norm = (source_strength - 1.0) / 1.0
        target_norm = (target_strength - 1.0) / 1.0

        # Maximum of source and target strengths (at least one strong basin)
        return max(source_norm, target_norm)

    def score_edge(
        self,
        edge: Dict[str, str],
        query: str,
        return_breakdown: bool = False,
        all_edges: Optional[List[Dict[str, str]]] = None,
    ) -> Dict:
        """
        Score a single edge with CLAUSE 5-signal formula.

        Args:
            edge: Edge dict with 'source', 'relation', 'target'
            query: Query string
            return_breakdown: Return signal breakdown
            all_edges: All edges for degree calculation (optional)

        Returns:
            Dict with 'score' and optionally 'breakdown'
        """
        # Calculate 5 signals
        phi_ent = self._calculate_phi_ent(edge, query)
        phi_rel = self._calculate_phi_rel(edge, query)
        phi_nbr = self._calculate_phi_nbr(edge)
        phi_deg = self._calculate_phi_deg(edge, all_edges)
        phi_basin = self._calculate_phi_basin(edge)

        # Weighted sum (CLAUSE formula)
        score = (
            0.25 * phi_ent +
            0.25 * phi_rel +
            0.20 * phi_nbr +
            0.15 * phi_deg +
            0.15 * phi_basin
        )

        result = {"score": score}

        if return_breakdown:
            result["breakdown"] = {
                "phi_ent": phi_ent,
                "phi_rel": phi_rel,
                "phi_nbr": phi_nbr,
                "phi_deg": phi_deg,
                "phi_basin": phi_basin,
            }

        return result

    def score_edges(
        self,
        edges: List[Dict[str, str]],
        query: str,
        signal_weights: Optional[Dict[str, float]] = None,
        vectorized: bool = True,
    ) -> Dict[str, float]:
        """
        Score multiple edges (uses NumPy vectorization by default for performance).

        Args:
            edges: List of edge dicts
            query: Query string
            signal_weights: Optional custom weights
            vectorized: Use NumPy vectorization (default True for performance)

        Returns:
            Dict mapping edge_key → score
        """
        if vectorized:
            return self.score_edges_vectorized(edges, query, signal_weights)

        # Fallback: non-vectorized scoring
        scores = {}
        for edge in edges:
            edge_key = f"{edge['source']}_{edge['relation']}_{edge['target']}"
            result = self.score_edge(
                edge, query, return_breakdown=False, all_edges=edges
            )
            scores[edge_key] = result["score"]

        return scores

    def score_edges_vectorized(
        self,
        edges: List[Dict[str, str]],
        query: str,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Score multiple edges using NumPy vectorization (T028).

        Performance target: <10ms for 1000 edges (100x faster than non-vectorized).

        Args:
            edges: List of edge dicts
            query: Query string
            signal_weights: Optional custom weights

        Returns:
            Dict mapping edge_key → score
        """
        if not edges:
            return {}

        weights = signal_weights or self.default_weights
        n_edges = len(edges)

        # Create signal matrix (N × 5)
        signal_matrix = np.zeros((n_edges, 5), dtype=np.float32)

        # Vectorize entity signal (φ_ent)
        for i, edge in enumerate(edges):
            source_sim = self._hash_based_similarity(edge['source'], query)
            target_sim = self._hash_based_similarity(edge['target'], query)
            signal_matrix[i, 0] = max(source_sim, target_sim)

        # Vectorize relation signal (φ_rel)
        for i, edge in enumerate(edges):
            signal_matrix[i, 1] = self._hash_based_similarity(edge['relation'], query)

        # Vectorize neighborhood signal (φ_nbr)
        # Build co-occurrence map for all edges
        co_occur_map = {}
        for edge in edges:
            source = edge['source']
            target = edge['target']
            source_basin = self.basin_tracker.get_basin(source) if source in self.basin_tracker._basins else None
            if source_basin and target in source_basin.get('co_occurring_concepts', {}):
                count = source_basin['co_occurring_concepts'][target]
                co_occur_map[f"{source}_{target}"] = min(count / 5.0, 1.0)

        for i, edge in enumerate(edges):
            edge_key = f"{edge['source']}_{edge['target']}"
            signal_matrix[i, 2] = co_occur_map.get(edge_key, 0.0)

        # Vectorize degree signal (φ_deg)
        # Count degrees from edges list
        degree_counts = {}
        for edge in edges:
            degree_counts[edge['source']] = degree_counts.get(edge['source'], 0) + 1
            degree_counts[edge['target']] = degree_counts.get(edge['target'], 0) + 1

        for i, edge in enumerate(edges):
            source_deg = degree_counts.get(edge['source'], 1)
            target_deg = degree_counts.get(edge['target'], 1)
            avg_deg = (source_deg + target_deg) / 2.0
            # Prefer moderate degree (peak at 10)
            signal_matrix[i, 3] = 1.0 - abs(avg_deg - 10.0) / 20.0
            signal_matrix[i, 3] = max(0.0, min(1.0, signal_matrix[i, 3]))

        # Vectorize basin signal (φ_basin)
        for i, edge in enumerate(edges):
            source = edge['source']
            target = edge['target']
            source_strength = 1.0
            target_strength = 1.0

            if source in self.basin_tracker._basins:
                source_strength = self.basin_tracker._basins[source].strength

            if target in self.basin_tracker._basins:
                target_strength = self.basin_tracker._basins[target].strength

            # Normalize to [0, 1]
            source_norm = (source_strength - 1.0) / 1.0
            target_norm = (target_strength - 1.0) / 1.0
            signal_matrix[i, 4] = max(source_norm, target_norm)

        # Weight vector [φ_ent, φ_rel, φ_nbr, φ_deg, φ_basin]
        weight_vector = np.array([
            weights['entity'],
            weights['relation'],
            weights['neighborhood'],
            weights['degree'],
            weights['basin'],
        ], dtype=np.float32)

        # Vectorized matrix multiplication: (N × 5) @ (5 × 1) → (N × 1)
        scores_array = signal_matrix @ weight_vector

        # Convert to dict
        scores = {}
        for i, edge in enumerate(edges):
            edge_key = f"{edge['source']}_{edge['relation']}_{edge['target']}"
            scores[edge_key] = float(scores_array[i])

        return scores
