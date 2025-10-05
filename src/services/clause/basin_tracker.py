#!/usr/bin/env python3
"""
T020: BasinTracker - Basin Frequency Strengthening

Implements basin strengthening with:
- Strength increment: +0.2 per activation (default)
- Strength cap: 2.0
- Co-occurrence tracking: symmetric updates
- Performance: <5ms per basin update
"""

import time
from typing import List, Dict
from datetime import datetime

from src.models.attractor_basin import AttractorBasin
from src.models import (
    BasinStrengtheningRequest,
    BasinStrengtheningResponse,
    BasinInfo,
)
from src.services.basin_cache import BasinCache


class BasinTracker:
    """
    Basin frequency strengthening tracker.

    Strengthens attractor basins based on concept appearance frequency,
    with symmetric co-occurrence tracking. Integrated with Redis cache
    for fast lookups (T027).
    """

    def __init__(self, basin_cache: BasinCache = None):
        """
        Initialize basin tracker with in-memory storage and optional cache.

        Args:
            basin_cache: Optional BasinCache instance for Redis caching
        """
        # In-memory basin storage (will integrate with Neo4j in T026)
        self._basins: Dict[str, AttractorBasin] = {}

        # Redis cache integration (T027)
        self._cache = basin_cache

    def strengthen_basins(
        self,
        concepts: List[str],
        document_id: str,
        increment: float = 0.2,
    ) -> Dict:
        """
        Strengthen basins for given concepts.

        Args:
            concepts: List of concept names to strengthen
            document_id: Document identifier for activation history
            increment: Strength increment (default 0.2)

        Returns:
            Dict with updated_basins, new_basins, cooccurrence_updates, timing
        """
        start_time = time.perf_counter()

        updated_basins: List[BasinInfo] = []
        new_basins: List[BasinInfo] = []
        cooccurrence_updates: Dict[str, List[str]] = {}

        # Process each concept
        for concept in concepts:
            basin_id = concept  # Use concept as basin_id

            # Get or create basin
            if basin_id in self._basins:
                basin = self._basins[basin_id]
                is_new = False
            else:
                # Create new basin with defaults
                basin = AttractorBasin(
                    basin_name=concept,
                    basin_type="conceptual",
                    stability=0.8,  # Default stability
                    depth=1.5,  # Default depth
                    activation_threshold=0.5,  # Default threshold
                    neural_field_influence={
                        "field_contribution": 0.0,
                        "spatial_extent": 1.0,
                        "temporal_persistence": 1.0,
                    },
                    strength=1.0,  # Initial strength
                    activation_count=0,
                    co_occurring_concepts={},
                )
                self._basins[basin_id] = basin
                is_new = True

            # Increment activation count
            basin.activation_count += 1

            # Strengthen basin (including first activation)
            if basin.activation_count > 1:
                # Increment strength starting from 2nd activation (cap at 2.0)
                basin.strength = min(basin.strength + increment, 2.0)

            # Update activation history
            activation_event = {
                "timestamp": datetime.utcnow().isoformat(),
                "document_id": document_id,
                "strength_after": basin.strength,
                "activation_count": basin.activation_count,
            }
            basin.activation_history.append(activation_event)

            # Invalidate cache on update (T027)
            if self._cache:
                self._cache.invalidate_basin(concept)

            # Create basin info
            basin_info = BasinInfo(
                basin_id=basin_id,
                basin_name=basin.basin_name,
                strength=basin.strength,
                activation_count=basin.activation_count,
                co_occurring_concepts=basin.co_occurring_concepts.copy(),
            )

            if is_new:
                new_basins.append(basin_info)
            else:
                updated_basins.append(basin_info)

        # Update co-occurrence (symmetric)
        for i, concept_a in enumerate(concepts):
            cooccurrence_updates[concept_a] = []

            for j, concept_b in enumerate(concepts):
                if i != j:  # Don't co-occur with self
                    # Update A → B
                    basin_a = self._basins[concept_a]
                    if concept_b not in basin_a.co_occurring_concepts:
                        basin_a.co_occurring_concepts[concept_b] = 0
                    basin_a.co_occurring_concepts[concept_b] += 1

                    # Track update
                    if concept_b not in cooccurrence_updates[concept_a]:
                        cooccurrence_updates[concept_a].append(concept_b)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        return {
            "updated_basins": [b.model_dump() for b in updated_basins],
            "new_basins": [b.model_dump() for b in new_basins],
            "cooccurrence_updates": cooccurrence_updates,
            "total_strengthening_time_ms": total_time_ms,
        }

    def get_basin(self, basin_id: str) -> Dict:
        """
        Get basin by ID.

        Checks cache first (T027), falls back to in-memory storage.

        Args:
            basin_id: Basin identifier

        Returns:
            Basin info dict

        Raises:
            KeyError: If basin not found
        """
        # Try cache first (T027)
        if self._cache:
            cached_data = self._cache.get_basin(basin_id)
            if cached_data:
                return cached_data

        # Fallback to in-memory storage
        if basin_id not in self._basins:
            raise KeyError(f"Basin not found: {basin_id}")

        basin = self._basins[basin_id]

        basin_data = {
            "basin_id": basin_id,
            "basin_name": basin.basin_name,
            "strength": basin.strength,
            "activation_count": basin.activation_count,
            "co_occurring_concepts": basin.co_occurring_concepts.copy(),
            "stability": basin.stability,
            "depth": basin.depth,
            "basin_type": basin.basin_type.value,
            "activation_history": basin.activation_history.copy(),
        }

        # Cache for future lookups
        if self._cache:
            self._cache.set_basin(basin_id, basin_data)

        return basin_data

    def get_all_basins(self) -> List[Dict]:
        """Get all basins"""
        return [self.get_basin(basin_id) for basin_id in self._basins.keys()]

    def clear_basins(self) -> None:
        """Clear all basins (for testing)"""
        self._basins.clear()
