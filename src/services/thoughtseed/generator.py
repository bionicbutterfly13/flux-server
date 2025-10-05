"""
T038: ThoughtSeed Generator Service

Implements ThoughtSeed generation per Spec 028.
Enables cross-document linking with basin context from Phase 1.

Key Features:
- Bulk generation during path exploration
- Basin context integration (strength, activation, co-occurrence)
- Similarity-based linking (threshold = 0.8)
- Redis caching for fast retrieval
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from src.models.clause.thoughtseed_models import ThoughtSeed, BasinContext

logger = logging.getLogger(__name__)


class ThoughtSeedGenerator:
    """
    ThoughtSeed generator for cross-document linking.

    Per Spec 028:
    - Generate ThoughtSeeds during exploration (not after)
    - Include basin context for similarity matching
    - Cache in Redis for fast lookup
    - Link across documents with similarity > threshold
    """

    def __init__(self, redis_client=None, neo4j_client=None, embedding_service=None):
        """
        Initialize ThoughtSeed generator.

        Args:
            redis_client: Redis client for caching
            neo4j_client: Neo4j client for persistence
            embedding_service: Embedding service for similarity matching
        """
        self.redis = redis_client
        self.neo4j = neo4j_client
        self.embedder = embedding_service

        logger.info("ThoughtSeed generator initialized")

    async def create(
        self,
        concept: str,
        source_doc: str,
        basin_context: Dict[str, Any],
        similarity_threshold: float = 0.8,
    ) -> str:
        """
        Create ThoughtSeed with basin context.

        Args:
            concept: Concept node ID
            source_doc: Source document/query
            basin_context: Basin context from AttractorBasin tracker
            similarity_threshold: Similarity threshold for linking

        Returns:
            ThoughtSeed ID
        """
        # Generate unique ID
        ts_id = f"ts_{concept}_{uuid.uuid4().hex[:8]}"

        # Create BasinContext model
        basin = BasinContext(
            strength=basin_context.get("strength", 1.0),
            activation_count=basin_context.get("activation_count", 0),
            co_occurring=basin_context.get("co_occurring", {}),
        )

        # Create ThoughtSeed
        thoughtseed = ThoughtSeed(
            id=ts_id,
            concept=concept,
            source_doc=source_doc,
            basin_context=basin,
            similarity_threshold=similarity_threshold,
            linked_documents=[],
            created_at=datetime.now(),
        )

        # Link to similar documents
        linked_docs = await self._find_similar_documents(
            concept=concept, threshold=similarity_threshold
        )
        thoughtseed.linked_documents = linked_docs

        # Persist to Neo4j
        if self.neo4j:
            await self._persist_to_neo4j(thoughtseed)

        # Cache in Redis (TTL = 1 hour)
        if self.redis:
            await self._cache_in_redis(thoughtseed, ttl=3600)

        logger.debug(f"Created ThoughtSeed {ts_id} with {len(linked_docs)} links")
        return ts_id

    async def get(self, thoughtseed_id: str) -> Optional[ThoughtSeed]:
        """
        Retrieve ThoughtSeed by ID.

        Args:
            thoughtseed_id: ThoughtSeed ID

        Returns:
            ThoughtSeed or None if not found
        """
        # Try Redis cache first
        if self.redis:
            cached = await self._get_from_redis(thoughtseed_id)
            if cached:
                return cached

        # Fallback to Neo4j
        if self.neo4j:
            return await self._get_from_neo4j(thoughtseed_id)

        return None

    async def _find_similar_documents(
        self, concept: str, threshold: float
    ) -> List[str]:
        """
        Find similar documents using embedding similarity.

        Args:
            concept: Concept to match
            threshold: Similarity threshold

        Returns:
            List of document IDs
        """
        # Placeholder - will use embedding service
        return []

    async def _persist_to_neo4j(self, thoughtseed: ThoughtSeed) -> None:
        """Persist ThoughtSeed to Neo4j"""
        if not self.neo4j:
            return

        # Cypher query to create ThoughtSeed node
        query = """
        MERGE (ts:ThoughtSeed {id: $id})
        SET ts.concept = $concept,
            ts.source_doc = $source_doc,
            ts.basin_context = $basin_context,
            ts.similarity_threshold = $similarity_threshold,
            ts.linked_documents = $linked_documents,
            ts.created_at = datetime($created_at)
        """

        await self.neo4j.execute(
            query,
            parameters={
                "id": thoughtseed.id,
                "concept": thoughtseed.concept,
                "source_doc": thoughtseed.source_doc,
                "basin_context": thoughtseed.basin_context.model_dump(),
                "similarity_threshold": thoughtseed.similarity_threshold,
                "linked_documents": thoughtseed.linked_documents,
                "created_at": thoughtseed.created_at.isoformat(),
            },
        )

    async def _cache_in_redis(self, thoughtseed: ThoughtSeed, ttl: int) -> None:
        """Cache ThoughtSeed in Redis"""
        if not self.redis:
            return

        key = f"thoughtseed:{thoughtseed.id}"
        value = thoughtseed.model_dump_json()
        await self.redis.setex(key, ttl, value)

    async def _get_from_redis(self, thoughtseed_id: str) -> Optional[ThoughtSeed]:
        """Get ThoughtSeed from Redis cache"""
        if not self.redis:
            return None

        key = f"thoughtseed:{thoughtseed_id}"
        value = await self.redis.get(key)

        if value:
            return ThoughtSeed.model_validate_json(value)

        return None

    async def _get_from_neo4j(self, thoughtseed_id: str) -> Optional[ThoughtSeed]:
        """Get ThoughtSeed from Neo4j"""
        if not self.neo4j:
            return None

        # Placeholder - would query Neo4j
        return None
