"""
CLAUSE Basin Cache - Redis caching layer for basin data

Implements T027: Redis basin caching with 1-hour TTL
- Cache basin data for fast lookups (<1ms target)
- Automatic invalidation on updates
- Batch loading on startup
"""

import json
import logging
from typing import Dict, List, Optional, Any
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class BasinCache:
    """
    Redis-based cache for AttractorBasin data.

    Provides <1ms basin lookups with 1-hour TTL and automatic
    invalidation on updates.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        """
        Initialize basin cache with Redis connection.

        Args:
            redis_url: Redis connection URL
            ttl_seconds: Time-to-live for cached basins (default 3600 = 1 hour)
        """
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._client: Optional[redis.Redis] = None
        self._connected = False

    @property
    def client(self) -> redis.Redis:
        """Lazy-load Redis client."""
        if self._client is None:
            try:
                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                # Test connection
                self._client.ping()
                self._connected = True
                logger.info(f"Redis basin cache connected: {self.redis_url}")
            except RedisError as e:
                logger.warning(f"Redis connection failed: {e}. Cache disabled.")
                self._connected = False
                # Return dummy client for graceful degradation
                raise

        return self._client

    def get_basin(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basin data from cache.

        Args:
            concept_id: Concept identifier

        Returns:
            Basin data dict if cached, None if not found or cache unavailable
        """
        if not self._connected:
            return None

        try:
            key = self._basin_key(concept_id)
            cached_data = self.client.get(key)

            if cached_data:
                logger.debug(f"Basin cache HIT: {concept_id}")
                return json.loads(cached_data)
            else:
                logger.debug(f"Basin cache MISS: {concept_id}")
                return None

        except RedisError as e:
            logger.warning(f"Redis get failed for {concept_id}: {e}")
            return None

    def set_basin(
        self,
        concept_id: str,
        basin_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set basin data in cache with TTL.

        Args:
            concept_id: Concept identifier
            basin_data: Basin data dictionary (serializable)
            ttl: Optional TTL override (default: self.ttl_seconds)

        Returns:
            True if cached successfully, False otherwise
        """
        if not self._connected:
            return False

        try:
            key = self._basin_key(concept_id)
            ttl_value = ttl or self.ttl_seconds

            # Serialize basin data
            serialized = json.dumps(basin_data)

            # Set with TTL
            self.client.setex(key, ttl_value, serialized)
            logger.debug(f"Basin cached: {concept_id} (TTL={ttl_value}s)")
            return True

        except (RedisError, TypeError, ValueError) as e:
            logger.warning(f"Redis set failed for {concept_id}: {e}")
            return False

    def invalidate_basin(self, concept_id: str) -> bool:
        """
        Invalidate (delete) basin from cache.

        Called when basin is updated to ensure consistency.

        Args:
            concept_id: Concept identifier

        Returns:
            True if invalidated, False if failed or not found
        """
        if not self._connected:
            return False

        try:
            key = self._basin_key(concept_id)
            deleted = self.client.delete(key)
            logger.debug(f"Basin invalidated: {concept_id} (deleted={deleted})")
            return bool(deleted)

        except RedisError as e:
            logger.warning(f"Redis delete failed for {concept_id}: {e}")
            return False

    def batch_set_basins(
        self,
        basins: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> int:
        """
        Batch load basins into cache.

        Used on startup to pre-populate cache with frequently accessed basins.

        Args:
            basins: List of basin data dicts (must include 'concept_id' key)
            ttl: Optional TTL override (default: self.ttl_seconds)

        Returns:
            Number of basins successfully cached
        """
        if not self._connected:
            return 0

        cached_count = 0
        ttl_value = ttl or self.ttl_seconds

        try:
            # Use pipeline for atomic batch operation
            pipe = self.client.pipeline()

            for basin_data in basins:
                concept_id = basin_data.get("concept_id")
                if not concept_id:
                    logger.warning("Basin missing concept_id, skipping")
                    continue

                key = self._basin_key(concept_id)
                serialized = json.dumps(basin_data)
                pipe.setex(key, ttl_value, serialized)
                cached_count += 1

            # Execute all at once
            pipe.execute()
            logger.info(
                f"Batch cached {cached_count} basins (TTL={ttl_value}s)"
            )
            return cached_count

        except (RedisError, TypeError, ValueError) as e:
            logger.error(f"Batch cache failed: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (connected, basin_count, memory_usage, etc.)
        """
        if not self._connected:
            return {
                "connected": False,
                "basin_count": 0,
                "memory_usage_mb": 0,
            }

        try:
            # Count basin keys
            basin_keys = self.client.keys("basin:*")
            basin_count = len(basin_keys)

            # Get memory usage
            info = self.client.info("memory")
            memory_mb = info.get("used_memory", 0) / (1024 * 1024)

            return {
                "connected": True,
                "basin_count": basin_count,
                "memory_usage_mb": round(memory_mb, 2),
                "ttl_seconds": self.ttl_seconds,
                "redis_url": self.redis_url,
            }

        except RedisError as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {
                "connected": False,
                "error": str(e),
            }

    def clear_all_basins(self) -> int:
        """
        Clear all basin data from cache.

        WARNING: This deletes all cached basins. Use with caution.

        Returns:
            Number of basins cleared
        """
        if not self._connected:
            return 0

        try:
            basin_keys = self.client.keys("basin:*")
            if basin_keys:
                deleted = self.client.delete(*basin_keys)
                logger.info(f"Cleared {deleted} basins from cache")
                return deleted
            return 0

        except RedisError as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    def _basin_key(self, concept_id: str) -> str:
        """Generate Redis key for basin."""
        return f"basin:{concept_id}"

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            try:
                self._client.close()
                logger.info("Redis basin cache connection closed")
            except RedisError as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self._client = None
                self._connected = False
