"""Redis configuration with TTL settings for ThoughtSeed pipeline."""

import redis
from typing import Optional
import logging
from .settings import settings

logger = logging.getLogger(__name__)

# TTL values from clarification session
NEURONAL_PACKET_TTL = 24 * 60 * 60  # 24 hours
ATTRACTOR_BASIN_TTL = 7 * 24 * 60 * 60  # 7 days
PROCESSING_RESULTS_TTL = 30 * 24 * 60 * 60  # 30 days

class RedisConfig:
    """Redis configuration and connection management."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        """Get Redis client connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                health_check_interval=30
            )
            # Test connection
            try:
                self._client.ping()
                logger.info("Redis connection established")
            except redis.ConnectionError as e:
                logger.error(f"Redis connection failed: {e}")
                raise
        return self._client

    def set_with_ttl(self, key: str, value: str, ttl_type: str) -> bool:
        """Set value with appropriate TTL based on data type."""
        ttl_mapping = {
            "neuronal_packet": NEURONAL_PACKET_TTL,
            "attractor_basin": ATTRACTOR_BASIN_TTL,
            "processing_results": PROCESSING_RESULTS_TTL,
        }

        ttl = ttl_mapping.get(ttl_type, PROCESSING_RESULTS_TTL)

        try:
            return self.client.setex(key, ttl, value)
        except redis.RedisError as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        try:
            return self.client.ttl(key)
        except redis.RedisError as e:
            logger.error(f"Redis TTL check failed for key {key}: {e}")
            return -1

    def cleanup_expired(self) -> int:
        """Clean up expired keys (Redis handles this automatically, but manual cleanup for monitoring)."""
        expired_count = 0
        try:
            # Get all keys with negative TTL (expired but not yet cleaned)
            for pattern in ["thoughtseed:*", "attractor:*", "results:*"]:
                keys = self.client.keys(pattern)
                for key in keys:
                    if self.client.ttl(key) == -1:  # No TTL set (shouldn't happen in our system)
                        logger.warning(f"Key {key} has no TTL set")
                    elif self.client.ttl(key) == -2:  # Key doesn't exist
                        expired_count += 1
            return expired_count
        except redis.RedisError as e:
            logger.error(f"Redis cleanup check failed: {e}")
            return 0

# Global Redis instance
redis_config = RedisConfig()

def get_redis_client() -> redis.Redis:
    """Get the global Redis client."""
    return redis_config.client