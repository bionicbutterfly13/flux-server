"""
T041a: CausalQueue for background causal processing

In-memory queue for background causal predictions with timeout fallback.
Per research.md decision 14: AsyncIO + in-memory queue (no Celery).

Background worker processes queued causal predictions without timeout constraints.
Results stored in memory dict keyed by query_hash for retrieval by PathNavigator.
"""

import asyncio
from collections import deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CausalQueueItem:
    """Item in the causal processing queue"""

    def __init__(
        self,
        query_hash: str,
        candidates: list[str],
        step_num: int,
    ):
        self.query_hash = query_hash
        self.candidates = candidates
        self.step_num = step_num


class CausalQueue:
    """
    In-memory queue for background causal reasoning.

    Per research.md decision 14:
    - In-memory deque (not Redis) for simplicity
    - Background worker processes queue every 10ms
    - Results stored in self.results[query_hash]
    - PathNavigator checks results on next step
    """

    def __init__(self, causal_reasoner=None):
        """
        Initialize CausalQueue.

        Args:
            causal_reasoner: CausalBayesianNetwork instance (injected)
        """
        self.queue: deque = deque()
        self.results: Dict[str, Dict[str, float]] = {}
        self.causal_reasoner = causal_reasoner
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def put(self, item: CausalQueueItem) -> None:
        """
        Add item to queue for background processing.

        Args:
            item: CausalQueueItem with query_hash, candidates, step_num
        """
        self.queue.append(item)
        logger.debug(
            f"Queued causal prediction: query_hash={item.query_hash[:8]}..., "
            f"candidates={len(item.candidates)}, step={item.step_num}"
        )

    async def get_result(self, query_hash: str) -> Optional[Dict[str, float]]:
        """
        Check if causal results available for query_hash.

        Args:
            query_hash: Hash from previous step

        Returns:
            Dict[candidate -> causal_score] if available, else None
        """
        return self.results.get(query_hash)

    async def start_worker(self) -> None:
        """Start background worker task"""
        if self._worker_task is None or self._worker_task.done():
            self._shutdown = False
            self._worker_task = asyncio.create_task(self.process_background())
            logger.info("CausalQueue background worker started")

    async def stop_worker(self) -> None:
        """Stop background worker task"""
        self._shutdown = True
        if self._worker_task and not self._worker_task.done():
            await self._worker_task
            logger.info("CausalQueue background worker stopped")

    async def process_background(self) -> None:
        """
        Background worker loop.

        Poll queue every 10ms, process items without timeout.
        Store results in self.results[query_hash].
        """
        logger.info("CausalQueue background processing started")

        while not self._shutdown:
            try:
                # Poll queue
                if len(self.queue) > 0:
                    item = self.queue.popleft()

                    # Process causal prediction (no timeout)
                    logger.debug(
                        f"Processing queued causal prediction: "
                        f"query_hash={item.query_hash[:8]}..."
                    )

                    if self.causal_reasoner:
                        try:
                            # Call causal reasoner (no timeout constraint)
                            scores = await self.causal_reasoner.predict(
                                item.candidates
                            )

                            # Store results
                            self.results[item.query_hash] = scores

                            logger.info(
                                f"Causal prediction complete: "
                                f"query_hash={item.query_hash[:8]}..., "
                                f"candidates={len(scores)}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Causal prediction failed: {e}", exc_info=True
                            )
                    else:
                        logger.warning(
                            "CausalBayesianNetwork not available, skipping prediction"
                        )

                else:
                    # Queue empty, sleep briefly
                    await asyncio.sleep(0.01)  # 10ms poll interval

            except Exception as e:
                logger.error(
                    f"Background worker error: {e}", exc_info=True
                )
                await asyncio.sleep(0.01)

        logger.info("CausalQueue background processing stopped")

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return len(self.queue)

    def get_results_count(self) -> int:
        """Get number of stored results"""
        return len(self.results)

    def clear_old_results(self, max_results: int = 1000) -> None:
        """
        Clear old results to prevent unbounded memory growth.

        Args:
            max_results: Maximum number of results to keep (FIFO eviction)
        """
        if len(self.results) > max_results:
            # Remove oldest entries (simple FIFO)
            excess = len(self.results) - max_results
            keys_to_remove = list(self.results.keys())[:excess]

            for key in keys_to_remove:
                del self.results[key]

            logger.info(f"Cleared {excess} old causal results")
