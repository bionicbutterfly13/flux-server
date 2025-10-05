"""
T049a: Conflict Monitor with 5% Threshold

Monitors transaction conflict rate over 1-minute sliding window.
Triggers read-only mode when conflict rate exceeds 5% (research.md decision 15).

Industry Context:
- Google Spanner: <1% conflict rate
- CockroachDB: 5-10% acceptable
- Neo4j: <5% recommended
- CLAUSE (3 agents, ~10 queries/sec): 5% = 0.5 conflicts/sec (manageable)
"""

import logging
import time
from collections import deque
from typing import Tuple

logger = logging.getLogger(__name__)


class ConflictMonitor:
    """
    Conflict rate monitor for CLAUSE multi-agent system.

    Per research.md decision 15:
    - Sliding window: 60 seconds
    - Threshold: 5% conflict rate
    - Action: Switch to read-only mode if exceeded
    - Rationale: Balance responsiveness with stability
    """

    def __init__(self, window_seconds: int = 60, threshold: float = 0.05):
        """
        Initialize conflict monitor.

        Args:
            window_seconds: Sliding window duration (default 60)
            threshold: Conflict rate threshold (default 0.05 = 5%)
        """
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.attempts = deque()  # (timestamp, success: bool)

        logger.info(
            f"ConflictMonitor initialized: window={window_seconds}s, threshold={threshold:.1%}"
        )

    def record_transaction(self, success: bool) -> None:
        """
        Record transaction attempt.

        Args:
            success: True if transaction succeeded, False if conflict
        """
        now = time.time()
        self.attempts.append((now, success))

        # Prune old attempts outside sliding window
        self._prune_old_attempts()

        # Log conflicts
        if not success:
            conflict_rate = self.get_conflict_rate()
            logger.warning(
                f"Transaction conflict recorded (rate now {conflict_rate:.1%})"
            )

    def get_conflict_rate(self) -> float:
        """
        Calculate conflict rate over sliding window.

        Returns:
            Conflict rate (0.0-1.0)
        """
        self._prune_old_attempts()

        if not self.attempts:
            return 0.0

        # Count conflicts (failed transactions)
        conflicts = sum(1 for _, success in self.attempts if not success)
        total = len(self.attempts)

        return conflicts / total

    def should_switch_to_readonly(self) -> bool:
        """
        Check if conflict rate exceeds threshold.

        Returns:
            True if should switch to read-only mode
        """
        conflict_rate = self.get_conflict_rate()
        exceeds_threshold = conflict_rate > self.threshold

        if exceeds_threshold:
            logger.error(
                f"Conflict rate {conflict_rate:.1%} exceeds threshold {self.threshold:.1%} - "
                f"recommend read-only mode"
            )

        return exceeds_threshold

    def get_stats(self) -> Tuple[float, int, int, int]:
        """
        Get monitoring statistics.

        Returns:
            (conflict_rate, total_attempts, conflicts, window_seconds)
        """
        self._prune_old_attempts()

        total = len(self.attempts)
        conflicts = sum(1 for _, success in self.attempts if not success)
        conflict_rate = conflicts / total if total > 0 else 0.0

        return (conflict_rate, total, conflicts, self.window_seconds)

    def _prune_old_attempts(self) -> None:
        """Remove attempts outside sliding window"""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove old attempts from left (oldest first)
        while self.attempts and self.attempts[0][0] < cutoff:
            self.attempts.popleft()

    def reset(self) -> None:
        """Reset monitor (clear all attempts)"""
        self.attempts.clear()
        logger.info("ConflictMonitor reset")


# Example usage for testing
if __name__ == "__main__":
    # Simulate CLAUSE system at 10 queries/sec
    monitor = ConflictMonitor(window_seconds=60, threshold=0.05)

    # Simulate 100 transactions (95 success, 5 conflicts)
    import random

    for i in range(100):
        success = random.random() > 0.05  # 5% conflict rate
        monitor.record_transaction(success)

        if i % 20 == 0:
            rate, total, conflicts, _ = monitor.get_stats()
            print(
                f"After {i} transactions: rate={rate:.1%}, conflicts={conflicts}/{total}"
            )

    # Check if should switch to read-only
    if monitor.should_switch_to_readonly():
        print("WARNING: Conflict rate exceeded 5% - switching to read-only mode")
    else:
        print("Conflict rate within acceptable range")
