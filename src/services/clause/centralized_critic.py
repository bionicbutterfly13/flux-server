"""
T032: Centralized Critic with 4 Value Heads

Implements CLAUSE LC-MAPPO centralized critic for multi-agent coordination.
Per research.md decision 13: shared encoder + 4 separate value heads.
"""

import torch
import torch.nn as nn
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class CLAUSECentralizedCritic(nn.Module):
    """
    T032: Centralized critic for LC-MAPPO coordination.

    Architecture (research.md decision 13):
    - Shared encoder: 2-layer MLP (state_dim → 256 → 256)
    - 4 value heads: Architect, Navigator, Curator, Coordinator
    - Each head: Linear(256 → 1)

    Used for CTDE (centralized training, decentralized execution):
    - Training: All agents use centralized critic for value estimation
    - Execution: Agents act independently with decentralized policies
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize centralized critic.

        Args:
            state_dim: Dimension of global state encoding (default 1154 from research.md)
            hidden_dim: Hidden layer dimension (default 256)
        """
        super().__init__()

        # Shared encoder: learns common global state representation
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 4 separate value heads (one per agent)
        self.architect_head = nn.Linear(hidden_dim, 1)
        self.navigator_head = nn.Linear(hidden_dim, 1)
        self.curator_head = nn.Linear(hidden_dim, 1)
        self.coordinator_head = nn.Linear(hidden_dim, 1)

        logger.info(
            f"CLAUSECentralizedCritic initialized: state_dim={state_dim}, hidden_dim={hidden_dim}"
        )

    def forward(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute value estimates for all 4 agents.

        Args:
            global_state: Global state tensor (batch_size, state_dim)

        Returns:
            Dict with value estimates for each agent:
            {
                "architect": (batch_size, 1),
                "navigator": (batch_size, 1),
                "curator": (batch_size, 1),
                "coordinator": (batch_size, 1)
            }
        """
        # Shared encoding
        features = self.encoder(global_state)

        # Compute values for each agent
        return {
            "architect": self.architect_head(features),
            "navigator": self.navigator_head(features),
            "curator": self.curator_head(features),
            "coordinator": self.coordinator_head(features),
        }


def compute_shaped_return(
    reward: float,
    cost: float,
    lambda_constraint: float,
    budget: float,
) -> float:
    """
    Compute shaped return for LC-MAPPO (research.md decision 13).

    Shaped return = reward - λ × constraint_violation

    Args:
        reward: Task accuracy reward (e.g., query answering performance)
        cost: Resource cost (e.g., edges traversed, tokens used, latency)
        lambda_constraint: Lagrangian dual variable (weight for constraint)
        budget: Budget limit for this resource

    Returns:
        Shaped return (can be negative if constraint violated)
    """
    # Constraint violation: max(0, cost - budget)
    constraint_violation = max(0.0, cost - budget)

    # Shaped return
    return reward - lambda_constraint * constraint_violation
