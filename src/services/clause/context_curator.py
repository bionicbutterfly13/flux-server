"""
T026-T031: Context Curator Service

ContextCurator agent for CLAUSE Phase 2 multi-agent system.
Implements listwise evidence selection with token budgets and provenance tracking.

Per Spec 035:
- Budget enforcement: β_tok (100-8192 tokens)
- Listwise scoring: Pairwise similarity + diversity penalty (research.md Decision 6)
- Token counting: tiktoken with 10% safety buffer (research.md Decision 7)
- Provenance tracking: 7 fields + 3 trust signals (Spec 032)
- Shaped utility: score - λ_tok × tokens (CLAUSE paper)
- Latency target: <100ms per curation
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import tiktoken

from src.models.clause.curator_models import (
    ContextCurationRequest,
    SelectedEvidence,
    ContextCurationResponse,
)
from src.models.clause.provenance_models import (
    ProvenanceMetadata,
    TrustSignals,
)

logger = logging.getLogger(__name__)


class ContextCurator:
    """
    ContextCurator agent - Phase 2 of CLAUSE 3-agent system.

    Implements listwise evidence selection with:
    - Pairwise redundancy detection
    - Token budget enforcement (tiktoken)
    - Learned stopping (shaped utility threshold)
    - Provenance tracking (Spec 032)
    """

    def __init__(
        self,
        embedding_service=None,
        provenance_tracker=None,
        neo4j_client=None,
    ):
        """
        Initialize ContextCurator with required dependencies.

        Args:
            embedding_service: Sentence transformer embedding service
            provenance_tracker: Provenance tracking service (Spec 032)
            neo4j_client: Neo4j client for evidence retrieval
        """
        self.embedder = embedding_service
        self.provenance_tracker = provenance_tracker
        self.neo4j = neo4j_client

        # T028: Initialize tiktoken encoder (GPT-4)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.safety_buffer = 0.1  # 10% safety buffer

        logger.info("ContextCurator initialized with tiktoken GPT-4 encoder")

    async def curate(
        self, request: ContextCurationRequest
    ) -> ContextCurationResponse:
        """
        Execute listwise evidence curation with provenance.

        Per research.md Decision 6-8:
        - Listwise scoring with pairwise similarity matrix
        - Diversity penalty to avoid redundancy
        - Token budget enforcement with tiktoken
        - Provenance metadata for trust signals

        Args:
            request: ContextCurationRequest with evidence pool, token budget

        Returns:
            ContextCurationResponse with selected evidence, metadata, performance
        """
        start_time = time.time()
        perf_tracker = {"provenance_overhead_ms": 0.0}

        # T027: Compute pairwise similarity matrix
        similarity_matrix = await self._compute_similarity_matrix(
            request.evidence_pool
        )

        # T031: Greedy selection with shaped utility
        selected_evidence = []
        selected_indices = []
        tokens_used = 0
        learned_stop_triggered = False

        # Adjusted token budget with safety buffer
        effective_budget = int(request.token_budget * (1 - self.safety_buffer))

        for i, evidence_text in enumerate(request.evidence_pool):
            # T028: Count tokens with tiktoken
            evidence_tokens = self._count_tokens(evidence_text)

            # Check token budget
            if tokens_used + evidence_tokens > effective_budget:
                logger.debug(f"Token budget exceeded at evidence {i}")
                break

            # T027: Compute relevance score
            relevance_score = await self._compute_relevance(evidence_text)

            # T027: Compute diversity penalty (similarity to already selected)
            diversity_penalty = 0.0
            for selected_idx in selected_indices:
                diversity_penalty += similarity_matrix[i, selected_idx]

            # Average penalty (avoid scaling with num selected)
            if selected_indices:
                diversity_penalty /= len(selected_indices)

            # Final score with diversity penalty
            final_score = relevance_score - 0.3 * diversity_penalty

            # T029: Shaped utility (score - cost)
            token_cost = request.lambda_tok * evidence_tokens
            shaped_utility = final_score - token_cost

            # T029: Learned stopping (stop when utility < 0)
            if shaped_utility < 0:
                logger.debug(f"Learned stop triggered at evidence {i} (utility={shaped_utility:.3f})")
                learned_stop_triggered = True
                break

            # T030: Add provenance metadata
            provenance = None
            if request.enable_provenance:
                prov_start = time.time()
                provenance = await self._generate_provenance(
                    evidence_text=evidence_text,
                    source_idx=i,
                )
                perf_tracker["provenance_overhead_ms"] += (time.time() - prov_start) * 1000

            # Add to selected evidence
            selected_evidence.append(
                SelectedEvidence(
                    text=evidence_text,
                    tokens=evidence_tokens,
                    score=final_score,
                    shaped_utility=shaped_utility,
                    provenance=provenance,
                )
            )
            selected_indices.append(i)
            tokens_used += evidence_tokens

        # Build response
        total_latency = (time.time() - start_time) * 1000

        return ContextCurationResponse(
            selected_evidence=selected_evidence,
            metadata={
                "tokens_used": tokens_used,
                "tokens_total": request.token_budget,
                "learned_stop_triggered": learned_stop_triggered,
                "evidence_pool_size": len(request.evidence_pool),
                "evidence_selected": len(selected_evidence),
            },
            performance={
                "latency_ms": total_latency,
                "provenance_overhead_ms": perf_tracker["provenance_overhead_ms"],
            },
        )

    # T027: Listwise Scoring
    async def _compute_similarity_matrix(
        self, evidence_pool: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix per research.md Decision 6.

        Creates n×n matrix where entry (i,j) is cosine similarity between
        evidence i and evidence j.

        Args:
            evidence_pool: List of evidence text snippets

        Returns:
            Symmetric similarity matrix (n×n)
        """
        n = len(evidence_pool)
        similarity_matrix = np.zeros((n, n))

        # Embed all evidence
        embeddings = []
        for evidence in evidence_pool:
            emb = await self._embed_text(evidence)
            embeddings.append(emb)

        # Compute pairwise cosine similarities
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric

        return similarity_matrix

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0

    # T028: Token Budget Enforcement
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken per research.md Decision 7.

        Uses GPT-4 tokenizer for accurate counting.

        Args:
            text: Text to tokenize

        Returns:
            Token count
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    # T030: Provenance Metadata
    async def _generate_provenance(
        self, evidence_text: str, source_idx: int
    ) -> ProvenanceMetadata:
        """
        Generate provenance metadata per Spec 032.

        Includes 7 required fields + 3 trust signals.

        Args:
            evidence_text: Evidence text
            source_idx: Index in evidence pool

        Returns:
            ProvenanceMetadata with full tracking
        """
        # Compute trust signals
        trust_signals = TrustSignals(
            reputation_score=0.85,  # Placeholder - will be computed from source
            recency_score=0.90,  # Placeholder - will be computed from timestamp
            semantic_consistency=0.88,  # Placeholder - will be computed from query
        )

        # Supporting evidence (truncated to 200 chars)
        supporting_evidence = evidence_text[:200]

        provenance = ProvenanceMetadata(
            source_uri=f"evidence_pool[{source_idx}]",  # Placeholder
            extraction_timestamp=datetime.now(),
            extractor_identity="ContextCurator-v2.0",
            supporting_evidence=supporting_evidence,
            verification_status="verified",  # Placeholder
            corroboration_count=1,  # Placeholder
            trust_signals=trust_signals,
        )

        return provenance

    # Helper methods (stubs - will be implemented in integration phase)
    async def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using sentence transformer (384-dim)"""
        if self.embedder:
            return await self.embedder.embed(text)
        return np.random.randn(384)  # Placeholder

    async def _compute_relevance(self, evidence_text: str) -> float:
        """Compute query relevance score (query not passed in current stub)"""
        # In real implementation, would use query from request context
        return np.random.uniform(0.5, 1.0)  # Placeholder
