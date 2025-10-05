"""
Query Engine - Orchestrates complete query processing pipeline
Per Spec 006: Natural language query → Search → Synthesis → Response

Updated 2025-10-01: Removed Qdrant, using Neo4j unified search only
Neo4j provides: graph relationships + vector similarity + full-text search

Per Constitution Article I, Section 1.4:
- Use direct imports (not imports with 'src.' prefix)
"""

from typing import Optional, Dict, Any
import asyncio
import time
import logging
from datetime import datetime

from src.models.query import Query
from src.models.response import QueryResponse
from src.services.neo4j_searcher import Neo4jSearcher
from src.services.response_synthesizer import ResponseSynthesizer

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Main query processing engine.

    Uses Neo4j unified search: graph + vector + full-text in one database.
    AutoSchemaKG integration for automatic knowledge graph construction.

    Performance target: <2s per query (per Spec 006)
    """

    def __init__(
        self,
        neo4j_searcher: Optional[Neo4jSearcher] = None,
        response_synthesizer: Optional[ResponseSynthesizer] = None
    ):
        """Initialize query engine with Neo4j searcher."""
        self.neo4j_searcher = neo4j_searcher or Neo4jSearcher()
        self.response_synthesizer = response_synthesizer or ResponseSynthesizer()
        self.default_result_limit = 10

    async def process_query(
        self,
        question: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        thoughtseed_id: Optional[str] = None
    ) -> QueryResponse:
        """
        Process natural language query end-to-end.

        Args:
            question: User's natural language question
            user_id: Optional user identifier
            context: Optional session context for follow-up questions
            thoughtseed_id: Optional ThoughtSeed ID for consciousness tracking

        Returns:
            Complete QueryResponse with synthesized answer and sources
        """
        start_time = time.time()

        try:
            # Create Query object
            query = Query(
                question=question,
                user_id=user_id,
                context=context or {},
                thoughtseed_id=thoughtseed_id,
                timestamp=datetime.now()
            )

            logger.info(f"Processing query {query.query_id}: {question[:50]}...")

            # Neo4j unified search (graph + vector + full-text)
            neo4j_results = await self.neo4j_searcher.search(query.question, self.default_result_limit)

            logger.info(f"Search complete: {len(neo4j_results)} results from Neo4j")

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Synthesize final response
            response = await self.response_synthesizer.synthesize(
                query=query,
                neo4j_results=neo4j_results,
                qdrant_results=[],  # No longer using Qdrant
                processing_time_ms=processing_time_ms
            )

            logger.info(
                f"Query {query.query_id} completed in {processing_time_ms}ms "
                f"(confidence: {response.confidence:.2f})"
            )

            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Return error response
            return QueryResponse(
                query_id="error",
                answer="I encountered an error processing your query. Please try again.",
                sources=[],
                confidence=0.0,
                processing_time_ms=processing_time_ms
            )

    async def process_batch_queries(self, questions: list[str]) -> list[QueryResponse]:
        """
        Process multiple queries efficiently.

        Args:
            questions: List of natural language questions

        Returns:
            List of QueryResponse objects
        """
        tasks = [self.process_query(q) for q in questions]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in batch
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch query {i} failed: {response}")
                results.append(QueryResponse(
                    query_id=f"batch-error-{i}",
                    answer="Query processing failed",
                    sources=[],
                    confidence=0.0,
                    processing_time_ms=0
                ))
            else:
                results.append(response)

        return results

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of query engine components.

        Returns:
            Status of Neo4j and overall system
        """
        health = {
            "neo4j": False,
            "overall": False
        }

        try:
            # Test Neo4j connection
            neo4j_test = await self.neo4j_searcher.search("test", limit=1)
            health["neo4j"] = True
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")

        health["overall"] = health["neo4j"]

        return health
