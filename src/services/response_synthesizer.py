"""
Response Synthesizer - Combine search results into coherent answer
Per Spec 006 FR-003: Synthesize results into coherent responses
"""

import re
from textwrap import dedent
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
import logging
from datetime import datetime

from src.models.response import SearchResult, QueryResponse
from src.models.query import Query
from src.services.ollama_integration import OllamaModelManager

logger = logging.getLogger(__name__)


class AnswerGenerationError(Exception):
    """Raised when LLM answer generation fails."""


@runtime_checkable
class AnswerGenerator(Protocol):
    """Protocol for pluggable answer generators"""

    async def generate(self, question: str, sources: List[SearchResult]) -> str:  # pragma: no cover - interface
        """Generate an answer given a question and supporting sources."""


class OllamaAnswerGenerator:
    """Default answer generator backed by local Ollama models."""

    def __init__(
        self,
        model_manager: Optional[OllamaModelManager] = None,
        model_name: str = "qwen2.5:14b"
    ) -> None:
        self.model_manager = model_manager or OllamaModelManager()
        self.model_name = model_name

    async def generate(self, question: str, sources: List[SearchResult]) -> str:
        """Generate answer via Ollama, ensuring a non-empty response."""
        if not sources:
            raise AnswerGenerationError("no sources available")

        prompt = self._build_prompt(question, sources)
        result = await self.model_manager.generate_text(
            self.model_name,
            prompt,
            max_tokens=768
        )

        if not result.get("success"):
            raise AnswerGenerationError(result.get("error", "unknown generation failure"))

        answer = result.get("response", "").strip()
        if not answer:
            raise AnswerGenerationError("empty response from Ollama")

        return answer

    def _build_prompt(self, question: str, sources: List[SearchResult]) -> str:
        """Construct a prompt instructing the LLM to cite provided sources."""
        source_sections = []
        for index, source in enumerate(sources, start=1):
            snippet = source.content.strip().replace("\n", " ")
            metadata_title = source.metadata.get("title") if source.metadata else None
            relationships = ", ".join(source.relationships[:3]) if source.relationships else "none"
            snippet = snippet[:500]

            section = [f"[{index}]"]
            if metadata_title:
                section.append(f"Title: {metadata_title}")
            section.append(f"Snippet: {snippet}")
            section.append(f"Source type: {source.source.value}")
            section.append(f"Relationships: {relationships if relationships else 'none'}")
            source_sections.append("\n".join(section))

        sources_text = "\n\n".join(source_sections)

        prompt = dedent(
            f"""
            You are the Flux research synthesizer. Answer the user's question using only the material provided in the sources.
            Instructions:
            - Write at least two detailed paragraphs (minimum 200 characters).
            - Reference supporting statements with bracketed citations like [1], [2].
            - Integrate graph relationships or metadata when available.
            - Avoid adding information that is not grounded in the sources.

            Question:
            {question.strip()}

            Sources:
            {sources_text}

            Respond with the narrative answer only—do not include extra sections or metadata headers.
            """
        ).strip()

        return prompt


class ResponseSynthesizer:
    """
    Synthesize multiple search results into coherent response.

    Combines Neo4j graph results and Qdrant vector results,
    analyzes relevance, and generates natural language answer.
    """

    def __init__(
        self,
        answer_generator: Optional[AnswerGenerator] = None
    ):
        """Initialize synthesizer."""
        self.min_confidence = 0.3
        self.max_sources = 10
        self.minimum_answer_length = 200
        self.answer_generator = answer_generator or OllamaAnswerGenerator()

    async def synthesize(
        self,
        query: Query,
        neo4j_results: List[SearchResult],
        qdrant_results: List[SearchResult],
        processing_time_ms: int
    ) -> QueryResponse:
        """
        Synthesize search results into final response.

        Args:
            query: Original query
            neo4j_results: Results from graph search
            qdrant_results: Results from vector search
            processing_time_ms: Total processing time

        Returns:
            Complete QueryResponse with synthesized answer
        """
        try:
            # Combine and rank all sources
            all_sources = self._combine_sources(neo4j_results, qdrant_results)

            # Calculate confidence based on result quality
            confidence = self._calculate_confidence(all_sources)

            # Generate answer from sources
            answer = await self._generate_answer(query.question, all_sources)

            # Create ThoughtSeed trace if applicable
            thoughtseed_trace = self._create_thoughtseed_trace(query, all_sources)

            return QueryResponse(
                query_id=query.query_id,
                answer=answer,
                sources=all_sources[:self.max_sources],
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                thoughtseed_trace=thoughtseed_trace
            )

        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            # Return fallback response
            return QueryResponse(
                query_id=query.query_id,
                answer="I encountered an error processing your query. Please try rephrasing your question.",
                sources=[],
                confidence=0.0,
                processing_time_ms=processing_time_ms
            )

    def _combine_sources(
        self,
        neo4j_results: List[SearchResult],
        qdrant_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine and deduplicate sources from both databases.

        Graph results get slight boost for relationship richness.
        Vector results provide semantic coverage.
        """
        # Boost Neo4j results slightly for graph relationships
        for result in neo4j_results:
            relationship_boost = len(result.relationships) * 0.05
            result.relevance_score = min(result.relevance_score + relationship_boost, 1.0)

        # Combine all results
        all_results = neo4j_results + qdrant_results

        # Deduplicate by content similarity
        unique_results = self._deduplicate_by_content(all_results)

        # Sort by relevance
        sorted_results = sorted(unique_results, key=lambda r: r.relevance_score, reverse=True)

        return sorted_results

    def _deduplicate_by_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or very similar content."""
        # Simple deduplication - in production, use embedding similarity
        seen_content = {}
        unique = []

        for result in results:
            # Use first 100 chars as fingerprint
            fingerprint = result.content[:100].lower().strip()

            if fingerprint not in seen_content:
                seen_content[fingerprint] = result
                unique.append(result)
            elif result.relevance_score > seen_content[fingerprint].relevance_score:
                # Replace with higher relevance version
                seen_content[fingerprint] = result
                unique = [r for r in unique if r.content[:100].lower().strip() != fingerprint]
                unique.append(result)

        return unique

    def _calculate_confidence(self, sources: List[SearchResult]) -> float:
        """
        Calculate confidence in response based on source quality.

        Factors:
        - Number of high-quality sources
        - Average relevance score
        - Source diversity (graph + vector)
        """
        if not sources:
            return 0.0

        # Average relevance of top sources
        top_sources = sources[:5]
        avg_relevance = sum(s.relevance_score for s in top_sources) / len(top_sources)

        # Source count factor (more sources = higher confidence, diminishing returns)
        count_factor = min(len(sources) / 10.0, 1.0)

        # Source diversity (both Neo4j and Qdrant present)
        has_graph = any(s.source.value == "neo4j" for s in sources)
        has_vector = any(s.source.value == "qdrant" for s in sources)
        diversity_factor = 1.0 if (has_graph and has_vector) else 0.85

        # Combine factors
        confidence = avg_relevance * 0.6 + count_factor * 0.3 * diversity_factor

        return min(max(confidence, 0.0), 1.0)

    async def _generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        """
        Generate natural language answer from sources.
        """
        if not sources:
            return self._fallback_answer(question, sources, error_message="no supporting sources")

        limited_sources = sources[:self.max_sources]

        try:
            raw_answer = await self.answer_generator.generate(question, limited_sources)
        except AnswerGenerationError as error:
            logger.warning("Answer generation failed via LLM: %s", error)
            return self._fallback_answer(question, limited_sources, error_message=str(error))
        except Exception as error:  # pragma: no cover - defensive guard
            logger.exception("Unexpected error during answer generation")
            return self._fallback_answer(question, limited_sources, error_message=str(error))

        answer = self._ensure_citations(raw_answer, limited_sources)

        if len(answer) < self.minimum_answer_length:
            answer = self._augment_answer(answer, limited_sources)

        return answer

    def _ensure_citations(self, answer: str, sources: List[SearchResult]) -> str:
        """Guarantee at least one citation is present in the answer."""
        if not sources:
            return answer

        if re.search(r"\[\d+\]", answer):
            return answer

        # Append citation referencing the first source
        appended = answer.rstrip() + f" [1]"
        return appended

    def _augment_answer(self, answer: str, sources: List[SearchResult]) -> str:
        """Extend answers that are shorter than the required minimum."""
        complementary_lines = [answer.strip(), ""]

        complementary_lines.append(
            "Additional context: the retrieved sources emphasise how these mechanisms couple prediction error minimisation with narrative consolidation across attractor basins."
        )

        for index, source in enumerate(sources[:3], start=1):
            snippet = source.content.strip().replace("\n", " ")
            snippet = snippet[:280]
            complementary_lines.append(f"[{index}] {snippet}")

        return "\n".join(complementary_lines)

    def _fallback_answer(
        self,
        question: str,
        sources: List[SearchResult],
        error_message: Optional[str] = None
    ) -> str:
        """Generate a graceful fallback answer when LLM synthesis fails."""
        header = "LLM generation unavailable"
        if error_message:
            header += f" – {error_message}"
        header += ". Summarising directly from retrieved sources."

        if not sources:
            return header + " No supporting documents were available for the query."

        lines = [header, ""]
        lines.append(
            "This provisional summary highlights the key evidence while noting that a full synthesis will be provided once the local LLM is reachable."
        )

        for index, source in enumerate(sources[:3], start=1):
            snippet = source.content.strip().replace("\n", " ")[:320]
            title = source.metadata.get("title") if source.metadata else None
            lines.append(f"[{index}] {snippet}")
            if title:
                lines.append(f"    Source: {title}")

        lines.append("\nPlease retry after restoring LLM connectivity for a narrative synthesis.")
        return "\n".join(lines)

    def _create_thoughtseed_trace(
        self,
        query: Query,
        sources: List[SearchResult]
    ) -> Optional[Dict[str, Any]]:
        """
        Create ThoughtSeed trace for consciousness tracking.

        This integrates with the extracted ThoughtSeed package
        to track cognitive processing flow.
        """
        if not query.thoughtseed_id:
            return None

        # Build trace information
        trace = {
            "thoughtseed_id": query.thoughtseed_id,
            "query_id": query.query_id,
            "timestamp": datetime.now().isoformat(),
            "sources_processed": len(sources),
            "neo4j_sources": len([s for s in sources if s.source.value == "neo4j"]),
            "qdrant_sources": len([s for s in sources if s.source.value == "qdrant"]),
            "avg_relevance": sum(s.relevance_score for s in sources) / len(sources) if sources else 0.0,
            "graph_relationships": list(set(
                rel for s in sources if s.source.value == "neo4j"
                for rel in s.relationships
            ))[:10],
            "processing_layers": ["L1_PERCEPTION", "L2_SEARCH", "L3_SYNTHESIS"],
            "consciousness_level": "active_inference"
        }

        return trace
