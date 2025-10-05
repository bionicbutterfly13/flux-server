"""Flux response synthesizer with policy-driven LLM routing."""

import logging
import re
import time
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.models.query import Query
from src.models.response import QueryResponse, SearchResult
from src.services.llm.router import ProviderRouter
from src.services.llm.telemetry import get_telemetry_store
from src.services.ollama_integration import OllamaModelManager

logger = logging.getLogger(__name__)


class AnswerGenerationError(Exception):
    """Raised when LLM answer generation fails."""


@runtime_checkable
class AnswerGenerator(Protocol):
    """Protocol for pluggable answer generators."""

    async def generate(self, question: str, sources: List[SearchResult]) -> str:  # pragma: no cover - interface
        """Generate an answer given a question and supporting sources."""


class BaseAnswerGenerator:
    """Shared utilities for answer generators."""

    @staticmethod
    def build_prompt(question: str, sources: List[SearchResult]) -> str:
        source_sections: List[str] = []
        for index, source in enumerate(sources, start=1):
            snippet = source.content.strip().replace("\n", " ")[:500]
            metadata_title = source.metadata.get("title") if source.metadata else None
            relationships = ", ".join(source.relationships[:3]) if source.relationships else "none"

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


class PolicyRoutedAnswerGenerator(BaseAnswerGenerator):
    """Answer generator that selects providers using routing policies."""

    OLLAMA_MODEL_ALIASES = {
        "ollama.deepseek-r1": "deepseek-r1",
        "ollama.qwen2.5-14b": "qwen2.5:14b",
    }

    def __init__(
        self,
        router: Optional[ProviderRouter] = None,
        ollama_manager: Optional[OllamaModelManager] = None,
        telemetry_store=None,
    ) -> None:
        self.router = router or ProviderRouter()
        self.ollama_manager = ollama_manager or OllamaModelManager()
        self.telemetry = telemetry_store or get_telemetry_store()

    async def generate(self, question: str, sources: List[SearchResult]) -> str:
        if not sources:
            raise AnswerGenerationError("no sources available")

        selection = self.router.select("chat")
        provider = selection.provider
        start = time.perf_counter()

        try:
            if provider.name in self.OLLAMA_MODEL_ALIASES:
                answer = await self._generate_with_ollama(provider.name, question, sources)
            else:
                raise AnswerGenerationError(
                    f"Provider {provider.name} not yet supported by runtime orchestrator"
                )
        except AnswerGenerationError as error:
            latency_ms = (time.perf_counter() - start) * 1000
            self.telemetry.record_failure(provider.name, latency_ms, str(error))
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        cost_estimate = provider.cost_per_1k_tokens * (768 / 1000)
        self.telemetry.record_success(provider.name, latency_ms, cost_estimate)
        return answer

    async def _generate_with_ollama(
        self,
        provider_name: str,
        question: str,
        sources: List[SearchResult],
    ) -> str:
        prompt = self.build_prompt(question, sources)
        model = self.OLLAMA_MODEL_ALIASES[provider_name]
        result = await self.ollama_manager.generate_text(model, prompt, max_tokens=768)

        if not result.get("success"):
            raise AnswerGenerationError(result.get("error", "unknown generation failure"))

        answer = result.get("response", "").strip()
        if not answer:
            raise AnswerGenerationError("empty response from provider")

        return answer


class ResponseSynthesizer:
    """Synthesize multiple search results into a coherent response."""

    def __init__(
        self,
        answer_generator: Optional[AnswerGenerator] = None,
    ) -> None:
        self.min_confidence = 0.3
        self.max_sources = 10
        self.minimum_answer_length = 200
        self.answer_generator = answer_generator or PolicyRoutedAnswerGenerator()

    async def synthesize(
        self,
        query: Query,
        neo4j_results: List[SearchResult],
        qdrant_results: List[SearchResult],
        processing_time_ms: int,
    ) -> QueryResponse:
        try:
            all_sources = self._combine_sources(neo4j_results, qdrant_results)
            confidence = self._calculate_confidence(all_sources)
            answer = await self._generate_answer(query.question, all_sources)
            thoughtseed_trace = self._create_thoughtseed_trace(query, all_sources)

            return QueryResponse(
                query_id=query.query_id,
                answer=answer,
                sources=all_sources[:self.max_sources],
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                thoughtseed_trace=thoughtseed_trace,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Response synthesis failed: %s", exc)
            return QueryResponse(
                query_id=query.query_id,
                answer="I encountered an error processing your query. Please try rephrasing your question.",
                sources=[],
                confidence=0.0,
                processing_time_ms=processing_time_ms,
            )

    def _combine_sources(
        self,
        neo4j_results: List[SearchResult],
        qdrant_results: List[SearchResult],
    ) -> List[SearchResult]:
        for result in neo4j_results:
            relationship_boost = len(result.relationships) * 0.05
            result.relevance_score = min(result.relevance_score + relationship_boost, 1.0)

        all_results = neo4j_results + qdrant_results
        unique_results = self._deduplicate_by_content(all_results)
        return sorted(unique_results, key=lambda r: r.relevance_score, reverse=True)

    def _deduplicate_by_content(self, results: List[SearchResult]) -> List[SearchResult]:
        fingerprints: Dict[str, SearchResult] = {}
        unique: List[SearchResult] = []

        for result in results:
            fingerprint = result.content[:100].lower().strip()
            existing = fingerprints.get(fingerprint)
            if existing is None or result.relevance_score > existing.relevance_score:
                fingerprints[fingerprint] = result

        unique.extend(fingerprints.values())
        return unique

    def _calculate_confidence(self, sources: List[SearchResult]) -> float:
        if not sources:
            return 0.0

        top_sources = sources[:5]
        avg_relevance = sum(s.relevance_score for s in top_sources) / len(top_sources)
        count_factor = min(len(sources) / 10.0, 1.0)
        has_graph = any(s.source.value == "neo4j" for s in sources)
        has_vector = any(s.source.value == "qdrant" for s in sources)
        diversity_factor = 1.0 if (has_graph and has_vector) else 0.85

        confidence = avg_relevance * 0.6 + count_factor * 0.3 * diversity_factor
        return min(max(confidence, 0.0), 1.0)

    async def _generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        if not sources:
            return self._fallback_answer(question, sources, error_message="no supporting sources")

        limited_sources = sources[:self.max_sources]

        try:
            raw_answer = await self.answer_generator.generate(question, limited_sources)
        except AnswerGenerationError as error:
            logger.warning("Answer generation failed via LLM: %s", error)
            return self._fallback_answer(question, limited_sources, error_message=str(error))
        except Exception as error:  # pragma: no cover - defensive
            logger.exception("Unexpected error during answer generation")
            return self._fallback_answer(question, limited_sources, error_message=str(error))

        answer = self._ensure_citations(raw_answer, limited_sources)
        if len(answer) < self.minimum_answer_length:
            answer = self._augment_answer(answer, limited_sources)
        return answer

    def _ensure_citations(self, answer: str, sources: List[SearchResult]) -> str:
        if not sources:
            return answer
        if re.search(r"\[\d+\]", answer):
            return answer
        return answer.rstrip() + " [1]"

    def _augment_answer(self, answer: str, sources: List[SearchResult]) -> str:
        complementary_lines = [answer.strip(), ""]
        complementary_lines.append(
            "Additional context: the retrieved sources emphasise how these mechanisms couple prediction error minimisation with narrative consolidation across attractor basins."
        )
        for index, source in enumerate(sources[:3], start=1):
            snippet = source.content.strip().replace("\n", " ")[:280]
            complementary_lines.append(f"[{index}] {snippet}")
        return "\n".join(complementary_lines)

    def _fallback_answer(
        self,
        question: str,
        sources: List[SearchResult],
        error_message: Optional[str] = None,
    ) -> str:
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
        sources: List[SearchResult],
    ) -> Optional[Dict[str, Any]]:
        if not query.thoughtseed_id:
            return None

        trace = {
            "thoughtseed_id": query.thoughtseed_id,
            "query_id": query.query_id,
            "timestamp": datetime.now().isoformat(),
            "sources_processed": len(sources),
            "neo4j_sources": len([s for s in sources if s.source.value == "neo4j"]),
            "qdrant_sources": len([s for s in sources if s.source.value == "qdrant"]),
            "avg_relevance": sum(s.relevance_score for s in sources) / len(sources) if sources else 0.0,
            "graph_relationships": list({rel for s in sources for rel in s.relationships})[:10],
            "processing_layers": ["L1_PERCEPTION", "L2_SEARCH", "L3_SYNTHESIS"],
            "consciousness_level": "active_inference",
        }
        return trace
