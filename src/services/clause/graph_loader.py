"""
CLAUSE Graph Loader - Neo4j to NetworkX subgraph extraction

Loads k-hop subgraphs from Neo4j for CLAUSE Subgraph Architect processing.
Implements NFR-005 retry logic with exponential backoff.
"""

import time
from typing import Dict, List, Optional, Set, Any
from neo4j import Driver
import networkx as nx
import logging

from src.config.neo4j_config import get_neo4j_driver

logger = logging.getLogger(__name__)


class GraphLoader:
    """
    Load k-hop subgraphs from Neo4j into NetworkX.

    Supports CLAUSE Phase 1 subgraph construction with retry logic
    for connection failures (NFR-005: 3 retries with exponential backoff).
    """

    def __init__(self, driver: Optional[Driver] = None):
        """Initialize with Neo4j driver."""
        self._driver = driver
        self._driver_initialized = driver is not None

    @property
    def driver(self) -> Driver:
        """Lazy-load Neo4j driver."""
        if not self._driver_initialized:
            try:
                self._driver = get_neo4j_driver()
                self._driver_initialized = True
            except Exception as e:
                logger.error(f"Failed to get Neo4j driver: {e}")
                raise

        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")

        return self._driver

    def load_subgraph_from_neo4j(
        self,
        query: str,
        hop_distance: int = 2,
        max_seed_nodes: int = 20,
    ) -> nx.MultiDiGraph:
        """
        Load k-hop subgraph from Neo4j based on query.

        Implements NFR-005 retry logic: 3 retries with exponential backoff
        (100ms, 200ms, 400ms) before raising exception.

        Args:
            query: Search query for finding seed nodes
            hop_distance: Maximum hops from seed nodes (default 2)
            max_seed_nodes: Maximum seed nodes to expand from (default 20)

        Returns:
            NetworkX MultiDiGraph with nodes and edges from Neo4j

        Raises:
            ConnectionError: After 3 failed retries (NFR-005)
        """
        retries = 3
        backoff_delays = [0.1, 0.2, 0.4]  # 100ms, 200ms, 400ms

        for attempt in range(retries):
            try:
                return self._load_subgraph_internal(
                    query, hop_distance, max_seed_nodes
                )
            except Exception as e:
                logger.warning(
                    f"Neo4j subgraph load attempt {attempt+1}/{retries} failed: {e}"
                )

                if attempt < retries - 1:
                    delay = backoff_delays[attempt]
                    logger.info(f"Retrying in {delay*1000:.0f}ms...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Neo4j subgraph load failed after {retries} retries"
                    )
                    raise ConnectionError(
                        f"Neo4j connection failed after {retries} retries "
                        f"with exponential backoff"
                    ) from e

        # Should never reach here, but satisfy type checker
        raise ConnectionError("Unexpected error in retry logic")

    def _load_subgraph_internal(
        self,
        query: str,
        hop_distance: int,
        max_seed_nodes: int,
    ) -> nx.MultiDiGraph:
        """Internal subgraph loading logic (retryable)."""

        # Step 1: Find seed nodes via full-text search
        seed_node_ids = self._get_seed_nodes(query, max_seed_nodes)

        if not seed_node_ids:
            logger.warning(f"No seed nodes found for query: {query}")
            return nx.MultiDiGraph()

        # Step 2: Expand k-hop neighborhood
        subgraph_nodes = self._expand_khop_neighborhood(
            seed_node_ids, hop_distance
        )

        # Step 3: Load subgraph edges
        graph = self._build_networkx_graph(subgraph_nodes)

        logger.info(
            f"Loaded subgraph: {len(graph.nodes)} nodes, "
            f"{len(graph.edges)} edges from {len(seed_node_ids)} seeds"
        )

        return graph

    def _get_seed_nodes(self, query: str, limit: int) -> List[str]:
        """
        Get seed nodes using full-text search on knowledge graph.

        Uses existing knowledge_search_index on KnowledgeTriple nodes.
        """
        cypher_query = """
        CALL db.index.fulltext.queryNodes('knowledge_search_index', $query)
        YIELD node, score
        WHERE node:KnowledgeTriple
        RETURN elementId(node) as node_id,
               node.subject as subject,
               score
        ORDER BY score DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(
                cypher_query,
                {"query": query, "limit": limit}
            )

            seed_ids = [record["node_id"] for record in result]
            return seed_ids

    def _expand_khop_neighborhood(
        self, seed_ids: List[str], hop_distance: int
    ) -> Set[str]:
        """
        Expand k-hop neighborhood from seed nodes using BFS.

        Note: Using manual BFS instead of APOC for compatibility.
        APOC optimization (apoc.path.subgraphNodes) can be added in T029.
        """
        all_nodes: Set[str] = set(seed_ids)
        current_layer = set(seed_ids)

        for hop in range(hop_distance):
            cypher_query = """
            MATCH (start)
            WHERE elementId(start) IN $node_ids
            MATCH (start)-[r]-(neighbor)
            RETURN DISTINCT elementId(neighbor) as neighbor_id
            """

            with self.driver.session() as session:
                result = session.run(cypher_query, {"node_ids": list(current_layer)})

                next_layer = {record["neighbor_id"] for record in result}
                next_layer -= all_nodes  # Remove already visited

                all_nodes.update(next_layer)
                current_layer = next_layer

                if not next_layer:
                    logger.info(f"BFS stopped at hop {hop+1} (no new nodes)")
                    break

        return all_nodes

    def _build_networkx_graph(self, node_ids: Set[str]) -> nx.MultiDiGraph:
        """
        Build NetworkX MultiDiGraph from Neo4j nodes and edges.

        Loads node attributes (concept_id, basin_id) and edge attributes
        (relation_type, weight) for CLAUSE edge scoring.
        """
        graph = nx.MultiDiGraph()

        # Load nodes with attributes
        node_query = """
        MATCH (n)
        WHERE elementId(n) IN $node_ids
        OPTIONAL MATCH (n)<-[:HAS_BASIN]-(basin:AttractorBasin)
        RETURN
            elementId(n) as node_id,
            labels(n) as labels,
            n.subject as subject,
            n.predicate as predicate,
            n.object as object,
            elementId(basin) as basin_id,
            basin.strength as basin_strength
        """

        with self.driver.session() as session:
            result = session.run(node_query, {"node_ids": list(node_ids)})

            for record in result:
                node_id = record["node_id"]

                # Extract concept_id from subject/object
                concept_id = (
                    record["subject"] or record["object"] or node_id
                )

                graph.add_node(
                    node_id,
                    concept_id=concept_id,
                    basin_id=record["basin_id"],
                    basin_strength=record["basin_strength"] or 1.0,
                    labels=record["labels"],
                    subject=record["subject"],
                    predicate=record["predicate"],
                    object=record["object"],
                )

        # Load edges with attributes
        edge_query = """
        MATCH (source)-[r]->(target)
        WHERE elementId(source) IN $node_ids
          AND elementId(target) IN $node_ids
        RETURN
            elementId(source) as source_id,
            elementId(target) as target_id,
            type(r) as relation_type,
            r.weight as weight
        """

        with self.driver.session() as session:
            result = session.run(edge_query, {"node_ids": list(node_ids)})

            for record in result:
                graph.add_edge(
                    record["source_id"],
                    record["target_id"],
                    relation=record["relation_type"],
                    weight=record["weight"] or 1.0,
                )

        return graph

    def get_basin_info(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basin information for a concept.

        Used by EdgeScorer to retrieve basin strength for edge scoring.
        """
        query = """
        MATCH (k:KnowledgeTriple)
        WHERE k.subject = $concept_id OR k.object = $concept_id
        OPTIONAL MATCH (k)<-[:HAS_BASIN]-(basin:AttractorBasin)
        RETURN
            elementId(basin) as basin_id,
            basin.strength as strength,
            basin.activation_count as activation_count,
            basin.co_occurring_concepts as co_occurring_concepts
        LIMIT 1
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, {"concept_id": concept_id})
                record = result.single()

                if record and record["basin_id"]:
                    return {
                        "basin_id": record["basin_id"],
                        "strength": record["strength"] or 1.0,
                        "activation_count": record["activation_count"] or 0,
                        "co_occurring_concepts": (
                            record["co_occurring_concepts"] or {}
                        ),
                    }
                else:
                    # No basin for this concept yet
                    return None

        except Exception as e:
            logger.error(f"Failed to get basin info for {concept_id}: {e}")
            return None
