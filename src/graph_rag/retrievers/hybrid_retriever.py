"""
Hybrid Retrieval System for Graph RAG
Combines vector search with graph traversal
"""
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger


class HybridRetriever:
    """Hybrid retrieval combining vector search and graph traversal"""

    def __init__(self, neo4j_handler, pinecone_handler, embedding_function):
        """
        Initialize hybrid retriever

        Args:
            neo4j_handler: Neo4j handler instance
            pinecone_handler: Pinecone handler instance
            embedding_function: Function to generate embeddings from text
        """
        self.neo4j = neo4j_handler
        self.pinecone = pinecone_handler
        self.embed = embedding_function

    def retrieve(self, query: str, top_k_vector: int = 5, top_k_graph: int = 10,
                max_hops: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve relevant information using hybrid approach

        Args:
            query: User query
            top_k_vector: Number of vector search results
            top_k_graph: Number of graph traversal results
            max_hops: Maximum hops in graph traversal

        Returns:
            Tuple of (vector_results, graph_results)
        """
        # Step 1: Vector search
        vector_results = self._vector_search(query, top_k_vector)

        # Step 2: Extract entities from query
        query_entities = self._extract_query_entities(query)

        # Step 3: Graph traversal from query entities
        graph_results = self._graph_traversal(query_entities, top_k_graph, max_hops)

        # Step 4: If graph results are sparse, expand from vector results
        if len(graph_results) < top_k_graph // 2:
            graph_results = self._expand_from_vector_results(
                vector_results, graph_results, top_k_graph, max_hops
            )

        logger.info(f"Retrieved {len(vector_results)} vector results and {len(graph_results)} graph results")

        return vector_results, graph_results

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of similar documents
        """
        try:
            results = self.pinecone.search_by_text(
                query_text=query,
                embedding_function=self.embed,
                top_k=top_k
            )

            logger.debug(f"Vector search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract potential entities from query using graph search

        Args:
            query: Query text

        Returns:
            List of entity names
        """
        # Search for entities matching query terms
        query_terms = query.lower().split()
        entities = []

        for term in query_terms:
            # Skip very short terms
            if len(term) < 3:
                continue

            # Search in graph
            matches = self.neo4j.search_entities(term, limit=3)
            entities.extend([
                match["properties"].get("name", "")
                for match in matches
                if match["properties"].get("name")
            ])

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        logger.debug(f"Extracted entities from query: {unique_entities}")
        return unique_entities

    def _graph_traversal(self, entity_names: List[str], top_k: int,
                        max_hops: int) -> List[Dict[str, Any]]:
        """
        Traverse graph from starting entities

        Args:
            entity_names: Starting entity names
            top_k: Maximum results
            max_hops: Maximum traversal hops

        Returns:
            List of related entities
        """
        all_related = []
        seen_ids = set()

        for entity_name in entity_names:
            # Find entity in graph
            entity = self.neo4j.find_entity_by_name("Entity", entity_name)

            # Try other common types if not found
            if not entity:
                for entity_type in ["Crop", "Disease", "Pest", "Technique", "Chemical"]:
                    entity = self.neo4j.find_entity_by_name(entity_type, entity_name)
                    if entity:
                        break

            if not entity:
                continue

            entity_id = entity["id"]

            # Get related entities
            related = self.neo4j.get_related_entities(
                entity_id,
                direction="both",
                max_hops=max_hops
            )

            # Add to results if not already seen
            for rel_entity in related:
                rel_id = rel_entity["id"]
                if rel_id not in seen_ids:
                    seen_ids.add(rel_id)
                    all_related.append(rel_entity)

                if len(all_related) >= top_k:
                    break

            if len(all_related) >= top_k:
                break

        return all_related[:top_k]

    def _expand_from_vector_results(self, vector_results: List[Dict[str, Any]],
                                   existing_graph_results: List[Dict[str, Any]],
                                   top_k: int, max_hops: int) -> List[Dict[str, Any]]:
        """
        Expand graph results by extracting entities from vector results

        Args:
            vector_results: Vector search results
            existing_graph_results: Already found graph results
            top_k: Target number of results
            max_hops: Maximum traversal hops

        Returns:
            Expanded graph results
        """
        # Extract entity mentions from vector result metadata
        entity_candidates = []

        for result in vector_results[:3]:  # Check top 3 vector results
            metadata = result.get("metadata", {})
            text = metadata.get("text", "")

            # Simple entity extraction: search for capitalized phrases
            words = text.split()
            for i, word in enumerate(words):
                if word and word[0].isupper():
                    entity_candidates.append(word)

        # Use these as starting points for graph traversal
        new_results = self._graph_traversal(entity_candidates, top_k, max_hops)

        # Merge with existing results (deduplicate by ID)
        existing_ids = {r["id"] for r in existing_graph_results}
        merged = list(existing_graph_results)

        for result in new_results:
            if result["id"] not in existing_ids:
                merged.append(result)
                existing_ids.add(result["id"])

            if len(merged) >= top_k:
                break

        return merged[:top_k]

    def retrieve_with_context(self, query: str, conversation_history: Optional[List[str]] = None,
                            top_k_vector: int = 5, top_k_graph: int = 10) -> Dict[str, Any]:
        """
        Retrieve with conversation context

        Args:
            query: Current query
            conversation_history: Previous queries/context
            top_k_vector: Number of vector results
            top_k_graph: Number of graph results

        Returns:
            Combined retrieval results with metadata
        """
        # Enhance query with conversation context
        enhanced_query = query
        if conversation_history:
            # Append recent context
            context = " ".join(conversation_history[-2:])
            enhanced_query = f"{context} {query}"

        # Retrieve
        vector_results, graph_results = self.retrieve(
            enhanced_query, top_k_vector, top_k_graph
        )

        return {
            "query": query,
            "enhanced_query": enhanced_query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "num_vector": len(vector_results),
            "num_graph": len(graph_results)
        }

    def rerank_results(self, vector_results: List[Dict[str, Any]],
                      graph_results: List[Dict[str, Any]],
                      vector_weight: float = 0.6) -> List[Dict[str, Any]]:
        """
        Rerank and combine vector and graph results

        Args:
            vector_results: Vector search results with scores
            graph_results: Graph traversal results
            vector_weight: Weight for vector scores (0-1)

        Returns:
            Combined and reranked results
        """
        graph_weight = 1.0 - vector_weight

        # Normalize vector scores
        if vector_results:
            max_vector_score = max(r.get("score", 0) for r in vector_results)
            for result in vector_results:
                result["normalized_score"] = result.get("score", 0) / max_vector_score if max_vector_score > 0 else 0
                result["source"] = "vector"

        # Assign scores to graph results based on relationship path length
        for i, result in enumerate(graph_results):
            # Shorter paths get higher scores
            path_length = len(result.get("relationship_path", []))
            score = 1.0 / (1.0 + path_length) if path_length > 0 else 0.8

            result["normalized_score"] = score
            result["source"] = "graph"

        # Combine and sort
        all_results = []

        for result in vector_results:
            result["final_score"] = result["normalized_score"] * vector_weight
            all_results.append(result)

        for result in graph_results:
            result["final_score"] = result["normalized_score"] * graph_weight
            all_results.append(result)

        # Sort by final score
        all_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        return all_results
