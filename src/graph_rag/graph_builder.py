"""
Knowledge Graph Builder for Graph RAG
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm import tqdm


class KnowledgeGraphBuilder:
    """Build knowledge graph from extracted entities and relationships"""

    def __init__(self, neo4j_handler):
        """
        Initialize graph builder

        Args:
            neo4j_handler: Neo4j handler instance
        """
        self.neo4j = neo4j_handler
        self.entity_cache = {}  # Cache entity IDs

    def build_from_extractions(self, extractions: List[Dict[str, Any]],
                              show_progress: bool = True) -> Dict[str, int]:
        """
        Build knowledge graph from extraction results

        Args:
            extractions: List of extraction results with entities and relationships
            show_progress: Whether to show progress bar

        Returns:
            Statistics about created entities and relationships
        """
        stats = {
            "entities_created": 0,
            "entities_found": 0,
            "relationships_created": 0,
            "errors": 0
        }

        # First pass: Create all entities
        logger.info("Creating entities...")
        entity_iterator = extractions if not show_progress else tqdm(extractions, desc="Creating entities")

        for extraction in entity_iterator:
            entities = extraction.get("entities", [])

            for entity in entities:
                try:
                    entity_id = self._create_or_get_entity(entity)
                    if entity_id:
                        if entity.get("name", "").lower() in self.entity_cache:
                            stats["entities_found"] += 1
                        else:
                            stats["entities_created"] += 1
                except Exception as e:
                    logger.error(f"Error creating entity {entity.get('name')}: {e}")
                    stats["errors"] += 1

        # Second pass: Create relationships
        logger.info("Creating relationships...")
        rel_iterator = extractions if not show_progress else tqdm(extractions, desc="Creating relationships")

        for extraction in rel_iterator:
            relationships = extraction.get("relationships", [])

            for rel in relationships:
                try:
                    success = self._create_relationship(rel)
                    if success:
                        stats["relationships_created"] += 1
                except Exception as e:
                    logger.error(f"Error creating relationship {rel}: {e}")
                    stats["errors"] += 1

        logger.info(f"Graph building complete: {stats}")
        return stats

    def _create_or_get_entity(self, entity: Dict[str, Any]) -> Optional[str]:
        """
        Create entity or get existing one

        Args:
            entity: Entity dict with 'name', 'type', and optional 'description'

        Returns:
            Entity ID
        """
        name = entity.get("name", "").strip()
        entity_type = entity.get("type", "Entity").strip()
        description = entity.get("description", "")

        if not name:
            return None

        # Check cache
        cache_key = name.lower()
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]

        # Prepare properties
        properties = {
            "name": name,
            "description": description
        }

        # Add any additional properties
        for key, value in entity.items():
            if key not in ["name", "type", "description", "source_chunk"]:
                properties[key] = value

        # Create or get entity
        entity_id = self.neo4j.get_or_create_entity(entity_type, properties)

        # Cache the ID
        self.entity_cache[cache_key] = entity_id

        return entity_id

    def _create_relationship(self, relationship: Dict[str, Any]) -> bool:
        """
        Create relationship between entities

        Args:
            relationship: Relationship dict with 'from', 'to', and 'type'

        Returns:
            Success status
        """
        from_name = relationship.get("from", "").strip().lower()
        to_name = relationship.get("to", "").strip().lower()
        rel_type = relationship.get("type", "RELATED_TO").strip().upper()

        # Replace spaces with underscores in relationship type
        rel_type = rel_type.replace(" ", "_")

        if not all([from_name, to_name, rel_type]):
            return False

        # Get entity IDs from cache
        from_id = self.entity_cache.get(from_name)
        to_id = self.entity_cache.get(to_name)

        if not from_id or not to_id:
            logger.warning(f"Entity not found for relationship: {from_name} -> {to_name}")
            return False

        # Create relationship
        properties = {}
        if "source_chunk" in relationship:
            properties["source"] = relationship["source_chunk"]

        return self.neo4j.create_relationship(from_id, to_id, rel_type, properties)

    def add_document_node(self, doc_id: str, metadata: Dict[str, Any]) -> str:
        """
        Add a document node to the graph

        Args:
            doc_id: Document ID
            metadata: Document metadata (title, path, etc.)

        Returns:
            Document node ID
        """
        properties = {
            "doc_id": doc_id,
            "name": metadata.get("title", doc_id),
            **metadata
        }

        return self.neo4j.create_entity_node("Document", properties)

    def link_entities_to_document(self, entity_names: List[str], doc_id: str) -> int:
        """
        Link entities to a document

        Args:
            entity_names: List of entity names mentioned in document
            doc_id: Document ID

        Returns:
            Number of links created
        """
        # Find document node
        doc_node = self.neo4j.find_entity_by_name("Document", doc_id)

        if not doc_node:
            logger.warning(f"Document node not found: {doc_id}")
            return 0

        doc_node_id = doc_node["id"]
        links_created = 0

        for entity_name in entity_names:
            entity_id = self.entity_cache.get(entity_name.lower())

            if entity_id:
                success = self.neo4j.create_relationship(
                    entity_id, doc_node_id, "MENTIONED_IN"
                )
                if success:
                    links_created += 1

        return links_created

    def enrich_graph_with_embeddings(self, pinecone_handler, embedding_function):
        """
        Add embeddings to graph entities for hybrid search

        Args:
            pinecone_handler: Pinecone handler instance
            embedding_function: Function to generate embeddings
        """
        # This could be used to store entity embeddings alongside graph
        # For now, we keep vectors and graph separate
        logger.info("Graph enrichment with embeddings not yet implemented")

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics

        Returns:
            Dictionary with graph stats
        """
        query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(*) as count
        """

        results = self.neo4j.execute_cypher(query)

        node_counts = {r["label"]: r["count"] for r in results if r.get("label")}

        # Get relationship counts
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
        """

        rel_results = self.neo4j.execute_cypher(rel_query)
        rel_counts = {r["type"]: r["count"] for r in rel_results}

        return {
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values())
        }

    def clear_cache(self):
        """Clear entity cache"""
        self.entity_cache.clear()
        logger.debug("Entity cache cleared")
