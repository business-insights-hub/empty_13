"""
Neo4j Graph Database Handler for Graph RAG
"""
import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from loguru import logger


class Neo4jHandler:
    """Handler for Neo4j graph database operations"""

    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None,
                 password: Optional[str] = None, database: Optional[str] = None):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j connection URI (defaults to env NEO4J_URI)
            username: Neo4j username (defaults to env NEO4J_USERNAME)
            password: Neo4j password (defaults to env NEO4J_PASSWORD)
            database: Neo4j database name (defaults to env NEO4J_DATABASE)
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not all([self.uri, self.username, self.password]):
            raise ValueError("Neo4j credentials not provided")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )
        logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def create_entity_node(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Create an entity node in the graph

        Args:
            entity_type: Type of entity (e.g., 'Crop', 'Disease', 'Technique')
            properties: Properties of the entity

        Returns:
            Node ID
        """
        with self.driver.session(database=self.database) as session:
            query = f"""
            CREATE (n:{entity_type} $properties)
            RETURN elementId(n) as id
            """
            result = session.run(query, properties=properties)
            record = result.single()
            node_id = record["id"] if record else None
            logger.debug(f"Created {entity_type} node: {properties.get('name', 'unnamed')}")
            return node_id

    def create_relationship(self, from_entity_id: str, to_entity_id: str,
                          rel_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a relationship between two entities

        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            rel_type: Type of relationship (e.g., 'TREATS', 'AFFECTS', 'PART_OF')
            properties: Optional properties for the relationship

        Returns:
            Success status
        """
        with self.driver.session(database=self.database) as session:
            props = properties or {}
            query = f"""
            MATCH (a), (b)
            WHERE elementId(a) = $from_id AND elementId(b) = $to_id
            CREATE (a)-[r:{rel_type} $properties]->(b)
            RETURN r
            """
            result = session.run(query, from_id=from_entity_id, to_id=to_entity_id, properties=props)
            success = result.single() is not None
            if success:
                logger.debug(f"Created relationship: {rel_type}")
            return success

    def find_entity_by_name(self, entity_type: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Find an entity by name

        Args:
            entity_type: Type of entity
            name: Name to search for

        Returns:
            Entity properties including ID, or None
        """
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (n:{entity_type})
            WHERE toLower(n.name) = toLower($name)
            RETURN elementId(n) as id, properties(n) as props
            LIMIT 1
            """
            result = session.run(query, name=name)
            record = result.single()
            if record:
                return {"id": record["id"], **record["props"]}
            return None

    def get_or_create_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Get existing entity by name or create new one

        Args:
            entity_type: Type of entity
            properties: Entity properties (must include 'name')

        Returns:
            Node ID
        """
        name = properties.get("name")
        if not name:
            raise ValueError("Entity properties must include 'name'")

        existing = self.find_entity_by_name(entity_type, name)
        if existing:
            return existing["id"]

        return self.create_entity_node(entity_type, properties)

    def get_related_entities(self, entity_id: str, relationship_type: Optional[str] = None,
                           direction: str = "both", max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Get entities related to a given entity

        Args:
            entity_id: Source entity ID
            relationship_type: Optional specific relationship type
            direction: 'outgoing', 'incoming', or 'both'
            max_hops: Maximum number of hops in the graph

        Returns:
            List of related entities with their relationships
        """
        with self.driver.session(database=self.database) as session:
            rel_pattern = f"[r:{relationship_type}]" if relationship_type else "[r]"

            if direction == "outgoing":
                path_pattern = f"(start)-{rel_pattern}*1..{max_hops}->(end)"
            elif direction == "incoming":
                path_pattern = f"(start)<-{rel_pattern}*1..{max_hops}-(end)"
            else:  # both
                path_pattern = f"(start)-{rel_pattern}*1..{max_hops}-(end)"

            query = f"""
            MATCH path = {path_pattern}
            WHERE elementId(start) = $entity_id
            RETURN DISTINCT elementId(end) as id, labels(end) as labels,
                   properties(end) as props,
                   [rel in relationships(path) | type(rel)] as rel_types
            LIMIT 50
            """

            result = session.run(query, entity_id=entity_id)
            entities = []
            for record in result:
                entities.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": record["props"],
                    "relationship_path": record["rel_types"]
                })

            logger.debug(f"Found {len(entities)} related entities")
            return entities

    def search_entities(self, search_text: str, entity_types: Optional[List[str]] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities by text in their properties

        Args:
            search_text: Text to search for
            entity_types: Optional list of entity types to filter
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        with self.driver.session(database=self.database) as session:
            label_filter = ""
            if entity_types:
                labels = "|".join(entity_types)
                label_filter = f":{labels}"

            query = f"""
            MATCH (n{label_filter})
            WHERE any(prop in keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($search_text))
            RETURN elementId(n) as id, labels(n) as labels, properties(n) as props
            LIMIT $limit
            """

            result = session.run(query, search_text=search_text, limit=limit)
            entities = []
            for record in result:
                entities.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": record["props"]
                })

            return entities

    def execute_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            Query results
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]

    def create_indexes(self):
        """Create recommended indexes for performance"""
        with self.driver.session(database=self.database) as session:
            # Create indexes for common entity types
            entity_types = ["Crop", "Disease", "Pest", "Technique", "Chemical", "Document"]

            for entity_type in entity_types:
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)")
                    logger.info(f"Created index for {entity_type}.name")
                except Exception as e:
                    logger.warning(f"Could not create index for {entity_type}: {e}")

    def clear_database(self):
        """Clear all nodes and relationships (use with caution!)"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Database cleared!")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
