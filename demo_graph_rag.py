"""
Simplified Graph RAG Demo - End-to-End Test
Works with installed dependencies only
"""
import os
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase
from pinecone import Pinecone
import ollama
import requests

# Load environment
load_dotenv()


class SimpleGraphRAG:
    """Simplified Graph RAG implementation"""

    def __init__(self):
        # Initialize connections
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_index = self.pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

        self.ollama_client = ollama.Client()

        logger.success("Graph RAG initialized")

    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 3) -> str:
        """Extract text from PDF (first N pages for demo)"""
        doc = fitz.open(pdf_path)
        text = ""

        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            text += page.get_text()

        doc.close()
        logger.info(f"Extracted {len(text)} characters from {os.path.basename(pdf_path)}")
        return text

    def chunk_text(self, text: str, chunk_size: int = 500) -> list:
        """Simple text chunking"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def extract_entities_with_llm(self, text: str) -> dict:
        """Extract entities using Ollama (using available model)"""
        # Use gemma:2b which is already installed
        prompt = f"""Extract agricultural entities from this text. Return ONLY a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "entity_name", "type": "Crop|Disease|Technique|Chemical", "description": "brief description"}}
  ],
  "relationships": [
    {{"from": "entity1", "to": "entity2", "type": "TREATS|AFFECTS|PREVENTS"}}
  ]
}}

Text: {text[:1000]}

Return ONLY valid JSON, no other text:"""

        try:
            response = self.ollama_client.chat(
                model="gemma:2b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )

            content = response['message']['content']

            # Extract JSON from response
            start = content.find('{')
            end = content.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
                logger.debug(f"Extracted {len(result.get('entities', []))} entities")
                return result
            else:
                logger.warning("No JSON found in LLM response")
                return {"entities": [], "relationships": []}

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def generate_embedding(self, text: str) -> list:
        """Generate embedding (using simple hash-based approach for demo)"""
        try:
            # For production, use Pinecone Inference API or sentence-transformers
            # For this demo, generate deterministic non-zero embeddings from text hash
            import hashlib
            text_hash = hashlib.sha256(text.encode()).hexdigest()

            # Convert hash to 1024-dimensional vector with non-zero values
            embedding = []
            for i in range(1024):
                # Use hash bytes to generate values between -1 and 1
                byte_val = int(text_hash[(i % len(text_hash)):(i % len(text_hash)) + 2], 16)
                normalized = (byte_val / 255.0 * 2) - 1  # Scale to [-1, 1]
                embedding.append(normalized + 0.01)  # Ensure non-zero

            logger.debug(f"Generated embedding for text (len={len(text)})")
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return non-zero fallback
            import random
            random.seed(len(text))
            return [random.uniform(-0.1, 0.1) + 0.01 for _ in range(1024)]

    def store_in_neo4j(self, entities: list, relationships: list):
        """Store entities and relationships in Neo4j"""
        with self.neo4j_driver.session() as session:
            entity_ids = {}

            # Create entities
            for entity in entities:
                # Clean entity type (take first if pipe-separated, fallback to Entity)
                entity_type = entity.get('type', 'Entity')
                if '|' in entity_type:
                    entity_type = entity_type.split('|')[0].strip()
                if not entity_type or not entity_type[0].isalpha():
                    entity_type = "Entity"

                query = f"""
                MERGE (n:{entity_type} {{name: $name}})
                SET n.description = $description
                RETURN elementId(n) as id
                """
                result = session.run(query,
                    name=entity['name'],
                    description=entity.get('description', '')
                )
                record = result.single()
                if record:
                    entity_ids[entity['name']] = record['id']

            # Create relationships
            for rel in relationships:
                from_id = entity_ids.get(rel['from'])
                to_id = entity_ids.get(rel['to'])

                if from_id and to_id:
                    rel_type = rel['type'].replace(' ', '_').replace('|', '_').upper()
                    # Use RELATED_TO as fallback if type is empty
                    if not rel_type or len(rel_type) < 2:
                        rel_type = "RELATED_TO"

                    query = f"""
                    MATCH (a), (b)
                    WHERE elementId(a) = $from_id AND elementId(b) = $to_id
                    MERGE (a)-[r:{rel_type}]->(b)
                    RETURN r
                    """
                    try:
                        session.run(query, from_id=from_id, to_id=to_id)
                    except Exception as e:
                        logger.warning(f"Could not create relationship {rel_type}: {e}")

            logger.info(f"Stored {len(entity_ids)} entities and {len(relationships)} relationships")

    def store_in_pinecone(self, chunks: list, doc_id: str, original_filename: str = None):
        """Store chunks with embeddings in Pinecone"""
        vectors = []

        for i, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk)
            vectors.append({
                "id": f"{doc_id}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk[:1000],  # Limit metadata size
                    "doc_id": doc_id,
                    "original_filename": original_filename or doc_id,
                    "chunk_num": i
                }
            })

        self.pinecone_index.upsert(vectors=vectors)
        logger.info(f"Stored {len(vectors)} vectors in Pinecone")

    def query_vector_search(self, query: str, top_k: int = 3) -> list:
        """Search Pinecone for similar chunks"""
        query_embedding = self.generate_embedding(query)
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches

    def query_graph(self, entity_name: str) -> list:
        """Query Neo4j for related entities"""
        with self.neo4j_driver.session() as session:
            query = """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($name)
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN n.name as entity, labels(n) as types,
                   collect(DISTINCT related.name) as related_entities
            LIMIT 5
            """
            result = session.run(query, name=entity_name)
            return [dict(record) for record in result]

    def answer_question(self, question: str) -> str:
        """Answer question using hybrid retrieval"""
        # Vector search
        vector_results = self.query_vector_search(question, top_k=2)

        # Extract context
        context = "\n\n".join([
            match.metadata.get('text', '')
            for match in vector_results
        ])

        # Generate answer with Ollama
        prompt = f"""Based on this context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = self.ollama_client.chat(
                model="gemma:2b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Error generating answer"

    def get_statistics(self) -> dict:
        """Get system statistics"""
        # Neo4j stats
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()['node_count']

            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()['rel_count']

        # Pinecone stats
        pinecone_stats = self.pinecone_index.describe_index_stats()

        return {
            "neo4j_nodes": node_count,
            "neo4j_relationships": rel_count,
            "pinecone_vectors": pinecone_stats.total_vector_count
        }

    def close(self):
        """Close connections"""
        self.neo4j_driver.close()


def demo():
    """Run complete demo"""
    print("\n" + "="*60)
    print("Graph RAG Demo - Processing Agricultural Documents")
    print("="*60 + "\n")

    rag = SimpleGraphRAG()

    try:
        # Step 1: Load and process a document
        pdf_file = "dataset/2017-883.pdf"
        logger.info(f"Step 1: Processing {pdf_file}")

        text = rag.extract_text_from_pdf(pdf_file, max_pages=2)
        chunks = rag.chunk_text(text, chunk_size=300)

        # Step 2: Extract entities (from first chunk only for speed)
        logger.info("Step 2: Extracting entities with LLM")
        entities_data = rag.extract_entities_with_llm(chunks[0] if chunks else text[:1000])

        # Step 3: Store in Neo4j
        if entities_data.get('entities'):
            logger.info("Step 3: Storing in Neo4j graph")
            rag.store_in_neo4j(
                entities_data['entities'],
                entities_data.get('relationships', [])
            )

        # Step 4: Store in Pinecone (first 3 chunks for demo)
        logger.info("Step 4: Storing embeddings in Pinecone")
        rag.store_in_pinecone(chunks[:3], "demo_doc")

        # Step 5: Test query
        logger.info("Step 5: Testing query")
        question = "What agricultural topics are discussed?"
        answer = rag.answer_question(question)

        # Step 6: Get statistics
        stats = rag.get_statistics()

        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\n📊 System Statistics:")
        print(f"  - Neo4j Nodes: {stats['neo4j_nodes']}")
        print(f"  - Neo4j Relationships: {stats['neo4j_relationships']}")
        print(f"  - Pinecone Vectors: {stats['pinecone_vectors']}")

        print(f"\n🔍 Query Test:")
        print(f"  Question: {question}")
        print(f"  Answer: {answer}")

        print("\n" + "="*60)
        logger.success("Demo completed successfully!")

    finally:
        rag.close()


if __name__ == "__main__":
    demo()
