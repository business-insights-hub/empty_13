"""
Main Graph RAG Orchestrator
Combines all components for end-to-end Graph RAG functionality
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from .handlers.neo4j_handler import Neo4jHandler
from .handlers.pinecone_handler import PineconeHandler, LlamaEmbedding
from .handlers.ollama_handler import OllamaHandler
from .extractors.entity_extractor import EntityExtractor, DocumentChunker, PDFProcessor
from .graph_builder import KnowledgeGraphBuilder
from .retrievers.hybrid_retriever import HybridRetriever


class GraphRAG:
    """Main Graph RAG orchestrator"""

    def __init__(self, ollama_model: str = "llama3.1"):
        """
        Initialize Graph RAG system

        Args:
            ollama_model: Ollama model to use (default: llama3.1)
        """
        # Load environment variables
        load_dotenv()

        logger.info("Initializing Graph RAG system...")

        # Initialize handlers
        self.neo4j = Neo4jHandler()
        self.pinecone = PineconeHandler()
        self.embedding = LlamaEmbedding()
        self.llm = OllamaHandler(model=ollama_model)

        # Initialize components
        self.extractor = EntityExtractor(self.llm)
        self.chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        self.pdf_processor = PDFProcessor()
        self.graph_builder = KnowledgeGraphBuilder(self.neo4j)
        self.retriever = HybridRetriever(self.neo4j, self.pinecone, self.embedding)

        logger.info("Graph RAG system initialized successfully!")

    def ingest_documents(self, document_paths: List[str], show_progress: bool = True) -> Dict[str, Any]:
        """
        Ingest documents into the Graph RAG system

        Args:
            document_paths: List of paths to PDF documents
            show_progress: Whether to show progress

        Returns:
            Statistics about ingestion
        """
        logger.info(f"Ingesting {len(document_paths)} documents...")

        all_chunks = []
        all_extractions = []
        all_embeddings = []

        # Process each document
        for doc_path in document_paths:
            logger.info(f"Processing: {doc_path}")

            try:
                # Extract text and chunk it
                chunks = self.pdf_processor.process_pdf(doc_path, self.chunker)
                all_chunks.extend(chunks)

                # Extract entities from chunks
                extractions = self.extractor.extract_from_chunks(chunks)
                all_extractions.extend(extractions)

                # Generate embeddings for chunks
                chunk_texts = [chunk["text"] for chunk in chunks]
                embeddings = self.embedding.embed_texts(chunk_texts)
                all_embeddings.extend(embeddings)

                logger.info(f"Processed {len(chunks)} chunks from {doc_path}")

            except Exception as e:
                logger.error(f"Error processing {doc_path}: {e}")
                continue

        # Build knowledge graph
        logger.info("Building knowledge graph...")
        graph_stats = self.graph_builder.build_from_extractions(all_extractions, show_progress)

        # Store embeddings in Pinecone
        logger.info("Storing embeddings in Pinecone...")
        metadata_list = [
            {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_num": chunk.get("chunk_num", 0),
                "text": chunk.get("text", "")[:1000]  # Limit text length
            }
            for chunk in all_chunks
        ]

        pinecone_stats = self.pinecone.upsert_document_chunks(
            all_chunks, all_embeddings, metadata_list
        )

        # Create indexes for better performance
        logger.info("Creating Neo4j indexes...")
        self.neo4j.create_indexes()

        stats = {
            "documents_processed": len(document_paths),
            "chunks_created": len(all_chunks),
            "embeddings_stored": pinecone_stats.get("upserted_count", 0),
            **graph_stats
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def query(self, question: str, top_k_vector: int = 5, top_k_graph: int = 10,
             return_context: bool = False) -> str:
        """
        Query the Graph RAG system

        Args:
            question: User question
            top_k_vector: Number of vector search results
            top_k_graph: Number of graph traversal results
            return_context: Whether to return retrieval context

        Returns:
            Answer (or dict with answer and context if return_context=True)
        """
        logger.info(f"Processing query: {question}")

        # Retrieve relevant information
        vector_results, graph_results = self.retriever.retrieve(
            question, top_k_vector, top_k_graph
        )

        # Generate answer
        answer = self.llm.synthesize_answer(question, vector_results, graph_results)

        if return_context:
            return {
                "answer": answer,
                "vector_results": vector_results,
                "graph_results": graph_results
            }

        return answer

    def chat(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Chat with conversation history

        Args:
            question: User question
            conversation_history: Previous conversation messages

        Returns:
            Answer
        """
        # Extract recent context from history
        query_context = []
        if conversation_history:
            for msg in conversation_history[-3:]:
                if msg.get("role") == "user":
                    query_context.append(msg.get("content", ""))

        # Retrieve with context
        result = self.retriever.retrieve_with_context(
            question, query_context, top_k_vector=5, top_k_graph=10
        )

        # Generate answer
        answer = self.llm.synthesize_answer(
            question,
            result["vector_results"],
            result["graph_results"]
        )

        return answer

    def search_entities(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities in the knowledge graph

        Args:
            search_text: Text to search for
            limit: Maximum results

        Returns:
            List of matching entities
        """
        return self.neo4j.search_entities(search_text, limit=limit)

    def get_entity_details(self, entity_name: str, entity_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get details about a specific entity and its relationships

        Args:
            entity_name: Name of the entity
            entity_type: Optional entity type

        Returns:
            Entity details with related entities
        """
        # Find entity
        if entity_type:
            entity = self.neo4j.find_entity_by_name(entity_type, entity_name)
        else:
            # Try common types
            entity = None
            for etype in ["Entity", "Crop", "Disease", "Pest", "Technique", "Chemical"]:
                entity = self.neo4j.find_entity_by_name(etype, entity_name)
                if entity:
                    break

        if not entity:
            return None

        # Get related entities
        related = self.neo4j.get_related_entities(entity["id"], max_hops=1)

        return {
            "entity": entity,
            "related_entities": related
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics

        Returns:
            Dictionary with graph and vector store stats
        """
        graph_stats = self.graph_builder.get_graph_statistics()
        pinecone_stats = self.pinecone.get_index_stats()

        return {
            "graph": graph_stats,
            "vector_store": pinecone_stats
        }

    def ingest_directory(self, directory_path: str, pattern: str = "*.pdf") -> Dict[str, Any]:
        """
        Ingest all documents from a directory

        Args:
            directory_path: Path to directory
            pattern: File pattern (default: *.pdf)

        Returns:
            Ingestion statistics
        """
        from glob import glob

        files = glob(os.path.join(directory_path, pattern))
        logger.info(f"Found {len(files)} files matching pattern '{pattern}'")

        if not files:
            logger.warning(f"No files found in {directory_path}")
            return {}

        return self.ingest_documents(files)

    def clear_all_data(self):
        """Clear all data from Neo4j and Pinecone (use with caution!)"""
        logger.warning("Clearing all data from Neo4j and Pinecone...")

        self.neo4j.clear_database()
        self.pinecone.clear_index()
        self.graph_builder.clear_cache()

        logger.info("All data cleared!")

    def close(self):
        """Close all connections"""
        self.neo4j.close()
        logger.info("Graph RAG system closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
