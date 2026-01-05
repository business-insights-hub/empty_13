"""
Pinecone Vector Store Handler for Graph RAG
"""
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
import requests


class PineconeHandler:
    """Handler for Pinecone vector database operations"""

    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None,
                 embedding_dimension: int = 1024):
        """
        Initialize Pinecone connection

        Args:
            api_key: Pinecone API key (defaults to env PINECONE_API_KEY)
            index_name: Pinecone index name (defaults to env PINECONE_INDEX_NAME)
            embedding_dimension: Dimension of embeddings (defaults to 1024 for llama-text-embed-v2)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.embedding_dimension = embedding_dimension

        if not all([self.api_key, self.index_name]):
            raise ValueError("Pinecone credentials not provided")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)

        # Get or create index
        self.index = self._get_or_create_index()
        logger.info(f"Connected to Pinecone index: {self.index_name}")

    def _get_or_create_index(self):
        """Get existing index or create new one"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
                )
            )

        return self.pc.Index(self.index_name)

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Upsert vectors into Pinecone

        Args:
            vectors: List of dicts with 'id', 'values', and optional 'metadata'

        Returns:
            Upsert statistics
        """
        try:
            response = self.index.upsert(vectors=vectors)
            logger.debug(f"Upserted {response.upserted_count} vectors")
            return {"upserted_count": response.upserted_count}
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            return {"upserted_count": 0}

    def upsert_document_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]],
                              metadata_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
        """
        Upsert document chunks with embeddings

        Args:
            chunks: List of document chunks with 'id' and 'text'
            embeddings: List of embedding vectors
            metadata_list: Optional metadata for each chunk

        Returns:
            Upsert statistics
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            metadata["text"] = chunk.get("text", "")
            metadata["chunk_id"] = chunk.get("id", f"chunk_{i}")

            vectors.append({
                "id": chunk.get("id", f"chunk_{i}"),
                "values": embedding,
                "metadata": metadata
            })

        return self.upsert_vectors(vectors)

    def query(self, query_vector: List[float], top_k: int = 5,
             filter_dict: Optional[Dict[str, Any]] = None,
             include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Query similar vectors

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            include_metadata: Whether to include metadata in results

        Returns:
            List of similar vectors with scores and metadata
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=include_metadata
            )

            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else {}
                })

            logger.debug(f"Found {len(matches)} similar vectors")
            return matches

        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            return []

    def search_by_text(self, query_text: str, embedding_function,
                      top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents by text query

        Args:
            query_text: Text query
            embedding_function: Function to convert text to embedding
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of similar documents with scores
        """
        # Generate embedding for query text
        query_embedding = embedding_function(query_text)

        # Query Pinecone
        return self.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
            include_metadata=True
        )

    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs

        Args:
            ids: List of vector IDs to delete

        Returns:
            Success status
        """
        try:
            self.index.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False

    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """
        Delete vectors matching filter

        Args:
            filter_dict: Metadata filter

        Returns:
            Success status
        """
        try:
            self.index.delete(filter=filter_dict)
            logger.info(f"Deleted vectors matching filter: {filter_dict}")
            return True
        except Exception as e:
            logger.error(f"Error deleting by filter: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics

        Returns:
            Index stats including vector count
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

    def fetch_vectors(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch vectors by IDs

        Args:
            ids: List of vector IDs

        Returns:
            Dictionary mapping IDs to vector data
        """
        try:
            response = self.index.fetch(ids=ids)
            vectors = {}
            for id, data in response.vectors.items():
                vectors[id] = {
                    "values": data.values,
                    "metadata": data.metadata
                }
            return vectors
        except Exception as e:
            logger.error(f"Error fetching vectors: {e}")
            return {}

    def clear_index(self):
        """Clear all vectors from index (use with caution!)"""
        try:
            self.index.delete(delete_all=True)
            logger.warning("Index cleared!")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")


class LlamaEmbedding:
    """Embedding function using Llama Text Embed v2 via Pinecone Inference API"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Llama embedding

        Args:
            api_key: Pinecone API key (defaults to env PINECONE_API_KEY)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.model = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")
        self.api_url = "https://api.pinecone.io/embed"

        if not self.api_key:
            raise ValueError("Pinecone API key not provided")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "inputs": texts
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            embeddings = [item["values"] for item in data["data"]]
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [[0.0] * 1024 for _ in texts]  # Return zero vectors on error

    def __call__(self, text: str) -> List[float]:
        """Make the class callable"""
        return self.embed_text(text)
