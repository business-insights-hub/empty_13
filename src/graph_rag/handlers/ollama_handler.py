"""
Ollama LLM Handler for Graph RAG
"""
import os
from typing import List, Dict, Any, Optional
import ollama
from loguru import logger


class OllamaHandler:
    """Handler for Ollama LLM operations"""

    def __init__(self, model: str = "llama3.1", host: Optional[str] = None):
        """
        Initialize Ollama handler

        Args:
            model: Model name (default: llama3.1)
            host: Ollama host URL (default: http://localhost:11434)
        """
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Initialize client
        self.client = ollama.Client(host=self.host)

        # Check if model is available
        self._ensure_model()
        logger.info(f"Initialized Ollama with model: {self.model}")

    def _ensure_model(self):
        """Ensure the model is available, pull if not"""
        try:
            models = self.client.list()
            available_models = [m['name'] for m in models.get('models', [])]

            if not any(self.model in m for m in available_models):
                logger.info(f"Model {self.model} not found. Pulling...")
                self.client.pull(self.model)
                logger.info(f"Model {self.model} pulled successfully")

        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """
        Generate text from prompt

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options
            )

            return response['message']['content']

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Chat with conversation history

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature}
            )

            return response['message']['content']

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return ""

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text

        Args:
            text: Input text

        Returns:
            Dictionary with extracted entities and relationships
        """
        system_prompt = """You are an expert agricultural knowledge extractor.
Extract entities and relationships from the given text.

Entities can be: Crops, Diseases, Pests, Techniques, Chemicals, Locations, etc.
Relationships can be: TREATS, AFFECTS, PREVENTS, PART_OF, LOCATED_IN, etc.

Return a JSON object with this structure:
{
  "entities": [
    {"name": "entity_name", "type": "entity_type", "description": "brief description"}
  ],
  "relationships": [
    {"from": "entity1", "to": "entity2", "type": "relationship_type"}
  ]
}

Only return valid JSON, nothing else."""

        prompt = f"Extract entities and relationships from this agricultural text:\n\n{text}"

        try:
            response = self.generate(prompt, system_prompt, temperature=0.3)

            # Try to parse JSON from response
            import json
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                logger.warning("No valid JSON found in response")
                return {"entities": [], "relationships": []}

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": [], "relationships": []}

    def answer_question(self, question: str, context: str, temperature: float = 0.7) -> str:
        """
        Answer a question based on provided context

        Args:
            question: User question
            context: Context information
            temperature: Sampling temperature

        Returns:
            Answer text
        """
        system_prompt = """You are a helpful agricultural assistant.
Answer questions based on the provided context.
If you don't know the answer based on the context, say so.
Be concise and accurate."""

        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        return self.generate(prompt, system_prompt, temperature)

    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text

        Args:
            text: Text to summarize
            max_length: Maximum summary length in words

        Returns:
            Summary text
        """
        system_prompt = "You are a helpful assistant that creates concise summaries."

        prompt = f"Summarize the following text in about {max_length} words:\n\n{text}"

        return self.generate(prompt, system_prompt, temperature=0.5, max_tokens=max_length * 2)

    def synthesize_answer(self, question: str, vector_results: List[Dict[str, Any]],
                         graph_results: List[Dict[str, Any]]) -> str:
        """
        Synthesize answer from vector and graph retrieval results

        Args:
            question: User question
            vector_results: Results from vector search
            graph_results: Results from graph traversal

        Returns:
            Synthesized answer
        """
        # Build context from vector results
        vector_context = "\n\n".join([
            f"Document {i+1}: {r.get('metadata', {}).get('text', '')}"
            for i, r in enumerate(vector_results[:3])
        ])

        # Build context from graph results
        graph_context = "\n\n".join([
            f"Related Entity: {r.get('properties', {}).get('name', 'Unknown')} "
            f"({', '.join(r.get('labels', []))})\n"
            f"Description: {r.get('properties', {}).get('description', 'No description')}"
            for r in graph_results[:5]
        ])

        system_prompt = """You are an expert agricultural assistant with access to both document knowledge and structured knowledge graph.

Use the provided context from both sources to answer the question comprehensively.
Mention specific entities and relationships when relevant.
If the context doesn't contain enough information, say so."""

        prompt = f"""Document Context:
{vector_context}

Knowledge Graph Context:
{graph_context}

Question: {question}

Provide a comprehensive answer:"""

        return self.generate(prompt, system_prompt, temperature=0.7)

    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible

        Returns:
            Connection status
        """
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
