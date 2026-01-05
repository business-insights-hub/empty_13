# Graph RAG Agricultural Assistant

A sophisticated Graph RAG (Retrieval-Augmented Generation) system for agricultural knowledge management and question answering. This system combines the power of knowledge graphs with vector embeddings to provide accurate, context-aware answers to agricultural questions.

## Architecture

The system uses a hybrid approach:

1. **Vector Search** (Pinecone): Fast semantic search over document chunks
2. **Knowledge Graph** (Neo4j): Structured relationships between agricultural entities
3. **LLM** (Ollama): Entity extraction and answer generation
4. **Embeddings** (Llama Text Embed v2): High-quality text embeddings

### How It Works

```
Documents (PDFs)
    ↓
PDF Processing & Chunking
    ↓
Entity Extraction (LLM) → Knowledge Graph (Neo4j)
    ↓
Text Embeddings → Vector Store (Pinecone)
    ↓
Query → Hybrid Retrieval (Vector + Graph)
    ↓
Answer Generation (LLM)
```

## Features

- **Document Ingestion**: Process agricultural PDFs and extract knowledge
- **Entity Recognition**: Automatically identify crops, diseases, pests, techniques, etc.
- **Knowledge Graph**: Build structured relationships between entities
- **Hybrid Search**: Combine semantic search with graph traversal
- **Conversational AI**: Chat interface with context awareness
- **Entity Explorer**: Search and explore entities in the knowledge graph

## Prerequisites

### 1. Neo4j Aura (Cloud Graph Database)

Already configured in `.env` file.

### 2. Pinecone (Vector Database)

Already configured in `.env` file.

### 3. Ollama (Local LLM)

Install Ollama from [ollama.ai](https://ollama.ai)

```bash
# macOS
brew install ollama

# Start Ollama
ollama serve

# Pull the model (in a new terminal)
ollama pull llama3.1
```

## Installation

1. **Clone and navigate to the project**:
```bash
cd /Users/ismatsamadov/agri_bot
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify environment variables** (already set in `.env`):
```bash
cat .env
```

## Usage

### Quick Start

Run the main script:
```bash
python main.py
```

This will show a menu with options:
1. Ingest documents from dataset folder
2. Run example queries
3. Search for entities
4. Get entity details
5. View system statistics
6. Interactive chat mode

### 1. Ingest Documents

First, ingest your agricultural PDFs:

```python
from src.graph_rag.graph_rag import GraphRAG

with GraphRAG() as graph_rag:
    # Ingest all PDFs from dataset directory
    stats = graph_rag.ingest_directory("dataset")
    print(stats)
```

This will:
- Extract text from PDFs
- Chunk documents into manageable pieces
- Extract entities and relationships using LLM
- Build knowledge graph in Neo4j
- Store embeddings in Pinecone

### 2. Query the System

```python
from src.graph_rag.graph_rag import GraphRAG

with GraphRAG() as graph_rag:
    answer = graph_rag.query("What are the main crops mentioned?")
    print(answer)
```

### 3. Interactive Chat

```python
from src.graph_rag.graph_rag import GraphRAG

with GraphRAG() as graph_rag:
    conversation_history = []

    question = "What diseases affect wheat?"
    answer = graph_rag.chat(question, conversation_history)
    print(answer)
```

### 4. Search Entities

```python
from src.graph_rag.graph_rag import GraphRAG

with GraphRAG() as graph_rag:
    entities = graph_rag.search_entities("wheat", limit=5)

    for entity in entities:
        print(f"- {entity['properties']['name']}")
```

### 5. Get Entity Details

```python
from src.graph_rag.graph_rag import GraphRAG

with GraphRAG() as graph_rag:
    details = graph_rag.get_entity_details("Wheat", "Crop")

    print(f"Entity: {details['entity']['name']}")
    print(f"Related: {len(details['related_entities'])} entities")
```

## Project Structure

```
agri_bot/
├── .env                          # Environment variables (credentials)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── main.py                       # Example usage script
├── dataset/                      # Agricultural PDF documents
│   └── *.pdf
└── src/
    └── graph_rag/
        ├── __init__.py
        ├── graph_rag.py          # Main orchestrator
        ├── graph_builder.py      # Knowledge graph builder
        ├── handlers/
        │   ├── neo4j_handler.py     # Neo4j connection
        │   ├── pinecone_handler.py  # Pinecone connection
        │   └── ollama_handler.py    # Ollama LLM
        ├── extractors/
        │   └── entity_extractor.py  # Entity extraction
        └── retrievers/
            └── hybrid_retriever.py  # Hybrid search
```

## Components

### Neo4j Handler

Manages graph database operations:
- Create entity nodes
- Create relationships
- Search entities
- Graph traversal
- Custom Cypher queries

### Pinecone Handler

Manages vector database operations:
- Store document embeddings
- Semantic search
- Hybrid filtering

### Ollama Handler

Manages LLM operations:
- Entity extraction
- Question answering
- Answer synthesis

### Entity Extractor

Processes documents:
- PDF text extraction
- Document chunking
- Entity and relationship extraction

### Knowledge Graph Builder

Builds the graph:
- Create entities from extractions
- Create relationships
- Deduplicate entities
- Link documents to entities

### Hybrid Retriever

Combines search methods:
- Vector similarity search
- Graph traversal from entities
- Result re-ranking

## Configuration

All configuration is in `.env`:

```bash
# Neo4j
NEO4J_URI=neo4j+s://9c0a7d96.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Pinecone
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=agribot
PINECONE_ENVIRONMENT=us-east-1

# Embedding
EMBEDDING_MODEL=llama-text-embed-v2

# Ollama (optional, defaults to localhost)
OLLAMA_HOST=http://localhost:11434
```

## Advanced Usage

### Custom Entity Types

Modify entity types in `handlers/neo4j_handler.py`:

```python
entity_types = ["Crop", "Disease", "Pest", "Technique", "Chemical", "YourType"]
```

### Adjust Chunking

Modify chunking parameters:

```python
chunker = DocumentChunker(
    chunk_size=1000,      # Larger for more context
    chunk_overlap=200      # More overlap for continuity
)
```

### Customize Retrieval

```python
answer = graph_rag.query(
    question,
    top_k_vector=10,      # More vector results
    top_k_graph=20,       # More graph results
    return_context=True   # Get retrieval details
)
```

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama
ollama serve

# In another terminal, verify it's running
ollama list
```

### Neo4j Connection Error

Wait 60 seconds after creating the Aura instance, or check status at [console.neo4j.io](https://console.neo4j.io)

### Pinecone API Error

Verify your API key in `.env` is correct.

### Out of Memory

Reduce batch size or chunk size in the ingestion process.

## Examples

### Example 1: Agriculture Q&A

```python
questions = [
    "What are common wheat diseases?",
    "How to prevent pest damage in crops?",
    "What fertilizers are recommended for grain cultivation?"
]

for q in questions:
    answer = graph_rag.query(q)
    print(f"Q: {q}\nA: {answer}\n")
```

### Example 2: Entity Exploration

```python
# Find all diseases
diseases = graph_rag.neo4j.execute_cypher(
    "MATCH (n:Disease) RETURN n.name as name, n.description as description"
)

for disease in diseases:
    print(f"{disease['name']}: {disease['description']}")
```

### Example 3: Graph Visualization

Use Neo4j Browser at your Aura instance URL:

```cypher
// Visualize crop-disease relationships
MATCH (c:Crop)-[r]->(d:Disease)
RETURN c, r, d
LIMIT 25
```

## Performance Tips

1. **Create Indexes**: Done automatically on first run
2. **Batch Processing**: Ingest documents in batches
3. **Cache Results**: Use conversation history for context
4. **Adjust Top-K**: Lower values for faster responses

## License

MIT License

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Support

For issues and questions, please check:
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Ollama Documentation](https://ollama.ai/docs)
