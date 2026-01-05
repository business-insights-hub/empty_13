"""
Simple test script for Graph RAG without complex dependencies
"""
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment
load_dotenv()

def test_neo4j():
    """Test Neo4j connection"""
    try:
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        logger.info(f"Testing Neo4j connection to {uri}")

        driver = GraphDatabase.driver(uri, auth=(username, password))

        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            assert record["test"] == 1

        driver.close()
        logger.success("✓ Neo4j connection successful!")
        return True

    except Exception as e:
        logger.error(f"✗ Neo4j connection failed: {e}")
        return False


def test_pinecone():
    """Test Pinecone connection"""
    try:
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")

        logger.info(f"Testing Pinecone connection to index: {index_name}")

        pc = Pinecone(api_key=api_key)
        indexes = [idx.name for idx in pc.list_indexes()]

        if index_name in indexes:
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            logger.success(f"✓ Pinecone connection successful! Vectors: {stats.total_vector_count}")
            return True
        else:
            logger.warning(f"Index '{index_name}' not found. Available: {indexes}")
            return False

    except Exception as e:
        logger.error(f"✗ Pinecone connection failed: {e}")
        return False


def test_ollama():
    """Test Ollama connection"""
    try:
        import ollama

        logger.info("Testing Ollama connection...")

        client = ollama.Client()
        models = client.list()

        model_names = [m['name'] for m in models.get('models', [])]
        logger.success(f"✓ Ollama connection successful! Models: {len(model_names)}")

        if model_names:
            logger.info(f"Available models: {', '.join(model_names[:3])}")
        else:
            logger.warning("No models found. Run 'ollama pull llama3.1'")

        return True

    except Exception as e:
        logger.error(f"✗ Ollama connection failed: {e}")
        logger.info("Make sure Ollama is running: ollama serve")
        return False


def test_pdf_reading():
    """Test PDF reading capability"""
    try:
        import fitz  # PyMuPDF
        import os

        logger.info("Testing PDF reading...")

        # Check if dataset exists
        pdf_files = [f for f in os.listdir("dataset") if f.endswith(".pdf")]

        if not pdf_files:
            logger.warning("No PDF files found in dataset/")
            return False

        # Try to read first PDF
        test_pdf = os.path.join("dataset", pdf_files[0])
        doc = fitz.open(test_pdf)

        text = ""
        for page in doc:
            text += page.get_text()

        doc.close()

        logger.success(f"✓ PDF reading successful! Read {len(text)} characters from {pdf_files[0]}")
        return True

    except Exception as e:
        logger.error(f"✗ PDF reading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Graph RAG System Tests")
    print("="*60 + "\n")

    results = {
        "Neo4j Connection": test_neo4j(),
        "Pinecone Connection": test_pinecone(),
        "Ollama Connection": test_ollama(),
        "PDF Reading": test_pdf_reading(),
    }

    print("\n" + "="*60)
    print("Test Results Summary:")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    print("="*60 + "\n")

    passed_count = sum(results.values())
    total_count = len(results)

    if passed_count == total_count:
        logger.success(f"All tests passed! ({passed_count}/{total_count})")
    else:
        logger.warning(f"Some tests failed. ({passed_count}/{total_count} passed)")


if __name__ == "__main__":
    main()
