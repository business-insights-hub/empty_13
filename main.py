"""
Graph RAG Agricultural Assistant - Main Script
Example usage of the Graph RAG system
"""
from loguru import logger
from src.graph_rag.graph_rag import GraphRAG


def ingest_data_example():
    """Example: Ingest agricultural documents"""
    logger.info("=== Ingesting Agricultural Documents ===")

    with GraphRAG() as graph_rag:
        # Ingest all PDFs from dataset directory
        stats = graph_rag.ingest_directory("dataset", pattern="*.pdf")

        print("\n" + "="*50)
        print("Ingestion Statistics:")
        print("="*50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("="*50)


def query_example():
    """Example: Query the Graph RAG system"""
    logger.info("=== Querying Graph RAG System ===")

    with GraphRAG() as graph_rag:
        # Example queries
        questions = [
            "What are the main crops mentioned in the documents?",
            "How can we prevent plant diseases?",
            "What techniques are recommended for grain cultivation?",
        ]

        for question in questions:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            print(f"{'='*50}")

            answer = graph_rag.query(question, top_k_vector=5, top_k_graph=10)

            print(f"\nAnswer:\n{answer}")
            print(f"{'='*50}\n")


def search_entities_example():
    """Example: Search for entities in the knowledge graph"""
    logger.info("=== Searching Entities ===")

    with GraphRAG() as graph_rag:
        search_terms = ["wheat", "disease", "fertilizer"]

        for term in search_terms:
            print(f"\n{'='*50}")
            print(f"Searching for: {term}")
            print(f"{'='*50}")

            entities = graph_rag.search_entities(term, limit=5)

            for entity in entities:
                labels = ", ".join(entity.get("labels", []))
                props = entity.get("properties", {})
                name = props.get("name", "Unknown")
                desc = props.get("description", "No description")

                print(f"\n- {name} ({labels})")
                print(f"  Description: {desc[:100]}...")


def get_entity_details_example():
    """Example: Get details about a specific entity"""
    logger.info("=== Getting Entity Details ===")

    with GraphRAG() as graph_rag:
        # Search for an entity first
        entities = graph_rag.search_entities("wheat", limit=1)

        if entities:
            entity_name = entities[0]["properties"].get("name", "")

            print(f"\n{'='*50}")
            print(f"Entity Details: {entity_name}")
            print(f"{'='*50}")

            details = graph_rag.get_entity_details(entity_name)

            if details:
                entity = details["entity"]
                related = details["related_entities"]

                print(f"\nEntity: {entity.get('name', 'Unknown')}")
                print(f"Description: {entity.get('description', 'No description')}")

                print(f"\nRelated Entities ({len(related)}):")
                for rel in related[:5]:
                    rel_name = rel.get("properties", {}).get("name", "Unknown")
                    rel_types = " -> ".join(rel.get("relationship_path", []))
                    print(f"  - {rel_name} (via {rel_types})")


def statistics_example():
    """Example: Get system statistics"""
    logger.info("=== System Statistics ===")

    with GraphRAG() as graph_rag:
        stats = graph_rag.get_statistics()

        print("\n" + "="*50)
        print("Knowledge Graph Statistics:")
        print("="*50)

        graph_stats = stats.get("graph", {})
        print(f"Total Nodes: {graph_stats.get('total_nodes', 0)}")
        print(f"Total Relationships: {graph_stats.get('total_relationships', 0)}")

        print("\nNode Counts by Type:")
        for node_type, count in graph_stats.get("node_counts", {}).items():
            print(f"  {node_type}: {count}")

        print("\nRelationship Counts by Type:")
        for rel_type, count in graph_stats.get("relationship_counts", {}).items():
            print(f"  {rel_type}: {count}")

        print("\n" + "="*50)
        print("Vector Store Statistics:")
        print("="*50)

        vector_stats = stats.get("vector_store", {})
        for key, value in vector_stats.items():
            print(f"{key}: {value}")

        print("="*50)


def interactive_chat():
    """Example: Interactive chat with the Graph RAG system"""
    logger.info("=== Interactive Chat Mode ===")

    print("\n" + "="*50)
    print("Agricultural Knowledge Assistant")
    print("="*50)
    print("Ask questions about agricultural topics.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    with GraphRAG() as graph_rag:
        conversation_history = []

        while True:
            try:
                question = input("\nYou: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                # Add to history
                conversation_history.append({
                    "role": "user",
                    "content": question
                })

                # Get answer
                answer = graph_rag.chat(question, conversation_history)

                print(f"\nAssistant: {answer}")

                # Add response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}")


def main():
    """Main function with menu"""
    print("\n" + "="*50)
    print("Graph RAG Agricultural Assistant")
    print("="*50)
    print("\nChoose an option:")
    print("1. Ingest documents from dataset folder")
    print("2. Run example queries")
    print("3. Search for entities")
    print("4. Get entity details")
    print("5. View system statistics")
    print("6. Interactive chat mode")
    print("0. Exit")
    print("="*50 + "\n")

    choice = input("Enter your choice (0-6): ").strip()

    if choice == "1":
        ingest_data_example()
    elif choice == "2":
        query_example()
    elif choice == "3":
        search_entities_example()
    elif choice == "4":
        get_entity_details_example()
    elif choice == "5":
        statistics_example()
    elif choice == "6":
        interactive_chat()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
