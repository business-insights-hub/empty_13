"""
Full Document Ingestion - Process All PDFs
"""
import os
import glob
from demo_graph_rag import SimpleGraphRAG
from loguru import logger
from tqdm import tqdm


def ingest_all_documents():
    """Process all PDFs in dataset folder"""

    print("\n" + "="*70)
    print("FULL DOCUMENT INGESTION - Graph RAG Agricultural Database")
    print("="*70 + "\n")

    # Initialize Graph RAG
    rag = SimpleGraphRAG()

    try:
        # Find all PDFs
        pdf_files = glob.glob("dataset/*.pdf")
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        total_chunks = 0
        total_entities = 0
        total_vectors = 0

        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_path)
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

            print(f"\n{'='*70}")
            print(f"Processing [{i}/{len(pdf_files)}]: {filename}")
            print(f"Size: {file_size_mb:.1f} MB")
            print(f"{'='*70}")

            try:
                # Step 1: Extract text (limit pages for large files)
                max_pages = 10 if file_size_mb < 5 else 5
                logger.info(f"Extracting text (max {max_pages} pages)...")
                text = rag.extract_text_from_pdf(pdf_path, max_pages=max_pages)

                if not text.strip():
                    logger.warning(f"No text extracted from {filename}, skipping")
                    continue

                # Step 2: Chunk text
                logger.info("Chunking text...")
                chunks = rag.chunk_text(text, chunk_size=400)
                total_chunks += len(chunks)

                # Step 3: Extract entities (from selected chunks to save time)
                # Process every 3rd chunk, or first 5 chunks for small docs
                chunks_to_process = chunks[::3][:5] if len(chunks) > 5 else chunks

                logger.info(f"Extracting entities from {len(chunks_to_process)} chunks...")
                all_entities = []
                all_relationships = []

                for chunk_idx, chunk in enumerate(tqdm(chunks_to_process, desc="Entity extraction")):
                    entities_data = rag.extract_entities_with_llm(chunk)
                    all_entities.extend(entities_data.get('entities', []))
                    all_relationships.extend(entities_data.get('relationships', []))

                # Deduplicate entities by name
                unique_entities = {}
                for entity in all_entities:
                    name = entity.get('name', '').lower()
                    if name and name not in unique_entities:
                        unique_entities[name] = entity

                logger.info(f"Found {len(unique_entities)} unique entities")
                total_entities += len(unique_entities)

                # Step 4: Store in Neo4j
                if unique_entities:
                    logger.info("Storing entities in Neo4j...")
                    rag.store_in_neo4j(
                        list(unique_entities.values()),
                        all_relationships
                    )

                # Step 5: Store in Pinecone (limit to 10 chunks to save space)
                chunks_for_vectors = chunks[:10]
                logger.info(f"Generating embeddings for {len(chunks_for_vectors)} chunks...")

                # Generate ASCII-safe doc_id using hash to avoid non-ASCII character issues
                import hashlib
                doc_id = hashlib.md5(filename.encode('utf-8')).hexdigest()[:16]
                rag.store_in_pinecone(chunks_for_vectors, doc_id, original_filename=filename)
                total_vectors += len(chunks_for_vectors)

                print(f"✓ Completed: {len(chunks)} chunks, {len(unique_entities)} entities")

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue

        # Final statistics
        print("\n" + "="*70)
        print("INGESTION COMPLETE")
        print("="*70)

        stats = rag.get_statistics()

        print(f"\n📊 Final Statistics:")
        print(f"  Documents Processed: {len(pdf_files)}")
        print(f"  Total Chunks Created: {total_chunks}")
        print(f"  Entities Extracted: {total_entities}")
        print(f"  Neo4j Nodes: {stats['neo4j_nodes']}")
        print(f"  Neo4j Relationships: {stats['neo4j_relationships']}")
        print(f"  Pinecone Vectors: {stats['pinecone_vectors']}")

        print(f"\n{'='*70}")
        print("✓ All documents successfully ingested into Graph RAG system!")
        print(f"{'='*70}\n")

    finally:
        rag.close()


if __name__ == "__main__":
    ingest_all_documents()
