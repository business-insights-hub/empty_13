"""
Entity Extraction Module for Graph RAG
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import re


class EntityExtractor:
    """Extract entities and relationships from text"""

    def __init__(self, llm_handler):
        """
        Initialize entity extractor

        Args:
            llm_handler: LLM handler (e.g., OllamaHandler)
        """
        self.llm = llm_handler

    def extract_from_text(self, text: str, chunk_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract entities and relationships from text

        Args:
            text: Input text
            chunk_id: Optional chunk ID for tracking

        Returns:
            Dictionary with entities and relationships
        """
        # Use LLM to extract entities
        result = self.llm.extract_entities(text)

        # Add source tracking
        if chunk_id:
            for entity in result.get("entities", []):
                entity["source_chunk"] = chunk_id
            for rel in result.get("relationships", []):
                rel["source_chunk"] = chunk_id

        return result

    def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple text chunks

        Args:
            chunks: List of chunks with 'id' and 'text' fields

        Returns:
            List of extraction results
        """
        results = []

        for chunk in chunks:
            chunk_id = chunk.get("id")
            text = chunk.get("text", "")

            if not text.strip():
                continue

            logger.debug(f"Extracting from chunk: {chunk_id}")

            try:
                result = self.extract_from_text(text, chunk_id)
                result["chunk_id"] = chunk_id
                results.append(result)
            except Exception as e:
                logger.error(f"Error extracting from chunk {chunk_id}: {e}")

        return results

    def merge_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple extraction results, deduplicating entities

        Args:
            extractions: List of extraction results

        Returns:
            Merged result
        """
        merged_entities = {}
        merged_relationships = []

        for extraction in extractions:
            # Merge entities (deduplicate by name)
            for entity in extraction.get("entities", []):
                name = entity.get("name", "").lower()
                if name and name not in merged_entities:
                    merged_entities[name] = entity
                elif name:
                    # Merge descriptions if both exist
                    existing = merged_entities[name]
                    new_desc = entity.get("description", "")
                    if new_desc and new_desc not in existing.get("description", ""):
                        existing["description"] = f"{existing.get('description', '')} {new_desc}".strip()

            # Collect all relationships
            merged_relationships.extend(extraction.get("relationships", []))

        # Deduplicate relationships
        unique_rels = []
        seen_rels = set()

        for rel in merged_relationships:
            rel_key = (
                rel.get("from", "").lower(),
                rel.get("to", "").lower(),
                rel.get("type", "")
            )

            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)

        return {
            "entities": list(merged_entities.values()),
            "relationships": unique_rels
        }


class DocumentChunker:
    """Split documents into chunks for processing"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document chunker

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks

        Args:
            text: Input text
            doc_id: Optional document ID

        Returns:
            List of chunks with metadata
        """
        chunks = []
        text_length = len(text)

        start = 0
        chunk_num = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence end markers
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)

                if break_point > start:
                    end = break_point + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "doc_id": doc_id,
                    "chunk_num": chunk_num
                })

                chunk_num += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end < text_length else text_length

        logger.debug(f"Created {len(chunks)} chunks from text of length {text_length}")
        return chunks

    def chunk_by_paragraphs(self, text: str, max_chunk_size: int = 1000,
                           doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks by paragraphs

        Args:
            text: Input text
            max_chunk_size: Maximum chunk size
            doc_id: Optional document ID

        Returns:
            List of chunks
        """
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_num = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds max size, save current chunk
            if current_chunk and len(current_chunk) + len(para) > max_chunk_size:
                chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk.strip(),
                    "doc_id": doc_id,
                    "chunk_num": chunk_num
                })
                chunk_num += 1
                current_chunk = para
            else:
                current_chunk += f"\n\n{para}" if current_chunk else para

        # Add remaining chunk
        if current_chunk:
            chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "doc_id": doc_id,
                "chunk_num": chunk_num
            })

        logger.debug(f"Created {len(chunks)} paragraph-based chunks")
        return chunks


class PDFProcessor:
    """Process PDF documents"""

    def __init__(self):
        """Initialize PDF processor"""
        try:
            import pymupdf
            self.pymupdf = pymupdf
        except ImportError:
            logger.warning("PyMuPDF not available, trying pypdf")
            try:
                from pypdf import PdfReader
                self.PdfReader = PdfReader
                self.pymupdf = None
            except ImportError:
                raise ImportError("Neither PyMuPDF nor pypdf is available")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        if self.pymupdf:
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        doc = self.pymupdf.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text

    def _extract_with_pypdf(self, pdf_path: str) -> str:
        """Extract text using pypdf"""
        reader = self.PdfReader(pdf_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        logger.info(f"Extracted {len(text)} characters from PDF")
        return text

    def process_pdf(self, pdf_path: str, chunker: DocumentChunker) -> List[Dict[str, Any]]:
        """
        Process PDF into chunks

        Args:
            pdf_path: Path to PDF file
            chunker: DocumentChunker instance

        Returns:
            List of text chunks
        """
        text = self.extract_text_from_pdf(pdf_path)

        # Get document ID from filename
        import os
        doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

        # Chunk the text
        chunks = chunker.chunk_text(text, doc_id)

        return chunks
