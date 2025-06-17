import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    A simple and clean RAG (Retrieval-Augmented Generation) system using ChromaDB
    and sentence-transformers for document ingestion and retrieval.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        db_path: str = "./chromadb"
    ):
        """
        Initialize the RAG system.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model name (free and lightweight)
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            db_path: Path to store ChromaDB data
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize sentence transformer for embeddings (free and good quality)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings for RAG"}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info("RAG system initialized successfully")

    def _read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""

    def _read_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            return ""

    def _read_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            return ""

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats."""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            return self._read_pdf(file_path)
        elif file_extension == '.docx':
            return self._read_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._read_txt(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""

    def ingest_documents(self, corpus_path: str) -> Dict[str, Any]:
        """
        Ingest documents from a corpus directory.

        Args:
            corpus_path: Path to directory containing documents

        Returns:
            Dictionary with ingestion statistics
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise ValueError(f"Corpus path does not exist: {corpus_path}")

        supported_extensions = ['.pdf', '.docx', '.txt', '.md']
        files_processed = 0
        chunks_created = 0

        logger.info(f"Starting document ingestion from: {corpus_path}")

        # Process all supported files in the corpus directory
        for file_path in corpus_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing file: {file_path}")

                # Extract text from file
                text = self._extract_text_from_file(str(file_path))
                if not text.strip():
                    logger.warning(f"No text extracted from {file_path}")
                    continue

                # Split text into chunks
                chunks = self.text_splitter.split_text(text)

                # Create embeddings and store in ChromaDB
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    # Generate embedding
                    embedding = self.embedding_model.encode([chunk])[0].tolist()

                    # Create unique ID for chunk
                    chunk_id = f"{file_path.name}_{i}"

                    # Store in ChromaDB
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "source": str(file_path),
                            "filename": file_path.name,
                            "chunk_index": i,
                            "file_type": file_path.suffix.lower()
                        }],
                        ids=[chunk_id]
                    )

                    chunks_created += 1

                files_processed += 1

        stats = {
            "files_processed": files_processed,
            "chunks_created": chunks_created,
            "total_documents_in_db": self.collection.count()
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on a query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant document chunks with metadata
        """
        if self.collection.count() == 0:
            logger.warning("No documents in collection. Please ingest documents first.")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "similarity_score": 1 - results['distances'][0][i]  # Convert distance to similarity
            })

        return formatted_results

    def get_context_for_query(self, query: str, max_context_length: int = 4000) -> str:
        """
        Get relevant context for a query, combining multiple chunks if needed.

        Args:
            query: User query
            max_context_length: Maximum length of context to return

        Returns:
            Combined context string
        """
        results = self.search(query, top_k=10)

        if not results:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for result in results:
            content = result['content']
            source = result['metadata']['filename']

            # Add source reference
            content_with_source = f"[Source: {source}]\n{content}\n"

            if current_length + len(content_with_source) <= max_context_length:
                context_parts.append(content_with_source)
                current_length += len(content_with_source)
            else:
                break

        return "\n".join(context_parts)

    def reset_collection(self):
        """Reset the document collection (useful for testing)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for RAG"}
        )
        logger.info("Collection reset successfully")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
