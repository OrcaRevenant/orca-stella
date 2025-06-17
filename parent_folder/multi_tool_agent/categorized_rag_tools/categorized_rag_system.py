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


class CategorizedRAGSystem:
    """
    A categorized RAG system that supports multiple separate document collections.
    Each category acts like a separate database table with complete isolation.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        db_path: str = "./categorized_chromadb"
    ):
        """
        Initialize the categorized RAG system.

        Args:
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks
            db_path: Path to store ChromaDB data
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_path = db_path

        # Initialize sentence transformer for embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True)
        )

        # Dictionary to store collection references
        self.collections = {}

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info("Categorized RAG system initialized successfully")

    def create_category(self, category_name: str, description: str = "") -> bool:
        """
        Create a new document category (like creating a new database table).

        Args:
            category_name: Name of the category
            description: Optional description of the category

        Returns:
            True if category was created successfully
        """
        try:
            # Sanitize category name for ChromaDB
            safe_name = category_name.lower().replace(" ", "_").replace("-", "_")

            # Create collection
            collection = self.client.get_or_create_collection(
                name=safe_name,
                metadata={
                    "description": description or f"Documents for {category_name}",
                    "category_name": category_name,
                    "created_by": "CategorizedRAGSystem"
                }
            )

            self.collections[category_name] = collection
            logger.info(f"Created category: {category_name} (collection: {safe_name})")
            return True

        except Exception as e:
            logger.error(f"Failed to create category {category_name}: {e}")
            return False

    def list_categories(self) -> List[Dict[str, Any]]:
        """
        List all available categories.

        Returns:
            List of category information
        """
        categories = []

        # Get all collections from ChromaDB
        all_collections = self.client.list_collections()

        for collection in all_collections:
            # Check if it's one of our categorized collections
            metadata = collection.metadata or {}
            if metadata.get("created_by") == "CategorizedRAGSystem":
                category_info = {
                    "name": metadata.get("category_name", collection.name),
                    "collection_name": collection.name,
                    "description": metadata.get("description", ""),
                    "document_count": collection.count()
                }
                categories.append(category_info)

                # Also add to our collections dict if not already there
                category_name = metadata.get("category_name", collection.name)
                if category_name not in self.collections:
                    self.collections[category_name] = collection

        return categories

    def get_category_stats(self, category_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific category.

        Args:
            category_name: Name of the category

        Returns:
            Dictionary with category statistics
        """
        if category_name not in self.collections:
            # Try to load the category
            categories = self.list_categories()
            found_category = next((c for c in categories if c["name"] == category_name), None)
            if not found_category:
                return {"error": f"Category '{category_name}' not found"}

        collection = self.collections[category_name]
        return {
            "category_name": category_name,
            "total_documents": collection.count(),
            "collection_name": collection.name,
            "metadata": collection.metadata
        }

    def ingest_documents_to_category(
        self,
        category_name: str,
        corpus_path: str,
        auto_create_category: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest documents into a specific category.

        Args:
            category_name: Name of the category to ingest documents into
            corpus_path: Path to directory containing documents
            auto_create_category: Whether to create category if it doesn't exist

        Returns:
            Dictionary with ingestion statistics
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise ValueError(f"Corpus path does not exist: {corpus_path}")

        # Ensure category exists
        if category_name not in self.collections:
            if auto_create_category:
                self.create_category(category_name, f"Auto-created for documents in {corpus_path}")
            else:
                raise ValueError(f"Category '{category_name}' does not exist")

        collection = self.collections[category_name]
        supported_extensions = ['.pdf', '.docx', '.txt', '.md']
        files_processed = 0
        chunks_created = 0

        logger.info(f"Starting document ingestion to category '{category_name}' from: {corpus_path}")

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

                # Create embeddings and store in the category's collection
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    # Generate embedding
                    embedding = self.embedding_model.encode([chunk])[0].tolist()

                    # Create unique ID for chunk
                    chunk_id = f"{category_name}_{file_path.name}_{i}"

                    # Store in ChromaDB collection for this category
                    collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "source": str(file_path),
                            "filename": file_path.name,
                            "chunk_index": i,
                            "file_type": file_path.suffix.lower(),
                            "category": category_name
                        }],
                        ids=[chunk_id]
                    )

                    chunks_created += 1

                files_processed += 1

        stats = {
            "category_name": category_name,
            "files_processed": files_processed,
            "chunks_created": chunks_created,
            "total_documents_in_category": collection.count()
        }

        logger.info(f"Ingestion complete for category '{category_name}': {stats}")
        return stats

    def search_in_category(
        self,
        category_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents within a specific category only.

        Args:
            category_name: Name of the category to search in
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant document chunks with metadata
        """
        if category_name not in self.collections:
            # Try to load the category
            categories = self.list_categories()
            found_category = next((c for c in categories if c["name"] == category_name), None)
            if not found_category:
                logger.warning(f"Category '{category_name}' not found")
                return []

        collection = self.collections[category_name]

        if collection.count() == 0:
            logger.warning(f"No documents in category '{category_name}'")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in the specific category's collection only
        results = collection.query(
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
                "similarity_score": 1 - results['distances'][0][i],
                "category": category_name
            })

        return formatted_results

    def get_context_for_query_in_category(
        self,
        category_name: str,
        query: str,
        max_context_length: int = 4000
    ) -> str:
        """
        Get relevant context for a query from a specific category only.

        Args:
            category_name: Name of the category to search in
            query: User query
            max_context_length: Maximum length of context to return

        Returns:
            Combined context string from the specified category
        """
        results = self.search_in_category(category_name, query, top_k=10)

        if not results:
            return f"No relevant documents found in category '{category_name}'."

        context_parts = []
        current_length = 0

        for result in results:
            content = result['content']
            source = result['metadata']['filename']

            # Add source reference with category
            content_with_source = f"[Category: {category_name}, Source: {source}]\n{content}\n"

            if current_length + len(content_with_source) <= max_context_length:
                context_parts.append(content_with_source)
                current_length += len(content_with_source)
            else:
                break

        return "\n".join(context_parts)

    def search_across_categories(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        top_k_per_category: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple categories and return results grouped by category.

        Args:
            query: Search query
            categories: List of categories to search (if None, searches all)
            top_k_per_category: Number of results per category

        Returns:
            Dictionary with category names as keys and results as values
        """
        if categories is None:
            categories = [cat["name"] for cat in self.list_categories()]

        results_by_category = {}

        for category_name in categories:
            try:
                results = self.search_in_category(category_name, query, top_k_per_category)
                if results:
                    results_by_category[category_name] = results
            except Exception as e:
                logger.error(f"Error searching in category {category_name}: {e}")

        return results_by_category

    def delete_category(self, category_name: str) -> bool:
        """
        Delete a category and all its documents.

        Args:
            category_name: Name of the category to delete

        Returns:
            True if category was deleted successfully
        """
        try:
            if category_name in self.collections:
                collection = self.collections[category_name]
                collection_name = collection.name

                # Delete the collection from ChromaDB
                self.client.delete_collection(collection_name)

                # Remove from our collections dict
                del self.collections[category_name]

                logger.info(f"Deleted category: {category_name}")
                return True
            else:
                logger.warning(f"Category '{category_name}' not found")
                return False

        except Exception as e:
            logger.error(f"Failed to delete category {category_name}: {e}")
            return False

    # Include the file reading methods from the original RAG system
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
