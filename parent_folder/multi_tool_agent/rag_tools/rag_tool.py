import os
from typing import Any, Dict
from .rag_system import RAGSystem
import logging

logger = logging.getLogger(__name__)

def search_documents(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for relevant information in the document corpus based on a query.

    Args:
        query: The search query or question to find relevant documents for
        max_results: Maximum number of relevant document chunks to return (default: 5)

    Returns:
        Dictionary containing search results with relevant document chunks and metadata
    """
    try:
        # Get or create global RAG instance
        if not hasattr(search_documents, "_rag_instance"):
            search_documents._rag_instance = RAGSystem()

            # Auto-initialize if corpus exists
            corpus_path = "./corpus"
            if os.path.exists(corpus_path):
                try:
                    if search_documents._rag_instance.collection.count() == 0:
                        logger.info(
                            f"Auto-initializing RAG with documents from {corpus_path}"
                        )
                        stats = search_documents._rag_instance.ingest_documents(
                            corpus_path
                        )
                        logger.info(f"RAG initialized: {stats}")
                except Exception as e:
                    logger.error(f"Auto-initialization failed: {e}")

        rag = search_documents._rag_instance

        # Perform search
        results = rag.search(query, top_k=max_results)

        if not results:
            return {
                "success": False,
                "message": "No relevant documents found. Please ensure documents are ingested in the corpus.",
                "results": [],
            }

        # Format results for agent consumption
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "content": result["content"],
                    "source": result["metadata"]["filename"],
                    "similarity_score": round(result["similarity_score"], 3),
                    "chunk_index": result["metadata"]["chunk_index"],
                }
            )

        return {
            "success": True,
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        logger.error(f"Error in document search: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to search documents",
        }


def get_document_context(query: str, max_context_length: int = 4000) -> Dict[str, Any]:
    """
    Get comprehensive context from documents for answering a specific query.
    This function combines multiple relevant document chunks into a single context.

    Args:
        query: The question or topic to get context for
        max_context_length: Maximum length of the combined context (default: 4000)

    Returns:
        Dictionary containing the combined context and metadata
    """
    try:
        # Get or create global RAG instance
        if not hasattr(get_document_context, "_rag_instance"):
            get_document_context._rag_instance = RAGSystem()

            # Auto-initialize if corpus exists
            corpus_path = "./corpus"
            if os.path.exists(corpus_path):
                try:
                    if get_document_context._rag_instance.collection.count() == 0:
                        logger.info(
                            f"Auto-initializing RAG with documents from {corpus_path}"
                        )
                        stats = get_document_context._rag_instance.ingest_documents(
                            corpus_path
                        )
                        logger.info(f"RAG initialized: {stats}")
                except Exception as e:
                    logger.error(f"Auto-initialization failed: {e}")

        rag = get_document_context._rag_instance

        # Get context
        context = rag.get_context_for_query(query, max_context_length)

        if context == "No relevant documents found.":
            return {
                "success": False,
                "message": "No relevant documents found for the query",
                "context": "",
            }

        return {
            "success": True,
            "query": query,
            "context": context,
            "context_length": len(context),
        }

    except Exception as e:
        logger.error(f"Error getting document context: {e}")
        return {"success": False, "error": str(e), "context": ""}

def get_rag_stats() -> Dict[str, Any]:
    """
    Get statistics about the current RAG system state.

    Returns:
        Dictionary containing RAG system statistics
    """
    try:
        # Get or create global RAG instance
        if not hasattr(get_rag_stats, "_rag_instance"):
            get_rag_stats._rag_instance = RAGSystem()

        rag = get_rag_stats._rag_instance
        stats = rag.get_stats()

        return {"success": True, "statistics": stats}

    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        return {"success": False, "error": str(e)}


# Export the tools for easy import
__all__ = [
    "search_documents",
    "get_document_context",
    "get_rag_stats",
]
