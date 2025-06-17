import os
from typing import Any, Dict, Optional, List
from .categorized_rag_system import CategorizedRAGSystem
import logging

logger = logging.getLogger(__name__)

def search_documents_in_category(
    category_name: str,
    query: str,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search for relevant information within a specific document category only.

    Args:
        category_name: Name of the category to search in
        query: The search query or question
        max_results: Maximum number of relevant document chunks to return

    Returns:
        Dictionary containing search results from the specified category
    """
    try:
        # Get or create global categorized RAG instance
        if not hasattr(search_documents_in_category, "_cat_rag_instance"):
            search_documents_in_category._cat_rag_instance = CategorizedRAGSystem()

        cat_rag = search_documents_in_category._cat_rag_instance

        # Perform search in specific category
        results = cat_rag.search_in_category(category_name, query, top_k=max_results)

        if not results:
            return {
                "success": False,
                "message": f"No relevant documents found in category '{category_name}'",
                "category": category_name,
                "results": []
            }

        # Format results for agent consumption
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result["content"],
                "source": result["metadata"]["filename"],
                "similarity_score": round(result["similarity_score"], 3),
                "chunk_index": result["metadata"]["chunk_index"],
                "category": result["category"]
            })

        return {
            "success": True,
            "query": query,
            "category": category_name,
            "total_results": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"Error searching in category: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to search in category '{category_name}'"
        }


def get_context_from_category(
    category_name: str,
    query: str,
    max_context_length: int = 4000
) -> Dict[str, Any]:
    """
    Get comprehensive context from a specific category for answering a query.

    Args:
        category_name: Name of the category to get context from
        query: The question or topic to get context for
        max_context_length: Maximum length of the combined context

    Returns:
        Dictionary containing the combined context from the specified category
    """
    try:
        # Get or create global categorized RAG instance
        if not hasattr(get_context_from_category, "_cat_rag_instance"):
            get_context_from_category._cat_rag_instance = CategorizedRAGSystem()

        cat_rag = get_context_from_category._cat_rag_instance

        # Get context from specific category
        context = cat_rag.get_context_for_query_in_category(
            category_name,
            query,
            max_context_length
        )

        if context.startswith("No relevant documents found"):
            return {
                "success": False,
                "message": f"No relevant documents found in category '{category_name}'",
                "category": category_name,
                "context": ""
            }

        return {
            "success": True,
            "query": query,
            "category": category_name,
            "context": context,
            "context_length": len(context)
        }

    except Exception as e:
        logger.error(f"Error getting context from category: {e}")
        return {
            "success": False,
            "error": str(e),
            "context": ""
        }


def search_across_categories(
    query: str,
    categories: Optional[List[str]] = None,
    max_results_per_category: int = 3
) -> Dict[str, Any]:
    """
    Search across multiple categories and return results grouped by category.

    Args:
        query: The search query
        categories: List of categories to search (if None, searches all)
        max_results_per_category: Number of results per category

    Returns:
        Dictionary containing search results grouped by category
    """
    try:
        # Get or create global categorized RAG instance
        if not hasattr(search_across_categories, "_cat_rag_instance"):
            search_across_categories._cat_rag_instance = CategorizedRAGSystem()

        cat_rag = search_across_categories._cat_rag_instance

        # Convert categories string to list if needed
        if categories and isinstance(categories, str):
            categories = [cat.strip() for cat in categories.split(",")]

        # Search across categories
        results_by_category = cat_rag.search_across_categories(
            query,
            categories,
            top_k_per_category=max_results_per_category
        )

        # Format results
        formatted_results = {}
        total_results = 0

        for category_name, results in results_by_category.items():
            formatted_category_results = []
            for result in results:
                formatted_category_results.append({
                    "content": result["content"],
                    "source": result["metadata"]["filename"],
                    "similarity_score": round(result["similarity_score"], 3),
                    "chunk_index": result["metadata"]["chunk_index"]
                })

            formatted_results[category_name] = formatted_category_results
            total_results += len(formatted_category_results)

        return {
            "success": True,
            "query": query,
            "categories_searched": list(results_by_category.keys()),
            "total_results": total_results,
            "results_by_category": formatted_results
        }

    except Exception as e:
        logger.error(f"Error searching across categories: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to search across categories"
        }


# Export the tools for easy import
__all__ = [
    "search_documents_in_category",
    "get_context_from_category",
    "search_across_categories",
]
