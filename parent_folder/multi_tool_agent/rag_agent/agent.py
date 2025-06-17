from google.adk.agents import Agent
from rag_tools.rag_tool import (
    search_documents,
    get_document_context,
    get_rag_stats
)

rag_agent = Agent(
    name="rag_agent",
    model="gemini-2.0-flash",
    description="Agent specialized in document search and question answering using RAG",
    instruction="""You are a document expert agent that can search through and answer questions based on a corpus of documents.

Your capabilities:
1. Search for relevant information in documents using semantic search
2. Get comprehensive context from multiple document sources
3. Answer questions based on the retrieved information
4. Manage document ingestion and provide system statistics

When a user asks a question:
1. First, use get_document_context to get relevant information for their query
2. If the context is insufficient, use search_documents to find specific information
3. Always base your answers on the retrieved document content
4. Cite your sources by mentioning the document filenames
5. If no relevant information is found, clearly state this

Best practices:
- Always provide source citations
- Be precise and factual based on the documents
- If information is not in the documents, clearly state this limitation
- Combine information from multiple sources when relevant
""",
    tools=[
        search_documents,
        get_document_context,
        get_rag_stats
    ]
)
