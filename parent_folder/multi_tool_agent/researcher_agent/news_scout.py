from google.adk.agents import Agent

def mock_search_tool(query: str) -> str:
    """Mock search tool for testing purposes - searches for news about cryptocurrency"""
    return f"""ARTICLE 1:
Title: Mock News Article about {query}
Content: This is a mock news article for testing purposes. In a real implementation, this would fetch actual news data.
Published Date: 2024-06-18
Source: Mock News Source
URL: https://example.com/mock-article

ARTICLE 2:
Title: Another Mock Article about {query}
Content: This is another mock article to demonstrate the news search functionality.
Published Date: 2024-06-17
Source: Mock Financial Times
URL: https://example.com/mock-article-2"""

news_scout_agent = Agent(
    name="news_scout_agent",
    model="gemini-2.0-flash",
    description="Agent to find today's news about a crypto currency",
    instruction="""Find recent news about the given cryptocurrency. ALWAYS format each article exactly as follows:

ARTICLE [NUMBER]:
Title: [Article headline]
Content: [Brief summary]
Published Date: [YYYY-MM-DD]
Source: [Publication name]
URL: [Full article URL]""",
    tools=[mock_search_tool],
    output_key="news_articles",
)
