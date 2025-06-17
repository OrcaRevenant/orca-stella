from google.adk.agents import Agent

analyzer_agent = Agent(
    name="analyzer_agent",
    model="gemini-2.0-flash",
    description="Agent to analyze news articles for crypto sentiment.",
    instruction=(
        "You are a cryptocurrency news analyzer agent. Your role is to analyze news articles and determine "
        "if they indicate bullish, bearish, or neutral sentiment for the cryptocurrency. "
        "Read the news articles from state['news_articles']. "
        "Provide a comprehensive analysis that includes: "
        "1. Overall sentiment (BULLISH/BEARISH/NEUTRAL) "
        "2. Key factors supporting this sentiment "
        "3. Potential market impact "
        "4. Confidence level (HIGH/MEDIUM/LOW) "
        "5. Supporting evidence from the news articles "
        "Be thorough and analytical in your reasoning. Only analyze when news articles are available."
    ),
    output_key="news_analysis",
)
