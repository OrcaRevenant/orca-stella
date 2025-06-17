from google.adk.agents import Agent

refiner_agent = Agent(
    name="refiner_agent",
    model="gemini-2.0-flash",
    description="Agent to refine news analysis based on critic feedback.",
    instruction=(
        "You are a news analysis refiner agent. Your role is to improve the analysis based on feedback from the critic_agent. "
        "Read the original news articles from state['news_articles'], the previous analysis from state['news_analysis'], "
        "and the critique feedback from state['critique_decision']. "
        "Create an improved analysis that addresses the critic's concerns while maintaining accuracy. "
        "Focus on: "
        "1. Providing clearer reasoning for bullish/bearish sentiment "
        "2. Addressing any missed important details "
        "3. Ensuring the conclusion is well-supported by evidence "
        "4. Making the analysis more comprehensive and actionable "
        "Output your refined analysis with a clear bullish/bearish/neutral conclusion."
    ),
    output_key="news_analysis",
)
