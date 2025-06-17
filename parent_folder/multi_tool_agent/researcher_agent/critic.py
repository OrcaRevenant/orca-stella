from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

critic_agent = Agent(
    name="critic_agent",
    model="gemini-2.0-flash",
    description="Agent to critique and validate news analysis for accuracy and quality.",
    instruction=(
        "You are a financial news critique agent. Your role is to review the analysis from the analyzer_agent "
        "and determine if the bullish/bearish sentiment assessment is accurate and well-reasoned. "
        "Review the news articles in state['news_articles'] and the analysis in state['news_analysis']. "
        "Check if: "
        "1. The sentiment analysis is supported by the actual news content "
        "2. The reasoning is sound and considers market impact "
        "3. The conclusion is clear (bullish, bearish, or neutral) "
        "4. Important details weren't missed "
        "Output 'APPROVED' if the analysis is good, or 'NEEDS_REFINEMENT' with specific feedback if it needs improvement."
    ),
    output_key="critique_decision",
)
