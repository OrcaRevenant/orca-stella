from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.loop_agent import LoopAgent
from .news_scout import news_scout_agent
from .analyzer import analyzer_agent
from .critic import critic_agent
from .refiner import refiner_agent
from .quality_checker import QualityCheckerAgent
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

refinement_loop = LoopAgent(
    name="analysis_refinement_loop",
    max_iterations=3,  # Limit to 3 refinement cycles
    sub_agents=[
        critic_agent,  # Critique the analysis
        QualityCheckerAgent(
            name="quality_checker_agent"
        ),  # Check if approved and escalate to stop
        refiner_agent,  # Refine based on feedback (only runs if not approved)
    ],
)


analyzer_agent_sequential = SequentialAgent(
    name="analyzer_agent_sequential",
    description="Agent to find the latest news about a crypto currency and analyze it for crypto sentiment.",
    sub_agents=[
        news_scout_agent,
        analyzer_agent,
        refinement_loop,
    ],
)

researcher_agent = Agent(
    name="researcher_agent",
    description="Agent to find the latest news about a crypto currency and analyze it for crypto sentiment.",
    instruction="""You are a research agent with access to two specific tools: try to give detailed answer for the user's question use tools responses to fulfill the user's question with details.
        1. analyzer_agent_sequential - Use ONLY when user explicitly asks for ANALYSIS or SENTIMENT analysis
        2. news_scout_agent - Use ONLY when user asks for NEWS or latest updates
""",
    tools=[
        AgentTool(agent=news_scout_agent),
        AgentTool(agent=analyzer_agent_sequential),
    ],
)
