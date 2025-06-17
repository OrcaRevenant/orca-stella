from google.adk.agents import Agent
from .orca_protocol import get_orca_protocol_toolset

orca_protocol_toolset = get_orca_protocol_toolset()

orca_protocol_agent = Agent(
    name="orca_protocol_agent",
    model="gemini-2.0-flash",
    description="Agent to use tools to interact with the Orca Protocol API",
    instruction="""You are an agent that can use the tools to interact with the Orca Protocol API.
    """,
    tools=[orca_protocol_toolset.get_tool("get_transactions")],
)
