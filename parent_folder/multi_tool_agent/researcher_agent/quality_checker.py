from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator


class QualityCheckerAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Check if critic approved the analysis
        critique_decision = ctx.session.state.get("critique_decision", "")
        should_stop = "APPROVED" in critique_decision.upper()

        current_analysis = ctx.session.state.get("news_analysis", "")

        # If approved, escalate to stop the loop
        yield Event(
            author=self.name,
            actions=EventActions(escalate=should_stop),
            content={
                "parts": [
                    {
                        "text": (
                            current_analysis
                            if should_stop
                            else "Analysis requires further refinement."
                        )
                    }
                ]
            },
        )
