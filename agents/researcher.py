from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent
from tools.tool_factory import ToolFactory
from config.settings import RESEARCHER_PROMPT

class ResearcherAgent(BaseAgent):
    """
    Researcher agent that gathers information using search tools.
    Specializes in information retrieval and synthesis.
    """
    
    def process(self, state: MessagesState) -> Command[Literal["validator"]]:
        """
        Process the current state to research and gather information.
        
        Args:
            state: The current workflow state
            
        Returns:
            A Command object routing to the validator with research results
        """
        # Create research tools
        research_tools = ToolFactory.create_research_tools()
        
        # Create a ReAct agent for research
        research_agent = create_react_agent(
            self.llm,
            tools=research_tools,
            state_modifier=RESEARCHER_PROMPT
        )
        
        # Invoke the research agent
        result = research_agent.invoke(state)
        
        # Log the transition
        self.log_transition("validator")
        
        # Return command with updated state and next destination
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content,
                        name="researcher"
                    )
                ]
            },
            goto="validator"
        )