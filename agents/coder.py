from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

from agents.base import BaseAgent
from tools.tool_factory import ToolFactory
from config.settings import CODER_PROMPT

class CoderAgent(BaseAgent):
    """
    Coder agent that handles technical tasks related to calculation,
    coding, data analysis, and problem-solving.
    """
    
    def process(self, state: MessagesState) -> Command[Literal["validator"]]:
        """
        Process the current state to perform coding, calculation, or analysis tasks.
        
        Args:
            state: The current workflow state
            
        Returns:
            A Command object routing to the validator with coding results
        """
        # Create coding tools
        coding_tools = ToolFactory.create_coding_tools()
        
        # Create a ReAct agent for coding
        code_agent = create_react_agent(
            self.llm,
            tools=coding_tools,
            state_modifier=CODER_PROMPT
        )
        
        # Invoke the code agent
        result = code_agent.invoke(state)
        
        # Log the transition
        self.log_transition("validator")
        
        # Return command with updated state and next destination
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content,
                        name="coder"
                    )
                ]
            },
            goto="validator"
        )