from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command

from agents.base import BaseAgent
from config.settings import ENHANCER_PROMPT

class EnhancerAgent(BaseAgent):
    """
    Enhancer agent that refines and clarifies user inputs.
    Improves query quality before further processing.
    """
    
    def process(self, state: MessagesState) -> Command[Literal["supervisor"]]:
        """
        Process the current state to enhance and clarify the user query.
        
        Args:
            state: The current workflow state
            
        Returns:
            A Command object routing back to the supervisor with enhanced query
        """
        # Prepare messages with the enhancer prompt
        messages = self.prepare_messages(ENHANCER_PROMPT, state)
        
        # Get response from the LLM
        enhanced_query = self.llm.invoke(messages)
        
        # Log the transition
        self.log_transition("supervisor")
        
        # Return command with updated state and next destination
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=enhanced_query.content,
                        name="enhancer"
                    )
                ]
            },
            goto="supervisor"
        )