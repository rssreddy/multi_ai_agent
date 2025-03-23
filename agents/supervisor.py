from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Command

from agents.base import BaseAgent
from core.models import Supervisor
from config.settings import SUPERVISOR_PROMPT
from utils.logger import logger

class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that routes tasks to the appropriate specialized agent.
    Acts as a coordinator in the workflow.
    """
    
    def process(self, state: MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:
        """
        Process the current state and determine which agent should handle the task next.
        
        Args:
            state: The current workflow state
            
        Returns:
            A Command object routing to the next appropriate agent
        """
        # Prepare messages with the supervisor prompt
        messages = self.prepare_messages(SUPERVISOR_PROMPT, state)
        
        # Get structured output from the LLM
        response = self.llm.with_structured_output(Supervisor).invoke(messages)
        
        # Extract routing decision and reason
        goto = response.next
        reason = response.reason
        
        # Log the transition
        self.log_transition(goto)
        
        # Return command with updated state and next destination
        return Command(
            update={
                "messages": [
                    HumanMessage(content=reason, name="supervisor")
                ]
            },
            goto=goto
        )